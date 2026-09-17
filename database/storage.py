"""Parquet 分区存储 + 断点记录

写入权限规则：
    frozen/  只读原始层 —— 仅下载器可写，须显式 Storage(..., allow_frozen=True)
    cleaned/ 加工层     —— 任何清洗脚本可写
被拒绝的写入会抛 FrozenWriteError，而不是静默污染只读层。

可靠性设计（P2 加固）
--------------------
1. **原子写**：先写 `.tmp`，再 `os.replace()` 换成正式文件。
   进程中断/断电只会留下临时文件，不会产生"半截 parquet"——
   之前直接 `to_parquet(最终路径)`，崩溃后该文件永久损坏且不会被修复
   （因为 `path.exists()` 会让后续运行直接跳过它）。
2. **断点文件容错**：`_checkpoint.json` 损坏时自动备份并重建，
   而不是让之后每次运行都抛 JSONDecodeError。
3. **断点缓存**：避免 mark_done 每次都重读整个 JSON（5000 只股票 = O(n²)）。
4. **磁盘保护不可被吞**：`DiskFullError` 继承 BaseException，
   因此下载器里常见的 `except Exception` 不会把它静默吃掉。
"""
import json
import os
import shutil
from pathlib import Path

import pandas as pd

from .config import (dir_of, recipe_dir, is_frozen, FrozenWriteError,
                     DISK_MIN_FREE, DB_ROOT)

# parquet 文件的魔数：头尾各 4 字节，都是 "PAR1"
_PARQUET_MAGIC = b"PAR1"


def is_valid_parquet(path) -> bool:
    """只读头尾 4 字节判断文件是不是完整的 parquet —— 极快

    为什么值得单独有个函数：非原子写被中断时留下的**半截文件**，头部魔数
    往往还在（`PAR1`），只有尾部被截断成零字节。所以两端都要查。
    实测一个真实的坏文件：`frozen/daily_raw/year=2026/601089.parquet`
    1570 字节，`head=b'PAR1'`、`tail=b'\\x00\\x00\\x00\\x00'`。
    """
    try:
        p = Path(path)
        size = p.stat().st_size
        if size < 8:
            return False
        with open(p, "rb") as f:
            head = f.read(4)
            f.seek(-4, 2)
            tail = f.read(4)
        return head == _PARQUET_MAGIC and tail == _PARQUET_MAGIC
    except OSError:
        return False


def find_corrupt_parquet(root, pattern="**/*.parquet"):
    """列出目录下所有损坏的 parquet（相对路径）

    只查魔数，不解析内容 —— 25 万文件约 5 分钟，因此 `--check parquet`
    默认只查最近年份，见 `scripts/validate_data.py`。
    """
    bad = []
    for f in Path(root).glob(pattern):
        if not is_valid_parquet(f):
            bad.append(f)
    return bad


def _verify_parquet(path):
    """写完后校验：能不能读出 footer

    比只查魔数更严 —— 顺带能抓住"魔数对但内容坏了"的情况。只读文件尾部的
    元数据，不加载数据，所以很便宜。
    """
    import pyarrow.parquet as pq
    pq.ParquetFile(path).metadata


def atomic_to_parquet(df: pd.DataFrame, path, verify: bool = True):
    """原子 + 可校验地写 parquet：临时文件 -> 校验 -> `os.replace`

    ⚠️ **所有写 parquet 的地方都该用它**，不要直接 `df.to_parquet(最终路径)`。
    直接写最终路径时，进程被杀 / 磁盘满 / 断电都会留下**半截文件**；更糟的是
    读取方常见的 `except Exception: continue` 会把它当成"这只股票没数据"跳过，
    于是坏文件**永远不会被重写**，一直烂到某次全量重建时炸掉。

    这正是 `scripts/daily_update.py::upsert` 踩过的坑：2026-09-17 的更新里
    `601089` 的 2026 分片是个 1570 字节的截断文件，下载阶段被 `except` 吞掉
    只打印一行，最后在重建清洗层时把整个更新任务打挂。

    verify=True 时用 `ParquetFile` 读一次 footer，确保写出来的**真的能读**；
    校验失败会抛异常，此时**最终文件不会被替换**（旧的好数据还在）。
    """
    path = Path(path)
    tmp = path.with_name(f"{path.name}.tmp{os.getpid()}")
    try:
        df.to_parquet(tmp, index=False)
        if verify:
            _verify_parquet(tmp)
        os.replace(tmp, path)
    finally:
        if tmp.exists():
            try:
                tmp.unlink()
            except OSError:
                pass


class Storage:
    """按 年份/代码 分区存储 Parquet，并维护断点文件

    参数:
        dataset:      数据集名（如 'daily_raw'、'daily'）
        recipe:       清洗配方名（显式指定时优先于 dataset）
        allow_frozen: 是否允许写入 frozen 层。**只有下载器**应设为 True；
                      清洗/分析代码保持默认 False，误写会立刻报错
    """

    def __init__(self, dataset: str = None, recipe: str = None,
                 allow_frozen: bool = False, root=None):
        if root is not None:
            # 显式指定目录（主要供测试使用，避免测试污染真实 db/）
            self.root = Path(root)
        elif recipe:
            self.root = recipe_dir(recipe)
        else:
            self.root = dir_of(dataset)
        self.dataset = dataset or recipe
        self.recipe = recipe
        self.layer = "frozen" if is_frozen(self.root) else "cleaned"
        self.allow_frozen = allow_frozen
        self.root.mkdir(parents=True, exist_ok=True)
        self.checkpoint_file = self.root / "_checkpoint.json"
        self._cp_cache = None

    # ---------- 写权限 ----------
    def assert_writable(self):
        """frozen 层默认只读，防止清洗/分析代码污染可信源"""
        if self.layer == "frozen" and not self.allow_frozen:
            raise FrozenWriteError(
                f"{self.root} 属于只读的 frozen 层。\n"
                f"  下载器请用 Storage({self.dataset!r}, allow_frozen=True)\n"
                f"  清洗产物请写入 cleaned/ 下的配方目录（见 database.config.RECIPES）"
            )

    # ---------- 磁盘监控 ----------
    @staticmethod
    def disk_free() -> int:
        """数据库所在卷的剩余空间（失败时返回大值，避免在异常平台误停下载）"""
        try:
            anchor = DB_ROOT.anchor or str(DB_ROOT)
            return shutil.disk_usage(anchor).free
        except Exception:
            return 10 ** 12

    def assert_disk_ok(self):
        free = self.disk_free()
        if free < DISK_MIN_FREE:
            raise DiskFullError(
                f"数据库所在盘剩余 {free / 2 ** 30:.1f}GB 不足 "
                f"{DISK_MIN_FREE / 2 ** 30:.0f}GB，停止下载"
            )

    # ---------- 原子写工具 ----------
    @staticmethod
    def _atomic_write_text(path: Path, text: str):
        """原子写文本：临时文件 + os.replace"""
        tmp = path.with_name(f"{path.name}.tmp{os.getpid()}")
        try:
            tmp.write_text(text, encoding="utf-8")
            os.replace(tmp, path)
        finally:
            if tmp.exists():
                try:
                    tmp.unlink()
                except OSError:
                    pass

    @staticmethod
    def _atomic_to_parquet(df: pd.DataFrame, path: Path):
        """原子写 parquet：临时文件 + os.replace

        直接写最终路径时，若进程在写一半时被杀（磁盘满/被 kill/断电），
        会留下无法读取的 parquet；而 save() 的 `path.exists()` 跳过逻辑
        会让它**永远不会被重写**，形成永久损坏。

        实现统一到模块级 `atomic_to_parquet`（带写后校验），这里只做转发。
        """
        atomic_to_parquet(df, path)

    # ---------- 断点 ----------
    def load_checkpoint(self) -> dict:
        """读取断点（带缓存 + 损坏容错）"""
        if self._cp_cache is not None:
            return self._cp_cache
        if not self.checkpoint_file.exists():
            self._cp_cache = {}
            return self._cp_cache
        try:
            self._cp_cache = json.loads(
                self.checkpoint_file.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError, UnicodeDecodeError):
            # 断电可能写坏断点文件；旧实现没有 try/except，
            # 一次损坏就会让之后每次运行都抛 JSONDecodeError。
            self._cp_cache = {}
            try:
                self.checkpoint_file.replace(
                    self.checkpoint_file.with_suffix(".json.corrupt"))
            except OSError:
                pass
        return self._cp_cache

    def save_checkpoint(self, state: dict):
        self._cp_cache = state
        self._atomic_write_text(
            self.checkpoint_file,
            json.dumps(state, ensure_ascii=False, indent=2),
        )

    def mark_done(self, key: str, span=None, empty: bool = False):
        """标记某个 key（通常是股票代码）已完成

        ⚠️ **只有真的写入了数据才该调用**（或确认"这只股票本来就没有数据"）。
        把"我以为成功了"标成完成会把缺失**永久固化** —— 断点续跑从此跳过它。
        这不是理论风险：`frozen/etf` 与 `frozen/options` 的断点里各有 84 个日期
        （其中 43 个是交易日），磁盘上**只有 7 天有数据**，其余 36 个交易日被标成
        done 却永远补不回来（问题清单 B4）。

        参数:
            span:  `(first_date, last_date)` —— 本次**实际写到磁盘**的数据范围。
                   记下来之后才能审计"断点说完成了，但数据只到 2015 年"
                   这种静默截断（tushare 返回半截历史时就是这么坏的）。
            empty: True 表示**已确认该 key 没有数据**（如退市股在区间内无行情）。
                   与"有数据但没记范围"区分开，避免 audit 把两者混为一谈。
        """
        cp = self.load_checkpoint()
        done = cp.setdefault("done", [])
        if key not in done:
            done.append(key)
        spans = cp.setdefault("spans", {})
        if span is not None:
            try:
                a, b = span
                spans[key] = [str(pd.Timestamp(a))[:10], str(pd.Timestamp(b))[:10]]
            except (TypeError, ValueError):
                pass
        elif empty:
            spans[key] = []           # 显式空：确认无数据
        self.save_checkpoint(cp)

    def is_done(self, key: str) -> bool:
        return key in self.load_checkpoint().get("done", [])

    @property
    def _spans(self) -> dict:
        return self.load_checkpoint().get("spans", {})

    def span_of(self, key: str):
        """断点记录的该 key 数据范围 -> (first, last) / () 表示确认无数据 / None 未知"""
        s = self._spans.get(key)
        if s is None:
            return None
        if len(s) == 0:
            return ()
        try:
            return pd.Timestamp(s[0]), pd.Timestamp(s[1])
        except (TypeError, ValueError, IndexError):
            return None

    def forget_done(self, keys) -> int:
        """把若干 key 从断点里移除（连同 span），让下次运行重新下载

        用于修复"标记完成但数据缺失"—— 这是 B4 的**补救路径**：
        光能检测不够，还得能把它变回"待下载"。返回实际移除的个数。
        """
        cp = self.load_checkpoint()
        done = list(cp.get("done", []))
        spans = cp.get("spans", {})
        keys = set(keys)
        cp["done"] = [k for k in done if k not in keys]
        for k in keys:
            spans.pop(k, None)
        self.save_checkpoint(cp)
        return len(done) - len(cp["done"])

    def audit_done(self, has_data=None) -> dict:
        """审计断点与磁盘是否一致（B4 的检测路径）

        参数:
            has_data: 可选 `f(key) -> bool`，判断磁盘上是否真有数据。
                      不传则用 `**/{key}.parquet` 是否存在判定。

        返回:
            n_done        断点里 key 的总数
            no_data       标记完成、磁盘上却没有数据 -> **应重新下载**
            confirmed_empty 明确记为"确认无数据"的（正常，不算问题）
            span_unknown  有数据但没记范围（旧断点）-> 无法核对是否被静默截断
            span_end      有范围记录的：{key: last_date}，供调用方与参考数据比对
        """
        done = list(self.load_checkpoint().get("done", []))
        no_data, confirmed_empty, span_unknown, span_end = [], [], [], {}
        for key in done:
            ok = (has_data(key) if has_data is not None
                  else any(self.root.glob(f"**/{key}.parquet")))
            sp = self.span_of(key)
            if not ok:
                (confirmed_empty if sp == () else no_data).append(key)
                continue
            if sp is None:
                span_unknown.append(key)
            elif sp:
                span_end[key] = sp[1]
        return {"n_done": len(done), "no_data": no_data,
                "confirmed_empty": confirmed_empty,
                "span_unknown": span_unknown, "span_end": span_end}

    # ---------- Parquet 读写 ----------
    def save(self, df: pd.DataFrame, year: int = None, code: str = None,
             force: bool = False) -> Path:
        """保存 DataFrame 到分区目录（原子写）

        分区规则: {root}/year={year}/{code}.parquet
                 （year=None 时写 {root}/{code}.parquet，用于非年度数据）
        """
        self.assert_writable()
        self.assert_disk_ok()
        if df is None or df.empty:
            return None

        if year is not None:
            part = self.root / f"year={year}"
            part.mkdir(parents=True, exist_ok=True)
            fname = f"{code}.parquet" if code is not None else "data.parquet"
        else:
            part = self.root
            fname = f"{code or 'data'}.parquet"

        path = part / fname
        if path.exists() and not force:
            return path  # 已存在则跳过（断点续跑语义）

        self._atomic_to_parquet(df, path)
        return path

    def load(self, year: int = None, code: str = None) -> pd.DataFrame:
        """读取分区数据（可只按年份或代码过滤）"""
        if year is not None:
            part = self.root / f"year={year}"
            if not part.exists():
                return pd.DataFrame()
            files = list(part.glob("*.parquet"))
        else:
            files = list(self.root.glob("**/*.parquet"))

        dfs = []
        for f in files:
            if code is not None and f.stem != code:
                continue
            dfs.append(pd.read_parquet(f))
        return pd.concat(dfs, ignore_index=True) if dfs else pd.DataFrame()


class DiskFullError(BaseException):
    """磁盘空间不足

    特意继承 BaseException 而非 Exception：下载器里到处是 `except Exception`
    兜底，若继承 Exception 会被静默吞掉，导致"磁盘满了还在继续跑"。
    继承 BaseException 让它一定向上传播，与 KeyboardInterrupt / SystemExit 同类。
    """
