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
        """
        tmp = path.with_name(f"{path.name}.tmp{os.getpid()}")
        try:
            df.to_parquet(tmp, index=False)
            os.replace(tmp, path)
        finally:
            if tmp.exists():
                try:
                    tmp.unlink()
                except OSError:
                    pass

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

    def mark_done(self, key: str):
        cp = self.load_checkpoint()
        done = cp.setdefault("done", [])
        if key in done:
            return
        done.append(key)
        self.save_checkpoint(cp)

    def is_done(self, key: str) -> bool:
        return key in self.load_checkpoint().get("done", [])

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
