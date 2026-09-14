"""Parquet 分区存储 + 断点记录

写入权限规则：
    frozen/  只读原始层 —— 仅下载器可写，须显式 Storage(..., allow_frozen=True)
    cleaned/ 加工层     —— 任何清洗脚本可写
被拒绝的写入会抛 FrozenWriteError，而不是静默污染只读层。
"""
import json
import shutil
import pandas as pd
from pathlib import Path
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
                 allow_frozen: bool = False):
        if recipe:
            self.root = recipe_dir(recipe)
        else:
            self.root = dir_of(dataset)
        self.dataset = dataset or recipe
        self.recipe = recipe
        self.layer = "frozen" if is_frozen(self.root) else "cleaned"
        self.allow_frozen = allow_frozen
        self.root.mkdir(parents=True, exist_ok=True)
        self.checkpoint_file = self.root / "_checkpoint.json"

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

    # ---------- 断点 ----------
    def load_checkpoint(self) -> dict:
        if self.checkpoint_file.exists():
            return json.loads(self.checkpoint_file.read_text(encoding="utf-8"))
        return {}

    def save_checkpoint(self, state: dict):
        self.checkpoint_file.write_text(
            json.dumps(state, ensure_ascii=False, indent=2), encoding="utf-8"
        )

    def mark_done(self, key: str):
        cp = self.load_checkpoint()
        cp.setdefault("done", [])
        if key not in cp["done"]:
            cp["done"].append(key)
        self.save_checkpoint(cp)

    def is_done(self, key: str) -> bool:
        return key in self.load_checkpoint().get("done", [])

    # ---------- Parquet 读写 ----------
    def save(self, df: pd.DataFrame, year: int = None, code: str = None,
             force: bool = False) -> Path:
        """保存 DataFrame 到分区目录。

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

        df.to_parquet(path, index=False)
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


class DiskFullError(Exception):
    pass
