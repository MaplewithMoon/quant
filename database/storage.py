"""Parquet 分区存储 + 断点记录"""
import json
import shutil
import pandas as pd
from pathlib import Path
from .config import dir_of, DISK_MIN_FREE


class Storage:
    """按 年份/代码 分区存储 Parquet，并维护断点文件"""

    def __init__(self, dataset: str):
        self.root = dir_of(dataset)
        self.root.mkdir(parents=True, exist_ok=True)
        self.checkpoint_file = self.root / "_checkpoint.json"

    # ---------- 磁盘监控 ----------
    @staticmethod
    def disk_free() -> int:
        try:
            return shutil.disk_usage("D:\\").free
        except Exception:
            return 10 ** 12

    def assert_disk_ok(self):
        free = self.disk_free()
        if free < DISK_MIN_FREE:
            raise DiskFullError(
                f"D盘剩余 {free / 2 ** 30:.1f}GB 不足 {DISK_MIN_FREE / 2 ** 30:.0f}GB，停止下载"
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
        分区规则: {dataset}/year={year}/code={code}.parquet 或 {dataset}/year={year}.parquet
        """
        self.assert_disk_ok()
        if df is None or df.empty:
            return None

        if year is not None:
            part = self.root / f"year={year}"
            part.mkdir(parents=True, exist_ok=True)
            if code is not None:
                fname = f"{code}.parquet"
            else:
                fname = f"data.parquet"
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
            files = list(self.root.glob("**/*.parquet")) if not self.root.name.startswith("year=") \
                else list(self.root.glob("*.parquet"))

        dfs = []
        for f in files:
            if code is not None and f.stem != code:
                continue
            dfs.append(pd.read_parquet(f))
        return pd.concat(dfs, ignore_index=True) if dfs else pd.DataFrame()


class DiskFullError(Exception):
    pass
