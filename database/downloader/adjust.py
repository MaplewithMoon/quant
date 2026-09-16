"""复权因子 + 分红送转下载器（tushare adj_factor / dividend）"""
import pandas as pd
from .base import BaseDownloader
from .tushare_client import get_client
from datetime import datetime


class AdjustDownloader(BaseDownloader):
    """复权因子（tushare adj_factor）"""

    def __init__(self):
        super().__init__("adjust", "adjust", calls_per_min=100)
        self._tc = None

    def _ts_code(self, code: str) -> str:
        return f"{code}.{'SZ' if code.startswith(('0','3')) else 'SH'}"

    def download_code(self, code: str) -> pd.DataFrame:
        self._tc = get_client()
        df = self._tc.adj_factor(
            self._ts_code(code),
            start=f"{self.start_year}0101", end="20500101",
        )
        if df is None or df.empty:
            # 确认无数据。必须与"有数据但没记范围"区分：audit_done 靠 empty 判定
            self.storage.mark_done(code, empty=True)
            return pd.DataFrame()
        df["code"] = code
        df["trade_date"] = pd.to_datetime(df["trade_date"])
        df = df.sort_values("trade_date")
        df["year"] = df["trade_date"].dt.year
        for y, g in df.groupby("year"):
            if y < self.start_year:
                continue
            self.save_year(g.drop(columns="year"), year=int(y), code=code, force=True)
        self.storage.mark_done(code, span=self._span_of(df))
        return df

    def download(self, codes: list = None, resume: bool = True):
        from tqdm import tqdm
        from .stocks import StockListDownloader
        if codes is None:
            meta = StockListDownloader().storage.load(year=2005)
            codes = meta["code"].tolist() if not meta.empty else []
        done = set(self.storage.load_checkpoint().get("done", []))
        todo = [c for c in codes if (not resume) or c not in done]
        self.logger.info(f"复权因子待下载 {len(todo)} 只")
        with tqdm(total=len(todo), desc="复权因子", unit="股", ncols=100) as pbar:
            for code in todo:
                self.storage.assert_disk_ok()
                try:
                    self.download_code(code)
                except Exception as e:
                    self.logger.error(f"{code} 失败: {e}")
                pbar.set_postfix(code=code)
                pbar.update(1)
        self.logger.info("复权因子下载完成")


class DividendDownloader(BaseDownloader):
    """分红送转明细（tushare dividend）"""

    def __init__(self):
        super().__init__("dividend", "dividend", calls_per_min=100)
        self._tc = None

    def _ts_code(self, code: str) -> str:
        return f"{code}.{'SZ' if code.startswith(('0','3')) else 'SH'}"

    def download_code(self, code: str) -> pd.DataFrame:
        self._tc = get_client()
        df = self._tc.call("dividend", ts_code=self._ts_code(code))
        if df is None or df.empty:
            self.storage.mark_done(code, empty=True)
            return pd.DataFrame()
        df["code"] = code
        self.save_year(df, year=2005, code=code, force=True)
        self.storage.mark_done(code, span=self._span_of(df))
        return df

    def download(self, codes: list = None, resume: bool = True):
        from tqdm import tqdm
        from .stocks import StockListDownloader
        if codes is None:
            meta = StockListDownloader().storage.load(year=2005)
            codes = meta["code"].tolist() if not meta.empty else []
        done = set(self.storage.load_checkpoint().get("done", []))
        todo = [c for c in codes if (not resume) or c not in done]
        self.logger.info(f"分红送转待下载 {len(todo)} 只")
        with tqdm(total=len(todo), desc="分红送转", unit="股", ncols=100) as pbar:
            for code in todo:
                self.storage.assert_disk_ok()
                try:
                    self.download_code(code)
                except Exception as e:
                    self.logger.error(f"{code} 失败: {e}")
                pbar.set_postfix(code=code)
                pbar.update(1)
        self.logger.info("分红送转下载完成")
