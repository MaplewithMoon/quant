"""交易日历下载器（tushare trade_cal）"""
import pandas as pd
from .base import BaseDownloader
from .tushare_client import get_client


class CalendarDownloader(BaseDownloader):
    """A股交易日历（tushare）"""

    def __init__(self):
        super().__init__("calendar", "calendar", calls_per_min=100)

    def download(self):
        self._tc = get_client()
        df = self._tc.call("trade_cal", exchange="SSE",
                           start_date="19900101", end_date="20500101")
        if df is None or df.empty:
            raise RuntimeError("交易日历获取失败")
        # 只保留交易日
        df = df[df["is_open"] == 1]
        df["cal_date"] = pd.to_datetime(df["cal_date"])
        df = df.rename(columns={"cal_date": "trade_date"})
        df = df.sort_values("trade_date").reset_index(drop=True)
        years = df["trade_date"].dt.year.unique()
        for y in years:
            self.save_year(df[df["trade_date"].dt.year == y], year=int(y), force=True)
        self.storage.mark_done("all")
        self.logger.info(f"交易日历完成: {len(df)} 天, {years.min()}~{years.max()}")
        return df
