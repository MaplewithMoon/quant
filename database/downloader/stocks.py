"""股票列表 + 上市日期下载器（tushare stock_basic）"""
import pandas as pd
from .base import BaseDownloader
from .tushare_client import get_client


class StockListDownloader(BaseDownloader):
    """A股股票代码、名称、行业、上市日期（tushare）"""

    def __init__(self):
        super().__init__("stocks", "stocks", calls_per_min=100)

    def download(self):
        self._tc = get_client()
        # 当前上市
        df = self._tc.call("stock_basic", list_status="L",
                           fields="ts_code,symbol,name,area,industry,list_date,market")
        # 已退市
        df_d = self._tc.call("stock_basic", list_status="D",
                             fields="ts_code,symbol,name,area,industry,list_date,market")
        # 暂停上市
        df_p = self._tc.call("stock_basic", list_status="P",
                             fields="ts_code,symbol,name,area,industry,list_date,market")

        all_df = pd.concat([df, df_d, df_p], ignore_index=True)
        all_df["code"] = all_df["symbol"]
        all_df["list_date"] = pd.to_datetime(all_df["list_date"], errors="coerce")

        self.save_year(all_df, year=2005, code="all", force=True)
        self.storage.mark_done("all")
        self.logger.info(f"股票列表完成: 上市{len(df)} + 退市{len(df_d)} + 暂停{len(df_p)} = {len(all_df)} 只")
        return all_df
