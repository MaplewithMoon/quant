"""三大财务报表下载器（tushare，季频）
资产负债表/利润表/现金流量表，每表约100+字段
分区: financial/{sheet}/year={year}/{sheet}_{code}.parquet
"""
import pandas as pd
from .base import BaseDownloader
from .tushare_client import get_client


class FinancialDownloader(BaseDownloader):
    """财务报表（tushare income/balancesheet/cashflow）"""

    SHEETS = {
        "balance": "balancesheet",
        "profit": "income",
        "cashflow": "cashflow",
    }

    def __init__(self):
        super().__init__("financial", "financial", calls_per_min=100)
        self._tc = None

    def _ts_code(self, code: str) -> str:
        return f"{code}.{'SZ' if code.startswith(('0','3')) else 'SH'}"

    def download_code(self, code: str) -> dict:
        self._tc = get_client()
        ts_code = self._ts_code(code)
        result = {}
        for sheet, api in self.SHEETS.items():
            try:
                df = self._tc.call(
                    api, ts_code=ts_code,
                    start_date=f"{self.start_year}0101", end_date="20500101",
                )
                if df is None or df.empty:
                    continue
                df["code"] = code
                if "end_date" in df.columns:
                    df["report_date"] = pd.to_datetime(df["end_date"])
                elif "f_ann_date" in df.columns:
                    df["report_date"] = pd.to_datetime(df["f_ann_date"])
                if "report_date" not in df.columns:
                    continue
                df["year"] = df["report_date"].dt.year
                # 每只股票三张表各存一个文件（表结构固定，不分年）
                self.save_year(df.drop(columns="year"), year=2005,
                               code=f"{sheet}_{code}", force=True)
                result[sheet] = df
            except Exception as e:
                self.logger.warning(f"{code} {sheet} 失败: {str(e)[:50]}")
        # 三张报表至少写成功一张才算完成；一张都没有 -> 记为确认无数据
        self.storage.mark_done(code, empty=not result)
        return result

    def download(self, codes: list = None, resume: bool = True):
        from tqdm import tqdm
        from .stocks import StockListDownloader
        if codes is None:
            meta = StockListDownloader().storage.load(year=2005)
            codes = meta["code"].tolist() if not meta.empty else []
        done = set(self.storage.load_checkpoint().get("done", []))
        todo = [c for c in codes if (not resume) or c not in done]
        self.logger.info(f"财务报表待下载 {len(todo)} 只, 后端: tushare")
        with tqdm(total=len(todo), desc="财务报表", unit="股", ncols=100) as pbar:
            for code in todo:
                self.storage.assert_disk_ok()
                try:
                    self.download_code(code)
                except Exception as e:
                    self.logger.error(f"{code} 失败: {e}")
                pbar.set_postfix(code=code)
                pbar.update(1)
        self.logger.info("财务报表下载完成")
