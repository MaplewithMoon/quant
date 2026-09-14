"""每日估值指标下载器：PE/PB/PS/总市值/流通市值/换手率
后端: tushare daily_basic（完整字段，优先）
      回退: akshare百度（仅PE/PB/总市值）
"""
import pandas as pd
from .base import BaseDownloader
from .tushare_client import get_client


class ValuationDownloader(BaseDownloader):
    """个股每日估值指标
    分区: valuation/year={year}/{code}.parquet
    """

    def __init__(self):
        super().__init__("valuation", "valuation", calls_per_min=100)
        self._tc = None

    def _ts_code(self, code: str) -> str:
        return f"{code}.{'SZ' if code.startswith(('0','3')) else 'SH'}"

    def _from_tushare(self, code: str) -> pd.DataFrame:
        self._tc = get_client()
        df = self._tc.daily_basic(
            self._ts_code(code),
            start=f"{self.start_year}0101", end="20500101",
        )
        if df is None or df.empty:
            return pd.DataFrame()
        # 保留 pe_ttm（如缺则用 pe）
        if "pe_ttm" not in df.columns and "pe" in df.columns:
            df["pe_ttm"] = df["pe"]
        if "turnover_rate" in df.columns:
            df["turnover"] = df["turnover_rate"]
        df["trade_date"] = pd.to_datetime(df["trade_date"])
        df["code"] = code
        # 市值单位: 万元 → 亿元
        for col in ["total_mv", "circ_mv"]:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors="coerce") / 10000
        keep = ["code", "trade_date", "pe_ttm", "pb", "ps_ttm",
                "total_mv", "circ_mv", "turnover"]
        df = df[[c for c in keep if c in df.columns]]
        return df.sort_values("trade_date")

    def _from_baidu(self, code: str) -> pd.DataFrame:
        import akshare as ak
        inds = {"pe_ttm": "市盈率(TTM)", "pb": "市净率", "total_mv": "总市值"}
        series = []
        for field, indicator in inds.items():
            try:
                df = self.fetch(ak.stock_zh_valuation_baidu,
                                symbol=code, indicator=indicator, period="全部")
                if df is None or df.empty:
                    continue
                df.columns = ["trade_date", field]
                df["trade_date"] = pd.to_datetime(df["trade_date"])
                df[field] = pd.to_numeric(df[field], errors="coerce")
                series.append(df)
            except Exception:
                continue
        if not series:
            return pd.DataFrame()
        merged = series[0]
        for s in series[1:]:
            merged = pd.merge(merged, s, on="trade_date", how="outer")
        merged["code"] = code
        return merged.sort_values("trade_date")

    def download_code(self, code: str) -> pd.DataFrame:
        df = pd.DataFrame()
        try:
            df = self._from_tushare(code)
        except Exception as e:
            self.logger.warning(f"{code} tushare失败，回退百度: {str(e)[:50]}")
            df = self._from_baidu(code)

        if df.empty:
            self.storage.mark_done(code)
            return df

        df["year"] = df["trade_date"].dt.year
        for y, g in df.groupby("year"):
            if y < self.start_year:
                continue
            self.save_year(g.drop(columns="year"), year=int(y), code=code, force=True)

        self.storage.mark_done(code)
        return df

    def download(self, codes: list = None, resume: bool = True):
        from tqdm import tqdm
        from .stocks import StockListDownloader
        if codes is None:
            meta = StockListDownloader().storage.load(year=2005)
            codes = meta["code"].tolist() if not meta.empty else []
        done = set(self.storage.load_checkpoint().get("done", []))
        todo = [c for c in codes if (not resume) or c not in done]
        self.logger.info(f"估值指标待下载 {len(todo)} 只, 后端: tushare")
        with tqdm(total=len(todo), desc="估值指标", unit="股", ncols=100) as pbar:
            for code in todo:
                self.storage.assert_disk_ok()
                try:
                    self.download_code(code)
                except Exception as e:
                    self.logger.error(f"{code} 失败: {e}")
                pbar.set_postfix(code=code)
                pbar.update(1)
        self.logger.info("估值指标下载完成")
