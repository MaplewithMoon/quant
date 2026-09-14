"""停牌 / ST状态 / 每日涨跌停价
ST状态: tushare namechange（历史名称变更含*ST/ST标记）
停牌:   tushare suspend_d
涨跌停价: 按板块规则计算（主板±10%，创业板/科创板±20%，ST±5%）
"""
import pandas as pd
from .base import BaseDownloader
from .tushare_client import get_client


class StStatusDownloader(BaseDownloader):
    """ST 状态历史（tushare namechange）"""
    def __init__(self):
        super().__init__("st", "st", calls_per_min=100)
        self._tc = None

    def _ts_code(self, code: str) -> str:
        return f"{code}.{'SZ' if code.startswith(('0','3')) else 'SH'}"

    def download_code(self, code: str) -> pd.DataFrame:
        self._tc = get_client()
        df = self._tc.call("namechange", ts_code=self._ts_code(code))
        if df is None or df.empty:
            self.storage.mark_done(code)
            return pd.DataFrame()
        df["code"] = code
        # 标记是否 ST
        df["is_st"] = df["name"].str.contains("ST", na=False).astype(int)
        self.save_year(df, year=2005, code=code, force=True)
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
        self.logger.info(f"ST状态待下载 {len(todo)} 只")
        with tqdm(total=len(todo), desc="ST状态", unit="股", ncols=100) as pbar:
            for code in todo:
                self.storage.assert_disk_ok()
                try:
                    self.download_code(code)
                except Exception as e:
                    self.logger.error(f"{code} 失败: {e}")
                pbar.set_postfix(code=code)
                pbar.update(1)
        self.logger.info("ST状态下载完成")


class SuspendDownloader(BaseDownloader):
    """停牌记录（tushare suspend_d）"""
    def __init__(self):
        super().__init__("suspend", "suspend", calls_per_min=100)
        self._tc = None

    def _ts_code(self, code: str) -> str:
        return f"{code}.{'SZ' if code.startswith(('0','3')) else 'SH'}"

    def download_code(self, code: str) -> pd.DataFrame:
        self._tc = get_client()
        df = self._tc.call("suspend_d", ts_code=self._ts_code(code),
                           start_date=f"{self.start_year}0101", end_date="20500101")
        if df is None or df.empty:
            self.storage.mark_done(code)
            return pd.DataFrame()
        df["code"] = code
        for c in ["suspend_date", "resume_date"]:
            if c in df.columns:
                df[c] = pd.to_datetime(df[c])
        self.save_year(df, year=2005, code=code, force=True)
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
        self.logger.info(f"停牌记录待下载 {len(todo)} 只")
        with tqdm(total=len(todo), desc="停牌", unit="股", ncols=100) as pbar:
            for code in todo:
                self.storage.assert_disk_ok()
                try:
                    self.download_code(code)
                except Exception as e:
                    self.logger.error(f"{code} 失败: {e}")
                pbar.set_postfix(code=code)
                pbar.update(1)
        self.logger.info("停牌记录下载完成")


class MetaDownloader(BaseDownloader):
    """每日涨跌停价（由日线前收盘按板块规则计算）"""

    def __init__(self):
        super().__init__("limit", "limit", calls_per_min=30)

    @staticmethod
    def limit_pct(code: str, is_st: bool = False) -> float:
        if is_st:
            return 0.05
        if code.startswith(("300", "301", "688", "689")):
            return 0.20
        return 0.10

    def download_code(self, code: str, daily: pd.DataFrame) -> pd.DataFrame:
        if daily.empty:
            self.storage.mark_done(code)
            return pd.DataFrame()
        df = daily.sort_values("trade_date").copy()
        df["pre_close"] = df["close"].shift(1)
        pct = self.limit_pct(code)
        df["limit_up"] = round(df["pre_close"] * (1 + pct), 2)
        df["limit_down"] = round(df["pre_close"] * (1 - pct), 2)
        out = df[["code", "trade_date", "pre_close", "limit_up", "limit_down"]].dropna()
        out["year"] = pd.to_datetime(out["trade_date"]).dt.year
        for y, g in out.groupby("year"):
            if y < self.start_year:
                continue
            self.save_year(g.drop(columns="year"), year=int(y), code=code, force=True)
        self.storage.mark_done(code)
        return out

    def download(self, codes: list = None, resume: bool = True):
        from tqdm import tqdm
        from .stocks import StockListDownloader
        from .daily import DailyDownloader
        if codes is None:
            meta = StockListDownloader().storage.load(year=2005)
            codes = meta["code"].tolist() if not meta.empty else []
        daily_store = DailyDownloader().storage
        done = set(self.storage.load_checkpoint().get("done", []))
        todo = [c for c in codes if (not resume) or c not in done]
        self.logger.info(f"涨跌停价待构建 {len(todo)} 只")
        with tqdm(total=len(todo), desc="涨跌停价", unit="股", ncols=100) as pbar:
            for code in todo:
                self.storage.assert_disk_ok()
                try:
                    d = daily_store.load(code=code)
                    self.download_code(code, d)
                except Exception as e:
                    self.logger.error(f"{code} 失败: {e}")
                pbar.set_postfix(code=code)
                pbar.update(1)
        self.logger.info("涨跌停价构建完成")
