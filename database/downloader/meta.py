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
            self.storage.mark_done(code, empty=True)
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
            self.storage.mark_done(code, empty=True)
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
        """⚠️ 已废弃：规则统一到 database/limit_rules.py

        旧实现有三处错：缺北交所 30%（920/43/83/87/88）、
        缺创业板 2020-08-24 前的 ±10%、且 download_code 里用 `close.shift(1)`
        当昨收（除权日会算错）。保留此方法只为兼容旧调用，内部转调唯一实现。
        """
        from database.limit_rules import limit_pct as _lp
        return _lp(code, None, is_st)

    def download_code(self, code: str, daily: pd.DataFrame) -> pd.DataFrame:
        from database.limit_rules import (apply_limit_prices, listing_windows,
                                          load_st_intervals)
        if daily.empty:
            self.storage.mark_done(code, empty=True)
            return pd.DataFrame()
        df = daily.sort_values("trade_date").copy()
        if "pre_close" not in df.columns:
            # 绝不能退回 close.shift(1)：那是未除权昨收，除权日必错
            self.logger.error(f"{code}: 缺官方 pre_close，跳过涨跌停计算")
            return pd.DataFrame()
        st_map = getattr(self, "_st_map", None)
        if st_map is None:
            st_map = self._st_map = load_st_intervals()
        windows = getattr(self, "_listing_windows", None)
        if windows is None:
            windows = self._listing_windows = listing_windows()
        out = apply_limit_prices(df, code, st_map,
                                 listing_rule=windows.get(code))
        # **不要 dropna**：limit 为空可能是"不设涨跌幅"（新股上市初期），
        # 与全量重建 / 增量更新保持同一口径（空值 = 无涨跌停限制）
        out = out[["code", "trade_date", "pre_close", "limit_up", "limit_down"]]
        out["year"] = pd.to_datetime(out["trade_date"]).dt.year
        for y, g in out.groupby("year"):
            if y < self.start_year:
                continue
            self.save_year(g.drop(columns="year"), year=int(y), code=code, force=True)
        self.storage.mark_done(code, span=self._span_of(out))
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
