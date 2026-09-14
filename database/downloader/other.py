"""其他数据：两融/北向/股东户数/ETF/期货/期权"""
import pandas as pd
from .base import BaseDownloader
from .tushare_client import get_client


class MarginDownloader(BaseDownloader):
    """两融余额（tushare margin，沪深两市日度）"""
    def __init__(self):
        super().__init__("margin", "margin", calls_per_min=200)

    def download(self, resume: bool = True):
        from tqdm import tqdm
        self._tc = get_client()
        frames = []
        # 按季度拉取
        quarters = []
        import datetime
        y = datetime.datetime.now().year
        for year in range(self.start_year, y + 1):
            for m_start in ("0101", "0401", "0701", "1001"):
                quarters.append(f"{year}{m_start}")
        with tqdm(total=len(quarters), desc="两融余额", ncols=100) as pbar:
            for i, q in enumerate(quarters):
                self.storage.assert_disk_ok()
                try:
                    df = self._tc.call("margin", start_date=q,
                                       end_date=f"{q[:4]}1231")
                    if df is not None and not df.empty:
                        frames.append(df)
                except Exception as e:
                    self.logger.warning(f"两融 {q} 失败: {e}")
                pbar.update(1)
        if frames:
            data = pd.concat(frames, ignore_index=True)
            data["trade_date"] = pd.to_datetime(data["trade_date"])
            data["year"] = data["trade_date"].dt.year
            for y, g in data.groupby("year"):
                self.save_year(g.drop(columns="year"), year=int(y), force=True)
            self.storage.mark_done("all")
            self.logger.info(f"两融下载完成: {len(data)} 行")


class NorthboundDownloader(BaseDownloader):
    """北向持股（tushare hk_hold）"""
    def __init__(self):
        super().__init__("northbound", "northbound", calls_per_min=200)

    def download(self, resume: bool = True):
        self._tc = get_client()
        try:
            # 沪深港通持股汇总（按交易日）
            df = self._tc.call("hsgt_top10", trade_date="20240101")
            # 北向资金历史行情
            import datetime
            df = self._tc.call("moneyflow_hsgt", start_date=f"{self.start_year}0101",
                               end_date="20500101")
            if df is not None and not df.empty:
                df["trade_date"] = pd.to_datetime(df["trade_date"])
                df["year"] = df["trade_date"].dt.year
                for y, g in df.groupby("year"):
                    self.save_year(g.drop(columns="year"), year=int(y), force=True)
                self.storage.mark_done("all")
                self.logger.info(f"北向资金完成: {len(df)} 行")
        except Exception as e:
            self.logger.error(f"北向资金失败: {e}")


class HolderDownloader(BaseDownloader):
    """股东户数（tushare stk_holdernumber）"""
    def __init__(self):
        super().__init__("holders", "holders", calls_per_min=100)

    def _ts_code(self, code: str) -> str:
        return f"{code}.{'SZ' if code.startswith(('0','3')) else 'SH'}"

    def download_code(self, code: str) -> pd.DataFrame:
        self._tc = get_client()
        df = self._tc.call("stk_holdernumber", ts_code=self._ts_code(code),
                           start_date=f"{self.start_year}0101", end_date="20500101")
        if df is None or df.empty:
            self.storage.mark_done(code)
            return pd.DataFrame()
        df["code"] = code
        if "end_date" in df.columns:
            df["end_date"] = pd.to_datetime(df["end_date"])
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
        self.logger.info(f"股东户数待下载 {len(todo)} 只")
        with tqdm(total=len(todo), desc="股东户数", unit="股", ncols=100) as pbar:
            for code in todo:
                self.storage.assert_disk_ok()
                try:
                    self.download_code(code)
                except Exception as e:
                    self.logger.error(f"{code} 失败: {e}")
                pbar.set_postfix(code=code)
                pbar.update(1)
        self.logger.info("股东户数下载完成")


class FundDownloader(BaseDownloader):
    """ETF/LOF 日线"""
    def __init__(self):
        super().__init__("etf", "etf", calls_per_min=20)

    def download(self, resume: bool = True):
        import akshare as ak
        try:
            # 场内基金列表
            df = self.fetch(ak.fund_etf_spot_em)
            if df is None or df.empty:
                return
            self.save_year(df, year=2005, code="fund_list")
            codes = df["代码"].head(20).tolist()  # 先下载部分ETF做验证
            for i, code in enumerate(codes):
                self.storage.assert_disk_ok()
                try:
                    d = self.fetch(ak.fund_etf_hist_em, symbol=code, period="daily",
                                   start_date=f"{self.start_year}0101", end_date="20500101",
                                   adjust="")
                    if d is not None and not d.empty:
                        d["code"] = code
                        self.save_year(d, year=2005, code=code)
                except Exception as e:
                    self.logger.warning(f"ETF {code} 失败: {e}")
            self.storage.mark_done("all")
            self.logger.info(f"ETF下载完成，已处理 {len(codes)} 只")
        except Exception as e:
            self.logger.error(f"ETF失败: {e}")


class FuturesDownloader(BaseDownloader):
    """股指/国债期货连续合约日线（tushare fut_daily 主连）"""
    MAIN_SYMBOLS = {"IF": "沪深300期货", "IH": "上证50期货", "IC": "中证500期货",
                    "IM": "中证1000期货", "T": "10年期国债期货",
                    "TF": "5年期国债期货", "TS": "2年期国债期货"}

    def __init__(self):
        super().__init__("futures", "futures", calls_per_min=100)
        self._tc = None

    def download(self, resume: bool = True):
        from tqdm import tqdm
        self._tc = get_client()
        done = set(self.storage.load_checkpoint().get("done", []))
        todo = [s for s in self.MAIN_SYMBOLS if (not resume) or s not in done]
        with tqdm(total=len(todo), desc="期货主连", unit="品种", ncols=100) as pbar:
            for sym in todo:
                self.storage.assert_disk_ok()
                name = self.MAIN_SYMBOLS[sym]
                try:
                    df = self._tc.call("fut_daily", ts_code=f"{sym}.CFX",
                                       start_date=f"{self.start_year}0101", end_date="20500101")
                    if df is None or df.empty:
                        self.logger.warning(f"{name} 无数据")
                        pbar.update(1)
                        continue
                    df["symbol"] = sym
                    df["trade_date"] = pd.to_datetime(df["trade_date"])
                    df["year"] = df["trade_date"].dt.year
                    for y, g in df.groupby("year"):
                        if y < self.start_year:
                            continue
                        self.save_year(g.drop(columns="year"), year=int(y),
                                       code=sym, force=True)
                    self.storage.mark_done(sym)
                    pbar.set_postfix(symbol=sym)
                except Exception as e:
                    self.logger.error(f"{name} 失败: {e}")
                pbar.update(1)
        self.logger.info("期货主连下载完成")


class OptionDownloader(BaseDownloader):
    """ETF期权（按合约下载）"""
    def __init__(self):
        super().__init__("options", "options", calls_per_min=15)

    def download(self, resume: bool = True, contract_codes: list = None):
        from tqdm import tqdm
        import akshare as ak
        if contract_codes is None:
            # 获取当前存续合约
            try:
                df = self.fetch(ak.option_current_em, symbol="510300")
                if df is not None and not df.empty:
                    contract_codes = df["option_code"].head(50).tolist()
                else:
                    contract_codes = []
            except Exception:
                contract_codes = []
        done = set(self.storage.load_checkpoint().get("done", []))
        todo = [c for c in contract_codes if (not resume) or c not in done]
        with tqdm(total=len(todo), desc="ETF期权", unit="合约", ncols=100) as pbar:
            for code in todo:
                self.storage.assert_disk_ok()
                try:
                    df = self.fetch(ak.option_hist_em, symbol=code)
                    if df is not None and not df.empty:
                        self.save_year(df, year=2005, code=code)
                        self.storage.mark_done(code)
                except Exception as e:
                    self.logger.warning(f"期权 {code} 失败: {e}")
                pbar.set_postfix(code=code)
                pbar.update(1)
        self.logger.info("ETF期权下载完成")
