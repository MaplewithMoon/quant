"""其他数据：两融/北向/股东户数/ETF/期货/期权"""
import pandas as pd
from .base import BaseDownloader
from .tushare_client import get_client

# 季末日期：`(起始月日, 结束月日)`
QUARTER_ENDS = (("0101", "0331"), ("0401", "0630"),
                ("0701", "0930"), ("1001", "1231"))


def quarter_ranges(start_year: int, end_year: int) -> list:
    """生成按季度切分、**互不重叠**的 `(start_date, end_date)` 列表

    ⚠️ 抽成独立函数是为了能测。原始写法把区间拼在 `download()` 里，出了这个
    bug 也没法回归测试：

        for m_start in ("0101", "0401", "0701", "1001"):
            self._tc.call("margin", start_date=q, end_date=f"{q[:4]}1231")

    季度起点配的是**年末**，于是 Q1 拉 1~12 月、Q2 拉 4~12 月、Q3 拉 7~12 月、
    Q4 拉 10~12 月 —— 10 月以后的数据被拉了 4 遍。`pd.concat` 之后又没去重，
    整行重复直接写进 parquet。实测 `frozen/margin` 2023 年 1,803 行里只有
    726 行是真实的，`sum(rzye)` 被放大 2.5 倍；而主键 `(trade_date, exchange_id)`
    仍然唯一，主键检查完全查不出来。

    区间必须**闭合在季末**，且相邻区间首尾相接、不重叠。
    """
    out = []
    for year in range(int(start_year), int(end_year) + 1):
        for m_start, m_end in QUARTER_ENDS:
            out.append((f"{year}{m_start}", f"{year}{m_end}"))
    return out


class MarginDownloader(BaseDownloader):
    """两融余额（tushare margin，沪/深/北三市日度）"""
    def __init__(self):
        super().__init__("margin", "margin", calls_per_min=200)

    def download(self, resume: bool = True):
        from tqdm import tqdm
        import datetime
        self._tc = get_client()
        frames = []
        # 区间构造见 quarter_ranges 的 docstring（那里记着放大 2.5 倍的旧 bug）
        quarters = quarter_ranges(self.start_year, datetime.datetime.now().year)
        with tqdm(total=len(quarters), desc="两融余额", ncols=100) as pbar:
            for i, (q0, q1) in enumerate(quarters):
                self.storage.assert_disk_ok()
                try:
                    df = self._tc.call("margin", start_date=q0, end_date=q1)
                    if df is not None and not df.empty:
                        frames.append(df)
                except Exception as e:
                    self.logger.warning(f"两融 {q0}~{q1} 失败: {e}")
                pbar.update(1)
        if frames:
            data = pd.concat(frames, ignore_index=True)
            n_raw = len(data)
            # 二次防线：即使将来又出现区间重叠的写法，也不会把重复写进库
            data = data.drop_duplicates()
            if len(data) != n_raw:
                self.logger.warning(f"两融去重: {n_raw} -> {len(data)} 行"
                                    f"（丢弃 {n_raw - len(data)} 行重复）")
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
