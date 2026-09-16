"""个股日线下载器：akshare(新浪) + baostock 双源交叉校验，前复权"""
import time
import pandas as pd
import numpy as np
from .base import BaseDownloader
from datetime import datetime

class DailyDownloader(BaseDownloader):
    """【已废弃】新浪源日线下载器，请改用 scripts/download_tushare_raw.py
    仅用于历史兼容，不再写入 frozen 层。
    """

    def __init__(self, use_qfq: bool = True):
        # 用独立的 dataset 名，输出到 db/cleaned/daily_akshare_legacy/，
        # 绝不与 daily_basic（清洗层标准日线）混用，避免复权口径混淆
        dataset = "daily_akshare"
        super().__init__(dataset, "daily", calls_per_min=50)
        self.use_qfq = use_qfq
        self._bs_logged_in = False

    # ---------- 单源获取 ----------
    def _from_akshare_sina(self, code: str) -> pd.DataFrame:
        import akshare as ak
        if code.startswith("920"):
            symbol = f"bj{code}"
        else:
            symbol = f"sz{code}" if code.startswith(("0", "3")) else f"sh{code}"
        df = self.fetch(ak.stock_zh_a_daily, symbol=symbol,
                        adjust="qfq" if self.use_qfq else "")
        if df is None or df.empty:
            return pd.DataFrame()
        df = df.rename(columns={"date": "trade_date"})
        df["code"] = code
        df["trade_date"] = pd.to_datetime(df["trade_date"])
        return df[["code", "trade_date", "open", "high", "low", "close", "volume", "amount"]]

    def _from_baostock(self, code: str) -> pd.DataFrame:
        import baostock as bs
        if code.startswith("920"):
            return pd.DataFrame()  # baostock 无北交所数据
        if not self._bs_logged_in:
            bs.login()
            self._bs_logged_in = True
        bcode = f"sz.{code}" if code.startswith(("0", "3")) else f"sh.{code}"
        rs = bs.query_history_k_data_plus(
            bcode, "date,open,high,low,close,volume,amount",
            start_date=f"{self.start_year}-01-01", end_date=datetime.now().strftime("%Y-%m-%d"),
            frequency="d", adjustflag="2" if self.use_qfq else "3",
        )
        data = rs.get_data()
        if data is None or data.empty:
            return pd.DataFrame()
        data = data.rename(columns={"date": "trade_date"})
        data["code"] = code
        data["trade_date"] = pd.to_datetime(data["trade_date"])
        for c in ["open", "high", "low", "close", "volume", "amount"]:
            data[c] = pd.to_numeric(data[c], errors="coerce")
        return data[["code", "trade_date", "open", "high", "low", "close", "volume", "amount"]]

    # ---------- 交叉校验 ----------
    def _cross_validate(self, df_ak, df_bs, code: str) -> pd.DataFrame:
        """合并双源，用收盘价比对；差异过大时以新浪为准并警告"""
        if df_ak.empty and df_bs.empty:
            return pd.DataFrame()
        if df_ak.empty:
            return df_bs
        if df_bs.empty:
            return df_ak

        merged = pd.merge(
            df_ak, df_bs, on="trade_date", suffixes=("_ak", "_bs"),
            how="outer",
        )
        merged["close_ak"] = pd.to_numeric(merged.get("close_ak"), errors="coerce")
        merged["close_bs"] = pd.to_numeric(merged.get("close_bs"), errors="coerce")

        # 用新浪的 OHLC 为基准，baostock 缺失时补全
        cols = ["open", "high", "low", "close", "volume", "amount"]
        for c in cols:
            merged[c] = merged[f"{c}_ak"].fillna(merged[f"{c}_bs"])

        diff = merged["close_ak"] - merged["close_bs"]
        tol = 0.005  # 0.5% 容差
        bad = merged["close_ak"].notna() & merged["close_bs"].notna() & (diff.abs() / merged["close_bs"] > tol)
        if bad.any():
            n_bad = bad.sum()
            self.logger.warning(f"{code} 双源差异超 {tol*100:.1f}% 的交易日 {n_bad} 个，以前复权新浪为准")

        merged["code"] = code
        merged["trade_date"] = pd.to_datetime(merged["trade_date"])
        out = merged[["code", "trade_date"] + cols].sort_values("trade_date").drop_duplicates("trade_date")
        return out

    # ---------- 下载单只股票 ----------
    def download_code(self, code: str) -> pd.DataFrame:
        df_ak = pd.DataFrame()
        df_bs = pd.DataFrame()
        try:
            df_ak = self._from_akshare_sina(code)
        except Exception as e:
            self.logger.warning(f"{code} akshare失败: {str(e)[:50]}")
        try:
            df_bs = self._from_baostock(code)
        except Exception as e:
            self.logger.warning(f"{code} baostock失败: {str(e)[:50]}")

        df = self._cross_validate(df_ak, df_bs, code)
        if df.empty:
            self.logger.warning(f"{code} 无数据，标记跳过")
            self.storage.mark_done(code, empty=True)
            return df

        # 按年分区存储
        df["year"] = df["trade_date"].dt.year
        for y, g in df.groupby("year"):
            if y < self.start_year:
                continue
            self.save_year(g.drop(columns="year"), year=int(y), code=code)

        self.storage.mark_done(code, span=self._span_of(df))
        return df

    def download(self, codes: list = None, resume: bool = True):
        """从新往旧下载。codes 为空时自动获取全部A股代码。"""
        from tqdm import tqdm
        from .stocks import StockListDownloader
        if codes is None:
            sl = StockListDownloader()
            meta = sl.storage.load(year=2005)
            codes = meta["code"].tolist() if not meta.empty else []
        if not codes:
            raise ValueError("无股票代码")

        done = set(self.storage.load_checkpoint().get("done", []))
        todo = [c for c in codes if (not resume) or c not in done]
        self.logger.info(f"待下载 {len(todo)} 只（已完成 {len(done)}）")

        with tqdm(total=len(todo), desc="日线下载", unit="股",
                  ncols=100, ascii=False) as pbar:
            for code in todo:
                self.storage.assert_disk_ok()
                try:
                    self.download_code(code)
                except Exception as e:
                    self.logger.error(f"{code} 下载失败，已记录断点: {e}")
                pbar.set_postfix(code=code)
                pbar.update(1)

        # 登出 baostock
        if self._bs_logged_in:
            import baostock as bs
            bs.logout()
            self._bs_logged_in = False
        self.logger.info(f"日线下载完成，共 {len(done) + len(todo)} 只")
