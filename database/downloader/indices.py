"""指数成分及权重历史 + 指数日K + 行业分类"""
import pandas as pd
from .base import BaseDownloader
from .tushare_client import get_client


INDEX_LIST = {
    "000300.SH": "沪深300",
    "000905.SH": "中证500",
    "000852.SH": "中证1000",
    "932000.CSI": "中证2000",
}


class IndexConsDownloader(BaseDownloader):
    """指数成分及权重历史（tushare index_weight，按季度快照）"""

    def __init__(self):
        super().__init__("index_cons", "index_cons", calls_per_min=100)
        self._tc = None

    def download(self, indices: dict = None, resume: bool = True):
        from tqdm import tqdm
        import datetime
        self._tc = get_client()
        indices = indices or INDEX_LIST
        done = set(self.storage.load_checkpoint().get("done", []))
        items = [c for c in indices if (not resume) or c not in done]

        # 季度范围列表（从新到旧）
        today = datetime.datetime.now()
        quarters = []
        for year in range(today.year, self.start_year - 1, -1):
            for m_start in ((1, 1), (4, 1), (7, 1), (10, 1)):
                m_end = m_start[0] + 2
                if m_start[0] == 10:
                    m_end = 12
                y_end = year if m_start[0] != 10 else year
                quarters.append((f"{year}{m_start[0]:02d}{m_start[1]:02d}",
                                 f"{y_end}{m_end:02d}31"))

        for code in items:
            name = indices[code]
            self.storage.assert_disk_ok()
            frames = []
            with tqdm(total=len(quarters), desc=f"{name}权重", unit="季度",
                      ncols=100, leave=False) as pbar:
                for q_start, q_end in quarters:
                    try:
                        df = self._tc.call("index_weight", index_code=code,
                                           start_date=q_start, end_date=q_end)
                        if df is not None and not df.empty:
                            frames.append(df)
                    except Exception as e:
                        self.logger.warning(f"{name} {q_start} 失败: {e}")
                    pbar.update(1)
            if not frames:
                self.logger.warning(f"{name} 无权重数据")
                continue
            data = pd.concat(frames, ignore_index=True)
            data = data.drop_duplicates(subset=["index_code", "con_code", "trade_date"])
            self.save_year(data, year=2005, code=code, force=True)
            self.storage.mark_done(code)
            self.logger.info(f"{name} 权重快照: {len(data)} 行, {len(frames)} 期")

        self.logger.info("指数成分权重下载完成")


class IndexDailyDownloader(BaseDownloader):
    """指数日K（不复权）——tushare index_daily"""

    def __init__(self):
        super().__init__("index_daily", "index_daily", calls_per_min=100)
        self._tc = None

    def download(self, indices: dict = None, resume: bool = True):
        from tqdm import tqdm
        self._tc = get_client()
        indices = indices or INDEX_LIST
        done = set(self.storage.load_checkpoint().get("done", []))
        items = [c for c in indices if (not resume) or c not in done]

        for code in items:
            name = indices[code]
            self.storage.assert_disk_ok()
            try:
                df = self._tc.call("index_daily", ts_code=code,
                                   start_date=f"{self.start_year}0101", end_date="20500101")
                if df is None or df.empty:
                    self.logger.warning(f"{name} 无数据")
                    continue
                df["index_code"] = code
                df["trade_date"] = pd.to_datetime(df["trade_date"])
                df["year"] = df["trade_date"].dt.year
                for y, g in df.groupby("year"):
                    if y < self.start_year:
                        continue
                    self.save_year(g.drop(columns="year"), year=int(y), code=code, force=True)
                self.storage.mark_done(code)
                self.logger.info(f"{name} 日K完成: {len(df)} 行")
            except Exception as e:
                self.logger.error(f"{name}({code}) 失败: {e}")

        self.logger.info("指数日K下载完成")


class IndustryDownloader(BaseDownloader):
    """行业分类（tushare index_classify 申万/中信 + stock_basic industry）"""

    def __init__(self):
        super().__init__("industry", "industry", calls_per_min=100)
        self._tc = None

    def download(self, resume: bool = True):
        self._tc = get_client()
        # 申万2021 一级行业
        try:
            df = self._tc.call("index_classify", level="L1", src="SW2021")
            self.save_year(df, year=2005, code="sw_l1", force=True)
            self.logger.info(f"申万一级行业: {len(df)} 个")
        except Exception as e:
            self.logger.warning(f"申万行业失败: {e}")

        # 股票所属行业（来自 stock_basic）
        try:
            df = self._tc.call("stock_basic", list_status="L",
                               fields="ts_code,name,industry,list_date")
            self.save_year(df, year=2005, code="stock_industry", force=True)
            self.logger.info(f"股票行业归属: {len(df)} 只")
        except Exception as e:
            self.logger.warning(f"股票行业失败: {e}")

        self.storage.mark_done("all")
