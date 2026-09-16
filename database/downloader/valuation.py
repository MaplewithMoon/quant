"""每日估值指标下载器：PE/PB/PS/总市值/流通市值/换手率
后端: tushare daily_basic（完整字段，优先）
      回退: akshare百度（仅PE/PB/总市值）

⚠️ **口径必须与 frozen/valuation 的唯一约定一致，否则会静默污染整个数据集**
   `frozen/valuation` 的约定（见 `database/config.py::FROZEN_DATASETS` 与
   `scripts/download_frozen_tushare.py`）：
       total_mv / circ_mv  单位 = **万元**（tushare 原样，不换算）
       换手率列名 = **turnover_rate**（不是 `turnover`）
   本文件早期版本在这里做了"万元 → 亿元"换算并把列名改成 `turnover`，
   而它写的是**同一个目录**、且 `save_year(force=True)` 会覆盖 —— 一旦有人跑
   `python scripts/run_download.py --redownload valuation`，就会得到
   单位与列名都不同的混合数据：`factors/panel.py` 取 `turnover_rate` 会全空，
   `total_mv` 会缩小 1 万倍（市值因子、中性化、行业权重全错）。
   历史遗留的 `db/_legacy/valuation_norm` 就是这种"亿元归一/turnover"口径，
   已经删除；这里不再制造第二份。
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
        df["trade_date"] = pd.to_datetime(df["trade_date"])
        df["code"] = code
        # 单位与列名**保持 tushare 原样**：万元 / turnover_rate
        keep = ["code", "trade_date", "pe_ttm", "pb", "ps_ttm",
                "total_mv", "circ_mv", "turnover_rate"]
        df = df[[c for c in keep if c in df.columns]]
        return df.sort_values("trade_date")

    def _from_baidu(self, code: str) -> pd.DataFrame:
        """akshare 百度估值（降级源）

        ⚠️ 两个必须处理的差异，否则会污染 frozen/valuation：
          1. akshare 的 `总市值` 单位是**亿元**，本数据集约定是**万元** → ×1e4
          2. 它只给 pe_ttm / pb / total_mv，缺 `turnover_rate` / `circ_mv` /
             `ps_ttm`。**必须补齐成 NaN**（而不是不写这几列），否则这些文件
             与其他文件列不一致，任何 `read_parquet(glob)` 选到缺列都会抛
             `schema mismatch`。
        """
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
        if "total_mv" in merged.columns:
            # 亿元 -> 万元，与 tushare 口径一致
            merged["total_mv"] = merged["total_mv"] * 1e4
        for col in ("pe_ttm", "pb", "ps_ttm", "total_mv", "circ_mv", "turnover_rate"):
            if col not in merged.columns:
                merged[col] = pd.NA
        return merged.sort_values("trade_date")

    def download_code(self, code: str) -> pd.DataFrame:
        df = pd.DataFrame()
        try:
            df = self._from_tushare(code)
        except Exception as e:
            self.logger.warning(f"{code} tushare失败，回退百度: {str(e)[:50]}")
            df = self._from_baidu(code)

        if df.empty:
            self.storage.mark_done(code, empty=True)
            return df

        df["year"] = df["trade_date"].dt.year
        for y, g in df.groupby("year"):
            if y < self.start_year:
                continue
            self.save_year(g.drop(columns="year"), year=int(y), code=code, force=True)

        self.storage.mark_done(code, span=self._span_of(df))
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
