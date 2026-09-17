"""股票列表 + 上市日期下载器（tushare stock_basic）—— 证券主表（security master）

【为什么这份表这么重要】
它是**所有股票池的根**。时点正确的可交易池必须靠三层过滤：

    ① exchange in ('SSE','SZSE')                      交易所归属（天然排除 NEEQ/BSE）
    ② list_date <= t                                    t 日必须已上市
    ③ delist_date 为空 或 delist_date > t                t 日必须未退市

而这三层各自的字段以前**一个都没存**：
    - 旧实现的 `fields` 只请求了 `ts_code,symbol,name,area,industry,list_date,market`
      —— **没有 `exchange`、没有 `delist_date`**；
    - L/D/P 三次调用 concat 到一起时**没记 `list_status`**，
      于是"上市 / 退市 / 暂停"这个最关键的区分在落盘时丢失了。
落盘的 `all.parquet` 只剩 5,889 行 × 8 列，只能靠"名字里含『退』"这种
启发式去猜退市股 —— 那既不准，方向也错（见下）。

【拉三种状态，避免幸存者偏差】
    L = 上市中 / D = 已退市 / P = 暂停上市
只拉 L 的话，回测历史时"当时存在、后来退市"的股票全都不在池子里，
组合表现会被系统性高估 —— 这就是**幸存者偏差**。

【为什么"名字含退"是错的判据】
它把"今天叫什么名字"当成"当时是否可交易"。真实的退市股不一定带"退"
（改名、重组、被吸收合并都会），而带"退"的也可能还在退市整理期可交易。
正确做法是看 `delist_date` 与 `t` 的关系 —— 一个纯粹的日期比较。
"""
import pandas as pd

from .base import BaseDownloader
from .tushare_client import get_client

# 证券主表需要的字段。
# ⚠️ 缺 `exchange` / `delist_date` 就没法做时点正确的股票池，见模块文档。
MASTER_FIELDS = ("ts_code,symbol,name,area,industry,market,"
                 "exchange,list_date,delist_date")


class StockListDownloader(BaseDownloader):
    """A股证券主表（代码、名称、交易所、上市/退市日期）"""

    def __init__(self):
        super().__init__("stocks", "stocks", calls_per_min=100)

    def download(self):
        self._tc = get_client()
        frames = []
        counts = {}
        for status in ("L", "D", "P"):
            d = self._tc.call("stock_basic", list_status=status,
                              fields=MASTER_FIELDS)
            if d is None or d.empty:
                counts[status] = 0
                continue
            d = d.copy()
            # **必须记下来**：L/D/P 是三次调用的语义，接口不返回这一列，
            # 不手动打标就会在 concat 时永久丢失
            d["list_status"] = status
            frames.append(d)
            counts[status] = len(d)
        if not frames:
            self.logger.error("stock_basic 三次调用都为空")
            return pd.DataFrame()

        all_df = pd.concat(frames, ignore_index=True)
        all_df["code"] = all_df["symbol"].astype(str).str.strip()
        all_df["list_date"] = pd.to_datetime(all_df["list_date"], errors="coerce")
        if "delist_date" in all_df.columns:
            # 未退市时 tushare 返回空串/NaN，统一成 NaT
            all_df["delist_date"] = pd.to_datetime(all_df["delist_date"],
                                                   errors="coerce")
        else:
            all_df["delist_date"] = pd.NaT
        all_df["exchange"] = all_df.get(
            "exchange", pd.Series(index=all_df.index, dtype=object))

        # 同一只股票理论上只出现在一个状态里；真重复时优先保留信息更全的
        # （L 有 delist_date 空、D 有具体退市日）
        before = len(all_df)
        all_df = all_df.sort_values("delist_date", na_position="first")
        all_df = all_df.drop_duplicates(subset=["ts_code"], keep="last")
        if len(all_df) != before:
            self.logger.warning(f"证券主表去重: {before} -> {len(all_df)} 行")

        self.save_year(all_df, year=2005, code="all", force=True)
        self.storage.mark_done("all")
        n_del = int(all_df["delist_date"].notna().sum())
        self.logger.info(
            f"证券主表完成: 上市{counts.get('L', 0)} + 退市{counts.get('D', 0)}"
            f" + 暂停{counts.get('P', 0)} = {len(all_df)} 只；"
            f"其中 delist_date 非空 {n_del} 只")
        return all_df
