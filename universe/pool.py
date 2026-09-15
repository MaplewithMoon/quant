# -*- coding: utf-8 -*-
"""股票池构建与过滤

为什么必须用「历史成分」而不是「当前成分」
------------------------------------------
用今天的沪深300成分股去回测 2018 年，等于事先知道了哪些股票 2018-2024 会
留在指数里（这些通常是表现好的）——这就是**幸存者偏差（survivorship bias）**，
会让回测收益系统性虚高。

本模块从 `frozen/index_cons` 读取**逐期快照**（沪深300 有 256 个季度快照，
2005 年至今），在每个调仓日只使用"当时已经公布的成分"。

过滤维度
--------
    指数成分   历史成分（消除幸存者偏差）
    上市天数   次新股（上市不足 N 个交易日）
    流动性     日均成交额下限（避免买到没法成交的票）
    ST         风险警示股（可能退市、涨跌幅限制不同）
    停牌       停牌日不可交易
    涨跌停     建仓日一字板买不进（可选）
"""
from dataclasses import dataclass

import numpy as np
import pandas as pd

from database.config import FROZEN_ROOT


@dataclass
class UniverseSpec:
    """股票池规则"""
    index_code: str = None        # 如 '000300.SH'；None = 全市场
    min_listed_days: int = 60     # 上市不足该交易日数的剔除（次新股）
    min_amount: float = 0.0       # 日均成交额下限（元）
    amount_window: int = 20       # 计算日均成交额的窗口
    exclude_st: bool = True
    exclude_suspended: bool = True
    top_n: int = 0                # 按流动性取前 N（0=不限）

    def describe(self) -> str:
        parts = [f"指数={self.index_code or '全市场'}"]
        if self.min_listed_days:
            parts.append(f"上市≥{self.min_listed_days}日")
        if self.min_amount:
            parts.append(f"日均成交额≥{self.min_amount/1e4:.0f}万")
        if self.exclude_st:
            parts.append("剔除ST")
        if self.exclude_suspended:
            parts.append("剔除停牌")
        if self.top_n:
            parts.append(f"取流动性前{self.top_n}")
        return " / ".join(parts)


# ============================================================
# 指数历史成分
# ============================================================
def load_index_members(index_code: str = "000300.SH") -> pd.DataFrame:
    """读取指数成分的历史快照（长表：trade_date, code, weight）"""
    from database.config import connect_duckdb
    glob = f"{FROZEN_ROOT.as_posix()}/index_cons/year=*/*.parquet"
    con = connect_duckdb()
    try:
        df = con.execute(f"""
            SELECT CAST(trade_date AS VARCHAR) AS trade_date, con_code, weight
            FROM read_parquet('{glob}')
            WHERE index_code = ?
        """, [index_code]).fetchdf()
    finally:
        con.close()
    if df.empty:
        return df
    # index_cons.trade_date 是 VARCHAR；走统一容错解析（tushare 偶有带时间的格式）
    from database.dates import parse_tushare_date
    df["trade_date"] = parse_tushare_date(df["trade_date"])
    df = df.dropna(subset=["trade_date"])
    # 'con_code' 形如 300750.SZ -> 提取 6 位代码
    df["code"] = df["con_code"].astype(str).str.split(".").str[0]
    return df[["trade_date", "code", "weight"]].sort_values("trade_date")


def _latest_snapshot_rows(snap: pd.DataFrame, dates: pd.DatetimeIndex) -> pd.DataFrame:
    """取每个交易日对应的"最近一次快照"的整行

    ⚠️ 这里不能用 `reindex(...).ffill()`：那样会把**每一列独立地**前向填充，
    于是某只股票被调出指数后（该快照里没有它 → NaN），仍会沿用上一次的权重，
    永远不会被剔除。实测这样会让沪深300 的成分数从 300 虚增到 865。

    正确做法：为每个日期找到 `<= 该日` 的最近一次快照，直接取那一行。
    """
    if snap.empty or len(dates) == 0:
        return pd.DataFrame(index=dates, columns=snap.columns, dtype=float)
    pos = snap.index.searchsorted(dates, side="right") - 1
    out = pd.DataFrame(np.nan, index=dates, columns=snap.columns, dtype=float)
    valid = pos >= 0
    if valid.any():
        out.iloc[np.flatnonzero(valid)] = snap.values[pos[valid]]
    return out


def index_member_panel(index_code: str, dates: pd.DatetimeIndex,
                       codes) -> pd.DataFrame:
    """历史成分掩码（宽表 bool）

    t 日使用 `trade_date <= t` 的**最近一次**快照 —— 即当时真实可得的成分，
    不使用未来快照，也不使用今天的成分。
    """
    members = load_index_members(index_code)
    if members.empty:
        return pd.DataFrame(True, index=dates, columns=list(codes))
    snap = members.pivot_table(index="trade_date", columns="code",
                               values="weight", aggfunc="first").sort_index()
    aligned = _latest_snapshot_rows(snap, dates)
    return aligned.reindex(columns=list(codes)).notna()


def index_weight_panel(index_code: str, dates: pd.DatetimeIndex,
                       codes) -> pd.DataFrame:
    """基准权重面板（宽表，逐日归一化到 1）

    用于业绩归因：t 日使用 <= t 的最近一次成分权重快照。
    """
    members = load_index_members(index_code)
    if members.empty:
        return pd.DataFrame()
    snap = members.pivot_table(index="trade_date", columns="code",
                               values="weight", aggfunc="first").sort_index()
    aligned = _latest_snapshot_rows(snap, dates).reindex(columns=list(codes))
    tot = aligned.sum(axis=1).replace(0, np.nan)
    return aligned.div(tot, axis=0)


# ============================================================
# ST / 停牌
# ============================================================
def st_panel(dates: pd.DatetimeIndex, codes) -> pd.DataFrame:
    """ST 状态掩码（宽表 bool）：该日该股是否处于 ST/*ST 状态

    转调 `database/limit_rules.py` 的**唯一** ST 区间实现（涨跌停价也用同一份）。
    旧实现自己写了一遍，并且 `except Exception: df = pd.DataFrame()` 静默吞错 ——
    一旦读取失败就返回全 False，**ST 股会被当成正常股留在股票池里**。
    现在数据集存在却解析不出内容时直接抛错。
    """
    from database.limit_rules import ST_DIR_DEFAULT, load_st_intervals, st_mask

    idx = pd.DatetimeIndex(dates)
    cols = [str(c).zfill(6) for c in codes]
    mask = pd.DataFrame(False, index=idx, columns=cols)
    if not ST_DIR_DEFAULT.exists():
        print("  [警告] frozen/st 不存在，ST 过滤无法生效")
        return mask
    st_map = load_st_intervals()
    if not st_map:
        raise RuntimeError("frozen/st 存在但未解析出任何 ST 区间 —— "
                           "拒绝静默返回全 False（会让 ST 股进入股票池）")
    for code in cols:
        if st_map.get(code):
            mask[code] = st_mask(idx, code, st_map)
    return mask


def suspended_panel(dates: pd.DatetimeIndex, codes) -> pd.DataFrame:
    """停牌掩码（宽表 bool）—— 转调唯一实现 database/status.py

    旧实现自己写了一遍，而且把 `suspend_type='R'`（**复牌日**，当天可交易）
    也当成停牌，会在复牌当天错误地把股票排除出股票池。
    """
    from database.status import suspended_wide
    return suspended_wide(dates, codes)


# ============================================================
# 股票池
# ============================================================
def listed_days_panel(panel: dict) -> pd.DataFrame:
    """每只股票截至当日的累计有效交易 bar 数（近似上市天数）"""
    from factors.panel import adjusted_close
    close = adjusted_close(panel)
    return close.notna().cumsum()


def build_universe(panel: dict, spec: UniverseSpec = None, **kw) -> pd.DataFrame:
    """构造逐日股票池掩码（宽表 bool，True=当日可选）

    参数:
        panel: factors.panel.load_panel() 的产出
        spec:  UniverseSpec；也可用关键字直接覆盖，如 build_universe(panel, index_code="000300.SH")
    """
    if spec is None:
        spec = UniverseSpec(**kw)
    elif kw:
        spec = UniverseSpec(**{**spec.__dict__, **kw})

    from factors.panel import adjusted_close
    close = adjusted_close(panel)
    dates, codes = close.index, close.columns

    mask = close.notna()                       # 有行情
    mask &= close.shift(1).notna()             # 至少两根 bar（能算收益）

    # 上市天数
    if spec.min_listed_days:
        mask &= listed_days_panel(panel) >= spec.min_listed_days

    # 流动性
    if spec.min_amount and "amount" in panel:
        avg_amt = panel["amount"].rolling(spec.amount_window, min_periods=5).mean()
        mask &= avg_amt >= spec.min_amount

    # 指数历史成分
    if spec.index_code:
        mask &= index_member_panel(spec.index_code, dates, codes)

    # ST / 停牌
    if spec.exclude_st:
        mask &= ~st_panel(dates, codes)
    if spec.exclude_suspended:
        mask &= ~suspended_panel(dates, codes)

    # 流动性前 N
    if spec.top_n:
        if "amount" not in panel:
            raise KeyError("top_n 需要 amount 面板")
        amt = panel["amount"].where(mask)
        rank = amt.rank(axis=1, ascending=False, method="first")
        mask &= rank <= spec.top_n

    return mask.fillna(False).astype(bool)


def universe_size(mask: pd.DataFrame) -> pd.Series:
    """每日可选股票数（用于检查股票池是否稳定）"""
    return mask.sum(axis=1)
