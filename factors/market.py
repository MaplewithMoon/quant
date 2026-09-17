# -*- coding: utf-8 -*-
"""市场级（时间序列）信号：两融 / 北向 / 股指期货基差 / 期权 PCR

【与 factors/panel.py 的分工】
    `panel.py` 给的是**横截面**面板（index=交易日, columns=股票），用来选股；
    本模块给的是**时间序列**信号（index=交易日, 单列），用来调**总仓位**。
    两者形状不同，用途也不同，不要混。

【为什么要有这些】
    项目里 `frozen/margin`、`frozen/northbound`、`frozen/futures`、
    `frozen/options`、`frozen/etf` 五个数据集此前**只有下载器、没有任何消费方**
    （B12）。下了不用既浪费又容易让人误以为在用。本模块把它们接成信号。

【PIT（这条最要紧）】
    这四个数据集都是**收盘后发布**的：两融要 T 日晚上甚至 T+1 才出，
    北向、期货结算、期权持仓同理。所以"某天的资金流"最早只能用于
    **下一个交易日**。所有构造函数默认 `lag=1`（把信号右移一个交易日）。
    不这么做就是典型前视 —— 拿今天收盘后才知道的数去决定今天怎么交易。

【已知局限（都来自数据本身，不是实现）】
    - `northbound` 只有 2025-05 起的 300 行，**不能用于更早的回测**
    - `options` 的 `opt_basic` 只覆盖 SSE（4.6% 的合约），PCR 只对这部分有效；
      `download_frozen_tushare.py --only opt_basic` 可以补齐（只需 8 次调用）
    - `futures` 只有主连，没有各月份合约，算不了期限结构
"""
from typing import Dict, Optional

import numpy as np
import pandas as pd

from database.config import FROZEN_ROOT, connect_duckdb, year_globs

__all__ = ["load_margin", "load_northbound", "load_futures_basis",
           "load_option_pcr", "market_signals", "exposure_from_signal",
           "apply_exposure", "load_etf_prices", "load_etf_basic",
           "FUTURES_INDEX_PAIRS"]

# 股指期货主连 → 对应现货指数（算基差用）
# 注意 IH(上证50) 没有对应指数行情（库里只有 000300/000905/000852/000985/
# 399101/932000），T/TF/TS 是国债期货也没有现货，所以都不在表里。
FUTURES_INDEX_PAIRS = {
    "IF": "000300.SH",      # 沪深300
    "IC": "000905.SH",      # 中证500
    "IM": "000852.SH",      # 中证1000
}

_DEFAULT_LAG = 1


# ============================================================
# 原始读取
# ============================================================
def load_margin(start=None, end=None) -> pd.DataFrame:
    """两融余额（**全市场日频**，SSE+SZSE 合计）

    tushare 的 `margin` 是**交易所级**汇总（不是逐股），字段:
        rzye   融资余额（元）      rzmre  融资买入额
        rzche  融资偿还额          rqye   融券余额
        rzrqye 融资融券余额        rqyl   融券余量
    返回: index=交易日, columns=[rzye, rzmre, rzche, rqye, rzrqye]
    """
    d = FROZEN_ROOT / "margin"
    if not d.exists():
        return pd.DataFrame()
    g = year_globs(d, *_years(start, end))
    if g == "[]":
        return pd.DataFrame()
    con = connect_duckdb()
    try:
        df = con.execute(f"""
            SELECT trade_date,
                   sum(rzye) AS rzye, sum(rzmre) AS rzmre, sum(rzche) AS rzche,
                   sum(rqye) AS rqye, sum(rzrqye) AS rzrqye
            FROM read_parquet({g})
            GROUP BY 1 ORDER BY 1
        """).fetchdf()
    finally:
        con.close()
    return _finish(df, start, end)


def load_northbound(start=None, end=None) -> pd.DataFrame:
    """北向资金（全市场日频）

    ⚠️ 库里只有 **2025-05-27 起** 约 300 个交易日 —— 交易所 2024-08 起
    停止披露逐股北向持股，这里只有市场级成交/净买入口径。更早的回测用不了。
    返回: index=交易日, columns=[hgt, sgt, north_money, south_money]
    """
    d = FROZEN_ROOT / "northbound"
    if not d.exists():
        return pd.DataFrame()
    g = year_globs(d, *_years(start, end))
    if g == "[]":
        return pd.DataFrame()
    con = connect_duckdb()
    try:
        df = con.execute(f"""
            SELECT trade_date, hgt, sgt, north_money, south_money
            FROM read_parquet({g}) ORDER BY 1
        """).fetchdf()
    finally:
        con.close()
    return _finish(df, start, end)


def load_futures_basis(codes=None, start=None, end=None) -> pd.DataFrame:
    """股指期货**基差**（期货相对现货的升贴水）

        基差 = (期货收盘 - 现货指数收盘) / 现货指数收盘

    A 股股指期货常年**贴水**（基差为负）。贴水收窄或转升水通常被解读为
    对冲/看空需求下降。返回 index=交易日, columns=[IF_basis, IC_basis, IM_basis]
    （百分比，如 -0.012 表示贴水 1.2%）
    """
    codes = list(codes or FUTURES_INDEX_PAIRS)
    fdir, idir = FROZEN_ROOT / "futures", FROZEN_ROOT / "index_daily"
    if not fdir.exists() or not idir.exists():
        return pd.DataFrame()
    y0, y1 = _years(start, end)
    gf, gi = year_globs(fdir, y0, y1), year_globs(idir, y0, y1)
    if gf == "[]" or gi == "[]":
        return pd.DataFrame()
    con = connect_duckdb()
    try:
        f = con.execute(f"""
            SELECT symbol, trade_date, close FROM read_parquet({gf})
            WHERE symbol IN ({','.join(['?'] * len(codes))})
        """, codes).fetchdf()
        idx = con.execute(f"""
            SELECT ts_code, trade_date, close FROM read_parquet({gi})
            WHERE ts_code IN ({','.join(['?'] * len(set(FUTURES_INDEX_PAIRS.values())))})
        """, sorted(set(FUTURES_INDEX_PAIRS.values()))).fetchdf()
    finally:
        con.close()
    if f.empty or idx.empty:
        return pd.DataFrame()
    f["trade_date"] = pd.to_datetime(f["trade_date"])
    idx["trade_date"] = pd.to_datetime(idx["trade_date"])
    out = {}
    for sym in codes:
        spot = FUTURES_INDEX_PAIRS.get(sym)
        if spot is None:
            continue
        a = f[f["symbol"] == sym][["trade_date", "close"]].rename(columns={"close": "fut"})
        b = idx[idx["ts_code"] == spot][["trade_date", "close"]].rename(columns={"close": "spot"})
        m = a.merge(b, on="trade_date", how="inner").dropna()
        if m.empty:
            continue
        out[f"{sym}_basis"] = (m.set_index("trade_date")["fut"]
                               / m.set_index("trade_date")["spot"] - 1.0)
    if not out:
        return pd.DataFrame()
    res = pd.DataFrame(out).sort_index()
    if start is not None:
        res = res[res.index >= pd.Timestamp(start)]
    if end is not None:
        res = res[res.index <= pd.Timestamp(end)]
    return res


def load_option_pcr(start=None, end=None) -> pd.DataFrame:
    """期权 **Put/Call Ratio**（认沽/认购成交与持仓之比）

    PCR 高 = 买认沽的多 = 情绪偏空（常用作**反向**指标）。
    返回: index=交易日, columns=[pcr_vol, pcr_oi, opt_vol, opt_oi]

    ⚠️ 覆盖率限制：`opt_basic` 目前**只有 SSE**（12,000 个合约，占日线合约的
    4.6%），所以这里算出来的 PCR 只代表上交所 ETF 期权。补齐要跑
    `python scripts/download_frozen_tushare.py --only opt_basic`（8 次调用）。
    覆盖率不足时会在返回值的 attrs 里标注 `coverage`。
    """
    d = FROZEN_ROOT / "options"
    basic = d / "opt_basic.parquet"
    if not d.exists() or not basic.exists():
        return pd.DataFrame()
    g = year_globs(d, *_years(start, end))
    if g == "[]":
        return pd.DataFrame()
    con = connect_duckdb()
    try:
        # 先看 opt_basic 覆盖了多少合约，避免"算出来了但其实只代表一小撮"
        tot = con.execute(f"SELECT count(DISTINCT ts_code) FROM read_parquet({g})").fetchone()[0]
        cov = con.execute(f"""
            SELECT count(DISTINCT o.ts_code) FROM read_parquet({g}) o
            JOIN read_parquet('{basic.as_posix()}') b USING (ts_code)
        """).fetchone()[0]
        df = con.execute(f"""
            SELECT o.trade_date, b.call_put,
                   sum(o.vol) AS v, sum(o.oi) AS oi
            FROM read_parquet({g}) o
            JOIN read_parquet('{basic.as_posix()}') b USING (ts_code)
            GROUP BY 1, 2 ORDER BY 1
        """).fetchdf()
    finally:
        con.close()
    if df.empty:
        return pd.DataFrame()
    df["trade_date"] = pd.to_datetime(df["trade_date"])
    w = df.pivot_table(index="trade_date", columns="call_put",
                       values=["v", "oi"], aggfunc="sum").fillna(0.0)
    res = pd.DataFrame(index=w.index)
    res["pcr_vol"] = _safe_ratio(w.get("v", {}).get("P"), w.get("v", {}).get("C"))
    res["pcr_oi"] = _safe_ratio(w.get("oi", {}).get("P"), w.get("oi", {}).get("C"))
    res["opt_vol"] = w["v"].sum(axis=1) if "v" in w else np.nan
    res["opt_oi"] = w["oi"].sum(axis=1) if "oi" in w else np.nan
    res = res.sort_index()
    res.attrs["coverage"] = float(cov) / float(tot) if tot else 0.0
    res.attrs["coverage_note"] = (
        f"opt_basic 覆盖 {cov:,}/{tot:,} = {res.attrs['coverage']:.1%} 的合约；"
        f"当前只含 SSE（上交所 ETF 期权），跑 `--only opt_basic` 可补齐其他交易所")
    if start is not None:
        res = res[res.index >= pd.Timestamp(start)]
    if end is not None:
        res = res[res.index <= pd.Timestamp(end)]
    return res


# ============================================================
# 信号合成（都带 PIT lag）
# ============================================================
def market_signals(start=None, end=None, lag: int = _DEFAULT_LAG,
                   margin_window: int = 20) -> pd.DataFrame:
    """把所有市场级信号拼成一张表（index=交易日）

    列:
        margin_chg    两融余额的 `margin_window` 日变化率（杠杆资金进出）
        margin_rzrq   两融余额的绝对水平（万元->亿元，仅供画图）
        north_net     `margin_window` 日北向净流入均值（仅 2025-05 起）
        IF_basis / IC_basis / IM_basis   股指期货基差
        pcr_vol / pcr_oi                 期权认沽认购比

    ⚠️ **所有列都已按 lag 右移**（默认 1 个交易日）——当天的值当天拿不到。
    缺失的列（如数据不足）不会造出来，调用方需自行判断。
    """
    parts = []
    mg = load_margin(start, end)
    if not mg.empty and "rzye" in mg:
        s = mg["rzye"].astype(float)
        parts.append(pd.DataFrame({
            "margin_chg": s.pct_change(margin_window),
            "margin_rzrq": s / 1e8,                     # 亿元，便于阅读
        }))
    nb = load_northbound(start, end)
    if not nb.empty and "north_money" in nb:
        parts.append(pd.DataFrame({
            "north_net": nb["north_money"].astype(float)
                           .rolling(margin_window).mean(),
        }))
    fb = load_futures_basis(start=start, end=end)
    if not fb.empty:
        parts.append(fb)
    try:
        pcr = load_option_pcr(start, end)
    except Exception:
        pcr = pd.DataFrame()
    if not pcr.empty:
        parts.append(pcr[["pcr_vol", "pcr_oi"]])
    if not parts:
        return pd.DataFrame()
    out = pd.concat(parts, axis=1).sort_index()
    # 期货/期权有数据的交易日，两融/北向不一定有（反之亦然）——不填充，
    # 让调用方按需要的列 dropna，避免"前向填充出一个假的当日值"
    return out.shift(lag) if lag else out


def exposure_from_signal(signal: pd.Series, low_q: float = 0.3,
                         high_q: float = 0.7,
                         min_exposure: float = 0.3,
                         max_exposure: float = 1.0,
                         expanding: bool = True) -> pd.Series:
    """把一个择时信号映射成**目标总仓位**（0~1）

    做法：按信号自身的分位数分档 —— 信号 ≤ `low_q` 分位 -> `min_exposure`，
    ≥ `high_q` 分位 -> `max_exposure`，中间线性。

    ⚠️ `expanding=True`（默认）用**扩张窗口**算分位，即第 t 天的阈值只用
    [起点, t] 的数据 —— 这是能在实盘复现的做法。用全样本分位就是前视：
    2018 年的信号会"知道"2025 年的分布。
    """
    s = signal.dropna()
    if s.empty:
        return pd.Series(dtype=float)
    if expanding:
        lo = s.expanding(min_periods=max(20, len(s) // 20)).quantile(low_q)
        hi = s.expanding(min_periods=max(20, len(s) // 20)).quantile(high_q)
    else:
        lo = pd.Series(s.quantile(low_q), index=s.index)
        hi = pd.Series(s.quantile(high_q), index=s.index)
    span = (hi - lo).replace(0, np.nan)
    x = ((s - lo) / span).clip(0, 1)
    expo = min_exposure + x * (max_exposure - min_exposure)
    return expo.fillna((min_exposure + max_exposure) / 2)


def apply_exposure(weights: pd.DataFrame, exposure: pd.Series,
                   renormalize: bool = False) -> pd.DataFrame:
    """按目标总仓位缩放权重面板

    参数:
        weights:   目标权重宽表（index=交易日, columns=代码），非调仓日为 NaN
        exposure:  目标总仓位（index=交易日, 0~1）。缺失的日期按 1.0 处理
                   （不因为择时信号缺失就默认空仓）
        renormalize:True 时缩放后再归一化到 1（即只做"选哪些股"的调整，
                   不改总仓位）—— 一般用不到，保留给"相对权重"场景

    语义：把**非 NaN 行**整体乘以当天的 exposure。注意 `weights` 里 NaN 表示
    "当天不调仓"，缩放不会把它变成非 NaN。
    """
    if weights.empty:
        return weights
    e = exposure.reindex(weights.index).fillna(1.0).clip(0.0, 1.0)
    out = weights.mul(e, axis=0)
    if renormalize:
        row = out.sum(axis=1)
        out = out.div(row.replace(0, np.nan), axis=0)
    return out


# ============================================================
# ETF（`frozen/etf` 是**按日期**分区的，与股票面板不同，需单独读）
# ============================================================
def load_etf_basic() -> pd.DataFrame:
    """ETF/基金基础信息（fund_basic）：name / fund_type / list_date / benchmark"""
    f = FROZEN_ROOT / "etf" / "fund_basic.parquet"
    if not f.exists():
        return pd.DataFrame()
    con = connect_duckdb()
    try:
        df = con.execute(f"SELECT * FROM read_parquet('{f.as_posix()}')").fetchdf()
    finally:
        con.close()
    return df


def load_etf_prices(codes=None, start=None, end=None,
                    field: str = "close") -> pd.DataFrame:
    """ETF 价格面板（宽表 index=交易日, columns=ts_code）

    ⚠️ `frozen/etf` 的分区是 `year={Y}/{日期}.parquet`（**按日期**），
    与股票日线的 `year={Y}/{代码}.parquet` 相反，所以不能走 `load_panel`。

    codes: ts_code 列表（如 '510300.SH'）；None = 全部（约 2,560 只，较慢）
    """
    d = FROZEN_ROOT / "etf"
    if not d.exists():
        return pd.DataFrame()
    g = year_globs(d, *_years(start, end))
    if g == "[]":
        return pd.DataFrame()
    where, params = [], []
    if codes:
        codes = list(codes)
        where.append(f"ts_code IN ({','.join(['?'] * len(codes))})")
        params += codes
    if start is not None:
        where.append("trade_date >= ?")
        params.append(pd.Timestamp(start))
    if end is not None:
        where.append("trade_date <= ?")
        params.append(pd.Timestamp(end))
    wsql = ("WHERE " + " AND ".join(where)) if where else ""
    con = connect_duckdb()
    try:
        df = con.execute(f"""
            SELECT ts_code, trade_date, {field} FROM read_parquet({g}) {wsql}
        """, params).fetchdf()
    finally:
        con.close()
    if df.empty:
        return pd.DataFrame()
    df["trade_date"] = pd.to_datetime(df["trade_date"])
    return df.pivot_table(index="trade_date", columns="ts_code",
                          values=field, aggfunc="last").sort_index()


# ============================================================
# 工具
# ============================================================
def _years(start, end):
    return (pd.Timestamp(start).year if start is not None else None,
            pd.Timestamp(end).year if end is not None else None)


def _finish(df: pd.DataFrame, start, end) -> pd.DataFrame:
    if df is None or df.empty:
        return pd.DataFrame()
    df["trade_date"] = pd.to_datetime(df["trade_date"])
    df = df.set_index("trade_date").sort_index()
    if start is not None:
        df = df[df.index >= pd.Timestamp(start)]
    if end is not None:
        df = df[df.index <= pd.Timestamp(end)]
    return df


def _safe_ratio(a, b):
    if a is None or b is None:
        return pd.Series(dtype=float)
    return (a / b.replace(0, np.nan))
