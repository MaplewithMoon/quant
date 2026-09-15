# -*- coding: utf-8 -*-
"""业绩归因：行业暴露 / 因子暴露 / Brinson 分解

回答三个问题：
    1. 我的组合押在哪些行业上？（industry_exposure）
    2. 我的组合在哪些风格因子上有暴露？（factor_exposure）
    3. 相对基准的超额收益，多少来自"配置"、多少来自"选股"？（brinson）

Brinson-Fachler 分解
--------------------
对每个行业 i：
    配置效应  = (w_p,i − w_b,i) × (r_b,i − r_b)
    选股效应  = w_b,i × (r_p,i − r_b,i)
    交互效应  = (w_p,i − w_b,i) × (r_p,i − r_b,i)
三者之和 = 组合收益 − 基准收益。
"""
from typing import Dict, Optional

import numpy as np
import pandas as pd


# ============================================================
# 行业暴露
# ============================================================
def load_industry_map() -> pd.DataFrame:
    """股票 -> 行业映射（来自 database.loader）"""
    from database.loader import load_industry_map as _f
    df = _f()
    if df.empty:
        return df
    if "code" not in df.columns:
        if "ts_code" in df.columns:
            df = df.assign(code=df["ts_code"].astype(str).str.split(".").str[0])
        else:
            return pd.DataFrame()
    for col in ("industry", "sw_l1", "name"):
        if col in df.columns:
            out = df[["code", col]].dropna().drop_duplicates("code")
            return out.rename(columns={col: "industry"})
    return pd.DataFrame()


def industry_exposure(holdings: pd.DataFrame,
                      industry_map: pd.DataFrame) -> pd.DataFrame:
    """逐日行业权重暴露（index=日期, columns=行业）

    holdings: 宽表，逐日个股权重（MultiBacktestResult.holdings）
    """
    if holdings.empty or industry_map.empty:
        return pd.DataFrame()
    m = industry_map.set_index("code")["industry"]
    ind = m.reindex(holdings.columns)
    out = {}
    for d, row in holdings.iterrows():
        r = row.dropna()
        if r.empty:
            continue
        out[d] = r.groupby(ind.reindex(r.index)).sum()
    return pd.DataFrame(out).T.sort_index().fillna(0.0)


# ============================================================
# 因子暴露
# ============================================================
def factor_exposure(holdings: pd.DataFrame, factor_z: pd.DataFrame,
                    mask: pd.DataFrame = None) -> pd.Series:
    """组合的因子暴露 = Σ 权重 × 因子横截面标准化值（逐日）

    正值表示组合在该因子上偏向"高分"一端，负值偏向"低分"一端。

    内部会对因子按日做横截面标准化（减均值、除标准差）：
      * 传原始因子（如 0..29 的名次）也能得到可解释的暴露；
      * 对已经标准化的因子是幂等的（均值 0、标准差 1 再标准化不变）。
    标准化的分母取**全池**有效样本，而不是仅持仓——否则暴露的含义会随持仓变化。
    """
    h = holdings.reindex(columns=factor_z.columns).fillna(0.0)
    z = factor_z.reindex(index=h.index)
    if mask is not None:
        z = z.where(mask.reindex(index=h.index, columns=z.columns))
    mu = z.mean(axis=1)
    sd = z.std(axis=1, ddof=0).replace(0, np.nan)
    z = z.sub(mu, axis=0).div(sd, axis=0)
    return (h * z).sum(axis=1, min_count=1)


def multi_factor_exposure(holdings: pd.DataFrame,
                          factor_panels: Dict[str, pd.DataFrame]) -> pd.DataFrame:
    """多个因子的暴露表（index=日期, columns=因子）"""
    out = {name: factor_exposure(holdings, z) for name, z in factor_panels.items()}
    return pd.DataFrame(out)


# ============================================================
# Brinson
# ============================================================
def _industry_returns(weights: pd.Series, asset_ret: pd.Series,
                      ind: pd.Series) -> pd.Series:
    """给定权重与个股收益，算各行业收益（行业内部按权重加权）"""
    r = asset_ret.reindex(weights.index)
    g = ind.reindex(weights.index)
    df = pd.DataFrame({"w": weights, "r": r, "ind": g}).dropna()
    if df.empty:
        return pd.Series(dtype=float)
    grp = df.groupby("ind")
    num = grp.apply(lambda x: (x["w"] * x["r"]).sum(), include_groups=False)
    den = grp["w"].sum()
    return (num / den.replace(0, np.nan)).dropna()


def brinson(holdings: pd.DataFrame, benchmark_weights: pd.DataFrame,
            returns: pd.DataFrame, industry_map: pd.DataFrame,
            rebalance_dates=None) -> pd.DataFrame:
    """Brinson-Fachler 归因

    参数:
        holdings:           组合逐日个股权重
        benchmark_weights:  基准逐日个股权重（如沪深300 权重）
        returns:            个股逐日收益（宽表）
        industry_map:       code -> industry
        rebalance_dates:    归因区间起点（默认用 holdings 中权重发生变化的日期）

    返回:
        DataFrame(index=行业, columns=[配置效应, 选股效应, 交互效应, 合计])，
        并附一行 "总计"。
    """
    if holdings.empty or benchmark_weights.empty or industry_map.empty:
        return pd.DataFrame()
    ind = industry_map.set_index("code")["industry"]

    # 归因区间：取权重发生变化的日期 + 最后一天
    if rebalance_dates is None:
        w = holdings.fillna(0.0)
        changed = (w.diff().abs().sum(axis=1) > 1e-9)
        rebalance_dates = list(w.index[changed])
    rebalance_dates = [d for d in rebalance_dates if d in holdings.index]
    # 权重全程不变（或只变过一次）时，退化为"整段一期"的归因
    if len(rebalance_dates) < 2:
        rebalance_dates = [holdings.index[0], holdings.index[-1]]
    elif rebalance_dates[-1] != holdings.index[-1]:
        rebalance_dates.append(holdings.index[-1])

    rows = []
    for t0, t1 in zip(rebalance_dates, rebalance_dates[1:]):
        seg = returns.loc[(returns.index > t0) & (returns.index <= t1)]
        if seg.empty:
            continue
        asset_ret = (1 + seg.fillna(0)).prod() - 1          # 区间个股收益
        wp = holdings.loc[t0].dropna()
        wb = benchmark_weights.loc[t0].dropna() if t0 in benchmark_weights.index else pd.Series(dtype=float)
        if wp.empty or wb.empty:
            continue
        wp = wp / wp.sum()
        wb = wb / wb.sum()

        rp = float((wp * asset_ret.reindex(wp.index).fillna(0)).sum())
        rb = float((wb * asset_ret.reindex(wb.index).fillna(0)).sum())
        rp_i = _industry_returns(wp, asset_ret, ind)
        rb_i = _industry_returns(wb, asset_ret, ind)

        wp_i = wp.groupby(ind.reindex(wp.index)).sum()
        wb_i = wb.groupby(ind.reindex(wb.index)).sum()
        inds = wp_i.index.union(wb_i.index)
        for i in inds:
            wpi = float(wp_i.get(i, 0.0))
            wbi = float(wb_i.get(i, 0.0))
            rbi = float(rb_i.get(i, 0.0))
            rpi = float(rp_i.get(i, rbi))
            alloc = (wpi - wbi) * (rbi - rb)
            select = wbi * (rpi - rbi)
            inter = (wpi - wbi) * (rpi - rbi)
            rows.append({"行业": i, "配置效应": alloc, "选股效应": select,
                         "交互效应": inter})
    if not rows:
        return pd.DataFrame()
    df = pd.DataFrame(rows).groupby("行业", as_index=True).sum()
    df["合计"] = df.sum(axis=1)
    total = df.sum().rename("总计")
    return pd.concat([df.sort_values("合计", ascending=False), total.to_frame().T])
