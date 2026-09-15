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


UNKNOWN_INDUSTRY = "未分类"


def _industry_series(industry_map: pd.DataFrame, columns=None) -> pd.Series:
    """code -> 行业；**缺失的一律落到"未分类"而不是 NaN**

    ⚠️ 为什么不能留 NaN：`groupby` 会直接丢掉 NaN 分组，
    于是这些股票的权重和收益在整个拆解里凭空消失，
    Brinson 恒等式 Σ(配置+选股+交互) = r_p − r_b 立刻不成立
    —— 实测全市场组合（持有大量已退市/非当前行业表的股票）时，
    总计只有 −131%，而真实超额是 −108%，差的就是这块被吞掉的权重。

    `industry_map` 是**当前**快照，退市股天然不在里面，所以这不是边角情况。
    """
    m = industry_map.set_index("code")["industry"]
    if columns is not None:
        m = m.reindex(list(columns))
    return m.fillna(UNKNOWN_INDUSTRY)


def industry_exposure(holdings: pd.DataFrame,
                      industry_map: pd.DataFrame) -> pd.DataFrame:
    """逐日行业权重暴露（index=日期, columns=行业）

    holdings: 宽表，逐日个股权重（MultiBacktestResult.holdings）
    """
    if holdings.empty or industry_map.empty:
        return pd.DataFrame()
    ind = _industry_series(industry_map, holdings.columns)
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
    """给定权重与个股收益，算各行业收益（行业内部按权重加权）

    ⚠️ 这里**不能用 dropna()** 过滤。调用方算组合/基准总收益时用的是
    `(w * r).fillna(0)`，也就是"缺收益按 0 计"；如果这里把缺数据的股票整行丢掉，
    行业收益的分母就变小，`Σ w_i · r_i ≠ 总收益`，Brinson 恒等式立刻不成立
    （实测真实持仓下逐期残差达 1.6e-2）。两边必须用同一套缺失值口径。
    """
    r = asset_ret.reindex(weights.index).fillna(0.0)
    g = ind.reindex(weights.index).fillna(UNKNOWN_INDUSTRY)
    df = pd.DataFrame({"w": weights.astype(float), "r": r, "ind": g})
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
    # ⚠️ 行业映射必须覆盖**组合与基准的并集**列空间。
    # `MultiBacktestResult.holdings` 只包含"实际持有过"的代码（几百只），
    # 而基准有 300 只成分股。若只用 holdings.columns 建映射，基准里那些不在
    # 组合持仓中的股票行业就是 NaN，会被丢掉 → 基准侧 Σw < 1 → 恒等式破裂
    # （实测逐期残差 1.6e-2）。
    ind = _industry_series(industry_map,
                           holdings.columns.union(benchmark_weights.columns))

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
    sum_excess = 0.0                 # Σ 各期 (r_p − r_b)：分解结果的正确对照物
    n_periods = 0
    max_gap = 0.0                    # 逐期 |Σ效应 − (r_p − r_b)| 的最大值，应≈0
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
        sum_excess += rp - rb
        n_periods += 1
        rp_i = _industry_returns(wp, asset_ret, ind)
        rb_i = _industry_returns(wb, asset_ret, ind)

        wp_i = wp.groupby(ind.reindex(wp.index)).sum()
        wb_i = wb.groupby(ind.reindex(wb.index)).sum()
        inds = wp_i.index.union(wb_i.index)
        per_effect = 0.0
        for i in inds:
            wpi = float(wp_i.get(i, 0.0))
            wbi = float(wb_i.get(i, 0.0))
            rbi = float(rb_i.get(i, 0.0))
            rpi = float(rp_i.get(i, rbi))
            alloc = (wpi - wbi) * (rbi - rb)
            select = wbi * (rpi - rbi)
            inter = (wpi - wbi) * (rpi - rbi)
            per_effect += alloc + select + inter
            rows.append({"行业": i, "配置效应": alloc, "选股效应": select,
                         "交互效应": inter})
        max_gap = max(max_gap, abs(per_effect - (rp - rb)))
    if not rows:
        return pd.DataFrame()
    df = pd.DataFrame(rows).groupby("行业", as_index=True).sum()
    df["合计"] = df.sum(axis=1)
    total = df.sum().rename("总计")
    out = pd.concat([df.sort_values("合计", ascending=False), total.to_frame().T])
    # 把对账口径挂在 attrs 上：分解之和应当等于各期 (r_p − r_b) 之和。
    # ⚠️ 不要拿它去和"几何超额"（策略累计 − 基准累计）比 —— 那是复利口径，
    # 波动越大差得越远；也别拿"日度算术超额"比，期数不同。
    out.attrs["sum_excess"] = float(sum_excess)
    out.attrs["n_periods"] = int(n_periods)
    out.attrs["max_period_gap"] = float(max_gap)
    return out
