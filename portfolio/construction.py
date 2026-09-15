# -*- coding: utf-8 -*-
"""组合构建：从因子打分到目标权重

流程:
    因子打分 → 股票池过滤 → 选股（前 N / 分位数）→ 权重分配 → 目标权重面板

权重方案:
    equal          等权
    market_cap     市值加权
    inv_vol        逆波动率加权（风险平价的简化版）

调仓频率:
    rebalance 可以是 "D"（每日）/ "W"（每周）/ "M"（每月）/ 整数（每 N 个交易日），
    或直接给一个调仓日列表。**非调仓日的目标权重为 NaN**，引擎据此判断
    "该日不调仓、让持仓自然漂移"。
"""
from dataclasses import dataclass

import numpy as np
import pandas as pd


# ============================================================
# 打分处理
# ============================================================
def winsorize(df: pd.DataFrame, lower: float = 0.01, upper: float = 0.99) -> pd.DataFrame:
    """横截面缩尾（逐日按分位数截断），降低极值对标准化的影响"""
    lo = df.quantile(lower, axis=1)
    hi = df.quantile(upper, axis=1)
    return df.clip(lower=lo, upper=hi, axis=0)


def zscore(df: pd.DataFrame, mask: pd.DataFrame = None,
           clip: float = 3.0) -> pd.DataFrame:
    """横截面标准化（逐日）：(x - 均值) / 标准差，可选缩尾到 ±clip"""
    x = df.where(mask) if mask is not None else df
    mu = x.mean(axis=1)
    sd = x.std(axis=1, ddof=0)
    z = x.sub(mu, axis=0).div(sd.replace(0, np.nan), axis=0)
    if clip:
        z = z.clip(-clip, clip)
    return z


def select_top_n(score: pd.DataFrame, mask: pd.DataFrame, n: int) -> pd.DataFrame:
    """逐日按打分选前 N（True=入选）"""
    s = score.where(mask)
    if n and n > 0:
        rank = s.rank(axis=1, ascending=False, method="first")
        return (rank <= n).fillna(False)
    return s.notna()


def select_quantile(score: pd.DataFrame, mask: pd.DataFrame,
                    q: float = 0.2) -> pd.DataFrame:
    """逐日按打分选前 q 分位"""
    s = score.where(mask)
    rank = s.rank(axis=1, ascending=False, pct=True)
    return (rank <= q).fillna(False)


# ============================================================
# 权重方案
# ============================================================
def weights_equal(selected: pd.DataFrame) -> pd.DataFrame:
    n = selected.sum(axis=1).replace(0, np.nan)
    return selected.div(n, axis=0)


def weights_market_cap(selected: pd.DataFrame, mv: pd.DataFrame) -> pd.DataFrame:
    v = mv.where(selected)
    tot = v.sum(axis=1).replace(0, np.nan)
    return v.div(tot, axis=0)


def weights_inv_vol(selected: pd.DataFrame, vol: pd.DataFrame) -> pd.DataFrame:
    """逆波动率加权：波动越小权重越高（风险平价在无相关性假设下的近似）"""
    inv = (1.0 / vol.replace(0, np.nan)).where(selected)
    tot = inv.sum(axis=1).replace(0, np.nan)
    return inv.div(tot, axis=0)


def cap_weights(w: pd.DataFrame, max_weight: float) -> pd.DataFrame:
    """单票权重上限：超出部分**按比例**摊给未触顶的票，迭代至收敛

    注意不能"给每只未触顶的票加同一个数"——那样加进去的总量不等于削掉的总量，
    会破坏"权重合计为 1"。正确做法是把未触顶的票整体放大 (room+excess)/room。
    """
    if not max_weight or max_weight >= 1:
        return w
    out = w.copy()
    for _ in range(50):
        over = out > max_weight + 1e-12
        if not over.any().any():
            break
        before = out.copy()
        excess = (out.where(over) - max_weight).sum(axis=1)
        out = out.mask(over, max_weight)               # 削顶
        room = out.notna() & ~over
        room_w = out.where(room)                       # 只保留未触顶的
        room_sum = room_w.sum(axis=1).replace(0, np.nan)
        scale = (1.0 + excess / room_sum).fillna(1.0)
        out = room_w.mul(scale, axis=0).where(room, out)
        if out.equals(before):
            break
    return out


# ============================================================
# 调仓日
# ============================================================
def rebalance_dates(index: pd.DatetimeIndex, rule="M") -> pd.DatetimeIndex:
    """按规则取调仓日

    rule: 'D' 每日 | 'W' 每周首个交易日 | 'M' 每月首个交易日 | 整数=每 N 个交易日
    """
    if isinstance(rule, (list, tuple, pd.DatetimeIndex)):
        return pd.DatetimeIndex(rule).intersection(index)
    if isinstance(rule, int) and not isinstance(rule, bool):
        return index[::max(rule, 1)]
    idx = pd.DatetimeIndex(index)
    s = pd.Series(idx, index=idx)
    if rule == "D":
        return idx
    if rule == "W":
        key = idx.to_period("W")
    elif rule == "M":
        key = idx.to_period("M")
    else:
        raise ValueError(f"未知调仓规则 {rule!r}（D/W/M 或整数）")
    first = ~pd.Series(key).duplicated()
    return idx[first.values]


# ============================================================
# 主入口
# ============================================================
def build_target_weights(score: pd.DataFrame, mask: pd.DataFrame,
                         n_hold: int = 50, weighting: str = "equal",
                         mv: pd.DataFrame = None, vol: pd.DataFrame = None,
                         rebalance="M", max_weight: float = 0.0,
                         quantile: float = 0.0) -> pd.DataFrame:
    """从因子打分构造目标权重面板

    参数:
        score:      因子打分（越大越看多），宽表
        mask:       股票池掩码（universe.build_universe 的产出）
        n_hold:     持股数量（quantile>0 时忽略）
        weighting:  'equal' / 'market_cap' / 'inv_vol'
        mv, vol:    市值 / 波动率面板（对应 weighting 需要时提供）
        rebalance:  'D'/'W'/'M' 或整数
        max_weight: 单票权重上限（0=不限）
        quantile:   >0 时按分位数选股（如前 20%），优先于 n_hold

    返回:
        宽表，**非调仓日为 NaN**（表示"当日不调仓"），调仓日为归一化权重。
    """
    reb = rebalance_dates(score.index, rebalance)
    selected = (select_quantile(score, mask, quantile) if quantile and quantile > 0
                else select_top_n(score, mask, n_hold))
    selected = selected.reindex(columns=score.columns).fillna(False)

    if weighting == "equal":
        w = weights_equal(selected)
    elif weighting == "market_cap":
        if mv is None:
            raise ValueError("market_cap 加权需要 mv 面板")
        w = weights_market_cap(selected, mv)
    elif weighting == "inv_vol":
        if vol is None:
            raise ValueError("inv_vol 加权需要 vol 面板")
        w = weights_inv_vol(selected, vol)
    else:
        raise ValueError(f"未知权重方案 {weighting!r}")

    if max_weight:
        w = cap_weights(w, max_weight)

    # 非调仓日置 NaN —— 引擎据此判断"该日不调仓、持仓自然漂移"
    out = pd.DataFrame(np.nan, index=score.index, columns=score.columns)
    out.loc[reb] = w.reindex(reb)
    return out


def turnover_of_weights(target: pd.DataFrame) -> pd.Series:
    """相邻两次调仓之间的权重换手（|Δw| 之和 / 2）"""
    reb = target.dropna(how="all")
    if len(reb) < 2:
        return pd.Series(dtype=float)
    diff = reb.fillna(0).diff().abs().sum(axis=1) / 2
    return diff.iloc[1:]
