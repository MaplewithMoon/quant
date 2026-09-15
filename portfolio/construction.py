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


def equal_weight_returns(panel: dict, mask: pd.DataFrame,
                         rebalance="M", price_key: str = "close_adj") -> pd.Series:
    """等权基准的日收益序列（解析解，不跑撮合引擎）

    逻辑：每期期初在股票池内等权买入，期内**买入持有**（权重自然漂移），
    到下一调仓日重新等权。

    为什么不用 `PortfolioBacktestEngine` 跑：
        股票池有 2000+ 只票时，引擎要逐日对每个持仓盯市，实测一次 20 分钟以上。
        而这个基准的用途只是回答"策略是不是只赚了市场 beta"，解析计算几秒就够，
        口径也更清楚（毛收益，不含费用；作为对照组足够，且不受整手约束扭曲）。

    返回: index=交易日, values=日收益（第一期期初为 0）
    """
    price = panel.get(price_key)
    if price is None:
        price = panel["close"]
    ret = price.astype(float).pct_change(fill_method=None)
    dates = ret.index
    if len(dates) < 2:
        return pd.Series(dtype=float)

    m = mask.reindex(index=dates, columns=ret.columns).fillna(False).astype(bool)
    reb = list(rebalance_dates(dates, rebalance))
    if not reb:
        return pd.Series(0.0, index=dates)

    out = pd.Series(np.nan, index=dates, dtype=float)
    for i, t0 in enumerate(reb):
        t1 = reb[i + 1] if i + 1 < len(reb) else None
        picked = m.loc[t0]
        cols = picked.index[picked.to_numpy()]
        if len(cols) == 0:
            continue
        seg = ret.loc[dates > t0] if t1 is None else ret.loc[(dates > t0) & (dates < t1)]
        if seg.empty:
            continue
        w = pd.Series(1.0 / len(cols), index=cols)
        cum = (1.0 + seg[cols].fillna(0.0)).cumprod()
        nav = (cum * w).sum(axis=1)                 # 期初净值 = 1.0
        out.loc[nav.index] = (nav / nav.shift(1).fillna(1.0) - 1.0).to_numpy()
    return out.fillna(0.0)
