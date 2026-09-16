# -*- coding: utf-8 -*-
"""因子评估：IC / RankIC / ICIR / 分层回测 / 换手率 / 因子衰减

评估口径
--------
对每个交易日 t：
    因子值 f(t)          —— 只用 t 日及以前的数据算出
    前瞻收益 r(t)        —— t+1 起持有 N 天的收益（见 panel.forward_returns）
    横截面相关 corr(f(t), r(t)) 即该日的 IC
IC 序列的均值衡量因子的预测力，标准差衡量稳定性，ICIR = 均值/标准差。

分层回测
--------
按因子值把股票分成 N 组，看每组的平均前瞻收益：
    - 单调性：Q1→QN 的收益是否单调递增/递减
    - 多空：最高组 − 最低组（按 direction 决定哪端是"多"）

这些都是"因子是否有效"的标准检验，比单看某个策略的收益曲线可靠得多。
"""
from dataclasses import dataclass, field
from typing import Dict, List

import numpy as np
import pandas as pd

from .panel import forward_returns, adjusted_close


def _corr(a: pd.Series, b: pd.Series, method: str = "spearman") -> float:
    """横截面相关系数

    spearman 用「先排名、再求 Pearson」实现 —— 这正是 Spearman 的定义，
    可以避免为了一个相关系数而引入 scipy 依赖（pandas 的
    `corr(method="spearman")` 内部会调用 scipy.stats）。
    """
    if method == "spearman":
        a, b = a.rank(), b.rank()
    return float(a.corr(b))


# ============================================================
# IC
# ============================================================
def ic_series(factor: pd.DataFrame, fwd_ret: pd.DataFrame,
              method: str = "spearman", min_count: int = 20) -> pd.Series:
    """逐日横截面相关（spearman = RankIC，默认）

    min_count: 当日有效股票数少于该值则跳过（样本太少的相关没有意义）
    """
    dates = factor.index.intersection(fwd_ret.index)
    out = {}
    for d in dates:
        f = factor.loc[d]
        r = fwd_ret.loc[d]
        m = f.notna() & r.notna()
        if int(m.sum()) < min_count:
            continue
        fv, rv = f[m], r[m]
        if fv.nunique() < 2 or rv.nunique() < 2:
            continue
        out[d] = _corr(fv, rv, method)
    return pd.Series(out, dtype=float).sort_index()


def ic_stats(ic: pd.Series, periods_per_year: int = 252) -> Dict:
    """IC 序列的统计特征"""
    ic = ic.dropna()
    n = len(ic)
    if n == 0:
        return {"IC均值": np.nan, "IC标准差": np.nan, "ICIR": np.nan,
                "t值": np.nan, "正IC占比": np.nan, "样本天数": 0,
                "年化ICIR": np.nan}
    mean, std = float(ic.mean()), float(ic.std(ddof=1)) if n > 1 else 0.0
    icir = mean / std if std > 0 else np.nan
    return {
        "IC均值": mean,
        "IC标准差": std,
        "ICIR": icir,
        "t值": icir * np.sqrt(n) if std > 0 else np.nan,
        "正IC占比": float((ic > 0).mean()),
        "样本天数": n,
        "年化ICIR": icir * np.sqrt(periods_per_year) if std > 0 else np.nan,
    }


# ============================================================
# 分层
# ============================================================
def quantile_returns(factor: pd.DataFrame, fwd_ret: pd.DataFrame,
                     n_quantiles: int = 5, direction: int = 1,
                     min_count: int = 20, stat: str = "mean") -> pd.DataFrame:
    """分层前瞻收益

    返回: index=日期, columns=['Q1'..'Qn']（Q1 = 因子值最低组）+ '多空'
          '多空' = 看多端 − 看空端，已按 direction 调整方向

    ⚠️ 关于 stat 参数（重要）
        默认用**算术均值**，但个股收益分布是**右偏**的（少数暴涨拉高均值），
        且高波动组的右尾更肥。因此算术均值分层可能**与秩 IC 符号相反**——
        实测 vol_60：算术均值 Q5 最高(0.093%)，而中位数随波动率单调下降
        (0.000%→-0.224%)，IC 为 -0.047。
        遇到背离时请用 stat="median" 复核，不要直接按均值下结论。
    """
    dates = factor.index.intersection(fwd_ret.index)
    rows = {}
    labels = [f"Q{i+1}" for i in range(n_quantiles)]
    agg = "median" if stat == "median" else "mean"
    for d in dates:
        f = factor.loc[d]
        r = fwd_ret.loc[d]
        m = f.notna() & r.notna()
        if int(m.sum()) < max(min_count, n_quantiles * 2):
            continue
        fv, rv = f[m], r[m]
        try:
            q = pd.qcut(fv.rank(method="first"), n_quantiles,
                        labels=labels, duplicates="drop")
        except ValueError:
            continue
        g = getattr(rv.groupby(q, observed=False), agg)()
        rows[d] = g
    df = pd.DataFrame(rows).T.sort_index()
    if df.empty:
        return df
    df = df.reindex(columns=labels)
    long_col = labels[-1] if direction > 0 else labels[0]
    short_col = labels[0] if direction > 0 else labels[-1]
    df["多空"] = df[long_col] - df[short_col]
    df.attrs["long"] = long_col
    df.attrs["short"] = short_col
    df.attrs["stat"] = agg
    return df


def monotonicity(qret: pd.DataFrame, n_quantiles: int = 5) -> float:
    """分层单调性：分层收益与组序号的 Spearman 相关（±1 为完全单调）"""
    cols = [f"Q{i+1}" for i in range(n_quantiles)]
    cols = [c for c in cols if c in qret.columns]
    if len(cols) < 3:
        return np.nan
    mean_by_q = qret[cols].mean()
    return float(pd.Series(range(len(cols))).rank().corr(
        pd.Series(mean_by_q.values).rank()))


def long_short_curve(qret: pd.DataFrame) -> pd.Series:
    """多空组合的累计净值（每日再平衡、等权）"""
    if qret.empty or "多空" not in qret:
        return pd.Series(dtype=float)
    return (1 + qret["多空"].fillna(0)).cumprod()


# ============================================================
# 换手 / 衰减
# ============================================================
def quantile_turnover(factor: pd.DataFrame, n_quantiles: int = 5,
                      min_count: int = 20) -> pd.Series:
    """分层换手率：相邻两期分组发生变化的比例（衡量交易成本压力）"""
    dates = factor.index
    labels = [f"Q{i+1}" for i in range(n_quantiles)]
    prev = None
    out = {}
    for d in dates:
        f = factor.loc[d]
        m = f.notna()
        if int(m.sum()) < max(min_count, n_quantiles * 2):
            continue
        try:
            q = pd.qcut(f[m].rank(method="first"), n_quantiles,
                        labels=labels, duplicates="drop")
        except ValueError:
            continue
        if prev is not None:
            common = q.index.intersection(prev.index)
            if len(common) > 0:
                changed = (q.loc[common] != prev.loc[common]).mean()
                out[d] = float(changed)
        prev = q
    return pd.Series(out, dtype=float).sort_index()


def factor_decay(factor: pd.DataFrame, close: pd.DataFrame,
                 periods: List[int] = (1, 5, 10, 20, 60),
                 method: str = "spearman", min_count: int = 20) -> pd.DataFrame:
    """因子衰减：不同持有期下的 IC 均值"""
    rows = []
    for p in periods:
        fwd = forward_returns(close, periods=p, lag=1)
        ic = ic_series(factor, fwd, method=method, min_count=min_count)
        s = ic_stats(ic)
        rows.append({"持有期": p, "IC均值": s["IC均值"], "ICIR": s["ICIR"],
                     "t值": s["t值"], "样本天数": s["样本天数"]})
    return pd.DataFrame(rows)


# ============================================================
# 汇总
# ============================================================
@dataclass
class FactorReport:
    name: str
    direction: int
    ic: pd.Series
    stats: Dict
    quantile: pd.DataFrame
    turnover: pd.Series
    decay: pd.DataFrame
    n_stocks: float = 0.0
    quantile_median: pd.DataFrame = field(default_factory=pd.DataFrame)

    @property
    def mean_ic(self) -> float:
        return float(self.stats.get("IC均值", np.nan))

    def mean_median_diverges(self) -> bool:
        """算术均值分层与中位数分层是否符号背离

        个股收益右偏，高波动组右尾更肥，会让算术均值分层与秩 IC 方向相反。
        背离时不能只按均值下结论。
        """
        if self.quantile.empty or self.quantile_median.empty:
            return False
        a = self.quantile["多空"].mean()
        b = self.quantile_median["多空"].mean()
        if pd.isna(a) or pd.isna(b):
            return False
        return (a > 0) != (b > 0)

    def summary(self) -> str:
        d = self.stats
        L = ["=" * 78, f"因子评估：{self.name}", "=" * 78]
        L.append(f"  方向: {'值越大越看多' if self.direction > 0 else '值越小越看多'}"
                 f"    平均股票数: {self.n_stocks:,.0f}")
        L.append("")
        L.append("  —— 预测力（IC，秩相关）——")
        L.append(f"    IC 均值   : {d.get('IC均值', np.nan):>8.4f}   "
                 f"（|IC|>0.03 通常算有效，>0.05 较强）")
        L.append(f"    IC 标准差 : {d.get('IC标准差', np.nan):>8.4f}")
        L.append(f"    ICIR      : {d.get('ICIR', np.nan):>8.4f}   （均值/标准差，>0.3 较稳）")
        L.append(f"    t 值      : {d.get('t值', np.nan):>8.2f}   （|t|>2 显著）")
        L.append(f"    正 IC 占比: {d.get('正IC占比', np.nan):>8.2%}")
        L.append(f"    样本天数  : {d.get('样本天数', 0):>8d}")
        L.append("")
        if not self.quantile.empty:
            n = len([c for c in self.quantile.columns if c.startswith("Q")])
            avg = self.quantile[[f"Q{i+1}" for i in range(n)]].mean()
            L.append("  —— 分层前瞻收益（Q1=因子值最低）——")
            if not self.quantile_median.empty:
                med = self.quantile_median[[f"Q{i+1}" for i in range(n)]].mean()
                L.append("    {:<5}{:>12}{:>14}".format("分层", "算术均值", "中位数"))
                for k in avg.index:
                    L.append("    {:<5}{:>12.4%}{:>14.4%}".format(k, avg[k], med[k]))
            else:
                for k, v in avg.items():
                    L.append(f"    {k:<4}: {v:>9.4%}")
            mono = monotonicity(self.quantile, n)
            L.append(f"    单调性(均值): {mono:>9.2f}   （±1 为完全单调）")
            if not self.quantile_median.empty:
                L.append(f"    单调性(中位): {monotonicity(self.quantile_median, n):>9.2f}")
            ls = self.quantile["多空"]
            L.append(f"    多空均值: {ls.mean():>9.4%}   "
                     f"t 值: {ls.mean() / ls.std() * np.sqrt(len(ls)) if ls.std() > 0 else np.nan:.2f}")
            if self.mean_median_diverges():
                L.append("")
                L.append("    ⚠️ 算术均值与中位数的多空方向相反！")
                L.append("       个股收益右偏、高波动组右尾更肥，会让等权算术均值被少数暴涨拉高。")
                L.append("       此时应以中位数/秩 IC 为准，或改用截尾均值，不要直接按均值下结论。")
        if not self.turnover.empty:
            L.append("")
            L.append("  —— 换手率 ——")
            L.append(f"    平均分层换手: {self.turnover.mean():>8.2%}   "
                     f"（越高交易成本压力越大）")
        if not self.decay.empty:
            L.append("")
            L.append("  —— 因子衰减（不同持有期的 IC）——")
            for _, r in self.decay.iterrows():
                icir = r["ICIR"] if pd.notna(r["ICIR"]) else float("nan")
                L.append(f"    持有 {int(r['持有期']):>3d} 日: IC={r['IC均值']:>8.4f}  "
                         f"ICIR={icir:>7.3f}")
        L.append("=" * 78)
        return "\n".join(L)


def evaluate(factor, panel: dict, periods: int = 1, n_quantiles: int = 5,
             method: str = "spearman", min_count: int = 20,
             with_decay: bool = True) -> FactorReport:
    """跑完一整套因子评估（同时给出均值与中位数分层，便于发现右偏导致的背离）"""
    f = factor.compute(panel) if hasattr(factor, "compute") else factor
    direction = getattr(factor, "direction", 1)
    name = getattr(factor, "name", "factor")

    close = adjusted_close(panel)
    f = f.reindex(index=close.index, columns=close.columns)

    fwd = forward_returns(close, periods=periods, lag=1)
    ic = ic_series(f, fwd, method=method, min_count=min_count)
    stats = ic_stats(ic)
    qret = quantile_returns(f, fwd, n_quantiles=n_quantiles,
                            direction=direction, min_count=min_count, stat="mean")
    qmed = quantile_returns(f, fwd, n_quantiles=n_quantiles,
                            direction=direction, min_count=min_count, stat="median")
    turn = quantile_turnover(f, n_quantiles=n_quantiles, min_count=min_count)
    decay = (factor_decay(f, close, method=method, min_count=min_count)
             if with_decay else pd.DataFrame())

    counts = f.notna().sum(axis=1)
    return FactorReport(name=name, direction=direction, ic=ic, stats=stats,
                        quantile=qret, turnover=turn, decay=decay,
                        n_stocks=float(counts.mean()) if len(counts) else 0.0,
                        quantile_median=qmed)
