# -*- coding: utf-8 -*-
"""基准相对绩效指标

把"一条净值曲线 vs 一条基准曲线"该说的东西一次算清楚：

    alpha / beta / R²           CAPM 回归（含 t 值与 p 值，不依赖 scipy）
    跟踪误差 / 信息比率          相对基准的主动管理能力
    上行/下行捕获                涨跌市里的进攻性与防守性
    分年度、分月收益表            收益的稳定性
    回撤明细                     幅度 / 区间 / 持续 / 恢复

【口径约定】
- 无风险利率 `rf` 是**年化**值，内部按 252 交易日折算到日频。
- `alpha` 按**日频回归截距 × 252** 年化（算术年化）。这是行业惯例；
  与"几何年化"会有微小差异，跨报告比较时注意口径一致。
- beta 全部基于**日频收益率**回归。用月频回归会得到不同的 beta。
- 所有方差/协方差用样本估计（ddof=1）。
"""
from dataclasses import dataclass, field
from typing import Dict, Optional, Tuple

import numpy as np
import pandas as pd

PERIODS = 252          # 年化交易日数
TRADING_DAYS = 252

# 波动率下限：常数序列的 std 不是精确 0（浮点残差 ~1e-19），
# 若直接拿它当分母，夏普会算出 1e16 这种天文数字。低于该阈值一律按 0 处理。
_EPS = 1e-12


# ============================================================
# 基础工具
# ============================================================
def to_returns(series: pd.Series) -> pd.Series:
    """价格/净值序列 -> 日收益率（首日 NaN 丢掉）"""
    s = pd.Series(series).astype(float).sort_index()
    return s.pct_change().dropna()


def align(equity: pd.Series, benchmark: pd.Series) -> pd.DataFrame:
    """按交易日对齐组合与基准的**净值**，并转成日收益

    返回 DataFrame，列 = ['port', 'bench']，索引为两者共同交易日。
    对齐用 inner join —— 基准缺数据的日期不参与比较，避免拿 0 填充制造假相关。
    """
    e = pd.Series(equity).astype(float).sort_index()
    b = pd.Series(benchmark).astype(float).sort_index()
    df = pd.DataFrame({"port": e, "bench": b}).dropna()
    if len(df) < 2:
        return pd.DataFrame(columns=["port", "bench"])
    return df.pct_change().dropna()


def cum_return(returns: pd.Series) -> pd.Series:
    """日收益 -> 累计净值（起点 1.0）"""
    r = pd.Series(returns).fillna(0.0)
    return (1.0 + r).cumprod()


def total_return(returns: pd.Series) -> float:
    r = pd.Series(returns).dropna()
    if r.empty:
        return 0.0
    return float((1.0 + r).prod() - 1.0)


def annual_return(returns: pd.Series, periods: int = PERIODS) -> float:
    """几何年化收益"""
    r = pd.Series(returns).dropna()
    if r.empty:
        return 0.0
    n = len(r)
    return float((1.0 + r).prod() ** (periods / n) - 1.0)


def annual_volatility(returns: pd.Series, periods: int = PERIODS) -> float:
    r = pd.Series(returns).dropna()
    if len(r) < 2:
        return 0.0
    return float(r.std(ddof=1) * np.sqrt(periods))


def sharpe_ratio(returns: pd.Series, rf: float = 0.0,
                 periods: int = PERIODS) -> float:
    r = pd.Series(returns).dropna()
    if len(r) < 2:
        return 0.0
    ex = r - rf / periods
    sd = ex.std(ddof=1)
    return float(ex.mean() / sd * np.sqrt(periods)) if sd > _EPS else 0.0


def sortino_ratio(returns: pd.Series, rf: float = 0.0,
                  periods: int = PERIODS) -> float:
    """索提诺比率：只用**下行**波动做分母（对右偏收益更公平）"""
    r = pd.Series(returns).dropna()
    if len(r) < 2:
        return 0.0
    ex = r - rf / periods
    downside = ex[ex < 0]
    dd = np.sqrt((downside ** 2).mean()) if len(downside) else 0.0
    return float(ex.mean() / dd * np.sqrt(periods)) if dd > _EPS else 0.0


def skewness(returns: pd.Series) -> float:
    r = pd.Series(returns).dropna()
    if len(r) < 3:
        return 0.0
    s = r.std(ddof=1)
    return float(((r - r.mean()) ** 3).mean() / s ** 3) if s > 0 else 0.0


def kurtosis(returns: pd.Series) -> float:
    """超额峰度（正态 = 0）"""
    r = pd.Series(returns).dropna()
    if len(r) < 4:
        return 0.0
    s = r.std(ddof=1)
    return float(((r - r.mean()) ** 4).mean() / s ** 4 - 3.0) if s > 0 else 0.0


# ============================================================
# 回撤
# ============================================================
@dataclass
class DrawdownInfo:
    max_drawdown: float = 0.0          # 负数，如 -0.1645
    peak_date: object = None           # 回撤前的高点
    trough_date: object = None         # 最低点
    recover_date: object = None        # 收复高点的日期（None = 期末仍未收复）
    duration_days: int = 0             # 高点 -> 最低点的自然交易日数
    recover_days: int = 0              # 高点 -> 收复的交易日数（0 = 未收复）
    max_drawdown_duration: int = 0     # 最长"未创新高"持续交易日数
    avg_drawdown: float = 0.0
    volatility_of_dd: float = 0.0
    dd_p95: float = 0.0                # 5% 分位（即最深 5% 的回撤水平）

    def to_dict(self) -> Dict:
        return {
            "max_drawdown": self.max_drawdown,
            "peak_date": str(self.peak_date)[:10] if self.peak_date is not None else "",
            "trough_date": str(self.trough_date)[:10] if self.trough_date is not None else "",
            "recover_date": str(self.recover_date)[:10] if self.recover_date is not None else "未收复",
            "duration_days": self.duration_days,
            "recover_days": self.recover_days,
            "max_drawdown_duration": self.max_drawdown_duration,
            "avg_drawdown": self.avg_drawdown,
            "volatility_of_dd": self.volatility_of_dd,
            "dd_p95": self.dd_p95,
        }


def drawdown_series(equity: pd.Series) -> pd.Series:
    """回撤序列（相对历史最高点，≤0）"""
    e = pd.Series(equity).astype(float).dropna()
    return (e - e.cummax()) / e.cummax()


def drawdown_info(equity: pd.Series) -> DrawdownInfo:
    e = pd.Series(equity).astype(float).dropna()
    if e.empty:
        return DrawdownInfo()
    dd = drawdown_series(e)
    trough = dd.idxmin()
    mdd = float(dd.min())
    peak = e.loc[:trough].idxmax()

    # 收复点：最低点之后首次回到前高
    after = e.loc[trough:]
    rec = after[after >= e.loc[peak]]
    recover = rec.index[0] if len(rec) else None

    pos = {d: i for i, d in enumerate(e.index)}
    dur = pos[trough] - pos[peak]
    rec_days = (pos[recover] - pos[peak]) if recover is not None else 0

    # 最长未创新高持续期
    new_high = e >= e.cummax()
    longest, cur = 0, 0
    for d, is_high in new_high.items():
        if is_high:
            cur = 0
        else:
            cur += 1
            longest = max(longest, cur)

    neg = dd[dd < 0]
    return DrawdownInfo(
        max_drawdown=mdd,
        peak_date=peak,
        trough_date=trough,
        recover_date=recover,
        duration_days=int(dur),
        recover_days=int(rec_days),
        max_drawdown_duration=int(longest),
        avg_drawdown=float(neg.mean()) if len(neg) else 0.0,
        volatility_of_dd=float(neg.std(ddof=1)) if len(neg) > 1 else 0.0,
        dd_p95=float(neg.quantile(0.05)) if len(neg) else 0.0,
    )


def calmar_ratio(returns: pd.Series, equity: pd.Series = None,
                 periods: int = PERIODS) -> float:
    ar = annual_return(returns, periods)
    dd = drawdown_info(equity if equity is not None else cum_return(returns)).max_drawdown
    return float(ar / abs(dd)) if dd != 0 else 0.0


# ============================================================
# alpha / beta（CAPM 回归，不依赖 scipy）
# ============================================================
def _betainc(a: float, b: float, x: float) -> float:
    """正则化不完全贝塔函数 I_x(a,b)（连分式展开，Numerical Recipes 6.4）

    只为算 t 分布的双尾 p 值；scipy 不是本项目的依赖，所以自带实现。
    """
    import math

    if x <= 0:
        return 0.0
    if x >= 1:
        return 1.0
    # ⚠️ 这个量是 -ln B(a,b)，不是 ln B(a,b)：
    #   lgamma(a+b) - lgamma(a) - lgamma(b) = -ln[Γ(a)Γ(b)/Γ(a+b)] = -ln B(a,b)
    # 因此指数里应当"加"它（NR 的 betai 也是 gammln(a+b)-gammln(a)-gammln(b)+...）。
    # 一开始写成减号，符号反了：I_0.5(2,3) 算出 0.9978（应为 0.6875）。
    neg_ln_beta = math.lgamma(a + b) - math.lgamma(a) - math.lgamma(b)

    def cf(a: float, b: float, x: float) -> float:
        tiny, eps = 1e-30, 3e-12
        qab, qap, qam = a + b, a + 1.0, a - 1.0
        c, d = 1.0, 1.0 - qab * x / qap
        if abs(d) < tiny:
            d = tiny
        d = 1.0 / d
        h = d
        for m in range(1, 300):
            m2 = 2 * m
            aa = m * (b - m) * x / ((qam + m2) * (a + m2))
            d = 1.0 + aa * d
            d = tiny if abs(d) < tiny else d
            c = 1.0 + aa / c
            c = tiny if abs(c) < tiny else c
            d = 1.0 / d
            h *= d * c
            aa = -(a + m) * (qab + m) * x / ((a + m2) * (qap + m2))
            d = 1.0 + aa * d
            d = tiny if abs(d) < tiny else d
            c = 1.0 + aa / c
            c = tiny if abs(c) < tiny else c
            d = 1.0 / d
            h *= d * c
            if abs(d * c - 1.0) < eps:
                break
        return h

    if x < (a + 1.0) / (a + b + 2.0):
        return (math.exp(math.log(x) * a + math.log(1 - x) * b + neg_ln_beta) / a
                * cf(a, b, x))
    return 1.0 - (math.exp(math.log(1 - x) * b + math.log(x) * a + neg_ln_beta) / b
                  * cf(b, a, 1 - x))


def t_pvalue(t: float, df: int) -> float:
    """双尾 p 值（Student-t），仅用标准库实现"""
    if df <= 0 or not np.isfinite(t):
        return float("nan")
    x = df / (df + t * t)
    return float(_betainc(df / 2.0, 0.5, x))


@dataclass
class AlphaBeta:
    alpha_daily: float = 0.0
    alpha_annual: float = 0.0
    alpha_tstat: float = 0.0
    alpha_pvalue: float = float("nan")
    beta: float = 0.0
    beta_tstat: float = 0.0
    r_squared: float = 0.0
    corr: float = 0.0
    residual_vol_annual: float = 0.0
    n_obs: int = 0

    def to_dict(self) -> Dict:
        return self.__dict__.copy()


def alpha_beta(returns: pd.Series, bench_returns: pd.Series,
               rf: float = 0.0, periods: int = PERIODS) -> AlphaBeta:
    """CAPM: r_p - rf = alpha + beta * (r_b - rf) + eps

    alpha 按 `periods` 年化（算术）。同时给出 t 值与双尾 p 值：
    |t| > 2（p < 0.05）才谈得上"显著的正 alpha"。
    """
    rp = pd.Series(returns).astype(float)
    rb = pd.Series(bench_returns).astype(float)
    df = pd.DataFrame({"p": rp, "b": rb}).dropna()
    n = len(df)
    if n < 3:
        return AlphaBeta(n_obs=n)

    rfd = rf / periods
    x = (df["b"] - rfd).to_numpy()
    y = (df["p"] - rfd).to_numpy()

    xm, ym = x.mean(), y.mean()
    sxx = float(((x - xm) ** 2).sum())
    sxy = float(((x - xm) * (y - ym)).sum())
    if sxx == 0:
        return AlphaBeta(n_obs=n)

    beta = sxy / sxx
    alpha_d = ym - beta * xm

    resid = y - (alpha_d + beta * x)
    dof = n - 2
    s2 = float((resid ** 2).sum()) / dof if dof > 0 else 0.0
    se_alpha = np.sqrt(s2 * (1.0 / n + xm ** 2 / sxx)) if s2 > 0 else 0.0
    se_beta = np.sqrt(s2 / sxx) if s2 > 0 else 0.0
    t_alpha = alpha_d / se_alpha if se_alpha > 0 else 0.0
    t_beta = beta / se_beta if se_beta > 0 else 0.0

    yc = y - ym
    sst = float((yc ** 2).sum())
    sse = float(((resid - resid.mean()) ** 2).sum())
    r2 = 1.0 - sse / sst if sst > 0 else 0.0
    corr = float(np.corrcoef(x, y)[0, 1]) if x.std() > 0 and y.std() > 0 else 0.0

    return AlphaBeta(
        alpha_daily=float(alpha_d),
        alpha_annual=float(alpha_d * periods),
        alpha_tstat=float(t_alpha),
        alpha_pvalue=t_pvalue(float(t_alpha), dof),
        beta=float(beta),
        beta_tstat=float(t_beta),
        r_squared=float(r2),
        corr=corr,
        residual_vol_annual=float(np.sqrt(s2) * np.sqrt(periods)),
        n_obs=n,
    )


# ============================================================
# 相对基准：跟踪误差 / 信息比率 / 捕获率
# ============================================================
def tracking_error(returns: pd.Series, bench_returns: pd.Series,
                   periods: int = PERIODS) -> float:
    df = pd.DataFrame({"p": returns, "b": bench_returns}).dropna()
    if len(df) < 2:
        return 0.0
    return float((df["p"] - df["b"]).std(ddof=1) * np.sqrt(periods))


def information_ratio(returns: pd.Series, bench_returns: pd.Series,
                      periods: int = PERIODS) -> float:
    """年化超额收益 / 跟踪误差

    注意：用**算术**年化超额（日超额均值 × 252），与跟踪误差口径一致。
    """
    df = pd.DataFrame({"p": returns, "b": bench_returns}).dropna()
    if len(df) < 2:
        return 0.0
    ex = df["p"] - df["b"]
    te = ex.std(ddof=1)
    return float(ex.mean() / te * np.sqrt(periods)) if te > 0 else 0.0


def excess_return(returns: pd.Series, bench_returns: pd.Series) -> float:
    """区间累计超额（几何）：(1+rp)/(1+rb) - 1"""
    df = pd.DataFrame({"p": returns, "b": bench_returns}).dropna()
    if df.empty:
        return 0.0
    return float((1 + df["p"]).prod() / (1 + df["b"]).prod() - 1.0)


def capture_ratios(returns: pd.Series, bench_returns: pd.Series) -> Tuple[float, float]:
    """(上行捕获, 下行捕获)

    上行捕获 = 基准上涨日的组合平均收益 / 基准平均收益（>1 进攻性强）
    下行捕获 = 基准下跌日的组合平均收益 / 基准平均收益（<1 防守性好）
    """
    df = pd.DataFrame({"p": returns, "b": bench_returns}).dropna()
    if df.empty:
        return 0.0, 0.0
    up, dn = df[df["b"] > 0], df[df["b"] < 0]
    uc = float(up["p"].mean() / up["b"].mean()) if len(up) and up["b"].mean() != 0 else 0.0
    dc = float(dn["p"].mean() / dn["b"].mean()) if len(dn) and dn["b"].mean() != 0 else 0.0
    return uc, dc


def win_rate_vs_bench(returns: pd.Series, bench_returns: pd.Series) -> float:
    df = pd.DataFrame({"p": returns, "b": bench_returns}).dropna()
    if df.empty:
        return 0.0
    return float((df["p"] > df["b"]).mean())


# ============================================================
# 收益表
# ============================================================
def annual_returns_table(returns: pd.Series, bench_returns: pd.Series = None) -> pd.DataFrame:
    """分年度收益表（几何）"""
    r = pd.Series(returns).dropna()
    if r.empty:
        return pd.DataFrame()
    g = r.groupby(r.index.year).apply(lambda x: (1 + x).prod() - 1)
    out = pd.DataFrame({"组合": g})
    if bench_returns is not None:
        b = pd.Series(bench_returns).dropna()
        gb = b.groupby(b.index.year).apply(lambda x: (1 + x).prod() - 1)
        out["基准"] = gb
        out["超额"] = out["组合"] - out["基准"]
    out.index.name = "年份"
    return out


def monthly_returns_table(returns: pd.Series) -> pd.DataFrame:
    """分月收益表（行=年, 列=月）"""
    r = pd.Series(returns).dropna()
    if r.empty:
        return pd.DataFrame()
    g = r.groupby([r.index.year, r.index.month]).apply(lambda x: (1 + x).prod() - 1)
    t = g.unstack()
    t.index.name = "年份"
    t.columns = [f"{m}月" for m in t.columns]
    return t


def rolling_metrics(returns: pd.Series, window: int = PERIODS,
                    rf: float = 0.0) -> pd.DataFrame:
    """滚动指标：年化收益 / 年化波动 / 夏普"""
    r = pd.Series(returns).dropna()
    if len(r) < window:
        return pd.DataFrame()
    ann = (1 + r).rolling(window).apply(np.prod, raw=True) ** (PERIODS / window) - 1
    vol = r.rolling(window).std(ddof=1) * np.sqrt(PERIODS)
    sharpe = (ann - rf) / vol.replace(0, np.nan)
    return pd.DataFrame({"ann_return": ann, "ann_vol": vol, "sharpe": sharpe})


# ============================================================
# 汇总
# ============================================================
def performance_summary(equity: pd.Series, benchmark: pd.Series = None,
                        rf: float = 0.0, periods: int = PERIODS) -> Dict:
    """一次性算出全部绝对 + 相对指标"""
    e = pd.Series(equity).astype(float).dropna()
    r = to_returns(e)
    dd = drawdown_info(e)
    out: Dict = {
        "n_days": int(len(r)),
        "total_return": total_return(r),
        "annual_return": annual_return(r, periods),
        "annual_volatility": annual_volatility(r, periods),
        "sharpe_ratio": sharpe_ratio(r, rf, periods),
        "sortino_ratio": sortino_ratio(r, rf, periods),
        "calmar_ratio": calmar_ratio(r, e, periods),
        "skewness": skewness(r),
        "kurtosis": kurtosis(r),
        "max_drawdown": dd.max_drawdown,
        "max_drawdown_peak": str(dd.peak_date)[:10] if dd.peak_date is not None else "",
        "max_drawdown_trough": str(dd.trough_date)[:10] if dd.trough_date is not None else "",
        "max_drawdown_recover": str(dd.recover_date)[:10] if dd.recover_date is not None else "未收复",
        "max_drawdown_duration": dd.max_drawdown_duration,
        "avg_drawdown": dd.avg_drawdown,
    }
    if benchmark is not None:
        b = pd.Series(benchmark).astype(float).dropna()
        bret = to_returns(b)
        ab = alpha_beta(r, bret, rf, periods)
        out.update({
            "bench_total_return": total_return(bret),
            "bench_annual_return": annual_return(bret, periods),
            "bench_annual_volatility": annual_volatility(bret, periods),
            "bench_sharpe_ratio": sharpe_ratio(bret, rf, periods),
            "bench_sortino_ratio": sortino_ratio(bret, rf, periods),
            "bench_calmar_ratio": calmar_ratio(bret, b, periods),
            "bench_max_drawdown": drawdown_info(b).max_drawdown,
            "excess_return": excess_return(r, bret),
            "alpha_annual": ab.alpha_annual,
            "alpha_tstat": ab.alpha_tstat,
            "alpha_pvalue": ab.alpha_pvalue,
            "beta": ab.beta,
            "r_squared": ab.r_squared,
            "residual_vol_annual": ab.residual_vol_annual,
            "tracking_error": tracking_error(r, bret, periods),
            "information_ratio": information_ratio(r, bret, periods),
            "win_rate_vs_bench": win_rate_vs_bench(r, bret),
        })
        uc, dc = capture_ratios(r, bret)
        out["up_capture"] = uc
        out["down_capture"] = dc
    return out
