# -*- coding: utf-8 -*-
"""风格分解与**残差 alpha**（T1·④）

【要解决什么问题】
A3 报的是「Alpha +2.40%，p=0.646，Beta 0.886，池子偏中小盘」——
**这个 alpha 没有意义，因为它混着风格暴露**。"选股能力"和"小盘 beta"
在这个数字里分不开，读者无法判断策略到底有没有 alpha。

【做法：横截面回归取残差】
对每个交易日 t 做一次横截面回归：

    r_i,t = a_t + Σ_k β_k,t · z_k,i,t + e_i,t        k ∈ {size, value, ...}

    β_k,t  = 该日因子 k 的**因子收益**（回归斜率）
    z_k,i,t = 个股 i 在因子 k 上的**横截面标准化**暴露

组合的风格暴露 E_k,t = Σ_i w_i,t · z_k,i,t（与 `factor_exposure` 同一口径）。
于是组合的风格中性残差收益：

    r_p,t^resid = r_p,t − Σ_k β_k,t · E_k,t

**残差 alpha** 就是 r_p,t^resid 的年化均值 + t 值。
它回答的是："把风格暴露解释掉之后，还剩下多少收益？"

【为什么这是"口径"问题而不是"算法"问题】
原口径 alpha 不是算错，而是**问的问题不对**。同一个策略：
  - 混口径 alpha：可能很漂亮（因为吃了小盘 beta）
  - 残差 alpha：可能不显著（说明收益来自风格，不是选股）
报告应当以**残差 alpha 为主口径**，混口径降为参考行。

【PIT】
风格面板必须按时点构建：
  - `size` 用当日的 `total_mv`（面板里本来就有，天然 PIT）
  - `value`（bp）按 `ann_date` 对齐后**前向填充**；不这么做就是拿未来财报
    去解释过去的收益（与 B7 同构的前视）。
"""
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

from .attribution import factor_exposure

# 默认风格因子：市值 / 价值。
# 加因子时注意：**每加一个就少一个自由度**，样本短的区间会很快吃掉 t 值。
DEFAULT_STYLES = ("size", "value")

STYLE_LABEL = {"size": "市值(对数)", "value": "价值(BP)"}


# ============================================================
# 风格面板
# ============================================================
def build_style_panels(panel: dict, dates, codes,
                       fundamentals: Dict[str, pd.DataFrame] = None,
                       style_names=DEFAULT_STYLES) -> Dict[str, pd.DataFrame]:
    """逐日风格面板 {name: DataFrame(index=日期, columns=代码)}

    - `size`  = log(total_mv)，直接取面板（逐日，天然 PIT）
    - `value` = bp（账面市值比）。取自 `fundamentals`（已按 ann_date PIT 对齐），
                在调仓日之间**前向填充** —— 财报没更新时账面价值不会变。
                ⚠️ 绝不能用"当前"财务数据回填历史（那是前视）。
    """
    idx = pd.DatetimeIndex(pd.to_datetime(dates))
    cols = [str(c).zfill(6) for c in codes]
    out: Dict[str, pd.DataFrame] = {}

    if "size" in style_names:
        mv = panel.get("total_mv")
        if mv is not None and not mv.empty:
            m = mv.reindex(index=idx, columns=cols).astype(float)
            out["size"] = np.log(m.where(m > 0))

    if "value" in style_names and fundamentals:
        bp = fundamentals.get("bp")
        if bp is not None and not bp.empty:
            b = bp.reindex(columns=cols).reindex(idx)
            # 只在财报更新日有值 -> 前向填充（PIT：不引入未来财报）
            out["value"] = b.ffill()
    return out


# ============================================================
# 因子收益（横截面回归斜率）
# ============================================================
def cross_section_factor_returns(returns: pd.DataFrame,
                                 factor_panels: Dict[str, pd.DataFrame],
                                 mask: pd.DataFrame = None,
                                 min_stocks: int = 50) -> pd.DataFrame:
    """逐日横截面回归，返回 {日期 × 因子} 的因子收益

    对每一天：
        r_i = a + Σ_k β_k · z_k,i + e_i
    其中 z 是**当日横截面标准化**后的因子值（减均值除标准差）。
    有效样本不足 `min_stocks` 的日子跳过（样本太少回归不稳）。

    为什么用回归斜率而不是"高低分组收益差"：回归同时给出了**联合**解释力，
    多个因子之间不会互相污染；分组法则要求因子正交，而 `bp` 与 `ep_ttm`
    在项目里已知高度相关。
    """
    names = list(factor_panels)
    if not names or returns is None or returns.empty:
        return pd.DataFrame(columns=names)
    idx = returns.index
    out = pd.DataFrame(np.nan, index=idx, columns=names, dtype=float)

    zs = {}
    for k in names:
        z = factor_panels[k].reindex(index=idx, columns=returns.columns)
        if mask is not None:
            z = z.where(mask.reindex(index=idx, columns=returns.columns))
        mu = z.mean(axis=1)
        sd = z.std(axis=1, ddof=0).replace(0, np.nan)
        zs[k] = z.sub(mu, axis=0).div(sd, axis=0)

    for d in idx:
        r = returns.loc[d]
        Z = pd.DataFrame({k: zs[k].loc[d] for k in names})
        ok = r.notna() & Z.notna().all(axis=1)
        if int(ok.sum()) < min_stocks:
            continue
        y = r[ok].to_numpy(dtype=float)
        X = Z[ok].to_numpy(dtype=float)
        X = np.column_stack([np.ones(len(X)), X])       # 加截距
        try:
            coef, *_ = np.linalg.lstsq(X, y, rcond=None)
        except np.linalg.LinAlgError:
            continue
        out.loc[d] = coef[1:]
    return out


# ============================================================
# 残差 alpha
# ============================================================
def _alpha_tstat(ret: pd.Series, ann: int = 252) -> dict:
    r = ret.dropna()
    n = len(r)
    if n < 20:
        return {"alpha_annual": np.nan, "t": np.nan, "p": np.nan, "n": n}
    mu = float(r.mean())
    sd = float(r.std(ddof=1))
    t = mu / (sd / np.sqrt(n)) if sd > 0 else np.nan
    # 正态近似双侧 p（样本量通常 ≥ 数百，够用）
    from math import erfc, sqrt
    p = float(erfc(abs(t) / sqrt(2))) if np.isfinite(t) else np.nan
    return {"alpha_annual": mu * ann, "t": t, "p": p, "n": n}


def residual_alpha(strategy_ret: pd.Series,
                   exposures: pd.DataFrame,
                   factor_rets: pd.DataFrame,
                   ann: int = 252) -> dict:
    """风格中性后的残差 alpha（**主口径**）

    返回 dict：
        mixed       混口径 alpha（原报告口径，降为参考行）
        residual    残差 alpha（主口径）
        contrib     各风格对收益的贡献（年化），用于解释"钱从哪来"
        exposure_mean  风格暴露时间均值
    """
    s = strategy_ret.dropna()
    out = {"mixed": _alpha_tstat(s, ann),
           "residual": {"alpha_annual": np.nan, "t": np.nan, "p": np.nan, "n": 0},
           "contrib": {}, "exposure_mean": {}, "n_days": int(len(s))}
    if s.empty or exposures is None or exposures.empty or factor_rets is None \
            or factor_rets.empty:
        return out

    names = [c for c in factor_rets.columns if c in exposures.columns]
    if not names:
        return out
    idx = s.index
    E = exposures.reindex(index=idx)[names].astype(float)
    B = factor_rets.reindex(index=idx)[names].astype(float)
    ok = E.notna().all(axis=1) & B.notna().all(axis=1)
    if int(ok.sum()) < 20:
        return out

    # 风格贡献 = β_k,t · E_k,t（逐日），残差 = r_p − Σ 贡献
    contrib = (B[ok] * E[ok])
    total_contrib = contrib.sum(axis=1)
    resid = s[ok] - total_contrib
    out["residual"] = _alpha_tstat(resid, ann)
    out["contrib"] = {k: float(contrib[k].mean() * ann) for k in names}
    out["exposure_mean"] = {k: float(E[k].mean()) for k in names}
    out["n_used"] = int(ok.sum())
    return out


def style_section(res: dict, holdings: pd.DataFrame = None,
                  panel: dict = None, fundamentals: dict = None,
                  returns: pd.DataFrame = None, mask: pd.DataFrame = None,
                  title: str = "六、风格分解（残差 alpha 为主口径）") -> List[str]:
    """渲染风格分解段落，供 result_report 的 `extra_sections` 使用"""
    L = ["", "-" * 78, title, "-" * 78]
    mix = res.get("mixed", {})
    rsl = res.get("residual", {})
    L.append("  口径说明：混口径 alpha 混着风格暴露，「选股能力」与「风格 beta」")
    L.append("            分不开；**残差 alpha** 才是对选股能力的检验。")
    L.append("")
    L.append(f"  {'口径':<14}{'年化':>10}{'t':>8}{'p':>9}{'样本':>8}")
    L.append(f"  {'混口径(参考)':<14}{mix.get('alpha_annual', float('nan')):>9.2%}"
             f"{mix.get('t', float('nan')):>8.2f}{mix.get('p', float('nan')):>9.3f}"
             f"{mix.get('n', 0):>8}")
    L.append(f"  {'残差(主口径)':<14}{rsl.get('alpha_annual', float('nan')):>9.2%}"
             f"{rsl.get('t', float('nan')):>8.2f}{rsl.get('p', float('nan')):>9.3f}"
             f"{rsl.get('n', 0):>8}")
    if res.get("exposure_mean"):
        L.append("")
        L.append("  风格暴露（时间均值；正=偏该因子高端）：")
        for k, v in res["exposure_mean"].items():
            c = res.get("contrib", {}).get(k)
            lab = STYLE_LABEL.get(k, k)
            L.append(f"    {lab:<12} 暴露 {v:>+7.3f}   年化贡献 "
                     f"{('%.2f%%' % (c * 100)) if c is not None else '-':>9}")
    return L


__all__ = ["DEFAULT_STYLES", "STYLE_LABEL", "build_style_panels",
           "cross_section_factor_returns", "residual_alpha", "style_section"]
