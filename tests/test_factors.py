# -*- coding: utf-8 -*-
"""因子库测试：面板构建 / 因子计算 / IC 与分层评估的正确性

重点守住"无未来函数"：因子只能用当日及以前的数据，前瞻收益从 t+1 起算。
"""
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from factors import (register, get_factor, list_factors, FACTORS,
                     ic_series, ic_stats, quantile_returns, quantile_turnover,
                     monotonicity, factor_decay, evaluate, forward_returns)
from factors.evaluation import _corr


def _panel(n_days=120, n_codes=40, seed=0):
    """构造一个可解析的合成面板（不依赖数据库）"""
    rng = np.random.default_rng(seed)
    dates = pd.bdate_range("2022-01-03", periods=n_days)
    codes = [f"{i:06d}" for i in range(1, n_codes + 1)]
    ret = rng.normal(0, 0.02, size=(n_days, n_codes))
    close = pd.DataFrame(100 * np.exp(np.cumsum(ret, axis=0)),
                         index=dates, columns=codes)
    volume = pd.DataFrame(rng.integers(1e5, 1e7, size=(n_days, n_codes)).astype(float),
                          index=dates, columns=codes)
    return {
        "close": close,
        "close_adj": close,
        "high_adj": close * 1.01,
        "low_adj": close * 0.99,
        "volume": volume,
        "amount": close * volume,
        "total_mv": pd.DataFrame(rng.uniform(1e5, 1e7, size=(n_days, n_codes)),
                                 index=dates, columns=codes),
        "pe_ttm": pd.DataFrame(rng.uniform(5, 80, size=(n_days, n_codes)),
                               index=dates, columns=codes),
        "pb": pd.DataFrame(rng.uniform(0.5, 10, size=(n_days, n_codes)),
                           index=dates, columns=codes),
        "ps_ttm": pd.DataFrame(rng.uniform(1, 30, size=(n_days, n_codes)),
                               index=dates, columns=codes),
        "turnover_rate": pd.DataFrame(rng.uniform(0.5, 15, size=(n_days, n_codes)),
                                      index=dates, columns=codes),
    }


# ============================================================
# 前瞻收益：不能有未来函数
# ============================================================
def test_forward_returns_uses_future_only():
    """fwd(t) 必须由 t 之后的价格决定，且与因子日对齐"""
    close = pd.DataFrame({"A": [10.0, 11.0, 12.0, 13.0]},
                         index=pd.bdate_range("2022-01-03", periods=4))
    f = forward_returns(close, periods=1, lag=1)
    # t=0: 从 t+1(=11) 到 t+2(=12) -> 12/11-1
    assert abs(f["A"].iloc[0] - (12.0 / 11.0 - 1)) < 1e-12
    assert pd.isna(f["A"].iloc[-1]) and pd.isna(f["A"].iloc[-2]), "尾部应无未来数据"
    f5 = forward_returns(close, periods=1, lag=0)
    assert abs(f5["A"].iloc[0] - (11.0 / 10.0 - 1)) < 1e-12
    try:
        forward_returns(close, lag=-1)
        raise AssertionError("负 lag 应报错")
    except ValueError:
        pass
    print("[OK] 前瞻收益：lag/periods 口径正确，尾部不产生未来数据")


def test_factor_uses_only_past_data():
    """把未来数据改掉，因子在改动点之前的值不应变化（因果性检验）"""
    p = _panel(seed=1)
    fac = get_factor("mom_20")
    full = fac.compute(p)

    cut = 80
    p2 = {k: (v.copy() if isinstance(v, pd.DataFrame) else v) for k, v in p.items()}
    for k in ("close", "close_adj", "high_adj", "low_adj", "volume", "amount"):
        if k in p2:
            # 把 cut 之后（含 cut）的数据整体放大，模拟"未来被改写"
            p2[k] = p2[k].copy()
            p2[k].iloc[cut:] = p2[k].iloc[cut:] * 3.0
    modified = fac.compute(p2)

    before = full.iloc[:cut]
    before2 = modified.iloc[:cut]
    diff = (before - before2).abs().max().max()
    assert diff < 1e-9 or pd.isna(diff), \
        f"改写未来数据影响了历史因子值（存在未来函数），最大差异 {diff}"
    print("[OK] 因果性：改写未来数据不影响历史因子值（无未来函数）")


# ============================================================
# 相关与 IC
# ============================================================
def test_corr_spearman_matches_manual():
    a = pd.Series([1.0, 2.0, 3.0, 4.0, 5.0])
    b = pd.Series([2.0, 1.0, 4.0, 3.0, 6.0])
    assert abs(_corr(a, b, "spearman") - _corr(a, b.rank(), "pearson")) < 1e-12
    assert abs(_corr(a, a, "spearman") - 1.0) < 1e-12
    assert abs(_corr(a, -a, "spearman") + 1.0) < 1e-12
    print("[OK] Spearman = 排名后 Pearson（无需 scipy），完全单调时 ±1")


def test_ic_series_detects_perfect_factor():
    """构造一个与未来收益完全正相关的因子，IC 应接近 1"""
    p = _panel(n_days=60, n_codes=30, seed=3)
    close = p["close_adj"]
    fwd = forward_returns(close, periods=1, lag=1)
    ic = ic_series(fwd, fwd, method="spearman", min_count=10)   # 因子=未来收益本身
    assert ic.mean() > 0.99, f"完美因子的 IC 应接近 1，实际 {ic.mean()}"
    s = ic_stats(ic)
    assert s["样本天数"] > 0 and s["正IC占比"] > 0.99
    print(f"[OK] IC 识别完美因子: 均值 {ic.mean():.4f}, t 值极大")


def test_ic_series_random_factor_near_zero():
    """随机因子（用过去数据构造）的 IC 应在 0 附近"""
    p = _panel(n_days=200, n_codes=60, seed=4)
    close = p["close_adj"]
    rng = np.random.default_rng(7)
    noise = pd.DataFrame(rng.normal(size=close.shape),
                         index=close.index, columns=close.columns)
    fwd = forward_returns(close, periods=1, lag=1)
    ic = ic_series(noise, fwd, method="spearman", min_count=20)
    assert abs(ic.mean()) < 0.05, f"随机因子 IC 应接近 0，实际 {ic.mean()}"
    print(f"[OK] 随机因子 IC ≈ 0: {ic.mean():.4f}（{len(ic)} 天）")


def test_ic_stats_edge_cases():
    empty = ic_stats(pd.Series(dtype=float))
    assert empty["样本天数"] == 0 and np.isnan(empty["IC均值"])
    one = ic_stats(pd.Series([0.1]))
    assert one["样本天数"] == 1
    print("[OK] IC 统计在空序列/单点时不崩")


# ============================================================
# 分层
# ============================================================
def test_quantile_returns_direction():
    """direction 应决定"多空"的方向，而分层列始终按因子值升序"""
    p = _panel(n_days=120, n_codes=60, seed=5)
    close = p["close_adj"]
    fwd = forward_returns(close, periods=1, lag=1)
    rng = np.random.default_rng(11)
    f = pd.DataFrame(rng.normal(size=close.shape), index=close.index,
                     columns=close.columns)

    q_up = quantile_returns(f, fwd, n_quantiles=5, direction=1, min_count=20)
    q_dn = quantile_returns(f, fwd, n_quantiles=5, direction=-1, min_count=20)
    assert list(q_up.columns) == ["Q1", "Q2", "Q3", "Q4", "Q5", "多空"]
    # 方向相反时多空应互为相反数
    diff = (q_up["多空"] + q_dn["多空"]).abs().max()
    assert diff < 1e-12, f"多空方向调整失败: {diff}"
    print("[OK] 分层多空方向：direction=-1 时多空取反，分层列仍按因子值升序")


def test_quantile_returns_perfect_factor_is_monotonic():
    """因子=未来收益时，分层收益应完全单调"""
    p = _panel(n_days=80, n_codes=50, seed=6)
    close = p["close_adj"]
    fwd = forward_returns(close, periods=1, lag=1)
    q = quantile_returns(fwd, fwd, n_quantiles=5, direction=1, min_count=20)
    assert monotonicity(q, 5) > 0.99, "完美因子应完全单调"
    assert q["多空"].mean() > 0
    print(f"[OK] 完美因子分层完全单调（单调性 {monotonicity(q, 5):.2f}），多空为正")


def test_quantile_turnover_range():
    p = _panel(n_days=100, n_codes=50, seed=7)
    rng = np.random.default_rng(3)
    f = pd.DataFrame(rng.normal(size=p["close"].shape), index=p["close"].index,
                     columns=p["close"].columns)
    t = quantile_turnover(f, n_quantiles=5, min_count=20)
    assert len(t) > 0
    assert ((t >= 0) & (t <= 1)).all(), "换手率应在 [0,1]"
    # 恒定因子 -> 排名不变 -> 换手为 0
    const = pd.DataFrame(1.0, index=f.index, columns=f.columns)
    t0 = quantile_turnover(const, n_quantiles=5, min_count=20)
    assert (t0.abs() < 1e-12).all() or len(t0) == 0, "恒定因子换手应为 0"
    print(f"[OK] 分层换手率: 均值 {t.mean():.2%}，恒定因子换手 0")


# ============================================================
# 注册表与端到端
# ============================================================
def test_registry_and_listing():
    df = list_factors()
    assert len(df) >= 15, f"内置因子应不少于 15 个，实际 {len(df)}"
    assert {"mom_20", "vol_20", "ep", "size"} <= set(FACTORS)
    try:
        get_factor("__not_exist__")
        raise AssertionError("未注册因子应报 KeyError")
    except KeyError:
        pass
    print(f"[OK] 因子注册表: {len(df)} 个因子，未知因子报错")


def test_register_custom_factor():
    @register("_test_custom", "测试因子", direction=1, category="测试")
    def _f(panel):
        c = panel["close_adj"]
        return c / c.shift(10) - 1

    assert "_test_custom" in FACTORS
    p = _panel(n_days=60, seed=8)
    v = get_factor("_test_custom").compute(p)
    assert v.shape == p["close"].shape
    print("[OK] 自定义因子注册与计算")


def test_quantile_median_and_divergence_detection():
    """收益右偏会让「算术均值分层」与「中位数分层」方向相反，必须能识别出来

    构造：低因子组每日稳定小赚；高因子组多数日子小亏、但偶尔暴涨。
    → 算术均值被暴涨拉高（高因子组更好），中位数仍显示高因子组更差。
    """
    from factors.base import Factor

    n_days, n_codes = 200, 40
    dates = pd.bdate_range("2022-01-03", periods=n_days)
    codes = [f"{i:06d}" for i in range(1, n_codes + 1)]
    f = pd.DataFrame(np.tile(np.arange(n_codes), (n_days, 1)).astype(float),
                     index=dates, columns=codes)

    ret = np.zeros((n_days, n_codes))
    ret[:, :20] = 0.001                      # 低因子组：稳定小赚
    ret[:, 20:] = -0.002                     # 高因子组：多数小亏
    rng = np.random.default_rng(0)
    spike = rng.random((n_days, 20)) < 0.10
    ret[:, 20:] = np.where(spike, 0.10, ret[:, 20:])   # 偶尔暴涨
    close = pd.DataFrame(100 * np.cumprod(1 + ret, axis=0), index=dates, columns=codes)
    panel = {"close": close, "close_adj": close}

    fac = Factor(name="_diverge", func=lambda p: f, direction=1)
    rep = evaluate(fac, panel, periods=1, n_quantiles=5, min_count=20,
                   with_decay=False)

    mean_ls = rep.quantile["多空"].mean()
    med_ls = rep.quantile_median["多空"].mean()
    assert mean_ls > 0, f"算术均值下高因子组应更好，实际 {mean_ls}"
    assert med_ls < 0, f"中位数下高因子组应更差，实际 {med_ls}"
    assert rep.mean_median_diverges(), "应识别出均值/中位数方向背离"
    assert "背离" in rep.summary() or "相反" in rep.summary()
    print(f"[OK] 右偏导致分层背离可被识别: 多空(均值)={mean_ls:.4%} "
          f"vs 多空(中位)={med_ls:.4%}")


def test_evaluate_end_to_end():
    p = _panel(n_days=150, n_codes=50, seed=9)
    rep = evaluate(get_factor("mom_20"), p, periods=1, n_quantiles=5,
                   min_count=20, with_decay=True)
    assert rep.name == "mom_20"
    assert rep.stats["样本天数"] > 0
    assert not rep.quantile.empty and "多空" in rep.quantile.columns
    assert not rep.decay.empty and set(rep.decay["持有期"]) >= {1, 5}
    s = rep.summary()
    assert "预测力" in s and "分层前瞻收益" in s and "换手率" in s
    assert not rep.quantile_median.empty, "应同时给出中位数分层"
    print(f"[OK] 端到端评估: IC={rep.stats['IC均值']:.4f}, "
          f"多空={rep.quantile['多空'].mean():.4%}, 衰减 {len(rep.decay)} 个持有期")


def test_all_builtin_factors_compute():
    """所有内置因子都应能在合成面板上算出结果（不报错、形状正确）"""
    p = _panel(n_days=140, n_codes=40, seed=10)
    shape = p["close"].shape
    bad = []
    for name in sorted(FACTORS):
        if name.startswith("_"):
            continue
        try:
            v = get_factor(name).compute(p)
            if v.shape != shape:
                bad.append(f"{name}: 形状 {v.shape} != {shape}")
        except Exception as e:
            bad.append(f"{name}: {type(e).__name__}: {e}")
    assert not bad, "以下因子计算失败:\n  " + "\n  ".join(bad)
    print(f"[OK] 全部内置因子在合成面板上计算成功（{len(FACTORS)} 个）")


if __name__ == "__main__":
    test_forward_returns_uses_future_only()
    test_factor_uses_only_past_data()
    test_corr_spearman_matches_manual()
    test_ic_series_detects_perfect_factor()
    test_ic_series_random_factor_near_zero()
    test_ic_stats_edge_cases()
    test_quantile_returns_direction()
    test_quantile_returns_perfect_factor_is_monotonic()
    test_quantile_turnover_range()
    test_quantile_median_and_divergence_detection()
    test_registry_and_listing()
    test_register_custom_factor()
    test_evaluate_end_to_end()
    test_all_builtin_factors_compute()
    print("\n全部因子库测试通过")
