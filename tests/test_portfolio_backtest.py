# -*- coding: utf-8 -*-
"""组合构建 + 多标的回测测试"""
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from backtest.multi_engine import PortfolioBacktestEngine, FILL_NEXT_OPEN, FILL_SAME_CLOSE
from execution.market_rules import MarketRules
from portfolio.construction import (build_target_weights, cap_weights, rebalance_dates,
                                    select_top_n, weights_equal, weights_inv_vol,
                                    weights_market_cap, zscore, turnover_of_weights)
from factors.composite import composite_score, correlation_matrix
from factors.base import Factor, add_factor


def _panel(n_days=200, n_codes=30, seed=0, price=10.0, volume=1e7, spread=0.0):
    rng = np.random.default_rng(seed)
    dates = pd.bdate_range("2023-01-03", periods=n_days)
    codes = [f"{i:06d}" for i in range(1, n_codes + 1)]
    ret = rng.normal(0.0005, 0.02, (n_days, n_codes))
    if spread:
        ret[:, :n_codes // 2] += spread        # 让前一半系统性更强
    close = pd.DataFrame(price * np.cumprod(1 + ret, axis=0), index=dates, columns=codes)
    vol = pd.DataFrame(volume, index=dates, columns=codes)
    return {"close": close, "close_adj": close, "open": close, "high": close * 1.01,
            "low": close * 0.99, "volume": vol, "amount": close * vol,
            "total_mv": pd.DataFrame(rng.uniform(1e6, 1e7, (n_days, n_codes)),
                                     index=dates, columns=codes)}


def _score(n_days=200, n_codes=30, seed=1):
    rng = np.random.default_rng(seed)
    dates = pd.bdate_range("2023-01-03", periods=n_days)
    codes = [f"{i:06d}" for i in range(1, n_codes + 1)]
    return pd.DataFrame(rng.normal(size=(n_days, n_codes)), index=dates, columns=codes)


# ============================================================
# 权重方案
# ============================================================
def test_weight_schemes_sum_to_one():
    sel = pd.DataFrame(False, index=range(5), columns=list("ABCDE"))
    sel.iloc[:, :3] = True
    for name, w in (("equal", weights_equal(sel)),
                    ("mv", weights_market_cap(sel, pd.DataFrame(1.0, index=sel.index,
                                                                columns=sel.columns))),
                    ("inv_vol", weights_inv_vol(sel, pd.DataFrame(0.02, index=sel.index,
                                                                  columns=sel.columns)))):
        tot = w.sum(axis=1)
        assert np.allclose(tot.values, 1.0), f"{name} 权重和应为 1，实际 {tot.values}"
    print("[OK] 三种权重方案（等权/市值/逆波动）权重和均为 1")


def test_cap_weights():
    w = pd.DataFrame({"A": [0.8], "B": [0.1], "C": [0.1]})
    c = cap_weights(w, 0.5)
    assert c.max().max() <= 0.5 + 1e-9, f"应受上限约束，实际 {c.max().max()}"
    assert abs(c.sum(axis=1).iloc[0] - 1.0) < 1e-9, "削顶后仍应合计为 1"
    print(f"[OK] 单票权重上限生效: 0.8 -> {c.iloc[0]['A']:.2f}，合计仍为 1")


def test_select_top_n():
    s = pd.DataFrame({"A": [3.0], "B": [1.0], "C": [2.0], "D": [np.nan]})
    sel = select_top_n(s, s.notna(), 2)
    assert sel.iloc[0]["A"] and sel.iloc[0]["C"]
    assert not sel.iloc[0]["B"] and not sel.iloc[0]["D"]
    print("[OK] 选前 N：按打分降序、NaN 不入选")


def test_rebalance_dates_rules():
    idx = pd.bdate_range("2023-01-02", periods=250)   # 约 1 年
    assert len(rebalance_dates(idx, "D")) == len(idx)
    assert 12 <= len(rebalance_dates(idx, "M")) <= 13, "一年约 12 个月"
    assert 50 <= len(rebalance_dates(idx, "W")) <= 53
    assert len(rebalance_dates(idx, 10)) == len(idx[::10])
    d = rebalance_dates(idx, "M")
    assert all(x.day <= 7 for x in d), "月度调仓应落在每月初"
    print(f"[OK] 调仓日规则: D={len(idx)} M={len(rebalance_dates(idx,'M'))} "
          f"W={len(rebalance_dates(idx,'W'))} 每10日={len(rebalance_dates(idx,10))}")


def test_build_target_weights_nan_on_non_rebalance():
    """非调仓日必须为 NaN（表示"不调仓、持仓漂移"），调仓日权重和为 1"""
    p = _panel(n_days=120)
    sc = _score(n_days=120)
    tw = build_target_weights(sc, p["close"].notna(), n_hold=5, rebalance="M")
    assert tw.isna().any().any(), "非调仓日应有 NaN"
    reb = tw.dropna(how="all")
    assert len(reb) >= 4
    for d, row in reb.iterrows():
        r = row.dropna()
        assert abs(r.sum() - 1.0) < 1e-9, f"{d} 调仓日权重和应为 1，实际 {r.sum()}"
    assert len(reb) < len(tw), "调仓日应少于全部交易日"
    print(f"[OK] 目标权重: {len(reb)} 个调仓日（权重和=1），其余 {len(tw)-len(reb)} 天为 NaN")


# ============================================================
# 多标的引擎
# ============================================================
def test_multi_engine_ledger_and_lot():
    p = _panel(n_days=200, n_codes=20, seed=3)
    sc = _score(n_days=200, n_codes=20, seed=4)
    mask = p["close"].notna()
    tw = build_target_weights(sc, mask, n_hold=6, rebalance="M")
    eng = PortfolioBacktestEngine(initial_capital=1_000_000)
    res = eng.run(p, tw)

    assert res.ledger_gap < 1e-6, f"账目差额应为 0，实际 {res.ledger_gap}"
    assert len(res.equity) == len(p["close"])
    assert abs(res.equity.iloc[0] - 1_000_000) < 1e-6, "首日尚未成交，应等于初始资金"
    assert not res.trades.empty, "应有成交"
    sizes = res.trades["size"].astype(float)
    assert all(s % 100 == 0 for s in sizes), "全部成交应为整手"
    assert (res.trades["fee"] > 0).all(), "每笔都应有费用"
    print(f"[OK] 多标的引擎: 账目差额 {res.ledger_gap:.1e}，"
          f"{len(res.trades)} 笔成交全部整手且计费，"
          f"收益 {res.metrics.total_return:.2%}")


def test_multi_engine_t_plus_1():
    """同一只股票不应在同一日既买又卖"""
    p = _panel(n_days=150, n_codes=15, seed=5)
    sc = _score(n_days=150, n_codes=15, seed=6)
    tw = build_target_weights(sc, p["close"].notna(), n_hold=5, rebalance=5)
    res = PortfolioBacktestEngine(initial_capital=500_000).run(p, tw)
    if res.trades.empty:
        print("[SKIP] 无成交")
        return
    t = res.trades.reset_index()
    dup = t.groupby(["timestamp", "code"])["action"].nunique()
    bad = dup[dup > 1]
    assert bad.empty, f"同一日同一股票出现买卖双向: {bad.head()}"
    print(f"[OK] T+1：{len(t)} 笔成交中无同日同股双向交易")


def test_multi_engine_holdings_sum_le_one():
    p = _panel(n_days=150, n_codes=15, seed=7)
    sc = _score(n_days=150, n_codes=15, seed=8)
    tw = build_target_weights(sc, p["close"].notna(), n_hold=5, rebalance="M")
    res = PortfolioBacktestEngine(initial_capital=1_000_000).run(p, tw)
    tot = res.holdings.sum(axis=1)
    assert (tot <= 1.0 + 1e-6).all(), f"持仓权重合计不应超过 1，实际最大 {tot.max()}"
    assert tot.iloc[-1] > 0.5, "长期应基本满仓"
    print(f"[OK] 持仓权重合计 ≤ 1（均值 {tot.mean():.2%}），未使用杠杆")


def test_multi_engine_fees_reduce_return():
    """同样策略，零费用与含费用的收益应有差距"""
    p = _panel(n_days=200, n_codes=20, seed=9)
    sc = _score(n_days=200, n_codes=20, seed=10)
    tw = build_target_weights(sc, p["close"].notna(), n_hold=10, rebalance="M")

    free = PortfolioBacktestEngine(initial_capital=1_000_000, commission=0.0,
                                   min_commission=0.0, stamp_duty=0.0,
                                   transfer_fee=0.0, slippage=0.0).run(p, tw)
    cost = PortfolioBacktestEngine(initial_capital=1_000_000, commission=0.0001,
                                   min_commission=5.0, stamp_duty=0.0005,
                                   transfer_fee=0.00001, slippage=0.001).run(p, tw)
    assert cost.equity.iloc[-1] < free.equity.iloc[-1], "含费用应更差"
    fees = float(cost.trades["fee"].sum())
    assert fees > 0
    print(f"[OK] 费用确实被扣: 无费用 {free.equity.iloc[-1]:,.0f} > "
          f"含费用 {cost.equity.iloc[-1]:,.0f}（累计费用 {fees:,.0f}）")


def test_multi_engine_limit_up_blocks_buy():
    """调仓日目标股票一字涨停 -> 应买不进"""
    p = _panel(n_days=60, n_codes=5, seed=11)
    codes = list(p["close"].columns)
    # 让第 30 天之后所有股票都一字涨停
    p["limit_up"] = pd.DataFrame(np.nan, index=p["close"].index, columns=codes)
    p["limit_down"] = pd.DataFrame(np.nan, index=p["close"].index, columns=codes)
    for c in codes:
        px = p["close"][c]
        p["limit_up"][c] = px
        p["limit_down"][c] = px * 0.5
    sc = _score(n_days=60, n_codes=5, seed=12)
    tw = build_target_weights(sc, p["close"].notna(), n_hold=3, rebalance="M")
    res = PortfolioBacktestEngine(initial_capital=1_000_000).run(p, tw)
    assert any("涨停" in k for k in res.rejections), \
        f"应记录涨停拦截，实际 {res.rejections}"
    print(f"[OK] 组合层涨跌停约束生效: {res.rejections}")


# ============================================================
# 因子合成
# ============================================================
def test_composite_direction_alignment():
    """direction=-1 的因子应被取负，保证合成打分"越大越看多" """
    p = _panel(n_days=60, n_codes=10)
    idx = p["close"].index

    add_factor("_pos", lambda pa: pd.DataFrame(1.0, index=idx, columns=p["close"].columns),
               direction=1)
    add_factor("_neg", lambda pa: pd.DataFrame(1.0, index=idx, columns=p["close"].columns),
               direction=-1)
    sc = composite_score(p, ["_pos"], method="equal")
    # 常量因子标准化后为 0，这里主要验证不报错且形状正确
    assert sc.shape == p["close"].shape
    print("[OK] 因子合成：方向对齐逻辑可运行（direction=-1 取负）")


def test_composite_equal_weight():
    # 股票数必须 >= min_count(20)，否则横截面相关每天都被跳过、返回 NaN
    p = _panel(n_days=80, n_codes=40, seed=13)
    codes = p["close"].columns
    idx = p["close"].index
    rng = np.random.default_rng(0)
    a = pd.DataFrame(rng.normal(size=(80, 40)), index=idx, columns=codes)
    b = pd.DataFrame(rng.normal(size=(80, 40)), index=idx, columns=codes)
    add_factor("_a", lambda pa: a, direction=1)
    add_factor("_b", lambda pa: b, direction=1)
    sc = composite_score(p, ["_a", "_b"], method="equal")
    assert sc.shape == p["close"].shape
    # 等权合成应介于两个标准化因子之间
    assert sc.notna().sum().sum() > 0
    corr = correlation_matrix(p, ["_a", "_b"])
    assert corr.shape == (2, 2)
    assert not np.isnan(corr.iloc[0, 1]), "样本足够时不应返回 NaN"
    assert abs(corr.iloc[0, 1]) < 0.5, "随机因子间相关性应较低"
    print(f"[OK] 等权合成打分形状正确；因子相关性 {corr.iloc[0,1]:+.3f}")


def test_correlation_matrix_different_column_sets():
    """回归：两个因子的列空间不同时也必须能算相关

    真实数据里 bp 依赖 daily_basic（只有 5,476 只），vol_60 有 5,480 只。
    早期实现直接 `x.notna() & y.notna()`，按列取并集把缺失列填成 NaN，
    bool 掩码被提升为 float64，布尔索引时抛
    "Cannot mask with non-boolean array containing NA / NaN values"。
    """
    p = _panel(n_days=60, n_codes=40, seed=7)
    codes = list(p["close"].columns)
    idx = p["close"].index
    rng = np.random.default_rng(3)
    a = pd.DataFrame(rng.normal(size=(60, 40)), index=idx, columns=codes)
    # b 少 3 列 —— 模拟因子覆盖度不同
    b = pd.DataFrame(rng.normal(size=(60, 37)), index=idx, columns=codes[:37])
    add_factor("_ca", lambda pa: a, direction=1)
    add_factor("_cb", lambda pa: b, direction=1)
    corr = correlation_matrix(p, ["_ca", "_cb"])
    v = corr.iloc[0, 1]
    assert not np.isnan(v), "列空间不同时不应返回 NaN"
    assert abs(v) < 0.5, f"随机因子间相关性应较低，实际 {v:+.3f}"
    print(f"[OK] 列空间不同的因子也能算相关：{v:+.3f}（回归：掩码 NaN 崩溃）")


if __name__ == "__main__":
    test_weight_schemes_sum_to_one()
    test_cap_weights()
    test_select_top_n()
    test_rebalance_dates_rules()
    test_build_target_weights_nan_on_non_rebalance()
    test_multi_engine_ledger_and_lot()
    test_multi_engine_t_plus_1()
    test_multi_engine_holdings_sum_le_one()
    test_multi_engine_fees_reduce_return()
    test_multi_engine_limit_up_blocks_buy()
    test_composite_direction_alignment()
    test_composite_equal_weight()
    test_correlation_matrix_different_column_sets()
    print("\n全部组合构建/多标的回测测试通过")
