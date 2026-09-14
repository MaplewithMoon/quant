# -*- coding: utf-8 -*-
"""滑点与市场冲击成本模型测试

核心结论要守住：**下单量越大，成交价越差**；固定滑点没有这个性质，
所以用固定滑点回测大资金策略会系统性高估收益。
"""
import os
import sys
from math import sqrt

import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from data.dataset import DataSet
from backtest.engine import BacktestEngine
from execution.broker import SimulatedBroker
from execution.impact import (NoSlippage, FixedSlippage, SqrtImpact, build_model)
from execution.order import Order, OrderSide
from strategy.base import Strategy, Signal, Action
from mocks import make_dataset


class BuyAt(Strategy):
    def __init__(self, k=0):
        super().__init__("BuyAt")
        self.k = k

    def on_bar(self, data, idx):
        return Signal(Action.BUY, size=1.0) if idx == self.k else Signal(Action.HOLD)


def _flat(n=6, price=10.0, volume=1_000_000.0):
    dates = pd.bdate_range("2024-01-01", periods=n)
    df = pd.DataFrame({
        "code": "TEST", "trade_date": dates,
        "open": [price] * n, "high": [price] * n,
        "low": [price] * n, "close": [price] * n,
        "volume": [volume] * n, "amount": [price * volume] * n,
    })
    return DataSet(symbol="TEST", data=df.set_index("trade_date"))


# ============================================================
# 模型本身
# ============================================================
def test_fixed_slippage_is_size_independent():
    m = FixedSlippage(0.002)
    assert m.impact_pct(100, 1_000_000) == 0.002
    assert m.impact_pct(500_000, 1_000_000) == 0.002
    p_small = m.adjust(10.0, OrderSide.BUY, 100, 1_000_000)
    p_big = m.adjust(10.0, OrderSide.BUY, 500_000, 1_000_000)
    assert p_small == p_big == 10.02
    print(f"[OK] 固定滑点与下单量无关: 小单 {p_small} == 大单 {p_big}")


def test_sqrt_impact_grows_with_size():
    m = SqrtImpact(k=0.1)
    p_small = m.adjust(10.0, OrderSide.BUY, 1_000, 1_000_000)     # 占 0.1%
    p_mid = m.adjust(10.0, OrderSide.BUY, 10_000, 1_000_000)      # 占 1%
    p_big = m.adjust(10.0, OrderSide.BUY, 100_000, 1_000_000)     # 占 10%
    assert p_small < p_mid < p_big, f"{p_small} {p_mid} {p_big}"
    print(f"[OK] 平方根冲击随下单量递增: {p_small} < {p_mid} < {p_big}")


def test_sqrt_impact_matches_formula():
    m = SqrtImpact(k=0.1)
    size, volume, price = 10_000.0, 1_000_000.0, 10.0
    expect_impact = 0.1 * sqrt(size / volume)          # 0.01
    expect_price = round(price * (1 + expect_impact), 2)
    got = m.adjust(price, OrderSide.BUY, size, volume)
    assert abs(got - expect_price) < 1e-9, f"期望 {expect_price}, 实际 {got}"
    assert abs(m.impact_pct(size, volume) - expect_impact) < 1e-12
    print(f"[OK] 与公式一致: 占比 {size/volume:.2%} → 冲击 {expect_impact:.4%} → 价 {got}")


def test_impact_capped_and_sell_symmetric():
    m = SqrtImpact(k=0.1, max_impact=0.05)
    # 极端大单：占比 100% → 原始冲击 0.1，应被 cap 到 0.05
    assert abs(m.impact_pct(1_000_000, 1_000_000) - 0.05) < 1e-12
    buy = m.adjust(10.0, OrderSide.BUY, 1_000_000, 1_000_000)
    sell = m.adjust(10.0, OrderSide.SELL, 1_000_000, 1_000_000)
    assert buy > 10.0 > sell, "买入加价、卖出折价"
    print(f"[OK] 冲击封顶 5%: 买 {buy} / 卖 {sell}（对称）")


def test_no_volume_means_no_impact():
    m = SqrtImpact(k=0.1)
    assert m.impact_pct(10_000, None) == 0.0
    assert m.impact_pct(10_000, 0) == 0.0
    print("[OK] 无成交量数据时冲击为 0（不静默编造价格）")


def test_build_model():
    assert isinstance(build_model("none"), NoSlippage)
    assert isinstance(build_model("fixed", rate=0.003), FixedSlippage)
    assert isinstance(build_model("sqrt", k=0.2), SqrtImpact)
    assert build_model("sqrt", k=0.2).k == 0.2
    try:
        build_model("bogus")
        raise AssertionError("应当报错")
    except ValueError:
        pass
    print("[OK] build_model 名称解析正常，未知名称报错")


# ============================================================
# 券商层：参与率上限
# ============================================================
def test_participation_cap():
    b = SimulatedBroker(max_participation=0.10, lot_size=100)
    assert b.max_tradable_size(1_000_000, is_buy=True) == 100_000
    assert b.max_tradable_size(1_000_000, is_buy=False) == 100_000
    b2 = SimulatedBroker(max_participation=0, lot_size=100)
    assert b2.max_tradable_size(1_000_000) == float("inf")
    assert b.max_tradable_size(None) == float("inf")
    print("[OK] 参与率上限=10% 生效；设为 0 或缺成交量时不限")


# ============================================================
# 引擎层：端到端
# ============================================================
def test_engine_respects_participation_cap():
    """单笔买入不得超过当日成交量的 10%"""
    ds = _flat(n=5, price=10.0, volume=50_000.0)      # 10% = 5,000 股
    engine = BacktestEngine(initial_capital=1_000_000, slippage=0.0,
                            max_participation=0.10)
    trades = engine.run(ds, BuyAt(0))
    assert not trades.empty
    size = float(trades["size"].iloc[0])
    assert size == 5_000, f"应被限制为 5,000 股，实际 {size:,.0f}"
    print(f"[OK] 参与率上限生效: 想买满仓却只成交 {size:,.0f} 股（当日量的 10%）")


def test_impact_reduces_return():
    """同一策略，启用冲击成本的收益应低于固定滑点"""
    ds = make_dataset(n=120, seed=21)

    def run(model):
        e = BacktestEngine(initial_capital=500_000, slippage_model=model,
                           max_participation=0.0)
        e.run(ds, BuyAt(0))
        return e.portfolio.total_value

    fixed = run(FixedSlippage(0.001))
    sqrt_ = run(SqrtImpact(k=0.5))       # 放大系数以便观察
    none = run(NoSlippage())
    assert fixed <= none, "有滑点不应优于零滑点"
    assert sqrt_ <= fixed, f"冲击成本应更贵: sqrt={sqrt_:,.0f} fixed={fixed:,.0f}"
    print(f"[OK] 零滑点 {none:,.0f} ≥ 固定滑点 {fixed:,.0f} ≥ 平方根冲击 {sqrt_:,.0f}")


def test_engine_impact_price_above_market():
    ds = _flat(n=5, price=10.0, volume=1_000_000.0)
    engine = BacktestEngine(initial_capital=1_000_000, slippage=0.0,
                            slippage_model=SqrtImpact(k=0.5),
                            max_participation=0.10)
    trades = engine.run(ds, BuyAt(0))
    px = float(trades["price"].iloc[0])
    assert px > 10.0, f"冲击成本应使买入价高于市价 10.0，实际 {px}"
    print(f"[OK] 引擎成交价含冲击: {px:.4f} > 市价 10.0")


if __name__ == "__main__":
    test_fixed_slippage_is_size_independent()
    test_sqrt_impact_grows_with_size()
    test_sqrt_impact_matches_formula()
    test_impact_capped_and_sell_symmetric()
    test_no_volume_means_no_impact()
    test_build_model()
    test_participation_cap()
    test_engine_respects_participation_cap()
    test_impact_reduces_return()
    test_engine_impact_price_above_market()
    print("\n全部冲击成本测试通过")
