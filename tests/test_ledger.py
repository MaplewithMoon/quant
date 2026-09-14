# -*- coding: utf-8 -*-
"""账目不变量测试

针对本项目历史上真实出现过的三个 P0 缺陷做的回归测试：
  1. 双账本 —— scripts/backtest.py 用成交记录重推净值，与 engine.portfolio
     实测差 0.96%（账户 1,040,192.63 vs 曲线 1,030,567.10）
  2. 卖出佣金符号反了 —— 佣金被加进 Order.filled_amount，卖出时又被当成
     成交价使用，于是佣金被当作收入加回现金，往返手续费净额≈0
  3. 胜率分母含零盈亏买入行 —— 实测报 19.05%，真实 40%

这些不变量一旦被破坏（例如以后有人又引入第二本账），CI 会直接失败。
"""
import os
import sys

import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from data.dataset import DataSet
from backtest.engine import BacktestEngine
from backtest.metrics import Metrics
from strategy.base import Strategy, Signal, Action
from mocks import make_dataset


def _flat_dataset(n=6, price=10.0):
    """价格恒定的行情：把"价格波动"这个变量消掉，只剩费用的影响"""
    dates = pd.bdate_range("2024-01-01", periods=n)
    df = pd.DataFrame({
        "code": "TEST", "trade_date": dates,
        "open": price, "high": price, "low": price, "close": price,
        "volume": 1_000_000.0, "amount": price * 1_000_000.0,
    })
    return DataSet(symbol="TEST", data=df.set_index("trade_date"))


class BuyThenSell(Strategy):
    """第 0 根 bar 满仓买入，第 1 根 bar 全部卖出"""

    def __init__(self):
        super().__init__("BuyThenSell")

    def on_bar(self, data, idx):
        if idx == 0:
            return Signal(Action.BUY, size=1.0)
        if idx == 1:
            return Signal(Action.SELL, size=1.0)
        return Signal(Action.HOLD)


class BuyAndHold(Strategy):
    """第 0 根 bar 满仓买入后一直持有"""

    def __init__(self):
        super().__init__("BuyAndHold")

    def on_bar(self, data, idx):
        return Signal(Action.BUY, size=1.0) if idx == 0 else Signal(Action.HOLD)


# ============================================================
# 不变量①：只有一本账
# ============================================================
def test_equity_curve_equals_portfolio():
    ds = make_dataset(n=60, seed=7)
    engine = BacktestEngine(initial_capital=100_000)
    engine.run(ds, BuyThenSell())

    eq = engine.equity_curve
    assert len(eq) == len(ds), f"每个 bar 都应有权益快照: {len(eq)} vs {len(ds)}"
    assert engine.ledger_gap < 1e-6, f"账目差额 {engine.ledger_gap:,.6f}"
    assert abs(eq.iloc[-1] - engine.portfolio.total_value) < 1e-6
    # next_open 模式下第 0 根 bar 只产生委托、不成交 → 权益仍等于初始资金
    assert abs(eq.iloc[0] - 100_000) < 1e-9, \
        f"第 0 根 bar 尚未成交，权益应为初始资金，实际 {eq.iloc[0]:,.2f}"
    print(f"[OK] 净值曲线与账户一致: {eq.iloc[-1]:,.2f} (gap={engine.ledger_gap:.2e}, "
          f"点数={len(eq)})")


def test_no_trades_equity_is_flat():
    """策略从不交易时，净值必须恒等于初始资金

    修正前 build_equity_curve 在 trades 为空时返回 close/close[0]*capital，
    即"不交易也跟随市场涨跌"，这是错的（账户里全是现金，不该有市场敞口）。
    """

    class NeverTrade(Strategy):
        def __init__(self):
            super().__init__("NeverTrade")

        def on_bar(self, data, idx):
            return Signal(Action.HOLD)

    ds = make_dataset(n=30, seed=5)
    engine = BacktestEngine(initial_capital=100_000)
    trades = engine.run(ds, NeverTrade())

    eq = engine.equity_curve
    assert trades.empty, "不应有成交"
    assert (eq == 100_000).all(), "无交易时净值应恒为初始资金"
    assert engine.portfolio.cash == 100_000
    print(f"[OK] 无交易时净值恒为 {eq.iloc[-1]:,.2f}（不随市场波动）")


# ============================================================
# 不变量②：费用真的被扣了（佣金符号回归测试）
# ============================================================
def test_round_trip_costs_money():
    """价格不变时，一次完整往返必须恰好亏掉两笔手续费

    旧实现下卖出会把佣金加回现金 → 往返净亏损为 0 → 本断言失败。
    """
    ds = _flat_dataset(n=4, price=10.0)
    engine = BacktestEngine(initial_capital=100_000, commission=0.0001,
                            slippage=0.0, min_commission=5.0)
    trades = engine.run(ds, BuyThenSell())

    final = engine.portfolio.total_value
    fees = float(trades["fee"].sum())        # 合计费用 = 佣金 + 印花税 + 过户费

    assert fees > 0, "应产生手续费"
    assert final < 100_000, f"价格不变时不应盈利，实际 {final:,.2f}"
    assert abs((100_000 - final) - fees) < 1e-6, (
        f"价格没动，亏损 {100_000 - final:,.6f} 应等于手续费 {fees:,.6f}")
    print(f"[OK] 往返亏损 {100_000 - final:,.4f} == 手续费合计 {fees:,.4f}")


def test_commission_sign_on_sell():
    """卖出收入必须小于成交额（费用是减项，不是加项）"""
    from execution.broker import SimulatedBroker
    from execution.order import Order, OrderSide

    broker = SimulatedBroker(commission=0.0001, min_commission=5.0)
    o = Order(symbol="T", side=OrderSide.SELL, size=1000, price=10.0)
    broker.execute(o, 10.0)

    assert o.filled_amount > 0
    assert o.commission > 0
    assert o.sell_proceeds < o.filled_amount, "卖出净收入必须小于成交额"
    assert abs(o.sell_proceeds - (o.filled_amount - o.total_fee)) < 1e-9
    print(f"[OK] 卖出: 成交额 {o.filled_amount:,.2f} - 费用 {o.total_fee:,.2f} "
          f"= 净收入 {o.sell_proceeds:,.2f}")


# ============================================================
# 不变量③：现金不为负 + 成交记录与实际持仓一致
# ============================================================
def test_cash_non_negative_and_no_silent_shrink():
    """静默缩量回归测试

    旧实现 Portfolio.buy 在现金不足时会偷偷把 size 改小，但引擎的成交记录
    仍按原始委托量写盘 → 记录 125,371 股 vs 实持 125,300 股 → 两本账漂移。
    """
    ds = make_dataset(n=40, seed=3)
    engine = BacktestEngine(initial_capital=50_000)
    trades = engine.run(ds, BuyAndHold())

    assert engine.portfolio.cash >= -1e-6, f"现金为负: {engine.portfolio.cash}"

    buy = trades[trades["action"] == "buy"]
    assert not buy.empty, "应至少有一笔买入"
    recorded = float(buy["size"].iloc[0])
    held = engine.portfolio.positions.get("TEST")
    actual = float(held.size) if held else 0.0
    assert abs(recorded - actual) < 1e-6, (
        f"成交记录 {recorded} != 实际持仓 {actual}（发生了静默缩量）")
    print(f"[OK] 成交记录 {recorded:,.0f} 股 == 实际持仓 {actual:,.0f} 股")


def test_buy_rejects_insufficient_cash():
    """现金不足必须报错，而不是静默缩量"""
    from portfolio.portfolio import Portfolio
    p = Portfolio(1_000)
    try:
        p.buy("TEST", 1000, 10.0, fees=5.0)   # 需要 10005，只有 1000
        raise AssertionError("应当抛出 ValueError")
    except ValueError as e:
        assert "现金不足" in str(e)
    assert p.cash == 1_000, "失败的买入不应改动现金"
    print("[OK] 现金不足时拒绝买入并保持现金不变")


# ============================================================
# 不变量④：胜率只在已平仓交易上计算
# ============================================================
def test_win_rate_uses_closed_trades_only():
    ds = make_dataset(n=120, seed=11)
    engine = BacktestEngine(initial_capital=100_000)
    trades = engine.run(ds, BuyThenSell())
    m = Metrics.compute(engine.equity_curve, trades)

    buys = int((trades["action"] == "buy").sum())
    sells = int((trades["action"] == "sell").sum())
    assert m.total_trades == sells, f"交易次数应等于平仓数 {sells}，实际 {m.total_trades}"

    closed = trades[trades["action"] == "sell"]
    expected = int((closed["pnl"] > 0).sum()) / max(sells, 1)
    assert abs(m.win_rate - expected) < 1e-9, f"胜率 {m.win_rate} != {expected}"
    print(f"[OK] 胜率 {m.win_rate:.2%} 基于 {sells} 笔平仓"
          f"（{buys} 笔买入未计入分母）")


def test_position_pct_is_a_ratio():
    """unrealized_pnl_pct 应该是"率"，不能被股数放大"""
    from portfolio.portfolio import Position
    pos = Position(symbol="T", size=1000, avg_cost=10.0, current_price=11.0)
    assert abs(pos.unrealized_pnl_pct - 0.1) < 1e-12, \
        f"应为 10%，实际 {pos.unrealized_pnl_pct}"
    print(f"[OK] 未实现收益率 {pos.unrealized_pnl_pct:.2%}（旧实现会输出 10000%）")


if __name__ == "__main__":
    test_equity_curve_equals_portfolio()
    test_no_trades_equity_is_flat()
    test_round_trip_costs_money()
    test_commission_sign_on_sell()
    test_cash_non_negative_and_no_silent_shrink()
    test_buy_rejects_insufficient_cash()
    test_win_rate_uses_closed_trades_only()
    test_position_pct_is_a_ratio()
    print("\n全部账目不变量测试通过")
