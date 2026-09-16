# -*- coding: utf-8 -*-
"""测试：回测引擎 + 订单执行"""
import sys
import io
import os
# 不在 import 阶段替换 sys.stdout（会破坏 pytest 的输出捕获，见 test_loader.py 注释）
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np
import pandas as pd
from mocks import make_dataset, make_rising_daily
from strategy.base import Strategy, Signal, Action
from backtest.engine import BacktestEngine
from backtest.metrics import Metrics
from execution.broker import SimulatedBroker
from execution.order import Order, OrderSide
from data.dataset import DataSet


class BuyAndHold(Strategy):
    """测试用：首日满仓买入持有"""
    def on_bar(self, data, idx):
        if idx == 0:
            return Signal(Action.BUY, size=1.0, reason="first_day")
        return Signal(Action.HOLD)


class Alternating(Strategy):
    """测试用：每隔N天切换买卖"""
    def __init__(self, every=10):
        super().__init__("Alternating")
        self.every = every
        self._prev = Action.HOLD

    def on_bar(self, data, idx):
        if idx % self.every == 0:
            if self._prev != Action.BUY:
                self._prev = Action.BUY
                return Signal(Action.BUY, size=1.0)
            self._prev = Action.HOLD
            return Signal(Action.SELL, size=1.0)
        return Signal(Action.HOLD)


def test_buy_and_hold_positive():
    """确定性上涨行情下 BuyAndHold 应盈利"""
    df = make_rising_daily(n=50, daily_ret=0.002)   # 每日+0.2%，确定性上涨
    ds = DataSet(symbol="TEST", data=df.set_index("trade_date"))
    engine = BacktestEngine(initial_capital=100_000)
    engine.run(ds, BuyAndHold())
    assert engine.portfolio.total_value > 100_000, "上涨行情应盈利"
    print(f"[OK] BuyAndHold 盈利: {engine.portfolio.total_value:,.0f}")


def test_commission_min():
    """最低佣金5元生效，且费用与成交额分开记账"""
    broker = SimulatedBroker(commission=0.0001, min_commission=5.0)
    o = Order(symbol="T", side=OrderSide.BUY, size=100, price=10.0)  # 1000元成交额
    broker.execute(o, 10.0)
    turnover = 100 * 10 * 1.001          # 含 0.1% 滑点
    # 修正：旧断言用 (filled_amount - turnover) 反推费用，等于把"佣金混进成交额"
    # 这个 bug 当成了规范。现在费用单列在 order.commission 上。
    assert abs(o.filled_amount - turnover) < 1e-6, "成交额不应包含佣金"
    assert abs(o.commission - 5.0) < 1e-6, f"应为最低5元, 实际{o.commission}"
    assert abs(o.avg_fill_price - 10.01) < 1e-6, "成交均价应为滑点后价格，不含费用"
    print("[OK] 最低佣金5元（成交额与费用分列）")


def test_engine_metrics():
    """回测后 Metrics 可计算且收益合理"""
    ds = make_dataset(n=100, seed=42)
    engine = BacktestEngine(initial_capital=100_000)
    trades = engine.run(ds, Alternating(every=20))
    # 用收盘价构建净值
    eq = pd.Series(100_000, index=ds.data.index, dtype=float)
    balance, pos = 100_000, 0.0
    t = trades.reset_index()
    t["ts"] = pd.to_datetime(t["timestamp"] if "timestamp" in t.columns else t[t.columns[0]])
    for i in range(len(ds.data)):
        date = ds.data.index[i]
        row = t[t["ts"] == date]
        if not row.empty:
            action, price, size = row["action"].iloc[0], row["price"].iloc[0], row["size"].iloc[0]
            if action == "buy":
                pos, balance = size, balance - size * price
            else:
                balance, pos = balance + size * price, 0.0
        eq.iloc[i] = balance + pos * ds.data["close"].iloc[i]
    m = Metrics.compute(eq, trades)
    assert m.total_return > -1.0   # 收益率不应低于 -100%
    assert m.total_trades > 0
    print(f"[OK] Metrics 计算正常: 收益={m.total_return:+.2%}, 交易={m.total_trades}")


def test_order_state_machine():
    """订单状态机：成交后变为 FILLED"""
    broker = SimulatedBroker()
    o = Order(symbol="T", side=OrderSide.BUY, size=100, price=10.0)
    assert o.is_active
    broker.execute(o, 10.0)
    assert o.status.value == "filled"
    assert not o.is_active
    print("[OK] 订单状态机")


if __name__ == "__main__":
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")  # 修正 Windows 控制台中文编码
    test_buy_and_hold_positive()
    test_commission_min()
    test_engine_metrics()
    test_order_state_machine()
    print("\n全部回测/订单测试通过")
