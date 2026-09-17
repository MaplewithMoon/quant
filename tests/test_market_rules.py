# -*- coding: utf-8 -*-
"""A股交易制度约束测试：T+1 / 涨跌停 / 停牌 / 一手取整 / 印花税 / 成交时点 / 数据截断

这些约束以前在回测里完全不存在，导致结果系统性偏乐观：
  - 当日买当日卖（无 T+1）
  - 涨停还能买、跌停还能卖
  - 停牌日照样成交
  - 能买零股（非整手）
  - 只算佣金，漏掉印花税（印花税是佣金的好几倍）
  - 用第 i 根 bar 收盘价决策并按同一收盘价成交（未来函数）
"""
import os
import sys

import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from data.dataset import DataSet
from backtest.engine import BacktestEngine, FILL_NEXT_OPEN, FILL_SAME_CLOSE
from execution.market_rules import MarketRules
from execution.broker import SimulatedBroker
from execution.order import Order, OrderSide
from portfolio.portfolio import Portfolio
from strategy.base import Strategy, Signal, Action
from mocks import make_dataset


def _dataset(opens, closes, limit_up=None, limit_down=None, suspended=None):
    n = len(opens)
    dates = pd.bdate_range("2024-01-01", periods=n)
    df = pd.DataFrame({
        "code": "TEST", "trade_date": dates,
        "open": opens,
        "high": [max(o, c) for o, c in zip(opens, closes)],
        "low": [min(o, c) for o, c in zip(opens, closes)],
        "close": closes,
        "volume": [1e6] * n,
        "amount": [c * 1e6 for c in closes],
    })
    if limit_up is not None:
        df["limit_up"] = limit_up
    if limit_down is not None:
        df["limit_down"] = limit_down
    if suspended is not None:
        df["suspended"] = suspended
    return DataSet(symbol="TEST", data=df.set_index("trade_date"))


class BuyAt(Strategy):
    """在第 k 根 bar 发买入信号"""

    def __init__(self, k=0):
        super().__init__("BuyAt")
        self.k = k

    def on_bar(self, data, idx):
        return Signal(Action.BUY, size=1.0) if idx == self.k else Signal(Action.HOLD)


class BuyThenSellAt(Strategy):
    """第 b 根 bar 买，第 s 根 bar 卖"""

    def __init__(self, b=0, s=1):
        super().__init__("BuyThenSellAt")
        self.b, self.s = b, s

    def on_bar(self, data, idx):
        if idx == self.b:
            return Signal(Action.BUY, size=1.0)
        if idx == self.s:
            return Signal(Action.SELL, size=1.0)
        return Signal(Action.HOLD)


# ============================================================
# T+1
# ============================================================
def test_t_plus_1_at_portfolio_level():
    p = Portfolio(100_000)
    p.new_day()
    p.buy("X", 100, 10.0, fees=0.0)
    assert p.sellable_size("X") == 0, "当日买入的股票当日不可卖（T+1）"
    try:
        p.sell("X", 100, 10.0)
        raise AssertionError("T+1 应当阻止当日卖出")
    except ValueError as e:
        assert "可卖" in str(e)
    p.new_day()                                   # 进入下一个交易日
    assert p.sellable_size("X") == 100, "次日应全部解锁"
    p.sell("X", 100, 11.0)
    assert p.position_size("X") == 0
    print("[OK] T+1：当日买入不可卖，次日起可卖")


def test_no_same_day_round_trip_in_trades():
    """回测成交记录里不应出现"同一日期既买又卖"的 T+0 往返"""
    ds = make_dataset(n=40, seed=9)
    engine = BacktestEngine(initial_capital=100_000)
    trades = engine.run(ds, BuyThenSellAt(b=5, s=6))
    if trades.empty:
        print("[OK] 无成交（跳过同日往返检查）")
        return
    dup = trades.index[trades.index.duplicated()].unique()
    assert len(dup) == 0, f"同一日期出现多笔成交（疑似 T+0）: {list(dup)}"
    print("[OK] 无同日买卖往返（T+1 生效）")


# ============================================================
# 涨跌停
# ============================================================
def test_limit_up_blocks_buy():
    """次日开盘即涨停 → 买不进"""
    ds = _dataset(opens=[10.0, 10.5, 10.5], closes=[10.0, 10.5, 10.5],
                  limit_up=[10.5, 10.5, 10.5])
    engine = BacktestEngine(initial_capital=100_000)
    trades = engine.run(ds, BuyAt(0))
    assert trades.empty, "涨停封板不应成交"
    summary = engine.rejections.summary()
    assert any("涨停" in k for k in summary), f"应记录涨停拦截: {summary}"
    print(f"[OK] 涨停不可买 → {list(summary)[0]}")


def test_limit_down_blocks_sell():
    """次日开盘即跌停 → 卖不掉"""
    ds = _dataset(opens=[10.0, 10.0, 9.0], closes=[10.0, 10.0, 9.0],
                  limit_down=[9.0, 9.0, 9.0])
    engine = BacktestEngine(initial_capital=100_000)
    trades = engine.run(ds, BuyThenSellAt(b=0, s=1))
    assert int((trades["action"] == "buy").sum()) == 1, "买入应成交"
    assert int((trades["action"] == "sell").sum()) == 0, "跌停封板不应卖出"
    summary = engine.rejections.summary()
    assert any("跌停" in k for k in summary), f"应记录跌停拦截: {summary}"
    print(f"[OK] 跌停不可卖 → {list(summary)[0]}")


def test_rules_disabled_allows_trade():
    """对照组：关闭制度约束后，同样的涨停数据就能成交"""
    ds = _dataset(opens=[10.0, 10.5, 10.5], closes=[10.0, 10.5, 10.5],
                  limit_up=[10.5, 10.5, 10.5])
    engine = BacktestEngine(initial_capital=100_000,
                            market_rules=MarketRules(enabled=False))
    trades = engine.run(ds, BuyAt(0))
    assert not trades.empty, "关闭规则后应能成交（证明拦截确实来自规则）"
    print("[OK] 关闭制度约束后涨停可成交（对照确认拦截来源）")


# ============================================================
# 停牌
# ============================================================
def test_suspended_blocks_trading():
    ds = _dataset(opens=[10.0, 10.0, 10.0], closes=[10.0, 10.0, 10.0],
                  suspended=[0, 1, 0])
    engine = BacktestEngine(initial_capital=100_000)
    trades = engine.run(ds, BuyAt(0))
    assert trades.empty, "停牌日不应成交"
    assert any("停牌" in k for k in engine.rejections.summary())
    print("[OK] 停牌不可交易")


# ============================================================
# 一手取整
# ============================================================
def test_lot_size_rounding():
    """A股必须整手（100 股）买入"""
    ds = make_dataset(n=30, seed=4)
    engine = BacktestEngine(initial_capital=123_456)   # 故意用不能整除的资金
    trades = engine.run(ds, BuyAt(0))
    assert not trades.empty
    size = float(trades["size"].iloc[0])
    assert size % 100 == 0, f"成交股数 {size} 不是 100 的整数倍"
    print(f"[OK] 整手买入: {size:,.0f} 股（100 的整数倍）")


def test_broker_max_affordable_is_lot_multiple():
    b = SimulatedBroker(slippage=0.0, commission=0.0001, min_commission=5.0,
                        stamp_duty=0.0005, transfer_fee=0.00001, lot_size=100)
    size = b.max_affordable_size(100_000, 33.33)
    assert size % 100 == 0
    assert b.estimate_buy_cost(size, 33.33) <= 100_000
    print(f"[OK] max_affordable_size = {size:,.0f} 股，且含费用后不超现金")


# ============================================================
# 印花税 / 过户费
# ============================================================
def test_stamp_duty_only_on_sell():
    b = SimulatedBroker(commission=0.0001, min_commission=5.0,
                        stamp_duty=0.0005, transfer_fee=0.00001, lot_size=100)
    buy = Order(symbol="T", side=OrderSide.BUY, size=1000, price=10.0)
    b.execute(buy, 10.0)
    sell = Order(symbol="T", side=OrderSide.SELL, size=1000, price=10.0)
    b.execute(sell, 10.0)

    assert buy.stamp_duty == 0, "买入不收印花税"
    assert sell.stamp_duty > 0, "卖出应收印花税"
    assert buy.transfer_fee > 0 and sell.transfer_fee > 0, "过户费双边"
    assert sell.total_fee > buy.total_fee, "卖出费用应高于买入（多了印花税）"
    print(f"[OK] 买入费用 {buy.total_fee:.2f}（无印花税） < "
          f"卖出费用 {sell.total_fee:.2f}（含印花税 {sell.stamp_duty:.2f}）")


# ============================================================
# 成交时点
# ============================================================
def test_next_open_uses_next_bar_open():
    """next_open：第 0 根 bar 决策，应按第 1 根 bar 的开盘价成交"""
    ds = _dataset(opens=[10.0, 20.0, 20.0], closes=[10.0, 20.0, 20.0])
    engine = BacktestEngine(initial_capital=100_000, slippage=0.0,
                            fill_timing=FILL_NEXT_OPEN)
    trades = engine.run(ds, BuyAt(0))
    assert not trades.empty
    px = float(trades["price"].iloc[0])
    assert abs(px - 20.0) < 1e-9, f"应按次日开盘 20.0 成交，实际 {px}"
    print(f"[OK] next_open 成交价 {px:.2f} == 次根 bar 开盘价（非决策日收盘 10.0）")


def test_same_close_uses_decision_bar_close():
    """same_close（旧行为）：按决策那根 bar 的收盘价成交 —— 存在未来函数"""
    ds = _dataset(opens=[10.0, 20.0, 20.0], closes=[10.0, 20.0, 20.0])
    engine = BacktestEngine(initial_capital=100_000, slippage=0.0,
                            fill_timing=FILL_SAME_CLOSE)
    trades = engine.run(ds, BuyAt(0))
    px = float(trades["price"].iloc[0])
    assert abs(px - 10.0) < 1e-9, f"应于决策日收盘 10.0 成交，实际 {px}"
    print(f"[OK] same_close 成交价 {px:.2f} == 决策日收盘价（旧行为，含未来函数）")


# ============================================================
# 数据截断（封堵策略偷看未来）
# ============================================================
class _PeekFuture(Strategy):
    """在 idx==5 时试图读取"下一根 bar"，用来看未来数据是否可达"""

    def __init__(self):
        super().__init__("PeekFuture")

    def on_bar(self, data, idx):
        if idx == 5:
            _ = data["close"].iloc[idx + 1]
        return Signal(Action.HOLD)


def test_truncate_blocks_future_access():
    ds = make_dataset(n=30, seed=1)
    engine = BacktestEngine(initial_capital=100_000, truncate_data=True)
    try:
        engine.run(ds, _PeekFuture())
        raise AssertionError("截断后策略不应能读到未来行")
    except IndexError:
        pass
    print("[OK] 截断模式下策略读未来行 → IndexError（通道被封堵）")


def test_untruncated_allows_future_access():
    """对照组：不截断时同样代码能读到未来行（证明截断确实起作用）"""
    ds = make_dataset(n=30, seed=1)
    engine = BacktestEngine(initial_capital=100_000, truncate_data=False)
    engine.run(ds, _PeekFuture())        # 不应抛异常
    print("[OK] 不截断模式下同样的策略能读到未来行（对照确认）")


def test_limit_price_space_alignment():
    """涨跌停价必须与回测价格处于同一复权空间

    cleaned/limit_price 由**不复权**清洗层派生，而回测默认用前复权价。
    若直接混比，有分红送转的股票会被误判（实测 002644 在 2024-03-12
    前复权开盘 8.2574 会被当成"超过不复权涨停 8.25"）。
    """
    from database.loader import load_trading_status, load_factor

    code, s, e = "002644", "2024-03-01", "2024-12-31"
    raw = load_trading_status(code, s, e, adjust="")
    qfq = load_trading_status(code, s, e, adjust="qfq")
    # ⚠️ 守卫不能只判 `.empty`：无数据时 loader 可能返回**没有 trade_date 列**
    # 的对象，`.empty` 未必为 True，随后 merge 直接 KeyError。
    # 这是"读真实数据的测试"，数据不在时必须**干净跳过**（CI 里没有 db/）。
    if (raw is None or qfq is None or raw.empty or qfq.empty
            or "trade_date" not in raw.columns
            or "trade_date" not in qfq.columns):
        print("[SKIP] 无涨跌停数据（或列不齐），跳过复权空间一致性检查")
        return

    fac = load_factor(code)
    # 复权因子也要有守卫：无数据时它同样可能缺 trade_date 列
    if (fac is None or fac.empty or "trade_date" not in getattr(fac, "columns", [])
            or "factor" not in getattr(fac, "columns", [])):
        print("[SKIP] 无复权因子数据，跳过复权空间一致性检查")
        return
    m = raw.merge(qfq, on="trade_date", suffixes=("_raw", "_qfq")).sort_values("trade_date")
    m = pd.merge_asof(m, fac.sort_values("trade_date"), on="trade_date",
                      direction="backward")
    m["factor"] = m["factor"].fillna(1.0)
    expect = (m["limit_up_raw"] * m["factor"]).round(4)
    diff = float((m["limit_up_qfq"] - expect).abs().max())

    assert diff < 1e-6, f"涨跌停价未换算到前复权空间，最大偏差 {diff}"
    assert (m["limit_up_raw"] != m["limit_up_qfq"]).any(), \
        "该股有分红，两个空间应当不同（否则本测试没有区分力）"
    print(f"[OK] 涨跌停价已按同一因子换算到前复权空间（最大偏差 {diff:.2e}）")


if __name__ == "__main__":
    test_t_plus_1_at_portfolio_level()
    test_no_same_day_round_trip_in_trades()
    test_limit_up_blocks_buy()
    test_limit_down_blocks_sell()
    test_rules_disabled_allows_trade()
    test_suspended_blocks_trading()
    test_lot_size_rounding()
    test_broker_max_affordable_is_lot_multiple()
    test_stamp_duty_only_on_sell()
    test_next_open_uses_next_bar_open()
    test_same_close_uses_decision_bar_close()
    test_truncate_blocks_future_access()
    test_untruncated_allows_future_access()
    test_limit_price_space_alignment()
    print("\n全部制度约束测试通过")
