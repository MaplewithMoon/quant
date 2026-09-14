# -*- coding: utf-8 -*-
"""回测试用案例合集：展示 scripts/backtest.py 和编程式调用的各种用法

运行:
    python scripts/backtest_demo.py
"""
import sys, io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
sys.path.insert(0, ".")

import json
import pandas as pd


def demo1_single_strategy():
    """案例1: 单策略回测（命令行）"""
    print("=" * 60)
    print("案例1: 单策略回测 - 宁德时代 双均线(5,20)")
    print("=" * 60)
    from scripts.backtest import load_dataset, make_strategy, run_backtest, fmt_metrics

    ds = load_dataset("300750", "2024-01-01", "2024-12-31", adjust="qfq")
    strat = make_strategy("MA", {"fast": 5, "slow": 20})
    _, trades, equity, m = run_backtest(ds, strat, 1_000_000, 0.0001, 0.001, risk=False)
    print(f"策略: {strat.name}, 交易 {len(trades)} 笔")
    print(fmt_metrics(m))
    print()


def demo2_custom_params():
    """案例2: 自定义策略参数"""
    print("=" * 60)
    print("案例2: 自定义参数 - 海龟策略(入场20/出场10)")
    print("=" * 60)
    from scripts.backtest import load_dataset, make_strategy, run_backtest, fmt_metrics

    ds = load_dataset("300750", "2024-01-01", "2024-12-31")
    strat = make_strategy("Turtle", {"entry_window": 20, "exit_window": 10})
    _, trades, equity, m = run_backtest(ds, strat, 1_000_000, 0.0001, 0.001, risk=False)
    print(f"策略: {strat.name}")
    print(fmt_metrics(m))
    print()


def demo3_with_risk():
    """案例3: 开启风控"""
    print("=" * 60)
    print("案例3: 开启风控 - 最大回撤15% + 单笔仓位30%")
    print("=" * 60)
    from scripts.backtest import load_dataset, make_strategy, run_backtest, fmt_metrics

    ds = load_dataset("300750", "2024-01-01", "2024-12-31")
    strat = make_strategy("MA", {"fast": 5, "slow": 20})
    _, trades, equity, m = run_backtest(ds, strat, 1_000_000, 0.0001, 0.001, risk=True)
    print(fmt_metrics(m))
    print("(对比: 无风控时 MA_5_20 回撤 -24.42%，开风控后回撤受限)")
    print()


def demo4_compare():
    """案例4: 多策略对比"""
    print("=" * 60)
    print("案例4: 多策略对比（按夏普排序）")
    print("=" * 60)
    from scripts.backtest import load_dataset, make_strategy, run_backtest

    ds = load_dataset("300750", "2024-01-01", "2024-12-31")
    rows = []
    for name in ["MA", "Turtle", "MeanRev", "BBand", "MACD", "Pullback", "AMA", "RSIDiv"]:
        try:
            strat = make_strategy(name, None)
            _, _, _, m = run_backtest(ds, strat, 1_000_000, 0.0001, 0.001, risk=False)
            rows.append({"策略": name, "收益%": round(m.total_return * 100, 2),
                         "夏普": round(m.sharpe_ratio, 2),
                         "回撤%": round(m.max_drawdown * 100, 2)})
        except Exception as e:
            print(f"  {name} 失败: {e}")
    df = pd.DataFrame(rows).sort_values("夏普", ascending=False)
    print(df.to_string(index=False))
    print()


def demo5_programmatic():
    """案例5: 编程式调用 - 直接在代码里用"""
    print("=" * 60)
    print("案例5: 编程式调用 - 自定义策略类")
    print("=" * 60)
    from strategy.base import Strategy, Signal, Action
    from data.dataset import DataSet
    from data.preprocessor import Preprocessor, fillna, add_technical_indicators
    from backtest.engine import BacktestEngine

    class MyStrategy(Strategy):
        """自定义策略: 收盘>开盘且涨超1%买入, 跌破5日均线卖出"""
        def __init__(self):
            super().__init__("MyCustom")
            self._in_pos = False

        def on_bar(self, data, idx):
            if idx < 6:
                return Signal(Action.HOLD)
            close = data["close"].iloc[idx]
            ma5 = data["close"].iloc[idx - 5:idx].mean()
            if not self._in_pos:
                if close > data["open"].iloc[idx] and data["pct_chg"].iloc[idx] > 1:
                    self._in_pos = True
                    return Signal(Action.BUY, reason="动量突破")
            else:
                if close < ma5:
                    self._in_pos = False
                    return Signal(Action.SELL, reason="跌破均线")
            return Signal(Action.HOLD)

    ds = DataSet.from_db("000001", "2024-01-01", "2024-12-31", adjust="qfq")
    ds = Preprocessor().add(fillna()).add(add_technical_indicators).run(ds)

    engine = BacktestEngine(initial_capital=1_000_000)
    trades = engine.run(ds, MyStrategy())
    print(f"自定义策略: {len(trades)} 笔交易, 最后净值 {engine.portfolio.total_value:,.0f}")
    print()


def demo6_json():
    """案例6: JSON 输出（供其他程序/Web使用）"""
    print("=" * 60)
    print("案例6: JSON 输出")
    print("=" * 60)
    from scripts.backtest import load_dataset, make_strategy, run_backtest

    ds = load_dataset("000001", "2024-01-01", "2024-12-31")
    strat = make_strategy("BBand", None)
    _, trades, equity, m = run_backtest(ds, strat, 1_000_000, 0.0001, 0.001, risk=False)
    print(json.dumps(m.to_dict(), ensure_ascii=False, indent=2))
    print()


if __name__ == "__main__":
    demo1_single_strategy()
    demo2_custom_params()
    demo3_with_risk()
    demo4_compare()
    demo5_programmatic()
    demo6_json()
