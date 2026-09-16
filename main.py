"""量化系统入口（演示）

main.py 是历史入口，回测功能已统一到 scripts/backtest.py（CLI + 编程式）。
本文件保持兼容：直接委托 scripts/backtest 的公共函数。

用法:
    python main.py               # 运行多策略对比（等效 scripts/backtest.py --compare）
    python main.py --symbol 600519 --strategy Turtle   # 单策略（等效 scripts/backtest.py）
"""
import sys
import io
sys.path.insert(0, ".")
if __name__ == "__main__":
    # 只在直接运行时切编码：模块顶层替换 sys.stdout 是有副作用的 import，
    # 会破坏 pytest 的输出捕获（详见 utils/console.py）
    from utils.console import force_utf8_stdout
    force_utf8_stdout()

import argparse
import scripts.backtest as bt


def run_single_backtest():
    """单策略回测（委托 scripts/backtest.py）"""
    bt.main()


def run_comparison():
    """多策略对比（委托 scripts/backtest.py 的对比逻辑）"""
    ds = bt.load_dataset("300750", "2024-01-01", "2024-12-31")
    from scripts.backtest import make_strategy, run_backtest
    rows = []
    for name in bt.STRATEGIES:
        try:
            strat = make_strategy(name, None)
            _, _, _, m = run_backtest(ds, strat, 1_000_000, 0.0001, 0.001, risk=False)
            rows.append({"策略": name, "收益": m.total_return, "夏普": m.sharpe_ratio,
                         "卡尔玛": m.calmar_ratio, "回撤": m.max_drawdown,
                         "交易": m.total_trades, "胜率": m.win_rate})
        except Exception as e:
            print(f"  {name} 失败: {e}")
    import pandas as pd
    df = pd.DataFrame(rows).sort_values("夏普", ascending=False)
    print(df.to_string(index=False))
    return rows


if __name__ == "__main__":
    # 支持命令行参数（--compare 走对比，否则单策略）
    sys.argv = [a for a in sys.argv if a not in ("--compare",)]
    if len(sys.argv) > 1:
        bt.main()
    else:
        run_comparison()
