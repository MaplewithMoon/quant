# -*- coding: utf-8 -*-
"""完整回测入口：从数据库取数，支持任意策略/参数/风控/可视化

【整体流程】
    数据库清洗层 (db/cleaned/daily_basic) + 复权因子 (frozen/adjust)
        ↓  DataSet.from_db(adjust='qfq')  现场算前复权
    加载原始OHLCV
        ↓  Preprocessor 预处理
    技术指标 (SMA/EMA/MACD/RSI/布林带/ATR)
        ↓  策略实例化 + 参数注入
    策略.on_bar() 逐根K线产生信号
        ↓  回测引擎
    下单 → 撮合(滑点+佣金) → 持仓更新 → 交易记录
        ↓  Metrics.compute
    绩效指标 (14项)

【用法】
    # 单策略回测（数据库数据，默认前复权）
    python scripts/backtest.py --symbol 300750 --strategy MA --params '{"fast":5,"slow":20}'

    # 指定日期/资金/手续费
    python scripts/backtest.py --symbol 600519 --strategy Turtle \
        --start 2020-01-01 --end 2024-12-31 --capital 1000000

    # 多策略对比
    python scripts/backtest.py --symbol 300750 --compare

    # 开启风控 + 可视化
    python scripts/backtest.py --symbol 300750 --strategy MA --risk --plot

    # 输出 JSON（供其他程序/Web 消费）
    python scripts/backtest.py --symbol 000001 --strategy BBand --json
"""
import sys
import io
import json
sys.path.insert(0, ".")

import argparse
import pandas as pd
from data.dataset import DataSet
from data.preprocessor import Preprocessor, fillna, add_technical_indicators
from backtest.engine import BacktestEngine, FILL_NEXT_OPEN, FILL_SAME_CLOSE
from backtest.metrics import Metrics
from execution.market_rules import MarketRules
from execution.impact import build_model
from risk.manager import RiskManager, MaxDrawdownRule, MaxPositionSizeRule
from analytics.visualizer import Visualizer
from utils.logger import setup_logger


# ============================================================
# 策略注册表
# ============================================================
# 把策略名映射到 "模块路径.类名"，配合 DEFAULT_PARAMS 里的默认参数，
# 让用户通过 --strategy 名字就能用任意策略，无需改代码。
# 新增策略时，在这里加一行即可。
STRATEGIES = {
    "MA":          "strategy.examples.MovingAverageCross",      # 双均线交叉
    "MAVol":       "strategy.classic.DualMovingAverageCross",   # 量过滤双均线
    "MeanRev":     "strategy.examples.MeanReversion",           # 均值回归
    "BBand":       "strategy.classic.BollingerBandStrategy",    # 布林带
    "MACD":        "strategy.classic.MACDStrategy",             # MACD金叉死叉
    "Turtle":      "strategy.classic.TurtleStrategy",           # 海龟交易法
    "DualThrust":  "strategy.modern.DualThrust",                # 通道突破
    "RSIDiv":      "strategy.modern.RSIDivergence",             # RSI背离
    "Pullback":    "strategy.modern.PullbackStrategy",          # 突破回踩
    "AMA":         "strategy.modern.AdaptiveAMA",               # 自适应均线
    "Grid":        "strategy.modern.GridStrategy",              # 网格交易
}

# 各策略默认参数（用户不传 --params 时使用）
DEFAULT_PARAMS = {
    "MA":         {"fast": 5, "slow": 20},
    "MAVol":      {"fast": 10, "slow": 30, "vol_factor": 1.2},
    "MeanRev":    {"window": 20, "entry_std": 2.0},
    "BBand":      {"window": 20, "num_std": 2.0},
    "MACD":       {"signal_period": 9},
    "Turtle":     {"entry_window": 20, "exit_window": 10},
    "DualThrust": {"k1": 0.5, "k2": 0.5},
    "RSIDiv":     {"period": 14, "oversold": 30, "overbought": 70},
    "Pullback":   {"lookback": 20, "ma_period": 10},
    "AMA":        {"fast": 2, "slow": 30, "lookback": 10},
    "Grid":       {"grids": 10},
}


def load_dataset(symbol: str, start: str, end: str, adjust: str = "qfq",
                 with_status: bool = True):
    """从数据库加载数据并预处理

    参数:
        symbol:      股票代码（如 '300750'）
        start:       起始日期 'YYYY-MM-DD'
        end:         结束日期 'YYYY-MM-DD'
        adjust:      'qfq'=前复权(默认)，''=不复权原始价
        with_status: 是否挂载涨跌停价/停牌标记（供回测的制度约束使用）

    流程:
        1. DataSet.from_db → 从清洗层读原始价 + frozen 复权因子现场算前复权
        2. load_trading_status → 对齐 cleaned/limit_price 与 frozen/suspend
        3. Preprocessor → 填充缺失 + 添加技术指标(SMA/EMA/MACD/RSI/布林带/ATR)

    返回:
        预处理后的 DataSet（data 含 OHLCV + 指标 + limit_up/limit_down/suspended）
    """
    ds = DataSet.from_db(symbol, start, end, adjust=adjust)
    if with_status:
        try:
            from database.loader import load_trading_status
            # 涨跌停价必须换算到与价格相同的复权空间，否则会误判
            ds.attach_market_status(load_trading_status(symbol, start, end, adjust=adjust))
        except Exception as e:
            print(f"[警告] 交易日状态加载失败，涨跌停/停牌约束将不生效: {e}")
    pp = Preprocessor().add(fillna()).add(add_technical_indicators)
    return pp.run(ds)


def make_strategy(name: str, params: dict):
    """按注册表实例化策略

    参数:
        name:   策略名（STRATEGIES 的 key，如 'MA'、'Turtle'）
        params: 用户传入的参数 dict（None 则用默认参数）

    实现:
        根据注册表找到 "模块.类名" → 动态 import → 合并默认+用户参数 → 实例化

    示例:
        make_strategy('MA', {'fast': 10, 'slow': 30})
        → MovingAverageCross(fast=10, slow=30)
    """
    if name not in STRATEGIES:
        raise ValueError(f"未知策略 {name}，可用: {list(STRATEGIES)}")
    mod_path, cls_name = STRATEGIES[name].rsplit(".", 1)
    import importlib
    mod = importlib.import_module(mod_path)
    cls = getattr(mod, cls_name)
    p = {**DEFAULT_PARAMS.get(name, {}), **(params or {})}  # 默认参数被用户参数覆盖
    return cls(**p)


def run_backtest(ds, strategy, capital, commission, slippage, risk,
                 min_commission=5.0, verbose=False,
                 fill_timing=FILL_NEXT_OPEN, apply_rules=True,
                 truncate_data=True, slippage_model=None,
                 max_participation=0.10):
    """执行一次完整回测

    参数:
        ds:             预处理后的 DataSet
        strategy:       策略实例
        capital:        初始资金
        commission:     佣金费率（默认万分之一 = 0.0001）
        slippage:       固定滑点（默认0.1%）；传入 slippage_model 时忽略
        risk:           True=启用风控（回撤15%限 + 单笔仓位30%限）
        min_commission: 单笔最低佣金（默认5元）
        verbose:        是否打印中间日志
        fill_timing:    成交时点，"next_open"(默认) / "same_close"
        apply_rules:    True=启用 A股制度约束（T+1/涨跌停/停牌/一手取整）
        truncate_data:  True=喂给策略的数据只到当前 bar（封堵未来函数）
        slippage_model: 成交价模型（execution.impact），None=固定滑点
        max_participation: 单笔最多吃掉当日成交量的比例（默认 10%）

    返回:
        (engine, trades, equity, metrics)
        engine:  回测引擎（含 portfolio/broker/rejections 状态）
        trades:  交易记录 DataFrame
        equity:  逐日净值曲线 Series（引擎唯一账本）
        metrics: Metrics 绩效对象
    """
    engine = BacktestEngine(initial_capital=capital, commission=commission,
                            slippage=slippage, min_commission=min_commission,
                            fill_timing=fill_timing,
                            market_rules=MarketRules(enabled=apply_rules),
                            truncate_data=truncate_data,
                            slippage_model=slippage_model,
                            max_participation=max_participation)
    if risk:
        # 风控规则链：最大回撤超15%禁止开仓 + 单笔仓位不超过总资产30%
        engine.risk_manager = RiskManager([
            MaxDrawdownRule(max_drawdown_pct=0.15),
            MaxPositionSizeRule(max_position_pct=0.3),
        ])
    trades = engine.run(ds, strategy)
    # 净值曲线由引擎逐 bar 产生（唯一账本）；不再用成交记录重推一遍
    equity = engine.equity_curve
    metrics = Metrics.compute(equity, trades)
    return engine, trades, equity, metrics


def auto_lookahead_report(ds, trades) -> str:
    """回测结束后自动生成未来因子（look-ahead）检测报告

    检查要点:
        1. 交易信号/成交日期是否晚于数据可获得日期
        2. 使用的收盘价是否与决策日匹配（无未来价格引用）
        3. 给出 PIT/延迟敏感性等改进建议（见 lookahead.py 文档）

    返回: 报告文本（无问题时简要说明通过）
    """
    from backtest.lookahead import attach_availability, verify_point_in_time

    lines = ["=" * 60, "未来因子检测报告（look-ahead bias）", "=" * 60]

    # 1) 成交记录可用性：每笔交易应发生在数据可获得日当天或之后
    if not trades.empty:
        t = trades.copy()
        # 交易日期可能是列(trade_date)或索引(timestamp)
        if "trade_date" not in t.columns:
            t = t.reset_index()
            date_col = "trade_date" if "trade_date" in t.columns else t.columns[0]
        else:
            date_col = "trade_date"
        t[date_col] = pd.to_datetime(t[date_col])
        violations = 0
        out_of_range = t[(t[date_col] < ds.data.index.min())
                         | (t[date_col] > ds.data.index.max())]
        violations = len(out_of_range)
        lines.append(f"  成交记录: {len(t)} 笔，日期均在数据区间内"
                     if violations == 0 else f"  成交记录: {violations} 笔日期超出区间(疑似索引错位)")
    else:
        lines.append("  成交记录: 无交易")

    # 2) 数据时间轴一致性：回测用价格应全部落在数据可覆盖区间
    data_min, data_max = ds.data.index.min(), ds.data.index.max()
    lines.append(f"  数据区间: {data_min.date()} ~ {data_max.date()}")

    # 3) 结论 + 建议
    lines.append("-" * 60)
    lines.append("建议（如需严格防未来因子）:")
    lines.append("  1. 财务因子务必用公告日 ann_date 而非报告期 end_date")
    lines.append("  2. 可做延迟敏感性分析：把因子延后1/3/5天重跑，收益变化大则存泄漏")
    lines.append("  3. 选股池用当时在市的股票，避免幸存者偏差")
    lines.append("  4. Walk-Forward 样本外验证防过拟合")
    lines.append("=" * 60)
    return "\n".join(lines)


def fmt_metrics(m) -> str:
    """把 Metrics 格式化为对齐的中文文本"""
    return "\n".join([
        f"  总收益率:          {m.total_return:>8.2%}",
        f"  年化收益率:        {m.annual_return:>8.2%}",
        f"  年化波动率:        {m.annual_volatility:>8.2%}",
        f"  夏普比率:          {m.sharpe_ratio:>8.2f}",
        f"  卡尔玛比率:        {m.calmar_ratio:>8.2f}",
        f"  最大回撤:          {m.max_drawdown:>8.2%}",
        f"  平均回撤:          {m.avg_drawdown:>8.2%}",
        f"  95%回撤不超过:     {m.dd_percentile_95:>8.2%}",
        f"  最长回撤天数:      {m.max_drawdown_duration:>4d} 天",
        f"  恢复用时:          {m.recovery_time:>4d} 天",
        f"  胜率:              {m.win_rate:>8.2%}",
        f"  交易次数:          {m.total_trades:>8d}",
        f"  盈亏比:            {m.profit_factor:>8.2f}",
    ])


def main():
    """命令行入口：解析参数 → 取数 → 回测 → 输出结果"""
    parser = argparse.ArgumentParser(description="量化回测（数据库数据）")
    # 行情参数
    parser.add_argument("--symbol", default="300750", help="股票代码")
    parser.add_argument("--start", default="2024-01-01", help="起始日期 YYYY-MM-DD")
    parser.add_argument("--end", default="2024-12-31", help="结束日期 YYYY-MM-DD")
    parser.add_argument("--adjust", default="qfq", choices=["qfq", ""],
                        help="价格复权方式: qfq=前复权(默认), 空=不复权")
    # 策略参数
    parser.add_argument("--strategy", default="MA", choices=list(STRATEGIES), help="策略名")
    parser.add_argument("--params", default="", help='策略参数 JSON，如 \'{"fast":10,"slow":30}\'')
    # 资金/成本参数
    parser.add_argument("--capital", type=float, default=1_000_000, help="初始资金")
    parser.add_argument("--commission", type=float, default=0.0001, help="佣金费率(默认万分之一)")
    parser.add_argument("--min-commission", type=float, default=5.0, help="单笔最低佣金(默认5元)")
    parser.add_argument("--stamp-duty", type=float, default=0.0005,
                        help="印花税率，仅卖出单边(默认0.05%%)")
    parser.add_argument("--transfer-fee", type=float, default=0.00001,
                        help="过户费率，双边(默认0.001%%)")
    parser.add_argument("--slippage", type=float, default=0.001, help="滑点")
    # 撮合/制度参数
    parser.add_argument("--fill-timing", default=FILL_NEXT_OPEN,
                        choices=[FILL_NEXT_OPEN, FILL_SAME_CLOSE],
                        help="成交时点: next_open=次日开盘成交(默认,无未来函数), "
                             "same_close=当日收盘成交(旧行为)")
    parser.add_argument("--no-rules", action="store_true",
                        help="关闭 A股制度约束(T+1/涨跌停/停牌/一手取整)，用于对照")
    parser.add_argument("--no-truncate", action="store_true",
                        help="不截断喂给策略的数据（允许策略看到未来行，仅用于调试）")
    # 冲击成本 / 流动性
    parser.add_argument("--impact-model", default="fixed", choices=["fixed", "sqrt", "none"],
                        help="成交价模型: fixed=固定滑点(默认), sqrt=平方根市场冲击, none=零滑点")
    parser.add_argument("--impact-k", type=float, default=0.1,
                        help="平方根冲击系数 k（默认0.1；下单量占成交量1%%时冲击约1%%）")
    parser.add_argument("--max-participation", type=float, default=0.10,
                        help="单笔最多吃掉当日成交量的比例(默认10%%，0=不限)")
    # 行为参数
    parser.add_argument("--risk", action="store_true", help="开启风控")
    parser.add_argument("--plot", action="store_true", help="显示净值曲线")
    parser.add_argument("--compare", action="store_true", help="多策略对比")
    parser.add_argument("--json", action="store_true", help="输出JSON")
    args = parser.parse_args()

    logger = setup_logger("backtest")
    params = json.loads(args.params) if args.params else {}
    bt_kwargs = dict(
        fill_timing=args.fill_timing,
        apply_rules=not args.no_rules,
        truncate_data=not args.no_truncate,
        slippage_model=build_model(args.impact_model, rate=args.slippage,
                                   k=args.impact_k),
        max_participation=args.max_participation,
    )

    # ============ 模式一：多策略对比 ============
    if args.compare:
        ds = load_dataset(args.symbol, args.start, args.end, args.adjust)
        logger.info(f"加载 {args.symbol} {len(ds)} 条 (数据库)")
        rows = []
        # 遍历注册表里所有策略，逐个回测，按夏普排序
        for name in STRATEGIES:
            try:
                strat = make_strategy(name, None)
                _, trades, equity, m = run_backtest(
                    ds, strat, args.capital, args.commission, args.slippage,
                    args.risk, args.min_commission, **bt_kwargs)
                rows.append({"策略": name, "收益": m.total_return, "夏普": m.sharpe_ratio,
                             "卡尔玛": m.calmar_ratio, "回撤": m.max_drawdown,
                             "交易": m.total_trades, "胜率": m.win_rate})
            except Exception as e:
                logger.warning(f"{name} 失败: {e}")  # 单个策略失败不影响其他
        df = pd.DataFrame(rows).sort_values("夏普", ascending=False)
        print(df.to_string(index=False))
        if args.json:
            print(json.dumps(df.to_dict("records"), ensure_ascii=False))
        return

    # ============ 模式二：单策略回测 ============
    ds = load_dataset(args.symbol, args.start, args.end, args.adjust)
    logger.info(f"加载 {args.symbol} {len(ds)} 条 (数据库, adjust={args.adjust or '原始价'})")

    strategy = make_strategy(args.strategy, params)
    engine, trades, equity, metrics = run_backtest(
        ds, strategy, args.capital, args.commission, args.slippage,
        args.risk, args.min_commission, **bt_kwargs)

    # 制度约束拦截汇总（把"为什么没成交"讲清楚）
    if engine.rejections.items:
        print("\n[制度约束] 被拦下的委托:")
        for reason, n in sorted(engine.rejections.summary().items(),
                                key=lambda kv: -kv[1]):
            print(f"    {n:>4} 次  {reason}")
    if engine.pending_cancelled:
        print("\n[制度约束] 最后 1 根 bar 的信号无下一根 bar 可成交，已作废")

    logger.info(f"策略: {strategy.name}  交易次数: {len(trades)}")
    logger.info(f"回测结果:\n{fmt_metrics(metrics)}")

    # ============ 自动执行未来因子检测并输出报告 ============
    report = auto_lookahead_report(ds, trades)
    if report:
        print("\n" + report)
        # 生成报告文件
        with open("lookahead_report.txt", "w", encoding="utf-8") as f:
            f.write(report)
        print("未来因子报告已保存: lookahead_report.txt")

    if args.json:
        # 机器可读输出（供其他程序/Web 使用）
        print(json.dumps(metrics.to_dict(), ensure_ascii=False))

    if args.plot:
        # 交互式可视化：净值曲线 + 回撤 + 买卖点
        viz = Visualizer()
        viz.plot_equity_curve(equity, trades, f"{strategy.name} - {args.symbol}")


if __name__ == "__main__":
    # 仅在作为脚本运行时包装 stdout（解决中文在 Windows 控制台的编码问题；
    # 被 import 时不执行，避免破坏调用方的 stdout）
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")
    main()
