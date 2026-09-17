# -*- coding: utf-8 -*-
"""板块动量轮动策略：完整回测 + 训练/测试集分离 + 绩效报告

【一条命令跑完】
    python scripts/sector_rotation_backtest.py \
        --start 2017-01-01 --split 2022-01-01 --end 2025-12-31 \
        --optimize --objective sharpe

【做了什么】
    1. 加载全市场价量面板 + 交易状态（涨跌停/停牌）
    2. 构造逐日股票池（剔除 ST/停牌/次新/流动性不足）
    3. **只在训练集上**做参数网格搜索（默认搜 动量窗口 / 板块数 / 每板块持股数）
    4. 用最优参数在**测试集**上跑一次（样本外，只跑一次），报出过拟合落差
    5. 全区间跑一遍出完整报告：夏普/回撤/Alpha/Beta/信息比率/跟踪误差/
       上行下行捕获/分年度分月收益/Brinson 归因/板块轮动轨迹
    6. 画图：累计净值 vs 沪深300、超额净值、回撤、分年度收益、滚动夏普

【无未来函数】
    - 板块动量只用 ≤ t 的数据；次级因子同样
    - 引擎 next_open：t 日收盘算权重 → t+1 开盘成交
    - 训练集/测试集严格按日期切分，测试集不参与任何参数选择
"""
import argparse
import io
import json
import os
import sys
import time

# 逐行刷新：不能只写 `io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")`，
# 那样会新建一个**块缓冲**包装器，把 `python -u` 的无缓冲设置覆盖掉 ——
# 长任务重定向到日志文件时，要等 8KB 缓冲满了才落盘，看不到进度。
sys.path.insert(0, ".")
if __name__ == "__main__":
    # 只在直接运行时切编码：模块顶层替换 sys.stdout 是有副作用的 import，
    # 会破坏 pytest 的输出捕获（详见 utils/console.py）
    from utils.console import force_utf8_stdout
    force_utf8_stdout()

import numpy as np
import pandas as pd

from analytics.performance import (annual_returns_table, drawdown_info,
                                   monthly_returns_table, performance_summary,
                                   to_returns)
from analytics.report import (fmt_value, format_returns_table, metrics_table,
                              plot_performance, plot_rolling, save_csv,
                              setup_font)
from analytics.attribution import (brinson, industry_exposure,
                                   load_industry_map)
from backtest.panel_data import load_price_panel
from backtest.metrics import Metrics
from database.loader import load_index_close
from optimizer.portfolio_search import (evaluate_portfolio, grid_search_portfolio,
                                        make_split, params_of, slice_benchmark,
                                        slice_mask, slice_panel)
from optimizer.search import ParamSpace
from portfolio.construction import equal_weight_returns
from strategy.sector_rotation import (SectorRotationSpec, build_rotation_weights,
                                      load_sector_map, plan_summary,
                                      sector_holding_history)
from universe import UniverseSpec, build_universe, universe_size

BENCH_CODE = "000300.SH"
BENCH_NAME = "沪深300"


# ============================================================
# 工具
# ============================================================
def banner(text: str, ch: str = "=", width: int = 92):
    print()
    print(ch * width)
    print(text)
    print(ch * width)


def truncate_result(res, eval_start: str):
    """把回测结果截断到评估窗口，并**重算**指标

    为什么需要：测试集回测要从 warmup 起点开始跑，策略在 split 那天才有
    正常的持仓状态（而不是从现金冷启动）。但绩效只能从 split 开始算，
    否则 warmup 段会把指标稀释掉。
    """
    ts = pd.Timestamp(eval_start)
    eq = res.equity[res.equity.index >= ts]
    tr = res.trades
    if tr is not None and not tr.empty and "timestamp" in tr.columns:
        tr = tr[pd.to_datetime(tr["timestamp"]) >= ts]
    if eq.empty:
        return eq, tr, Metrics()
    return eq, tr, Metrics.compute(eq, tr)


def run_segment(panel: dict, mask: pd.DataFrame, factory, params: dict,
                warmup_start: str, eval_start: str, eval_end: str,
                engine_kwargs: dict) -> dict:
    """跑 [warmup_start, eval_end]，但只评估 [eval_start, eval_end]"""
    seg = slice_panel(panel, warmup_start, eval_end)
    seg_mask = slice_mask(mask, warmup_start, eval_end)
    out = evaluate_portfolio(seg, seg_mask, factory, params, objective="sharpe",
                             engine_kwargs=engine_kwargs)
    if out["result"] is None:
        raise RuntimeError(f"回测失败: {out['_error']}")
    eq, tr, m = truncate_result(out["result"], eval_start)
    out.update({"equity": eq, "trades": tr, "metrics": m,
                "result": out["result"]})
    return out


# ============================================================
# main
# ============================================================
def main():
    ap = argparse.ArgumentParser(description="板块动量轮动策略回测 + 样本外验证")
    # 区间
    ap.add_argument("--start", default="2017-01-01", help="训练集开始")
    ap.add_argument("--split", default="2022-01-01", help="训练/测试分界")
    ap.add_argument("--end", default="2025-12-31", help="测试集结束")
    ap.add_argument("--warmup-days", type=int, default=420, help="预热交易日（算动量用）")
    ap.add_argument("--benchmark", default=BENCH_CODE)
    # 资金与费用
    ap.add_argument("--capital", type=float, default=1_000_000)
    # 执行参数（成交时点 / 滑点 / 冲击模型 / 全部费用）由公共层统一提供 ——
    # 原先各脚本各写一遍，这个脚本连 --fill-timing 都没有（见 execution/setup.py）
    from execution.setup import add_execution_args
    add_execution_args(ap, fill_default="next_open")
    # 股票池
    ap.add_argument("--universe-index", default="", help="股票池指数（空=全市场）")
    ap.add_argument("--min-listed-days", type=int, default=120)
    ap.add_argument("--min-amount", type=float, default=5e7)
    ap.add_argument("--no-st-filter", action="store_true")
    # 策略
    ap.add_argument("--rebalance", default="M", help="调仓频率 D/W/M/整数")
    ap.add_argument("--top-sectors", type=int, default=5)
    ap.add_argument("--stocks-per-sector", type=int, default=4)
    ap.add_argument("--lookback", type=int, default=60)
    ap.add_argument("--skip", type=int, default=5)
    ap.add_argument("--secondary", default="amount",
                    choices=["amount", "low_vol", "large_cap", "neutral_cap", "none"])
    ap.add_argument("--direction", default="momentum",
                    choices=["momentum", "reversal"],
                    help="momentum=买动量最强的板块；reversal=买跌得最惨的板块")
    ap.add_argument("--sector-weighting", default="equal",
                    choices=["equal", "momentum"])
    ap.add_argument("--max-weight", type=float, default=0.0)
    ap.add_argument("--no-vol-adjust", action="store_true")
    # ---- 改进 1：风险控制 ----
    ap.add_argument("--abs-threshold", type=float, default=None,
                    help="绝对动量门槛：打分低于此值的板块不持有（momentum 下设 0 = 只买上涨的板块）")
    ap.add_argument("--scaled-exposure", action="store_true",
                    help="按合格板块数线性降仓（只选出 2/5 个 -> 仓位 40%%，其余持币）")
    ap.add_argument("--min-exposure", type=float, default=0.0, help="目标仓位下限")
    ap.add_argument("--market-ma", type=int, default=0,
                    help=">0 启用市场趋势过滤：指数在 N 日均线上方才满仓")
    ap.add_argument("--defensive-exposure", type=float, default=0.0,
                    help="趋势过滤判定为风险期时的目标仓位（0=空仓持币）")
    # 优化
    ap.add_argument("--optimize", action="store_true", help="在训练集上网格搜索")
    ap.add_argument("--objective", default="sharpe",
                    choices=["sharpe", "calmar", "annual_return", "total_return",
                             "information_ratio"])
    ap.add_argument("--grid", default="full", choices=["full", "quick", "improve"],
                    help="full=48 组；quick=4 组；improve=6 组（改进项验证用）")
    ap.add_argument("--optimize-direction", action="store_true",
                    help="把 momentum/reversal 也纳入搜索（组数翻倍）")
    ap.add_argument("--compare-direction", action="store_true",
                    help="用最优参数把 momentum / reversal 两个方向在训练集和测试集上各跑一遍，出对照表")
    ap.add_argument("--compare-improvements", action="store_true",
                    help="把风险控制改进项（绝对动量/趋势过滤/降仓/风格中性）在训练集和测试集上各跑一遍")
    ap.add_argument("--compare-only", action="store_true",
                    help="跑完对照实验就结束，不再出完整报告（对照实验本身要十几分钟）")
    ap.add_argument("--n-grid", type=int, default=10)
    # 输出
    ap.add_argument("--outdir", default="results/sector_rotation")
    ap.add_argument("--cache-dir", default=".cache/panel",
                    help="面板磁盘缓存目录（调参时能省掉每次 ~8 分钟的加载）；传空字符串关闭")
    ap.add_argument("--refresh-cache", action="store_true", help="忽略已有缓存，强制重新加载")
    ap.add_argument("--no-attribution", action="store_true")
    args = ap.parse_args()

    t_start = time.time()
    split = make_split(args.start, args.split, args.end)
    warmup_start = (pd.Timestamp(args.start)
                    - pd.Timedelta(days=int(args.warmup_days * 1.45))).strftime("%Y-%m-%d")
    os.makedirs(args.outdir, exist_ok=True)
    zh = setup_font()

    from execution.setup import build_execution, describe_execution
    engine_kwargs = dict(initial_capital=args.capital, **build_execution(args))

    banner("板块动量轮动策略  |  完整回测 + 样本外验证")
    print(f"  区间        : {split.describe()}")
    print(f"  预热起点    : {warmup_start}（只用于算动量，不计入绩效）")
    print(f"  基准        : {args.benchmark} {BENCH_NAME}")
    print(f"  初始资金    : {args.capital:,.0f}    {describe_execution(engine_kwargs)}")
    print(f"  调仓频率    : {args.rebalance}")

    # ---------- 1. 数据 ----------
    banner("1. 加载数据", "-")
    t0 = time.time()
    panel = load_price_panel(warmup_start, args.end,
                             cache_dir=(args.cache_dir or None),
                             refresh_cache=args.refresh_cache)
    if not panel:
        raise SystemExit("未取到行情数据")
    close = panel["close"]
    print(f"  价量面板 : {close.shape[0]} 交易日 × {close.shape[1]} 只股票  "
          f"({time.time()-t0:.0f}s)")

    bench = load_index_close(args.benchmark, warmup_start, args.end)
    if bench.empty:
        raise SystemExit(f"未取到基准 {args.benchmark} 的行情（frozen/index_daily）")
    print(f"  基准行情 : {len(bench)} 个交易日  "
          f"{str(bench.index[0])[:10]} ~ {str(bench.index[-1])[:10]}")

    sector_map = load_sector_map(load_industry_map())
    covered = close.columns.isin(sector_map.index).mean()
    print(f"  板块划分 : {sector_map.nunique()} 个行业，覆盖 {covered:.1%} 的股票")

    # ---------- 2. 股票池 ----------
    banner("2. 构造股票池", "-")
    spec_u = UniverseSpec(index_code=(args.universe_index or None),
                          min_listed_days=args.min_listed_days,
                          min_amount=args.min_amount,
                          exclude_st=not args.no_st_filter,
                          exclude_suspended=True)
    t0 = time.time()
    mask = build_universe(panel, spec_u)
    sz = universe_size(mask)
    print(f"  规则     : {spec_u.describe()}")
    print(f"  每日可选 : 中位 {sz.median():.0f}, 最少 {int(sz.min())}, 最多 {int(sz.max())}"
          f"  ({time.time()-t0:.0f}s)")

    # ---------- 3. 策略工厂 ----------
    # 板块收益只跟面板和分类有关，与策略参数无关 —— 预计算一次，参数间复用。
    # 不做这层缓存，网格搜索每组参数都要重算 ~5000 列的分组均值（每组 +15s）。
    t0 = time.time()
    from strategy.sector_rotation import SectorData
    sd = SectorData.build(panel, sector_map, min_stocks=5)
    print(f"  板块数据预计算: {sd.sector_ret.shape[1]} 个板块 × "
          f"{sd.sector_ret.shape[0]} 交易日  ({time.time()-t0:.0f}s)")

    def factory(pnl, msk, params):
        kw = {k: v for k, v in params.items()}
        spec = SectorRotationSpec(rebalance=args.rebalance,
                                  secondary=args.secondary,
                                  sector_weighting=args.sector_weighting,
                                  max_weight=args.max_weight,
                                  abs_threshold=args.abs_threshold,
                                  scaled_exposure=args.scaled_exposure,
                                  min_exposure=args.min_exposure,
                                  market_ma=args.market_ma,
                                  defensive_exposure=args.defensive_exposure,
                                  **kw)
        return build_rotation_weights(pnl, msk, sector_map, spec,
                                      sector_data=sd, market_close=bench).weights

    base_params = dict(
        lookback=args.lookback, skip=args.skip,
        top_sectors=args.top_sectors, stocks_per_sector=args.stocks_per_sector,
        vol_adjust=not args.no_vol_adjust, direction=args.direction,
    )

    # ---------- 4. 训练集寻优 ----------
    search = None
    best_params = dict(base_params)
    if args.optimize:
        banner("3. 训练集参数寻优（样本外不参与）", "-")
        if args.grid == "quick":
            space = ParamSpace({
                "lookback": [20, 60, 120, 250],
                "top_sectors": [5],
                "stocks_per_sector": [5],
                "vol_adjust": [True],
            })
        elif args.grid == "improve":
            # 改进项搜索：风控开关 × 选股方式（比 full 更聚焦，组数可控）
            space = ParamSpace({
                "lookback": [60, 120, 250],
                "top_sectors": [5, 8],
                "stocks_per_sector": [5],
                "vol_adjust": [True],
            })
        else:
            space = ParamSpace({
                "lookback": [20, 60, 120, 250],
                "top_sectors": [3, 5, 8],
                "stocks_per_sector": [3, 5],
                "vol_adjust": [True, False],
            })
        if args.optimize_direction:
            space.spec["direction"] = ["momentum", "reversal"]
        panel_tr = slice_panel(panel, warmup_start, split.split)
        mask_tr = slice_mask(mask, warmup_start, split.split)
        bench_tr = slice_benchmark(bench, warmup_start, split.split)
        t0 = time.time()
        search = grid_search_portfolio(
            panel_tr, mask_tr, factory, space, objective=args.objective,
            benchmark=bench_tr, engine_kwargs=engine_kwargs,
            n_grid=args.n_grid, verbose=True, label="训练集")
        if not np.isfinite(search.iloc[0]["_score"]):
            raise SystemExit("训练集搜索全部失败，检查参数空间与数据")
        best_params = params_of(search.iloc[0])
        print(f"  用时 {time.time()-t0:.0f}s")
        print()
        show = search.head(10).copy()
        # 截断回测指标列，只保留关键列，便于阅读
        keep = ["lookback", "top_sectors", "stocks_per_sector", "vol_adjust",
                "direction", "annual_return", "sharpe_ratio", "max_drawdown",
                "total_trades"]
        show = show[[c for c in keep if c in show.columns]]
        print("  训练集 Top10:")
        print(show.to_string(index=False))
        print()
        print(f"  >>> 最优参数: {best_params}")
        search.to_csv(os.path.join(args.outdir, "train_search.csv"),
                      index=False, encoding="utf-8-sig")

    # ---------- 4b. 方向对照（动量 vs 反转）----------
    if args.compare_direction:
        banner("4b. 方向对照：动量 vs 反转（同样参数，训练/测试各跑一遍）", "-")
        rows = []
        for d in ("momentum", "reversal"):
            p_d = {**best_params, "direction": d}
            r_tr = run_segment(panel, mask, factory, p_d, warmup_start,
                               split.start, split.split, engine_kwargs)
            r_te = run_segment(panel, mask, factory, p_d, warmup_start,
                               split.split, split.end, engine_kwargs)
            rows.append({
                "方向": "动量(买最强)" if d == "momentum" else "反转(买最弱)",
                "样本内年化": r_tr["metrics"].annual_return,
                "样本内夏普": r_tr["metrics"].sharpe_ratio,
                "样本内回撤": r_tr["metrics"].max_drawdown,
                "样本外年化": r_te["metrics"].annual_return,
                "样本外夏普": r_te["metrics"].sharpe_ratio,
                "样本外回撤": r_te["metrics"].max_drawdown,
            })
            print(f"  {rows[-1]['方向']}: 样本内 夏普 {rows[-1]['样本内夏普']:+.3f} "
                  f"年化 {rows[-1]['样本内年化']:+.2%}  |  "
                  f"样本外 夏普 {rows[-1]['样本外夏普']:+.3f} "
                  f"年化 {rows[-1]['样本外年化']:+.2%}")
        cmp_df = pd.DataFrame(rows)
        cmp_df.to_csv(os.path.join(args.outdir, "direction_comparison.csv"),
                      index=False, encoding="utf-8-sig")
        print()
        show_cmp = cmp_df.copy()
        for c in show_cmp.columns:
            if "年化" in c or "回撤" in c:
                show_cmp[c] = show_cmp[c].map(lambda x: f"{x:+.2%}")
            elif "夏普" in c:
                show_cmp[c] = show_cmp[c].map(lambda x: f"{x:+.3f}")
        print(show_cmp.to_string(index=False))

    # ---------- 4c. 改进项对照 ----------
    if args.compare_improvements:
        banner("4c. 改进项对照（同样参数，训练/测试各跑一遍）", "-")
        # 每一行 = 在基线之上叠加一组风控/风格开关。
        # ⚠️ 方法论：只在训练集上挑"哪个改进有效"，测试集的结果只用于验证，
        #    不允许回头再改选择。下面的排序与结论都基于样本内。
        variants = [
            ("基线（无风控）", {}),
            ("+绝对动量>0", {"abs_threshold": 0.0}),
            ("+趋势过滤 MA200", {"market_ma": 200}),
            ("+绝对动量+趋势过滤", {"abs_threshold": 0.0, "market_ma": 200}),
            ("+按合格数降仓", {"abs_threshold": 0.0, "scaled_exposure": True}),
            ("+大盘选股", {"secondary": "large_cap"}),
            ("+市值中性选股", {"secondary": "neutral_cap"}),
            ("+趋势过滤+市值中性", {"market_ma": 200, "secondary": "neutral_cap"}),
        ]

        def make_factory(ov):
            def fac(pnl, msk, params):
                spec = SectorRotationSpec(
                    rebalance=args.rebalance,
                    secondary=ov.get("secondary", args.secondary),
                    sector_weighting=args.sector_weighting,
                    max_weight=args.max_weight,
                    abs_threshold=ov.get("abs_threshold", args.abs_threshold),
                    scaled_exposure=ov.get("scaled_exposure", args.scaled_exposure),
                    min_exposure=args.min_exposure,
                    market_ma=ov.get("market_ma", args.market_ma),
                    defensive_exposure=args.defensive_exposure,
                    **{**best_params, **params})
                return build_rotation_weights(pnl, msk, sector_map, spec,
                                              sector_data=sd,
                                              market_close=bench).weights
            return fac

        rows = []
        for label, ov in variants:
            fac = make_factory(ov)
            t0 = time.time()
            r_tr = run_segment(panel, mask, fac, {}, warmup_start,
                               split.start, split.split, engine_kwargs)
            r_te = run_segment(panel, mask, fac, {}, warmup_start,
                               split.split, split.end, engine_kwargs)
            rows.append({
                "改进项": label,
                "内_年化": r_tr["metrics"].annual_return,
                "内_夏普": r_tr["metrics"].sharpe_ratio,
                "内_回撤": r_tr["metrics"].max_drawdown,
                "外_年化": r_te["metrics"].annual_return,
                "外_夏普": r_te["metrics"].sharpe_ratio,
                "外_回撤": r_te["metrics"].max_drawdown,
                "外_波动": r_te["metrics"].annual_volatility,
            })
            print(f"  {label:<20} 内 夏普 {rows[-1]['内_夏普']:+.3f} "
                  f"回撤 {rows[-1]['内_回撤']:+.1%}  |  "
                  f"外 夏普 {rows[-1]['外_夏普']:+.3f} "
                  f"回撤 {rows[-1]['外_回撤']:+.1%}   ({time.time()-t0:.0f}s)")
        imp = pd.DataFrame(rows)
        imp.to_csv(os.path.join(args.outdir, "improvements.csv"),
                   index=False, encoding="utf-8-sig")
        print()
        show = imp.copy()
        for c in show.columns:
            if "回撤" in c or "年化" in c or "波动" in c:
                show[c] = show[c].map(lambda x: f"{x:+.2%}")
            elif "夏普" in c:
                show[c] = show[c].map(lambda x: f"{x:+.3f}")
        print(show.to_string(index=False))
        # 只用样本内排序 —— 样本外是验证，不是选择依据
        best_in = imp.sort_values("内_夏普", ascending=False).iloc[0]
        print(f"\n  >>> 样本内最优改进项: {best_in['改进项']}  "
              f"(内 夏普 {best_in['内_夏普']:+.3f})")
        base = imp.iloc[0]
        print(f"      相对基线：样本内夏普 {base['内_夏普']:+.3f} -> {best_in['内_夏普']:+.3f}，"
              f"样本外夏普 {base['外_夏普']:+.3f} -> {best_in['外_夏普']:+.3f}")
        print(f"      对应样本外回撤 {base['外_回撤']:+.1%} -> {best_in['外_回撤']:+.1%}")

    if args.compare_only and (args.compare_improvements or args.compare_direction):
        banner("对照实验完成（--compare-only，跳过完整报告）")
        return 0

    # ---------- 5. 样本外 + 全区间 ----------
    banner("4. 样本外验证", "-")
    tr_out = run_segment(panel, mask, factory, best_params, warmup_start,
                         split.start, split.split, engine_kwargs)
    te_out = run_segment(panel, mask, factory, best_params, warmup_start,
                         split.split, split.end, engine_kwargs)
    print(f"  样本内 {split.start}~{split.split}:  夏普 {tr_out['metrics'].sharpe_ratio:+.3f}  "
          f"年化 {tr_out['metrics'].annual_return:+.2%}  "
          f"回撤 {tr_out['metrics'].max_drawdown:+.2%}")
    print(f"  样本外 {split.split}~{split.end}:  夏普 {te_out['metrics'].sharpe_ratio:+.3f}  "
          f"年化 {te_out['metrics'].annual_return:+.2%}  "
          f"回撤 {te_out['metrics'].max_drawdown:+.2%}")
    gap = tr_out['metrics'].sharpe_ratio - te_out['metrics'].sharpe_ratio
    print(f"  夏普落差: {gap:+.3f}  "
          f"({'落差偏大，警惕过拟合' if gap > 0.5 else '落差可接受'})")

    banner("5. 全区间回测", "-")
    full = run_segment(panel, mask, factory, best_params, warmup_start,
                       split.start, split.end, engine_kwargs)
    res = full["result"]
    eq = full["equity"]
    tr = full["trades"]
    print(f"  账目差额 : {res.ledger_gap:.2e}  (应=0)")
    print(f"  成交笔数 : {len(tr):,}")
    if res.rejections:
        print("  —— 被 A股制度约束拦下 ——")
        for k, v in sorted(res.rejections.items(), key=lambda kv: -kv[1]):
            print(f"    {v:>6} 次  {k}")

    bench_full = slice_benchmark(bench, split.start, split.end)
    summ = performance_summary(eq, bench_full)

    # 轮动计划（归因要用它的调仓日程；轨迹打印在第 8 节）
    plan = build_rotation_weights(
        panel, mask, sector_map,
        SectorRotationSpec(rebalance=args.rebalance, secondary=args.secondary,
                           sector_weighting=args.sector_weighting,
                           max_weight=args.max_weight,
                           abs_threshold=args.abs_threshold,
                           scaled_exposure=args.scaled_exposure,
                           min_exposure=args.min_exposure,
                           market_ma=args.market_ma,
                           defensive_exposure=args.defensive_exposure,
                           **best_params),
        sector_data=sd, market_close=bench)

    # ---------- 5b. 对照组：全市场等权 ----------
    banner("5b. 对照组：全市场等权", "-")
    # ⚠️ 这个对照非常关键：本策略的股票池是**全市场**（市值偏小），
    # 而沪深300 是大盘蓝筹。2017-2025 这两个群体的走势长期背离，
    # 只跟沪深300 比会得出"策略很差"或"策略很强"的错误结论。
    # 等权市场才是策略真正的 beta 参照。
    ew_ret = equal_weight_returns(panel, mask, args.rebalance)
    ew_ret = ew_ret[(ew_ret.index >= pd.Timestamp(split.start))
                    & (ew_ret.index <= pd.Timestamp(split.end))]
    ctrl_eq = (1.0 + ew_ret).cumprod() * args.capital
    ctrl_eq = ctrl_eq.reindex(eq.index).ffill()
    ctrl_m = Metrics.compute(ctrl_eq, pd.DataFrame())
    ctrl_summ = performance_summary(ctrl_eq, bench_full)
    vs_ctrl = performance_summary(eq, ctrl_eq)
    print(f"  全市场等权: 年化 {ctrl_m.annual_return:+.2%}  夏普 {ctrl_m.sharpe_ratio:+.3f}  "
          f"回撤 {ctrl_m.max_drawdown:+.2%}   （解析法，含 {len(ew_ret)} 个交易日）")
    print(f"  板块轮动  : 年化 {full['metrics'].annual_return:+.2%}  "
          f"夏普 {full['metrics'].sharpe_ratio:+.3f}  "
          f"回撤 {full['metrics'].max_drawdown:+.2%}")
    print(f"  >>> 策略相对全市场等权的年化 Alpha = {vs_ctrl['alpha_annual']:+.2%} "
          f"(t={vs_ctrl['alpha_tstat']:+.2f}, p={vs_ctrl['alpha_pvalue']:.3f})")
    print(f"      Beta vs 等权市场 = {vs_ctrl['beta']:.3f}  "
          f"R² = {vs_ctrl['r_squared']:.3f}")
    print("      这一项剔除了小盘 beta 的影响，才是'选板块'真正贡献的钱。")

    # ---------- 6. 绩效报告 ----------
    banner("6. 绩效报告", "-")
    print("  三方对照:")
    bench_summ = {
        "total_return": summ["bench_total_return"],
        "annual_return": summ["bench_annual_return"],
        "annual_volatility": summ["bench_annual_volatility"],
        "sharpe_ratio": summ["bench_sharpe_ratio"],
        "calmar_ratio": summ["bench_calmar_ratio"],
        "max_drawdown": summ["bench_max_drawdown"],
    }
    rows = ["累计收益", "年化收益", "年化波动", "夏普比率", "最大回撤", "卡玛比率"]
    cells = {r: [] for r in rows}
    for label, s in (("板块轮动策略", summ), ("基准 " + BENCH_NAME, bench_summ),
                     ("全市场等权", ctrl_summ)):
        cells["累计收益"].append(fmt_value(s["total_return"], "pct"))
        cells["年化收益"].append(fmt_value(s["annual_return"], "pct"))
        cells["年化波动"].append(fmt_value(s["annual_volatility"], "pct"))
        cells["夏普比率"].append(fmt_value(s.get("sharpe_ratio"), "num"))
        cells["最大回撤"].append(fmt_value(s["max_drawdown"], "pct"))
        cells["卡玛比率"].append(fmt_value(s.get("calmar_ratio"), "num"))
    comp = pd.DataFrame(cells, index=["板块轮动策略", "基准 " + BENCH_NAME, "全市场等权"]).T
    print(comp.to_string())
    print()
    print(metrics_table(summ, BENCH_NAME))

    ann, mon = annual_returns_table(to_returns(eq),
                                    to_returns(bench_full)), monthly_returns_table(to_returns(eq))
    print(format_returns_table(ann, "分年度收益（策略 vs 沪深300）"))
    print(format_returns_table(mon, "分月收益（策略，%）"))

    dd = drawdown_info(eq)
    print()
    print("=" * 78)
    print("最大回撤明细")
    print("-" * 78)
    for k, v in dd.to_dict().items():
        print(f"  {k:<24} {v}")
    print("=" * 78)

    # ---------- 7. 归因 ----------
    if not args.no_attribution:
        banner("7. 归因分析", "-")
        ind_map = load_industry_map()
        exp = industry_exposure(res.holdings, ind_map)
        if not exp.empty:
            avg = exp.mean().sort_values(ascending=False)
            print("  行业暴露（区间平均权重，前 15）:")
            for name, w in avg.head(15).items():
                if w > 0.001:
                    print(f"    {str(name):<12} {w:>7.2%}")
            print(f"  （共涉及 {int((avg > 0).sum())} 个行业；"
                  f"平均单行业 {avg[avg>0].mean():.2%}）")
            exp.to_csv(os.path.join(args.outdir, "industry_exposure.csv"),
                       encoding="utf-8-sig", index_label="trade_date")

        # Brinson：基准用沪深300 成分权重
        from universe import index_weight_panel
        from factors.panel import adjusted_close
        try:
            bench_w = index_weight_panel(args.benchmark, eq.index, close.columns)
            bench_w = bench_w.ffill().reindex(index=eq.index, columns=close.columns)
            ret = adjusted_close(panel).pct_change(fill_method=None).reindex(index=eq.index)
            # ⚠️ 必须显式传真实调仓日程。holdings 逐日随价格漂移，
            # brinson 的自动区间识别（权重有变化就切一期）会切出 2500 个"日度区间"，
            # 于是归因表变成 2500 个日度效应之和 —— 那是**算术**超额，
            # 不能拿去和几何口径的 (策略收益 − 基准收益) 对比。
            reb_dates = list(plan.weights.dropna(how="all").index)
            br = brinson(res.holdings.reindex(eq.index), bench_w, ret, ind_map,
                         rebalance_dates=reb_dates)
            if not br.empty:
                print()
                print(f"  Brinson 归因 vs {BENCH_NAME}（按 {len(reb_dates)} 个调仓期拆解，前 12）:")
                show = br.reindex(br["合计"].abs().sort_values(ascending=False).index).head(12)
                s2 = show.copy()
                for c in s2.columns:
                    s2[c] = s2[c].map(lambda x: f"{x:+.4%}")
                print(s2.to_string())
                # ⚠️ br 里已经有一行"总计"，求和时必须排除它，否则翻倍
                detail = br[br.index != "总计"] if "总计" in br.index else br
                total_eff = float(detail["合计"].sum())
                # 正确对照物由 brinson 自己带出来：Σ 各期 (r_p − r_b)
                ref = float(br.attrs.get("sum_excess", float("nan")))
                gap_internal = float(br.attrs.get("max_period_gap", float("nan")))
                print(f"  逐期恒等式自检 max|Σ效应 − (rp−rb)| = {gap_internal:.2e}  "
                      f"（应≈0，共 {br.attrs.get('n_periods')} 期）")
                print(f"  合计效应 {total_eff:+.4%}  ==  Σ 各期超额 {ref:+.4%}   "
                      f"(差 {total_eff - ref:.2e})")
                print(f"  参考：几何口径 策略 {summ['total_return']:+.2%} − 基准 "
                      f"{summ['bench_total_return']:+.2%} = {summ['total_return'] - summ['bench_total_return']:+.2%}"
                      f"（复利口径，与上面两项都不同，波动越大差越多）")
                br.to_csv(os.path.join(args.outdir, "brinson.csv"),
                          encoding="utf-8-sig")
        except Exception as e:
            print(f"  [跳过 Brinson] {type(e).__name__}: {e}")

    # ---------- 8. 轮动轨迹 ----------
    banner("8. 板块轮动轨迹", "-")
    print(plan_summary(plan))
    if not plan.holding_detail.empty:
        # 紧凑格式：一行一个调仓日，列出所选板块（宽表 71 列打印出来没法看）
        print("  各期选中的板块:")
        g = plan.holding_detail.groupby("date")
        for d, sub in g:
            if pd.Timestamp(d) < eq.index[0]:
                continue
            secs = "、".join(f"{r.sector}" for r in sub.itertuples())
            print(f"    {str(d)[:10]}  {secs}")
    hist = sector_holding_history(plan)
    if not hist.empty:
        hist.to_csv(os.path.join(args.outdir, "sector_rotation_trace.csv"),
                    encoding="utf-8-sig", index_label="date")

    # ---------- 9. 图 ----------
    banner("9. 生成图表", "-")
    tag = f"{split.start}_{split.end}"
    p1 = os.path.join(args.outdir, f"equity_vs_csi300_{tag}.png")
    plot_performance(eq, bench_full, p1,
                     title=f"板块动量轮动 vs 沪深300   {split.start} ~ {split.end}",
                     split_date=split.split, zh=zh)
    print(f"  累计净值/超额/回撤/年度收益 -> {p1}")
    p2 = os.path.join(args.outdir, f"rolling_{tag}.png")
    if plot_rolling(eq, bench_full, p2, window=252, title="滚动绩效（窗口 1 年）"):
        print(f"  滚动夏普/超额净值           -> {p2}")

    # ---------- 10. 落盘 ----------
    banner("10. 保存结果", "-")
    out = pd.DataFrame({"equity": eq, "benchmark": bench_full.reindex(eq.index),
                        "control_ew": ctrl_eq.reindex(eq.index)})
    out["nav"] = out["equity"] / out["equity"].iloc[0]
    out["bench_nav"] = out["benchmark"] / out["benchmark"].iloc[0]
    out["control_nav"] = out["control_ew"] / out["control_ew"].iloc[0]
    out["excess_nav"] = out["nav"] / out["bench_nav"]
    out["drawdown"] = (out["equity"] / out["equity"].cummax() - 1)
    save_csv(out, os.path.join(args.outdir, "equity.csv"))
    print(f"  净值曲线 -> {os.path.join(args.outdir, 'equity.csv')}")

    if not tr.empty:
        save_csv(tr, os.path.join(args.outdir, "trades.csv"), index_label="idx")
        print(f"  成交明细 -> {os.path.join(args.outdir, 'trades.csv')}")

    # 逐日持仓（只存调仓日，全量 2600×5700 太大）与板块暴露，便于事后复盘
    reb_w = plan.weights.dropna(how="all")
    if not reb_w.empty:
        save_csv(reb_w, os.path.join(args.outdir, "holdings_rebalance.csv"))
        print(f"  调仓日持仓 -> {os.path.join(args.outdir, 'holdings_rebalance.csv')}")

    report = {
        "区间": {"训练集": list(split.train), "测试集": list(split.test)},
        "数据": {"股票数": int(close.shape[1]), "交易日": int(close.shape[0]),
                 "板块数": int(sector_map.nunique()),
                 "每日可选中位数": float(sz.median())},
        "策略参数": {k: (bool(v) if isinstance(v, (np.bool_, bool)) else v)
                     for k, v in best_params.items()},
        "股票池规则": spec_u.describe(),
        "全区间指标": {k: (None if isinstance(v, float) and not np.isfinite(v) else v)
                       for k, v in summ.items()},
        "样本内指标": tr_out["metrics"].to_dict(),
        "样本外指标": te_out["metrics"].to_dict(),
        "对照组_全市场等权_指标": {k: (None if isinstance(v, float) and not np.isfinite(v) else v)
                              for k, v in ctrl_summ.items()},
        "对照组_说明": "全市场等权（解析法，月度再平衡，不含费用）；策略股票池偏小盘，此对照才是真正的 beta 参照",
        "策略相对对照组_alpha": float(vs_ctrl["alpha_annual"]),
        "策略相对对照组_alpha_p": float(vs_ctrl["alpha_pvalue"]),
        "策略相对对照组_beta": float(vs_ctrl["beta"]),
        "策略相对对照组_r2": float(vs_ctrl["r_squared"]),
        "过拟合落差_夏普": float(gap),
        "账目差额": float(res.ledger_gap),
        "被拒委托": res.rejections,
        "分年度收益": {str(k): float(v) for k, v in ann["组合"].items()},
        "分年度超额": {str(k): float(v) for k, v in ann["超额"].items()} if "超额" in ann else {},
    }
    with open(os.path.join(args.outdir, "report.json"), "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2, default=str)
    print(f"  汇总指标 -> {os.path.join(args.outdir, 'report.json')}")

    banner(f"完成，总用时 {time.time()-t_start:.0f}s")
    return 0


if __name__ == "__main__":
    sys.exit(main())
