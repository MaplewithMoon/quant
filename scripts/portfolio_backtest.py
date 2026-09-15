# -*- coding: utf-8 -*-
"""组合回测 CLI：股票池 → 因子合成 → 组合构建 → 多标的回测 → 业绩归因

【用法】
    # 单因子选股，沪深300 池内月度调仓
    python scripts/portfolio_backtest.py --index 000300.SH --factor bp \
        --n-hold 30 --rebalance M --start 2022-01-01 --end 2024-12-31

    # 多因子合成（按 ICIR 加权）
    python scripts/portfolio_backtest.py --index 000300.SH \
        --factor vol_60,turn_20,bp --method icir_weight --n-hold 30

    # 全市场 + 流动性过滤，双周调仓，逆波动率加权
    python scripts/portfolio_backtest.py --factor bp --top-n 500 \
        --min-amount 5e7 --weighting inv_vol --rebalance 10

    # 关闭制度约束做对照
    python scripts/portfolio_backtest.py --factor bp --no-rules

【说明】
- 股票池默认使用**历史指数成分**（逐期快照），不是今天的成分，消除幸存者偏差。
- 成交时点默认 next_open：T 日收盘算权重 → T+1 开盘成交，无未来函数。
- 业绩归因对比基准为该指数的权重（同样取历史快照）。
"""
import sys, io, os, argparse
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
sys.path.insert(0, ".")

import numpy as np
import pandas as pd

from backtest.multi_engine import PortfolioBacktestEngine, FILL_NEXT_OPEN, FILL_SAME_CLOSE
from backtest.panel_data import load_price_panel
from execution.market_rules import MarketRules
from execution.impact import build_model
from factors.composite import composite_score, correlation_matrix
from portfolio.construction import build_target_weights, turnover_of_weights
from universe import UniverseSpec, build_universe, universe_size, index_weight_panel
from analytics.attribution import (industry_exposure, multi_factor_exposure,
                                   brinson, load_industry_map)
from factors.base import Factor


def parse_symbols(text):
    import re
    out = []
    for p in re.split(r"[,;\s]+", (text or "").strip()):
        if p.isdigit() and len(p) <= 6:
            out.append(p.zfill(6))
        elif p:
            out.append(p)
    return out


def main():
    ap = argparse.ArgumentParser(description="多标的组合回测")
    ap.add_argument("--factor", default="bp", help="因子名，逗号分隔（见 scripts/factor_eval.py --list）")
    ap.add_argument("--method", default="equal",
                    choices=["equal", "ic_weight", "icir_weight"], help="因子合成方式")
    ap.add_argument("--index", default="000300.SH", help="股票池指数（空=全市场）")
    ap.add_argument("--start", default="2022-01-01")
    ap.add_argument("--end", default="2024-12-31")
    ap.add_argument("--n-hold", type=int, default=30, help="持股数量")
    ap.add_argument("--quantile", type=float, default=0.0, help="按分位选股（>0 时覆盖 n-hold）")
    ap.add_argument("--weighting", default="equal",
                    choices=["equal", "market_cap", "inv_vol"], help="权重方案")
    ap.add_argument("--max-weight", type=float, default=0.0, help="单票权重上限")
    ap.add_argument("--rebalance", default="M", help="调仓频率：D/W/M 或整数（交易日）")
    ap.add_argument("--min-listed-days", type=int, default=60)
    ap.add_argument("--min-amount", type=float, default=5e7, help="日均成交额下限（元）")
    ap.add_argument("--top-n", type=int, default=0, help="股票池按流动性取前 N")
    ap.add_argument("--no-st-filter", action="store_true", help="不剔除 ST")
    ap.add_argument("--capital", type=float, default=1_000_000)
    ap.add_argument("--commission", type=float, default=0.0001)
    ap.add_argument("--stamp-duty", type=float, default=0.0005)
    ap.add_argument("--slippage", type=float, default=0.001)
    ap.add_argument("--impact-model", default="fixed", choices=["fixed", "sqrt", "none"])
    ap.add_argument("--impact-k", type=float, default=0.1)
    ap.add_argument("--max-participation", type=float, default=0.10)
    ap.add_argument("--fill-timing", default=FILL_NEXT_OPEN,
                    choices=[FILL_NEXT_OPEN, FILL_SAME_CLOSE])
    ap.add_argument("--no-rules", action="store_true", help="关闭 A股制度约束")
    ap.add_argument("--no-attribution", action="store_true", help="跳过归因分析")
    ap.add_argument("--out", default="", help="净值曲线保存为 CSV")
    args = ap.parse_args()

    factors = parse_symbols(args.factor) if "," in args.factor else [args.factor.strip()]
    print("=" * 96)
    print(f"组合回测  {args.start} ~ {args.end}   资金 {args.capital:,.0f}")
    print(f"因子: {factors}   合成: {args.method}   持股 {args.n_hold}   "
          f"调仓 {args.rebalance}   加权 {args.weighting}")
    print("=" * 96)

    # ---- 1. 数据 ----
    panel = load_price_panel(args.start, args.end, limit=(args.top_n or None))
    if not panel:
        raise SystemExit("未取到数据")
    close = panel["close"]
    print(f"面板: {close.shape[0]} 交易日 × {close.shape[1]} 只股票")

    # ---- 2. 股票池 ----
    spec = UniverseSpec(index_code=(args.index or None),
                        min_listed_days=args.min_listed_days,
                        min_amount=args.min_amount,
                        exclude_st=not args.no_st_filter,
                        top_n=args.top_n)
    mask = build_universe(panel, spec)
    sz = universe_size(mask)
    print(f"股票池规则: {spec.describe()}")
    print(f"每日可选股票数: 中位 {sz.median():.0f}, 最少 {sz.min()}, 最多 {sz.max()}")

    # ---- 3. 因子合成 ----
    if len(factors) > 1:
        corr = correlation_matrix(panel, factors, mask)
        print("\n因子相关性（合成前应确认不过高）:")
        print(corr.round(3).to_string())
    score = composite_score(panel, factors, method=args.method, panel_mask=mask)
    print(f"\n合成打分覆盖: 日均 {score.notna().sum(axis=1).mean():.0f} 只")

    # ---- 4. 组合构建 ----
    vol = None
    if args.weighting == "inv_vol":
        from factors.panel import adjusted_close
        vol = adjusted_close(panel).pct_change(fill_method=None).rolling(60).std()
    tw = build_target_weights(
        score, mask, n_hold=args.n_hold, weighting=args.weighting,
        mv=panel.get("total_mv"), vol=vol,
        rebalance=(int(args.rebalance) if args.rebalance.isdigit() else args.rebalance),
        max_weight=args.max_weight, quantile=args.quantile)
    reb_days = tw.dropna(how="all").index
    print(f"调仓次数: {len(reb_days)}   平均换手: "
          f"{turnover_of_weights(tw).mean():.1%}" if len(reb_days) > 1 else "调仓次数不足")

    # ---- 5. 回测 ----
    eng = PortfolioBacktestEngine(
        initial_capital=args.capital, commission=args.commission,
        stamp_duty=args.stamp_duty, slippage=args.slippage,
        slippage_model=build_model(args.impact_model, rate=args.slippage, k=args.impact_k),
        max_participation=args.max_participation,
        fill_timing=args.fill_timing,
        market_rules=MarketRules(enabled=not args.no_rules))
    res = eng.run(panel, tw)
    print()
    print(res.summary())

    # ---- 6. 归因 ----
    if not args.no_attribution:
        ind_map = load_industry_map()
        if not ind_map.empty and not res.holdings.empty:
            print()
            exp = industry_exposure(res.holdings, ind_map)
            if not exp.empty:
                avg = exp.mean().sort_values(ascending=False)
                print("行业暴露（区间平均权重，前 10）:")
                for k, v in avg.head(10).items():
                    print(f"    {str(k):<12} {v:>7.2%}")

            if args.index:
                bw = index_weight_panel(args.index, close.index, close.columns)
                ret = close.pct_change(fill_method=None)
                br = brinson(res.holdings, bw, ret, ind_map)
                if not br.empty:
                    print()
                    print(f"Brinson 归因 vs {args.index}（区间累计）:")
                    show = br.copy()
                    for c in show.columns:
                        show[c] = show[c].map(lambda x: f"{x:+.4%}")
                    print(show.to_string())

    if args.out:
        parent = os.path.dirname(os.path.abspath(args.out))
        os.makedirs(parent, exist_ok=True)
        res.equity.to_csv(args.out, header=["equity"], encoding="utf-8-sig")
        print(f"\n净值已保存: {args.out}")


if __name__ == "__main__":
    main()
