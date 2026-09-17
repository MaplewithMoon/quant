# -*- coding: utf-8 -*-
"""聚宽策略移植回测：在本项目引擎上跑聚宽策略，用于与聚宽平台结果对照

用法:
    python scripts/jq_backtest.py --start 2024-01-01 --end 2025-12-31
    python scripts/jq_backtest.py --only v1 --fill close --capital 200000
"""
import argparse
import importlib
import io
import json
import os
import sys
import time

sys.path.insert(0, ".")
if __name__ == "__main__":
    # 只在直接运行时切编码：模块顶层替换 sys.stdout 是有副作用的 import，
    # 会破坏 pytest 的输出捕获（详见 utils/console.py）
    from utils.console import force_utf8_stdout
    force_utf8_stdout()

import numpy as np
import pandas as pd

from analytics.performance import drawdown_info, performance_summary, to_returns
from analytics.report import annual_returns_table, format_returns_table, \
    metrics_table, plot_performance, save_csv, setup_font
from backtest.panel_data import load_price_panel
from database.loader import load_index_close

STRATEGIES = {
    "v1": ("动态持仓+止损线9+单价≤50（国九小市值）", "strategies.jq.guojiu_smallcap"),
    "v2": ("国九小市值改进（中证全指+择时）", "strategies.jq.guojiu_smallcap_v2"),
}
BENCH = {"v1": "399101.XSHE", "v2": "000985.XSHG"}   # 与原策略 set_benchmark 一致


def banner(t, ch="=", w=94):
    print()
    print(ch * w)
    print(t)
    print(ch * w)


def main():
    ap = argparse.ArgumentParser(description="聚宽策略移植回测")
    ap.add_argument("--start", default="2024-01-01")
    ap.add_argument("--end", default="2025-12-31")
    ap.add_argument("--warmup-days", type=int, default=420,
                    help="预热自然日（MA10/前收要用）")
    ap.add_argument("--capital", type=float, default=100_000,
                    help="聚宽示例回测常用 10 万本金")
    ap.add_argument("--fill", default="auto", choices=["auto", "open", "close"])
    ap.add_argument("--etf-yield", type=float, default=0.02,
                    help="货币 ETF（511880）合成年化收益；库里没有它的日线")
    ap.add_argument("--only", default="", help="只跑其中一个：v1 / v2")
    ap.add_argument("--no-rules", action="store_true", help="关闭 A 股制度约束（对照用）")
    ap.add_argument("--slippage", type=float, default=0.001,
                    help="固定滑点（默认 1bp）。⚠️ 原先引擎里硬编码 0.0，"
                         "零滑点会让两个策略的结果偏乐观")
    ap.add_argument("--impact-model", default="none", choices=["none", "fixed", "sqrt"],
                    help="市场冲击模型；sqrt 需配合加大 --capital 做容量分析")
    ap.add_argument("--impact-k", type=float, default=0.1, help="平方根冲击系数")
    ap.add_argument("--verbose", action="store_true", help="打印策略日志")
    ap.add_argument("--outdir", default="results/jq")
    args = ap.parse_args()

    t0 = time.time()
    os.makedirs(args.outdir, exist_ok=True)
    start, end = pd.Timestamp(args.start), pd.Timestamp(args.end)
    warmup = (start - pd.Timedelta(days=int(args.warmup_days))).strftime("%Y-%m-%d")
    setup_font()

    banner("聚宽策略移植回测  |  与本项目引擎对照")
    print(f"  区间 {args.start} ~ {args.end}   本金 {args.capital:,.0f}   "
          f"成交价 {args.fill}   制度约束 {'关' if args.no_rules else '开'}")
    print(f"  滑点 {args.slippage:.2%}   冲击模型 {args.impact_model}"
          + (f"(k={args.impact_k})" if args.impact_model == "sqrt" else ""))

    banner("1. 加载数据", "-")
    t1 = time.time()
    panel = load_price_panel(warmup, args.end, cache_dir=".cache/panel")
    if not panel:
        raise SystemExit("未取到行情数据")
    print(f"  面板 {panel['close'].shape[0]} 交易日 × {panel['close'].shape[1]} 只"
          f"  ({time.time()-t1:.0f}s)")

    rows, results = [], {}
    keys = [args.only] if args.only else list(STRATEGIES)
    for key in keys:
        label, modname = STRATEGIES[key]
        banner(f"2. 运行 {key}: {label}", "-")
        from joinquant.api import run_strategy, log as jqlog
        jqlog.level = 30 if not args.verbose else 10
        mod = importlib.import_module(modname)
        importlib.reload(mod)                 # 每次重跑都要清掉 g / 任务表
        t2 = time.time()
        res = run_strategy(mod, panel, args.start, args.end,
                           initial_cash=args.capital, fill=args.fill,
                           etf_yield=args.etf_yield, rules=not args.no_rules,
                           verbose=args.verbose, slippage=args.slippage,
                           impact_model=args.impact_model,
                           impact_k=args.impact_k)
        eq = res["equity"]
        eng = res["engine"]
        print(f"  用时 {time.time()-t2:.0f}s   交易日 {len(eq)}   "
              f"最终权益 {eq.iloc[-1]:,.0f}   拒绝委托 {sum(res['rejections'].values())} 笔")
        if res["rejections"]:
            top = sorted(res["rejections"].items(), key=lambda kv: -kv[1])[:6]
            for k, v in top:
                print(f"      {v:>5} 次  {k}")

        # 基准：原策略自己的 set_benchmark
        bench_code = BENCH[key]
        bench = None
        # ⚠️ 必须取全序列：index_close 默认 count=1，只拿一个点的话
        # 组合与基准没有重叠交易日，画图和算 alpha 都会直接失败
        bd = eng.data.index_close(bench_code, end, count=10 ** 6)
        if len(bd):
            bench = bd.reindex(eq.index).ffill()
        summ = performance_summary(eq, bench) if bench is not None else \
            performance_summary(eq)
        dd = drawdown_info(eq)
        print(f"  累计 {summ['total_return']:+.2%}  年化 {summ['annual_return']:+.2%}  "
              f"波动 {summ['annual_volatility']:.2%}  夏普 {summ['sharpe_ratio']:+.3f}  "
              f"最大回撤 {dd.max_drawdown:+.2%}")
        if bench is not None:
            print(f"  基准 {bench_code}: 累计 {summ['bench_total_return']:+.2%}  "
                  f"年化 {summ['bench_annual_return']:+.2%}  "
                  f"| Alpha {summ['alpha_annual']:+.2%} "
                  f"(t={summ['alpha_tstat']:+.2f}, p={summ['alpha_pvalue']:.3f})  "
                  f"Beta {summ['beta']:+.3f}")
        print(f"  卡玛 {summ['calmar_ratio']:+.3f}  索提诺 {summ['sortino_ratio']:+.3f}  "
              f"换手相关：持仓 {eng.pf.positions and len(eng.pf.positions)} 只")

        # ---- 聚宽风格 HTML 报告 ----
        from analytics.jq_report import build_jq_report
        html_path = os.path.join(args.outdir, f"report_{key}.html")
        note = ("本报告由本项目引擎生成。与聚宽对照前请先看 "
                "docs/聚宽策略移植报告.md 的「近似与差异」："
                "指数已改用真实日线与真实月度成分、ETF 用真实日线；"
                "但成交价为日线近似（上午开盘 / 下午收盘），聚宽为分钟撮合。")
        build_jq_report(eq, bench if bench is not None else eq, label,
                        subtitle=f"{args.start} ~ {args.end}   本金 "
                                 f"{args.capital:,.0f}   成交价 {args.fill}   "
                                 f"基准 {bench_code}",
                        trades=res.get("trades"),
                        out_path=html_path, extra_notes=note)
        print(f"  聚宽风格报告 -> {html_path}")

        rows.append({
            "策略": key, "名称": label, "基准": bench_code,
            "累计收益": summ["total_return"], "年化收益": summ["annual_return"],
            "年化波动": summ["annual_volatility"], "夏普": summ["sharpe_ratio"],
            "最大回撤": dd.max_drawdown, "卡玛": summ["calmar_ratio"],
            "基准年化": summ.get("bench_annual_return"),
            "Alpha": summ.get("alpha_annual"), "Beta": summ.get("beta"),
            "Alpha_p": summ.get("alpha_pvalue"),
            "拒绝委托": int(sum(res["rejections"].values())),
            "最终权益": float(eq.iloc[-1]),
        })
        results[key] = {"res": res, "summ": summ, "bench": bench, "label": label}

        # 落盘
        tag = f"{key}_{args.start}_{args.end}"
        out = pd.DataFrame({"equity": eq})
        if bench is not None:
            out["benchmark"] = bench.reindex(eq.index)
            out["nav"] = out["equity"] / out["equity"].iloc[0]
            out["bench_nav"] = out["benchmark"] / out["benchmark"].iloc[0]
            out["excess_nav"] = out["nav"] / out["bench_nav"]
        out["drawdown"] = out["equity"] / out["equity"].cummax() - 1
        save_csv(out, os.path.join(args.outdir, f"equity_{tag}.csv"))
        if bench is not None:
            plot_performance(eq, bench, os.path.join(args.outdir, f"equity_{tag}.png"),
                             title=f"{label}  {args.start} ~ {args.end}", zh=True)
        ann = annual_returns_table(to_returns(eq),
                                   to_returns(bench) if bench is not None else None)
        print(format_returns_table(ann, "分年度收益"))
        print(metrics_table(summ, bench_code))

    banner("3. 对照汇总", "-")
    cmp = pd.DataFrame(rows)
    cmp.to_csv(os.path.join(args.outdir, "compare.csv"), index=False,
               encoding="utf-8-sig")
    show = cmp.copy()
    for c in ("累计收益", "年化收益", "年化波动", "最大回撤", "基准年化", "Alpha"):
        show[c] = show[c].map(lambda x: f"{x:+.2%}" if pd.notna(x) else "-")
    for c in ("夏普", "卡玛", "Beta"):
        show[c] = show[c].map(lambda x: f"{x:+.3f}" if pd.notna(x) else "-")
    print(show[["策略", "名称", "累计收益", "年化收益", "夏普", "最大回撤",
                "基准年化", "Alpha", "Alpha_p", "拒绝委托", "最终权益"]].to_string(index=False))
    print("\n  ⚠️ 与聚宽对比时请先看 docs/聚宽策略移植报告.md 的「近似与差异」一节：")
    print("     指数成分/点位为规则重建、ETF 用现金合成、分钟撮合用日线近似。")

    with open(os.path.join(args.outdir, "report.json"), "w", encoding="utf-8") as fh:
        json.dump({"区间": [args.start, args.end], "本金": args.capital,
                   "成交价模式": args.fill, "制度约束": not args.no_rules,
                   "策略": cmp.replace({np.nan: None}).to_dict("records")},
                  fh, ensure_ascii=False, indent=2, default=str)
    banner(f"完成，总用时 {time.time()-t0:.0f}s")
    return 0


if __name__ == "__main__":
    sys.exit(main())
