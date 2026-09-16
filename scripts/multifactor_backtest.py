# -*- coding: utf-8 -*-
"""多因子选股策略回测（基本面因子，训练/测试集严格分离）

【方法论：预注册规则 → 样本外只跑一次】
    1. 先在训练集上算全部因子的 IC / 分层（scripts/fundamental_ic.py）
    2. 用**只看样本内**的规则挑因子：样本内 p<0.05 且 多空>0 且 单调性>0.5
    3. 本脚本在训练集上选参数（持股数、加权方式），在测试集上**只跑一次**
    4. 同时报告"规则组合"与"先验组合"（价值+流动性+筹码）两条线：
       - 规则组合 = 预注册规则的诚实样本外结果
       - 先验组合 = 由既有文献/常识（价值、流动性、筹码集中）选出的因子，
                    不是看着测试集挑的

用法:
    python scripts/multifactor_backtest.py --start 2017-01-01 --split 2022-01-01 --end 2025-12-31
"""
import argparse
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
from analytics.report import (annual_returns_table, format_returns_table,
                              metrics_table, plot_performance, plot_rolling,
                              save_csv, setup_font)
from backtest.metrics import Metrics
from backtest.multi_engine import PortfolioBacktestEngine
from backtest.panel_data import load_price_panel
from database.loader import load_index_close
from factors.fundamental import (FACTOR_META, load_all_factors,
                                 neutralized_score, combine_by_rule)
from factors.panel import adjusted_close
from portfolio.construction import build_target_weights, equal_weight_returns, \
    turnover_of_weights
from universe import UniverseSpec, build_universe, universe_size

BENCH_CODE, BENCH_NAME = "000300.SH", "沪深300"


def banner(t, ch="=", w=92):
    print()
    print(ch * w)
    print(t)
    print(ch * w)


def fmt_pct(x):
    return f"{x:+.2%}" if pd.notna(x) else "-"


def fmt_num(x):
    return f"{x:+.3f}" if pd.notna(x) else "-"


def run_one(panel, mask, score, spec_kwargs, engine_kwargs, warmup_start,
            eval_start, eval_end, capital):
    """跑一段回测并截断到评估窗口"""
    sub = {k: v.loc[(v.index >= pd.Timestamp(warmup_start))
                    & (v.index <= pd.Timestamp(eval_end))]
           for k, v in panel.items() if isinstance(v, pd.DataFrame)}
    sc = score.loc[(score.index >= pd.Timestamp(warmup_start))
                   & (score.index <= pd.Timestamp(eval_end))]
    mk = mask.loc[(mask.index >= pd.Timestamp(warmup_start))
                  & (mask.index <= pd.Timestamp(eval_end))]
    tw = build_target_weights(sc, mk, **spec_kwargs)
    eng = PortfolioBacktestEngine(**engine_kwargs)
    res = eng.run(sub, tw)
    ts = pd.Timestamp(eval_start)
    eq = res.equity[res.equity.index >= ts]
    tr = res.trades
    if tr is not None and not tr.empty and "timestamp" in tr.columns:
        tr = tr[pd.to_datetime(tr["timestamp"]) >= ts]
    m = Metrics.compute(eq, tr) if not eq.empty else Metrics()
    return {"equity": eq, "trades": tr, "metrics": m, "result": res,
            "target": tw, "ledger_gap": res.ledger_gap}


def main():
    ap = argparse.ArgumentParser(description="多因子选股策略回测（基本面因子）")
    ap.add_argument("--start", default="2017-01-01")
    ap.add_argument("--split", default="2022-01-01")
    ap.add_argument("--end", default="2025-12-31")
    ap.add_argument("--warmup-days", type=int, default=609,
                    help="预热自然日；⚠️ 改这个值会导致面板缓存未命中（重扫约 8 分钟）")
    ap.add_argument("--benchmark", default=BENCH_CODE)
    ap.add_argument("--capital", type=float, default=1_000_000)
    ap.add_argument("--commission", type=float, default=0.0001)
    ap.add_argument("--stamp-duty", type=float, default=0.0005)
    ap.add_argument("--min-commission", type=float, default=5.0)
    ap.add_argument("--slippage", type=float, default=0.0)
    ap.add_argument("--min-listed-days", type=int, default=120)
    ap.add_argument("--min-amount", type=float, default=5e7)
    ap.add_argument("--rebalance", default="M")
    ap.add_argument("--n-hold-grid", default="30,50,80",
                    help="训练集上搜索的持股数（逗号分隔）")
    ap.add_argument("--factor-table", default="results/fundamental/factor_ic.csv",
                    help="fundamental_ic.py 产出的因子统计表（用于预注册规则选因子）")
    ap.add_argument("--cache-dir", default=".cache/panel")
    ap.add_argument("--lag-days", type=int, default=1)
    ap.add_argument("--outdir", default="results/multifactor")
    args = ap.parse_args()

    t0 = time.time()
    os.makedirs(args.outdir, exist_ok=True)
    start, split, end = (pd.Timestamp(args.start), pd.Timestamp(args.split),
                         pd.Timestamp(args.end))
    warmup = (start - pd.Timedelta(days=int(args.warmup_days))).strftime("%Y-%m-%d")
    setup_font()

    engine_kwargs = dict(initial_capital=args.capital, commission=args.commission,
                         min_commission=args.min_commission,
                         stamp_duty=args.stamp_duty, slippage=args.slippage)

    banner("多因子选股策略  |  训练/测试集分离  |  全部因子行业+市值中性")
    print(f"  训练集 {args.start} ~ {args.split}     测试集 {args.split} ~ {args.end}")
    print(f"  资金 {args.capital:,.0f}   滑点 {args.slippage:.2%}   调仓 {args.rebalance}")

    # ---------- 1. 数据 ----------
    banner("1. 加载数据", "-")
    panel = load_price_panel(warmup, args.end, cache_dir=(args.cache_dir or None))
    if not panel:
        raise SystemExit("未取到行情数据")
    close = adjusted_close(panel)
    mask = build_universe(panel, UniverseSpec(min_listed_days=args.min_listed_days,
                                              min_amount=args.min_amount))
    bench = load_index_close(args.benchmark, warmup, args.end)
    print(f"  面板 {close.shape[0]} 日 × {close.shape[1]} 只；"
          f"每日可选中位 {universe_size(mask).median():.0f} 只")

    # 因子只需要调仓日的取值（横截面打分本身也是调仓日才算）
    from portfolio.construction import rebalance_dates
    reb = rebalance_dates(close.index, args.rebalance)
    t1 = time.time()
    factors = load_all_factors(panel, pd.DatetimeIndex(reb), close.columns,
                               lag_days=args.lag_days, verbose=True)
    print(f"  因子 {len(factors)} 个（PIT 对齐 {time.time()-t1:.0f}s）")

    from analytics.attribution import load_industry_map, _industry_series
    panel_ind = _industry_series(load_industry_map(), close.columns)
    mv = panel.get("total_mv")

    # ---------- 2. 预注册规则挑因子 ----------
    banner("2. 因子集合（预注册规则 + 先验组合）", "-")
    rule_factors, prior_factors = [], ["bp", "ep_ttm", "turnover_20", "holder_chg"]
    if os.path.isfile(args.factor_table):
        tab = pd.read_csv(args.factor_table)
        rule_factors = combine_by_rule(tab)
        print(f"  规则组合（样本内 p<0.05 且 多空>0 且 单调性>0.5）: "
              f"{len(rule_factors)} 个")
        print(f"    {rule_factors}")
    else:
        print(f"  [警告] 未找到 {args.factor_table}，跳过规则组合。"
              f"请先跑 scripts/fundamental_ic.py")
    print(f"  先验组合（价值/流动性/筹码，来自既有文献而非本次数据）: {prior_factors}")
    sets = []
    if rule_factors:
        sets.append(("规则组合", [f for f in rule_factors if f in factors]))
    sets.append(("先验组合", [f for f in prior_factors if f in factors]))

    # ---------- 3. 训练集选持股数，测试集只跑一次 ----------
    banner("3. 训练集选参数 -> 测试集验证", "-")
    n_grid = [int(x) for x in args.n_hold_grid.split(",") if x.strip()]
    rows, results = [], {}
    for label, names in sets:
        if not names:
            continue
        print(f"\n  === {label}: {names} ===")
        score = neutralized_score(factors, names, mask, panel_ind, mv)
        best = None
        for n in n_grid:
            r = run_one(panel, mask, score,
                        dict(n_hold=n, weighting="equal", rebalance=args.rebalance),
                        engine_kwargs, warmup, start, split, args.capital)
            m = r["metrics"]
            print(f"    持股 {n:>3}  内 夏普 {fmt_num(m.sharpe_ratio)}  "
                  f"年化 {fmt_pct(m.annual_return)}  回撤 {fmt_pct(m.max_drawdown)}")
            if best is None or m.sharpe_ratio > best[1].sharpe_ratio:
                best = (n, m)
        n_best = best[0]
        tr = run_one(panel, mask, score,
                     dict(n_hold=n_best, weighting="equal", rebalance=args.rebalance),
                     engine_kwargs, warmup, start, split, args.capital)
        te = run_one(panel, mask, score,
                     dict(n_hold=n_best, weighting="equal", rebalance=args.rebalance),
                     engine_kwargs, warmup, split, end, args.capital)
        full = run_one(panel, mask, score,
                       dict(n_hold=n_best, weighting="equal", rebalance=args.rebalance),
                       engine_kwargs, warmup, start, end, args.capital)
        print(f"    >>> 训练集最优持股数 {n_best}")
        print(f"    样本内: 夏普 {fmt_num(tr['metrics'].sharpe_ratio)} "
              f"年化 {fmt_pct(tr['metrics'].annual_return)} "
              f"回撤 {fmt_pct(tr['metrics'].max_drawdown)}")
        print(f"    样本外: 夏普 {fmt_num(te['metrics'].sharpe_ratio)} "
              f"年化 {fmt_pct(te['metrics'].annual_return)} "
              f"回撤 {fmt_pct(te['metrics'].max_drawdown)}")
        rows.append({
            "组合": label, "因子": ",".join(names), "持股数": n_best,
            "内年化": tr["metrics"].annual_return, "内夏普": tr["metrics"].sharpe_ratio,
            "内回撤": tr["metrics"].max_drawdown,
            "外年化": te["metrics"].annual_return, "外夏普": te["metrics"].sharpe_ratio,
            "外回撤": te["metrics"].max_drawdown,
            "外波动": te["metrics"].annual_volatility,
            "账目差额": full["ledger_gap"],
        })
        results[label] = {"train": tr, "test": te, "full": full,
                          "names": names, "n_hold": n_best, "score": score}

    cmp = pd.DataFrame(rows)
    cmp.to_csv(os.path.join(args.outdir, "compare.csv"), index=False,
               encoding="utf-8-sig")

    # ---------- 4. 完整报告（用先验组合，训练/测试都报）----------
    for label in [s[0] for s in sets]:
        if label not in results:
            continue
        R = results[label]
        banner(f"4. 完整报告 —— {label}（{R['names']}，持股 {R['n_hold']}）", "-")
        for seg, key in (("全区间", "full"), ("样本外", "test")):
            r = R[key]
            ev = r["equity"]
            if ev.empty:
                continue
            b = bench.reindex(ev.index).ffill()
            summ = performance_summary(ev, b)
            ctrl_ret = equal_weight_returns(
                {k: v.loc[v.index.isin(ev.index)] for k, v in panel.items()
                 if isinstance(v, pd.DataFrame)},
                mask.reindex(ev.index), args.rebalance)
            ctrl_eq = (1 + ctrl_ret).cumprod() * args.capital
            vs_ctrl = performance_summary(ev, ctrl_eq.reindex(ev.index).ffill())
            print(f"\n  —— {seg} ——")
            print(f"  策略年化 {fmt_pct(summ['annual_return'])}  夏普 "
                  f"{fmt_num(summ['sharpe_ratio'])}  回撤 {fmt_pct(summ['max_drawdown'])}"
                  f"   |  沪深300 年化 {fmt_pct(summ['bench_annual_return'])}  "
                  f"夏普 {fmt_num(summ['bench_sharpe_ratio'])}")
            print(f"  Alpha(vs 沪深300) {fmt_pct(summ['alpha_annual'])} "
                  f"(t={fmt_num(summ['alpha_tstat'])}, p={summ['alpha_pvalue']:.3f})  "
                  f"Beta {fmt_num(summ['beta'])}  R² {fmt_num(summ['r_squared'])}")
            print(f"  信息比率 {fmt_num(summ['information_ratio'])}  "
                  f"跟踪误差 {fmt_pct(summ['tracking_error'])}  "
                  f" |  vs 全市场等权 Alpha {fmt_pct(vs_ctrl['alpha_annual'])} "
                  f"(p={vs_ctrl['alpha_pvalue']:.3f})")
            if key == "full":
                print()
                print(metrics_table(summ, BENCH_NAME))
                ann = annual_returns_table(to_returns(ev), to_returns(b))
                print(format_returns_table(ann, "分年度收益"))
                dd = drawdown_info(ev)
                print(f"\n  最大回撤 {dd.max_drawdown:+.2%}  "
                      f"({str(dd.peak_date)[:10]} -> {str(dd.trough_date)[:10]}, "
                      f"收复 {str(dd.recover_date)[:10]})")
                print(f"  账目差额 {r['ledger_gap']:.2e}   成交 {len(r['trades']):,} 笔")
                to = turnover_of_weights(r["target"])   # r == R["full"]，不是 R 本身
                if len(to):
                    print(f"  平均换手 {to.mean():.1%}")

                # 图与落盘
                tag = f"{args.start}_{args.end}"
                p1 = os.path.join(args.outdir, f"equity_{label}_{tag}.png")
                plot_performance(ev, b, p1,
                                 title=f"多因子选股（{label}）vs 沪深300   {args.start} ~ {args.end}",
                                 split_date=args.split, zh=True)
                p2 = os.path.join(args.outdir, f"rolling_{label}_{tag}.png")
                plot_rolling(ev, b, p2, 252, title=f"滚动绩效（{label}）")
                print(f"  图 -> {p1}")
                out = pd.DataFrame({"equity": ev, "benchmark": b.reindex(ev.index)})
                out["nav"] = out["equity"] / out["equity"].iloc[0]
                out["bench_nav"] = out["benchmark"] / out["benchmark"].iloc[0]
                out["excess_nav"] = out["nav"] / out["bench_nav"]
                save_csv(out, os.path.join(args.outdir, f"equity_{label}.csv"))
                if r["trades"] is not None and not r["trades"].empty:
                    save_csv(r["trades"],
                             os.path.join(args.outdir, f"trades_{label}.csv"),
                             index_label="idx")

    banner("5. 对照汇总（结论看这里）", "-")
    show = cmp.copy()
    for c in ("内年化", "外年化", "内回撤", "外回撤", "外波动"):
        show[c] = show[c].map(fmt_pct)
    for c in ("内夏普", "外夏普"):
        show[c] = show[c].map(fmt_num)
    print(show[["组合", "持股数", "内年化", "内夏普", "内回撤",
                "外年化", "外夏普", "外回撤", "外波动"]].to_string(index=False))
    base = cmp[cmp["组合"] == "先验组合"]
    if not base.empty:
        print(f"\n  先验组合样本外: 年化 {base.iloc[0]['外年化']:+.2%}  "
              f"夏普 {base.iloc[0]['外夏普']:+.3f}  回撤 {base.iloc[0]['外回撤']:+.2%}")
    print("\n  提醒：持股数与加权方式在训练集上选，测试集只跑一次；"
          "因子集合用预注册规则或先验确定，不参与调参。")

    with open(os.path.join(args.outdir, "report.json"), "w", encoding="utf-8") as fh:
        json.dump({
            "区间": {"训练集": [args.start, args.split], "测试集": [args.split, args.end]},
            "因子集合": {k: v["names"] for k, v in results.items()},
            "持股数": {k: v["n_hold"] for k, v in results.items()},
            "对照": cmp.replace({np.nan: None}).to_dict("records"),
        }, fh, ensure_ascii=False, indent=2, default=str)
    print(f"\n  产出 -> {args.outdir}/")
    banner(f"完成，总用时 {time.time()-t0:.0f}s")
    return 0


if __name__ == "__main__":
    sys.exit(main())
