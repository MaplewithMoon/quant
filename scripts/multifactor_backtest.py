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


def style_report(R, close, style_panels, factor_rets):
    """风格分解段落：**残差 alpha 为主口径**，混口径降为参考行

    A3 报的「Alpha +2.40%，p=0.646」之所以没意义，是因为它混着风格暴露 ——
    "选股能力"与"小盘 beta"在那个数字里分不开。这里把风格解释掉再看残差。
    """
    from analytics.attribution import multi_factor_exposure
    from analytics.style import residual_alpha, style_section

    hold = R["full"]["result"].holdings
    if hold is None or hold.empty or not style_panels:
        return None
    ret = close.pct_change()
    port_ret = (hold.reindex(columns=ret.columns).fillna(0.0)
                * ret.reindex(columns=hold.columns).fillna(0.0)).sum(axis=1)
    expo = multi_factor_exposure(hold, style_panels)
    st = residual_alpha(port_ret, expo, factor_rets)
    if not st.get("residual", {}).get("n"):
        return None
    return style_section(st)


def run_one(panel, mask, score, spec_kwargs, engine_kwargs, warmup_start,
            eval_start, eval_end, capital, risk_config=None, ctx=None):
    """跑一段回测并截断到评估窗口

    ⚠️ `risk_config` 传的是**配置**而不是已建好的 RiskManager：
    `PortfolioRiskManager` 内部有 `_prev_weights` 状态（换手类规则要用它算
    "相对上期的变动"），同一个实例跨 训练集/测试集/全区间 复用会让状态串味 ——
    测试集的第一次调仓会拿训练集末尾的持仓当"上期持仓"，凭空触发换手限制。
    所以每段回测都在这里现建一个。
    """
    sub = {k: v.loc[(v.index >= pd.Timestamp(warmup_start))
                    & (v.index <= pd.Timestamp(eval_end))]
           for k, v in panel.items() if isinstance(v, pd.DataFrame)}
    sc = score.loc[(score.index >= pd.Timestamp(warmup_start))
                   & (score.index <= pd.Timestamp(eval_end))]
    mk = mask.loc[(mask.index >= pd.Timestamp(warmup_start))
                  & (mask.index <= pd.Timestamp(eval_end))]
    tw = build_target_weights(sc, mk, **spec_kwargs)
    eng = PortfolioBacktestEngine(**engine_kwargs)
    from risk import build_risk_manager
    res = eng.run(sub, tw, risk_manager=build_risk_manager(risk_config),
                  context=ctx) if ctx is not None else \
        eng.run(sub, tw, risk_manager=build_risk_manager(risk_config))
    ts = pd.Timestamp(eval_start)
    eq = res.equity[res.equity.index >= ts]
    tr = res.trades
    if tr is not None and not tr.empty and "timestamp" in tr.columns:
        tr = tr[pd.to_datetime(tr["timestamp"]) >= ts]
    m = Metrics.compute(eq, tr) if not eq.empty else Metrics()
    # ⚠️ 把结果对象的 equity/metrics 换成**评估窗口**（截掉预热期）的那份：
    # 引擎内部算的 metrics 覆盖 warmup_start~eval_end 全程，直接拿去渲染会
    # 报出与脚本其它地方不一致的数字。统一口径后再交给渲染层。
    res.equity = eq
    res.metrics = m
    return {"equity": eq, "trades": tr, "metrics": m, "result": res,
            "target": tw, "ledger_gap": res.ledger_gap,
            "lookahead_violations": res.lookahead_violations,
            "risk_events": res.risk_events}


def main():
    ap = argparse.ArgumentParser(description="多因子选股策略回测（基本面因子）")
    ap.add_argument("--start", default="2017-01-01")
    ap.add_argument("--split", default="2022-01-01")
    ap.add_argument("--end", default="2025-12-31")
    ap.add_argument("--warmup-days", type=int, default=609,
                    help="预热自然日；⚠️ 改这个值会导致面板缓存未命中（重扫约 8 分钟）")
    ap.add_argument("--benchmark", default=BENCH_CODE)
    ap.add_argument("--capital", type=float, default=1_000_000)
    # 执行参数（成交时点 / 滑点 / 冲击模型 / 全部费用）由公共层统一提供 ——
    # 原先散落在各脚本里各写一遍，抄漏一处就永久缺失（见 execution/setup.py）
    from execution.setup import add_execution_args
    add_execution_args(ap)
    ap.add_argument("--min-listed-days", type=int, default=120)
    ap.add_argument("--min-amount", type=float, default=5e7)
    ap.add_argument("--rebalance", default="M")
    ap.add_argument("--n-hold-grid", default="30,50,80",
                    help="训练集上搜索的持股数（逗号分隔）")
    ap.add_argument("--factor-table", default="results/fundamental/factor_ic.csv",
                    help="fundamental_ic.py 产出的因子统计表（用于预注册规则选因子）")
    ap.add_argument("--cache-dir", default=".cache/panel")
    ap.add_argument("--lag-days", type=int, default=1)
    ap.add_argument("--horizon", type=int, default=20,
                    help="因子门禁用的前瞻收益窗口（交易日）")
    ap.add_argument("--no-gate", action="store_true",
                    help="跳过因子有效性门禁。⚠️ 跳过时报告头部会显式标注"
                         "「未执行门禁」，不允许静默略过")
    ap.add_argument("--outdir", default="results/multifactor")
    args = ap.parse_args()

    t0 = time.time()
    os.makedirs(args.outdir, exist_ok=True)
    start, split, end = (pd.Timestamp(args.start), pd.Timestamp(args.split),
                         pd.Timestamp(args.end))
    warmup = (start - pd.Timedelta(days=int(args.warmup_days))).strftime("%Y-%m-%d")
    setup_font()

    from execution.setup import build_execution
    engine_kwargs = dict(initial_capital=args.capital, **build_execution(args))

    # ⚠️ C6：这个脚本原先**完全没有接冲击成本模型**，而且 `--slippage` 默认 0.0。
    # 后果是：把 --capital 从 100 万调到 1 亿，回测结果**一分钱都不会变** ——
    # 也就是说 A4「turnover_20 的容量」根本无法用这个脚本验证。
    # 现在按其他回测脚本的口径统一：fixed 滑点（默认 1bp），要评估容量就
    # 换成 --impact-model sqrt（冲击 ∝ 下单量/成交量，会随资金规模放大）。
    # 组合层风控配置（None = 不启用，行为与改动前一致）
    risk_config = {
        "max_weight": args.max_weight,
        "max_drawdown_pct": args.max_drawdown,
        "min_exposure": args.derisk_min_exposure,
        "max_turnover": args.max_turnover,
    }
    if not any(risk_config.values()):
        risk_config = None

    banner("多因子选股策略  |  训练/测试集分离  |  全部因子行业+市值中性")
    print(f"  训练集 {args.start} ~ {args.split}     测试集 {args.split} ~ {args.end}")
    print(f"  资金 {args.capital:,.0f}   滑点 {args.slippage:.2%}   "
          f"冲击模型 {args.impact_model}"
          + (f"(k={args.impact_k})" if args.impact_model == "sqrt" else "")
          + f"   调仓 {args.rebalance}")
    if risk_config:
        print(f"  组合风控: 单票上限 {args.max_weight or '—'}   "
              f"回撤降仓 {args.max_drawdown or '—'}   "
              f"换手上限 {args.max_turnover or '—'}")
    else:
        print("  组合风控: 未启用（--max-weight / --max-drawdown / --max-turnover 可开）")

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

    # 行业归属：**优先用 PIT**（index_member_all 建的逐日行业面板）
    #
    # 用当前快照（stock_basic.industry）做历史中性化是**前视**：全库有 1,646 只
    # 股票换过行业（最多 6 段），拿"它现在属于哪个行业"去中性化它十年前的因子值，
    # 等于把未来信息提前用了。PIT 面板用 `in_date`/`out_date` 还原当时的归属。
    from database.industry import has_pit_data, industry_pit_panel
    if has_pit_data():
        panel_ind = industry_pit_panel(reb, close.columns)
        print(f"  行业归属: PIT 面板 {panel_ind.shape[0]} 调仓日 × "
              f"{panel_ind.shape[1]} 只（{panel_ind.attrs['source']}）"
              f"  覆盖率 {panel_ind.attrs['pit_coverage']:.1%}"
              f"  快照兜底 {panel_ind.attrs['snapshot_fallback_ratio']:.2%}")
    else:
        from analytics.attribution import _industry_series, load_industry_map
        panel_ind = _industry_series(load_industry_map(), close.columns)
        print("  ⚠ 行业归属: **当前快照**（对换过行业的股票构成前视）")
        print("     补 PIT 数据: python scripts/download_frozen_tushare.py "
              "--only sw_member")
    mv = panel.get("total_mv")

    # ---- 回测上下文：用于**自动匹配已知缺陷**（见 database/defects.py）
    # 行业来源如实填：用 PIT 就填 "pit"，用当前快照就填 "snapshot" ——
    # 填错会让渲染层漏标/误标 B7（前视）那条。
    ind_src = "pit" if has_pit_data() else "snapshot"
    bt_ctx = {"start": args.start, "end": args.end,
              # 本脚本股票池是**全市场**（不挂指数），所以 index_code=None；
              # benchmark 只是对照基准，不是选股池，别混
              "index_code": None,
              "exchanges": ("SSE", "SZSE"),
              "codes": list(close.columns),
              "industry_source": ind_src}

    # 早期一行提示（完整附注在最后的统一报告里，见 analytics/result_report.py）
    from database.defects import match_defects
    _def = match_defects(bt_ctx)
    print(f"  ⚠ 已知缺陷: 本次触及 {len(_def)} 项 "
          f"({', '.join(d.key for d in _def) or '无'})，详见结尾报告")

    # ---- 风格面板（PIT）：用于风格分解与**残差 alpha**（T1·④）----
    # size 取面板逐日 total_mv（天然 PIT）；value 取 bp，已按 ann_date 对齐，
    # 调仓日之间前向填充 —— 绝不用"当前"财报回填历史
    from analytics.style import build_style_panels, cross_section_factor_returns
    style_panels = build_style_panels(panel, close.index, close.columns,
                                      fundamentals=factors)
    _factor_rets = (cross_section_factor_returns(close.pct_change(), style_panels,
                                                 mask=mask)
                    if style_panels else None)
    print(f"  风格面板: {list(style_panels)}"
          + (f"   因子收益 {_factor_rets.shape[0]} 日"
             if _factor_rets is not None else ""))

    # ---- 因子有效性门禁（T1·③）----
    # 已有的选因子规则（combine_by_rule）三个条件**全是样本内**，样本外没有任何
    # 自动检查 —— A8 说的正是这种："样本外 IC 显著但多空为负、单调性≈0"。
    # ⚠️ 门禁**只标记不剔除**：静默剔除会造出"被审查过的幸存者因子库"，
    #    与股票池的幸存者偏差是同构的病。
    _gate = {}
    if args.no_gate:
        print("  ⚠ 因子门禁: **未执行**（--no-gate）—— 结果未经样本外有效性检查")
    else:
        from factors.gate import gate_check, gate_summary
        from factors.panel import forward_returns
        _fwd = forward_returns(adjusted_close(panel), periods=args.horizon)
        _reps, _scored = [], {}
        for _name, _f in factors.items():
            if _f is None or _f.empty:
                continue
            _r = gate_check(_f, _fwd, split_point=args.split, label=_name)
            _reps.append(_r)
            _scored[_name] = _r
        _gate = gate_summary(_reps)
        if _gate:
            print(f"  因子门禁: {_gate['status']} — {_gate['summary']}")
            for _x in _gate["reasons"][:5]:
                print(f"      ✗ {_x}")

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

    # 门禁结果**只标记不剔除**：静默剔除会造出"被审查过的幸存者因子库"，
    # 与股票池的幸存者偏差是同构的病。
    for _lab, _names in sets:
        _bad = [_n for _n in _names
                if _scored.get(_n, {}).get("status") in ("FAIL", "WARN")]
        if _bad:
            print(f"  ⚠ {_lab} 含未通过门禁的因子（**保留，仅标记**）: {_bad}")

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
                        engine_kwargs, warmup, start, split, args.capital,
                        risk_config=risk_config, ctx=bt_ctx)
            m = r["metrics"]
            print(f"    持股 {n:>3}  内 夏普 {fmt_num(m.sharpe_ratio)}  "
                  f"年化 {fmt_pct(m.annual_return)}  回撤 {fmt_pct(m.max_drawdown)}")
            if best is None or m.sharpe_ratio > best[1].sharpe_ratio:
                best = (n, m)
        n_best = best[0]
        tr = run_one(panel, mask, score,
                     dict(n_hold=n_best, weighting="equal", rebalance=args.rebalance),
                     engine_kwargs, warmup, start, split, args.capital,
                     risk_config=risk_config, ctx=bt_ctx)
        te = run_one(panel, mask, score,
                     dict(n_hold=n_best, weighting="equal", rebalance=args.rebalance),
                     engine_kwargs, warmup, split, end, args.capital,
                     risk_config=risk_config, ctx=bt_ctx)
        full = run_one(panel, mask, score,
                       dict(n_hold=n_best, weighting="equal", rebalance=args.rebalance),
                       engine_kwargs, warmup, start, end, args.capital,
                       risk_config=risk_config, ctx=bt_ctx)
        print(f"    >>> 训练集最优持股数 {n_best}")
        print(f"    样本内: 夏普 {fmt_num(tr['metrics'].sharpe_ratio)} "
              f"年化 {fmt_pct(tr['metrics'].annual_return)} "
              f"回撤 {fmt_pct(tr['metrics'].max_drawdown)}")
        print(f"    样本外: 夏普 {fmt_num(te['metrics'].sharpe_ratio)} "
              f"年化 {fmt_pct(te['metrics'].annual_return)} "
              f"回撤 {fmt_pct(te['metrics'].max_drawdown)}")
        # 前视自检：引擎能保证"成交不早于决策"。>0 就说明结果不可信，必须显眼。
        la = full["lookahead_violations"]
        if la:
            print(f"    ⚠⚠ 前视自检发现 {la} 笔成交早于决策日 —— 本次结果不可信！")
        else:
            print("    ✓ 前视自检通过（成交均不早于决策日）")
        if full["risk_events"]:
            print(f"    风控触发 {len(full['risk_events'])} 次（"
                  f"{full['risk_events'][0]['rule']} 等）")
        rows.append({
            "组合": label, "因子": ",".join(names), "持股数": n_best,
            "内年化": tr["metrics"].annual_return, "内夏普": tr["metrics"].sharpe_ratio,
            "内回撤": tr["metrics"].max_drawdown,
            "外年化": te["metrics"].annual_return, "外夏普": te["metrics"].sharpe_ratio,
            "外回撤": te["metrics"].max_drawdown,
            "外波动": te["metrics"].annual_volatility,
            "账目差额": full["ledger_gap"],
            "前视违规": la,
            "风控触发": len(full["risk_events"]),
        })
        results[label] = {"train": tr, "test": te, "full": full,
                          "names": names, "n_hold": n_best, "score": score}

    # 把门禁结果挂到**回测结果对象**上，报告头部会显示（⑤ 的载体直接生效）
    for _r in results.values():
        _r["full"]["result"].gate = _gate

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
                # **统一渲染**：缺陷附注 / 前视自检 / 风控触发 / 被拦委托
                # 都由 analytics/result_report.py 出，脚本不再各印一套（T1·⑤）
                _extra = []
                _st = style_report(R, close, style_panels, _factor_rets)
                if _st:
                    _extra += _st
                # 成交时点对照：**默认跑**（不是"可以传开关跑两版"）。
                # 不跑就永远不知道结论对成交时点有多敏感，F1 的归因也无从谈起。
                if not args.no_fill_compare:
                    from backtest.fill_compare import compare_fill_timing
                    from risk import build_risk_manager
                    _evp = {k: v.loc[(v.index >= pd.Timestamp(warmup))
                                     & (v.index <= pd.Timestamp(args.end))]
                            for k, v in panel.items() if isinstance(v, pd.DataFrame)}
                    _cmp = compare_fill_timing(
                        lambda **kw: PortfolioBacktestEngine(**kw),
                        _evp, R["full"]["target"],
                        base_kwargs=dict(engine_kwargs),
                        run_kwargs={"risk_manager": build_risk_manager(risk_config),
                                    "context": bt_ctx})
                    if _cmp["text"]:
                        _extra += _cmp["text"].split("\n")
                print(r["result"].render(
                    title=f"{label}（{R['names']}，持股 {R['n_hold']}）"
                          f"  {args.start} ~ {args.end}",
                    extra_sections=_extra))
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
        json.dump({            "区间": {"训练集": [args.start, args.split], "测试集": [args.split, args.end]},
            "因子集合": {k: v["names"] for k, v in results.items()},
            "持股数": {k: v["n_hold"] for k, v in results.items()},
            "对照": cmp.replace({np.nan: None}).to_dict("records"),
            # 已知缺陷与护栏状态一并落盘：没有这些，报告数字脱离前提就没法解读
            "已知缺陷": [{"key": d.key, "标题": d.title, "严重度": d.severity,
                          "需额外数据": d.needs_data} for d in _def],
            "交易成本": {"滑点": args.slippage, "冲击模型": args.impact_model,
                         "impact_k": args.impact_k if args.impact_model == "sqrt" else None},
            "组合风控": risk_config,
        }, fh, ensure_ascii=False, indent=2, default=str)
    # ---- 结果快照（T3·E4）：记下 commit + 数据指纹 + 参数 + 结果哈希 ----
    # 这个项目已两次因数据修复而结论变化，但没有机制知道**哪些结论过期了**。
    # 快照落到 outdir，`analytics.snapshot.stale_snapshots()` 能直接报出来。
    try:
        from analytics.snapshot import write_snapshot
        _first = next(iter(results.values()), None)
        if _first:
            _p = write_snapshot(
                _first["full"]["result"], args.outdir,
                label=f"多因子 {args.start}~{args.end}",
                params={"start": args.start, "split": args.split, "end": args.end,
                        "capital": args.capital, "slippage": args.slippage,
                        "impact_model": args.impact_model,
                        "fill_timing": args.fill_timing,
                        "rebalance": args.rebalance,
                        "risk_config": risk_config},
                extra={"组合": {k: v["names"] for k, v in results.items()}})
            print(f"  结果快照 -> {_p}")
    except Exception as _e:
        print(f"  （结果快照写入失败，不影响回测: {type(_e).__name__}）")

    print(f"\n  产出 -> {args.outdir}/")
    banner(f"完成，总用时 {time.time()-t0:.0f}s")
    return 0


if __name__ == "__main__":
    sys.exit(main())
