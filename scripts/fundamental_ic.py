# -*- coding: utf-8 -*-
"""基本面因子 IC 评估（训练集 / 测试集分开报）

【为什么先做这一步，而不是直接搭策略】
上一轮的教训（docs/板块轮动改进报告.md）：在样本内选出来的"最优改进"，
样本外几乎全部失效。策略回测把"信号有没有信息"和"组合怎么构建、费用多少、
约束怎么拦"混在一起，很难归因。

所以这一轮先只回答一个问题：**这个因子在横截面上有没有预测力？**
    指标 = 调仓日横截面 RankIC，训练集与测试集分别统计
    - IC 均值、ICIR、t 值、p 值、正 IC 占比
    - 分层（N 组）多空价差与单调性
    只有在**测试集上也站得住**的因子，才值得进入策略层。

用法:
    python scripts/fundamental_ic.py --start 2017-01-01 --split 2022-01-01 --end 2025-12-31
    python scripts/fundamental_ic.py ... --neutralize     # 行业+市值中性（强烈建议）
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

from analytics.performance import t_pvalue
from backtest.panel_data import load_price_panel
from factors.evaluation import ic_series, quantile_returns
from factors.fundamental import FACTOR_META, load_all_factors, factor_summary
from factors.panel import adjusted_close, forward_returns
from portfolio.construction import rebalance_dates
from universe import UniverseSpec, build_universe


def banner(text, ch="=", width=94):
    print()
    print(ch * width)
    print(text)
    print(ch * width)


def neutralize_rows(f: pd.Series, ind: pd.Series, size: pd.Series,
                    n_size: int = 5) -> pd.Series:
    """把因子在**行业内**和**市值组内**去均值（秩中性化）

    为什么必须做：营收增速、ROE 这类指标有很强的行业属性
    （白酒天然 90% 毛利率、银行天然高杠杆），不中性化的话
    "因子有效"很可能是"这个行业这两年好"。市值同理 —— 小盘股成长性天然更高。

    做法：先取横截面百分位秩，再减去同行业/同市值组的均值。
    用秩而不是原值，是为了不被 np_yoy 那种 P5 = −245% 的极端值带偏。
    """
    r = f.rank(pct=True)
    if ind is not None and len(ind):
        g = ind.reindex(r.index)
        m = r.groupby(g).transform("mean")
        r = r - m
    if size is not None and len(size):
        s = size.reindex(r.index)
        try:
            q = pd.qcut(s.rank(method="first"), n_size, labels=False)
            r = r - r.groupby(q).transform("mean")
        except (ValueError, IndexError):
            pass
    return r


def prep_factor(name: str, factor: pd.DataFrame, reb: pd.DatetimeIndex,
                mask: pd.DataFrame, ind_map=None, mv=None,
                neutralize: bool = False, min_count: int = 30) -> pd.DataFrame:
    """把因子整理成"调仓日 × 股票"的打分矩阵（统一成越大越好 + 可选中性化）"""
    direction = FACTOR_META.get(name, ("", 1))[1]
    f = factor.where(mask) if mask is not None else factor
    if direction < 0:
        f = -f
    if not neutralize:
        return f.reindex(index=reb)
    vals = {}
    for d in reb:
        if d not in f.index:
            continue
        row = f.loc[d]
        if int(row.notna().sum()) < min_count:
            continue
        sz = mv.loc[d] if (mv is not None and d in mv.index) else None
        vals[d] = neutralize_rows(row, ind_map, sz)
    return pd.DataFrame(vals).T if vals else pd.DataFrame(index=reb)


def stats_of(ic: pd.Series) -> dict:
    ic = ic.dropna()
    n = len(ic)
    if n < 2:
        return {"IC": np.nan, "ICIR": np.nan, "t": np.nan, "p": np.nan,
                "正IC占比": np.nan, "期数": n}
    mean, std = float(ic.mean()), float(ic.std(ddof=1))
    icir = mean / std if std > 0 else np.nan
    t = icir * np.sqrt(n) if std > 0 else np.nan
    return {"IC": mean, "ICIR": icir, "t": t,
            "p": t_pvalue(t, n - 1) if np.isfinite(t) else np.nan,
            "正IC占比": float((ic > 0).mean()), "期数": n}


def main():
    ap = argparse.ArgumentParser(description="基本面因子 IC 评估（样本内外分开）")
    ap.add_argument("--start", default="2017-01-01")
    ap.add_argument("--split", default="2022-01-01")
    ap.add_argument("--end", default="2025-12-31")
    ap.add_argument("--horizon", type=int, default=21, help="前瞻收益的交易日数")
    ap.add_argument("--rebalance", default="M")
    ap.add_argument("--neutralize", action="store_true",
                    help="行业+市值中性化后再算 IC（推荐）")
    ap.add_argument("--n-quantiles", type=int, default=5)
    ap.add_argument("--min-amount", type=float, default=5e7)
    ap.add_argument("--min-listed-days", type=int, default=120)
    ap.add_argument("--warmup-days", type=int, default=609,
                    help="预热自然日数。⚠️ 面板缓存按 (起始日, 结束日) 做键，"
                         "改这个值会导致缓存未命中并重扫全量（约 8 分钟）")
    ap.add_argument("--cache-dir", default=".cache/panel")
    ap.add_argument("--outdir", default="results/fundamental")
    ap.add_argument("--lag-days", type=int, default=1,
                    help="财报公告后多少天才能用（point-in-time 缓冲）")
    args = ap.parse_args()

    t0 = time.time()
    os.makedirs(args.outdir, exist_ok=True)
    start, split, end = (pd.Timestamp(args.start), pd.Timestamp(args.split),
                         pd.Timestamp(args.end))
    warmup = (start - pd.Timedelta(days=int(args.warmup_days))).strftime("%Y-%m-%d")

    banner("基本面因子 IC 评估  |  训练集与测试集分开统计")
    print(f"  训练集 {args.start} ~ {args.split}     测试集 {args.split} ~ {args.end}")
    print(f"  前瞻收益 {args.horizon} 交易日   调仓频率 {args.rebalance}   "
          f"中性化 {'开' if args.neutralize else '关'}")
    print(f"  财报生效滞后 {args.lag_days} 天（point-in-time 缓冲，"
          f"对齐键是 ann_date 不是 end_date）")

    # ---------- 1. 数据 ----------
    banner("1. 加载数据", "-")
    panel = load_price_panel(warmup, args.end, cache_dir=(args.cache_dir or None))
    if not panel:
        raise SystemExit("未取到行情数据")
    close = adjusted_close(panel)
    mask = build_universe(panel, UniverseSpec(
        min_listed_days=args.min_listed_days, min_amount=args.min_amount))
    print(f"  价量面板: {close.shape[0]} 交易日 × {close.shape[1]} 只股票")
    print(f"  股票池  : 每日可选中位 {mask.sum(axis=1).median():.0f} 只")

    reb_all = rebalance_dates(close.index, args.rebalance)
    reb = pd.DatetimeIndex(reb_all[(reb_all >= start) & (reb_all <= end)])
    print(f"  调仓日  : {len(reb)} 期（{str(reb[0])[:10]} ~ {str(reb[-1])[:10]}）")

    t1 = time.time()
    factors = load_all_factors(panel, reb, close.columns,
                               lag_days=args.lag_days, verbose=True)
    print(f"  因子    : {len(factors)} 个（加载 + PIT 对齐 {time.time()-t1:.0f}s）")
    print()
    print(factor_summary(factors).to_string(index=False))

    fwd = forward_returns(close, periods=args.horizon, lag=1).loc[reb] \
        .where(mask.loc[reb])
    print(f"\n  前瞻窗口 {args.horizon} 日，有效样本中位 "
          f"{fwd.notna().sum(axis=1).median():.0f} 只/期")

    ind_map = mv = None
    if args.neutralize:
        # 注意用 analytics.attribution 的包装版：它把 ts_code 转成 6 位 code，
        # 直接拿 database.loader.load_industry_map() 的原表会因为只有 ts_code 而报错
        from analytics.attribution import load_industry_map as _load_ind
        from analytics.attribution import _industry_series
        ind_map = _industry_series(_load_ind(), close.columns)
        mv = panel.get("total_mv")
        if mv is None:
            raise SystemExit("中性化需要 total_mv（请用 with_valuation 面板）")

    # ---------- 2. 逐因子评估（IC 序列只算一次，再按区间切片）----------
    banner("2. 因子 IC（调仓日横截面 RankIC）"
           + ("  行业+市值中性" if args.neutralize else ""), "-")
    rows = []
    for name, f in factors.items():
        t2 = time.time()
        sc = prep_factor(name, f, reb, mask, ind_map, mv, args.neutralize)
        ic = ic_series(sc, fwd, method="spearman", min_count=30)
        a, b, full = stats_of(ic.loc[ic.index < split]), \
            stats_of(ic.loc[ic.index >= split]), stats_of(ic)

        # 分层：样本内 / 样本外分别算多空价差与单调性。
        # ⚠️ 选因子只能用**样本内**这两个数，所以必须都报出来 ——
        # 只报样本外的多空/单调性，等于变相用测试集选因子。
        qr = quantile_returns(sc, fwd, n_quantiles=args.n_quantiles,
                              direction=1, min_count=30)

        def _ls_mono(sub):
            if qr is None or qr.empty or sub.empty:
                return np.nan, np.nan
            cols = list(qr.columns)
            ls = float((sub[cols[-1]] - sub[cols[0]]).mean() * (252 / args.horizon))
            means = sub.mean()
            mono = float(pd.Series(means.values).rank().corr(
                pd.Series(np.arange(len(means))).rank()))
            return ls, mono

        in_ls, in_mono = _ls_mono(qr.loc[qr.index < split] if qr is not None
                                  and not qr.empty else pd.DataFrame())
        out_ls, out_mono = _ls_mono(qr.loc[qr.index >= split] if qr is not None
                                    and not qr.empty else pd.DataFrame())

        rows.append({
            "因子": name, "中文名": FACTOR_META.get(name, ("", 0))[0],
            "方向": FACTOR_META.get(name, ("", 1))[1],
            "覆盖": float(sc.notna().mean().mean()),
            "内IC": a["IC"], "内t": a["t"], "内p": a["p"], "内期数": a["期数"],
            "内ICIR": a["ICIR"], "内多空年化": in_ls, "内单调性": in_mono,
            "外IC": b["IC"], "外t": b["t"], "外p": b["p"],
            "外ICIR": b["ICIR"], "外正IC占比": b["正IC占比"], "外期数": b["期数"],
            "外多空年化": out_ls, "外单调性": out_mono,
            "全IC": full["IC"], "全t": full["t"],
            "_秒": time.time() - t2,
        })
        print(f"  {name:<14} 内IC {a['IC']:+.4f}(t{a['t']:+.2f}) 内多空 {in_ls:+.1%}  "
              f"| 外IC {b['IC']:+.4f}(t{b['t']:+.2f}) 外多空 {out_ls:+.1%}   "
              f"({time.time()-t2:.1f}s)")

    tab = pd.DataFrame(rows).sort_values("外IC", key=lambda s: s.abs(),
                                         ascending=False).reset_index(drop=True)
    tab.to_csv(os.path.join(args.outdir, "factor_ic.csv"),
               index=False, encoding="utf-8-sig")

    # ---------- 3. 汇总表 ----------
    banner("3. 因子 IC 汇总（按样本外 |IC| 排序，仅供阅读）", "-")
    show = tab[["因子", "中文名", "内IC", "内t", "内p", "内多空年化", "内单调性",
                "外IC", "外t", "外p", "外多空年化", "外单调性"]].copy()
    for c in ("内IC", "外IC"):
        show[c] = show[c].map(lambda x: f"{x:+.4f}" if pd.notna(x) else "-")
    for c in ("内t", "外t", "内单调性", "外单调性"):
        show[c] = show[c].map(lambda x: f"{x:+.2f}" if pd.notna(x) else "-")
    for c in ("内p", "外p"):
        show[c] = show[c].map(lambda x: f"{x:.3f}" if pd.notna(x) else "-")
    for c in ("内多空年化", "外多空年化"):
        show[c] = show[c].map(lambda x: f"{x:+.1%}" if pd.notna(x) else "-")
    print(show.to_string(index=False))

    # ---------- 4. 预注册规则选因子 ----------
    # ⚠️ 这一步是整份评估的关键：**规则必须只用样本内数据**，
    # 样本外的数字只用于事后验证，不允许回头改规则。
    banner("4. 按*样本内*规则选因子（预注册，不看样本外）", "-")
    rule = ((tab["内p"] < 0.05) & (tab["内多空年化"] > 0) & (tab["内单调性"] > 0.5))
    picked = tab[rule].sort_values("内IC", ascending=False)
    print("  规则：样本内 p<0.05  且  样本内多空>0  且  样本内单调性>0.5")
    print(f"  入选 {len(picked)} 个: {list(picked['因子'])}")
    if not picked.empty:
        cols = ["因子", "内IC", "内t", "内多空年化", "内单调性",
                "外IC", "外t", "外p", "外多空年化", "外单调性"]
        s2 = picked[cols].copy()
        for c in ("内IC", "外IC"):
            s2[c] = s2[c].map(lambda x: f"{x:+.4f}")
        for c in ("内t", "外t", "内单调性", "外单调性"):
            s2[c] = s2[c].map(lambda x: f"{x:+.2f}")
        s2["外p"] = s2["外p"].map(lambda x: f"{x:.3f}")
        for c in ("内多空年化", "外多空年化"):
            s2[c] = s2[c].map(lambda x: f"{x:+.1%}")
        print()
        print("  —— 入选因子的样本外表现（这才是干净的样本外验证）——")
        print(s2.to_string(index=False))
        surv = picked[picked["外p"] < 0.05]
        print(f"\n  其中样本外仍显著 (p<0.05): {len(surv)}/{len(picked)} 个 -> "
              f"{list(surv['因子'])}")
        print(f"  样本外多空仍为正        : "
              f"{int((picked['外多空年化'] > 0).sum())}/{len(picked)} 个")
        picked.to_csv(os.path.join(args.outdir, "selected_factors.csv"),
                      index=False, encoding="utf-8-sig")

    # ---------- 5. 全部因子的事后对照（仅供参考，不用于选择）----------
    banner("5. 事后对照（用测试集看，不能用于选因子）", "-")
    sig_in = tab[tab["内p"] < 0.05]
    sig_out = tab[tab["外p"] < 0.05]
    both = sorted(set(sig_in["因子"]) & set(sig_out["因子"]))
    same = tab[(np.sign(tab["内IC"]) == np.sign(tab["外IC"]))
               & tab["内IC"].notna() & tab["外IC"].notna()]
    print(f"  样本内显著 (p<0.05): {len(sig_in)} 个 -> {list(sig_in['因子'])}")
    print(f"  样本外显著 (p<0.05): {len(sig_out)} 个 -> {list(sig_out['因子'])}")
    print(f"  两个区间都显著     : {len(both)} 个 -> {both if both else '无'}")
    print(f"  样本内外 IC 同号   : {len(same)}/{len(tab)} 个")
    print()
    print("  判读标准：")
    print("    1) 只看样本内显著的因子 = 一定会踩过拟合")
    print("    2) 样本外 IC 同号 且 |t| > 2 才算站得住")
    print("    3) 月度 IC 只有 40~60 期，t 值本身不稳 —— 宁可严格")
    print("    4) IC 显著但多空为负 / 单调性差 = 因子只在极端分位有效，不能直接做组合")

    with open(os.path.join(args.outdir, "ic_report.json"), "w", encoding="utf-8") as fh:
        json.dump({
            "区间": {"训练集": [args.start, args.split], "测试集": [args.split, args.end]},
            "前瞻收益交易日": args.horizon, "中性化": bool(args.neutralize),
            "财报滞后天数": args.lag_days, "调仓期数": int(len(reb)),
            "因子": tab.replace({np.nan: None}).to_dict("records"),
        }, fh, ensure_ascii=False, indent=2, default=str)
    print(f"\n  结果 -> {args.outdir}/factor_ic.csv, ic_report.json")
    banner(f"完成，总用时 {time.time()-t0:.0f}s")
    return 0


if __name__ == "__main__":
    sys.exit(main())
