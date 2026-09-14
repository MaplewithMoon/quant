# -*- coding: utf-8 -*-
"""因子有效性评估 CLI

【用法】
    # 列出所有已注册因子
    python scripts/factor_eval.py --list

    # 评估单个因子
    python scripts/factor_eval.py --factor mom_20 --start 2020-01-01 --end 2024-12-31

    # 批量评估全部因子（按 |IC| 排序）
    python scripts/factor_eval.py --all --start 2020-01-01 --end 2024-12-31 --limit 800

    # 自定义持有期 / 分层数
    python scripts/factor_eval.py --factor rev_5 --periods 5 --quantiles 10

【说明】
- 默认股票池为"区间内日均成交额最高的 N 只"（--limit），探索阶段可控制计算量；
  设为 0 表示用全部股票。
- 因子值只用当日及以前数据；前瞻收益从 t+1 起算，二者不重叠（无未来函数）。
"""
import sys, io, argparse
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
sys.path.insert(0, ".")

import numpy as np
import pandas as pd

from factors import (load_panel, get_factor, evaluate, list_factors,
                     adjusted_close)
from factors.panel import forward_returns
from factors.evaluation import ic_series, ic_stats, quantile_returns, monotonicity


def main():
    p = argparse.ArgumentParser(description="因子有效性评估")
    p.add_argument("--factor", default="", help="因子名（见 --list）")
    p.add_argument("--all", action="store_true", help="评估全部已注册因子")
    p.add_argument("--list", action="store_true", help="列出所有因子后退出")
    p.add_argument("--start", default="2020-01-01")
    p.add_argument("--end", default="2024-12-31")
    p.add_argument("--limit", type=int, default=800,
                   help="股票池取日均成交额前 N 只（0=全部）")
    p.add_argument("--periods", type=int, default=1, help="前瞻收益持有天数")
    p.add_argument("--quantiles", type=int, default=5, help="分层数")
    p.add_argument("--min-count", type=int, default=20, help="当日最少有效股票数")
    p.add_argument("--no-decay", action="store_true", help="跳过因子衰减计算")
    p.add_argument("--out", default="", help="把汇总表保存为 CSV")
    args = p.parse_args()

    if args.list:
        df = list_factors()
        print(f"已注册因子 {len(df)} 个：")
        print(df.to_string(index=False))
        return

    names = sorted(__import__("factors").FACTORS) if args.all else [args.factor]
    names = [n for n in names if n]
    if not names:
        raise SystemExit("请用 --factor 指定因子，或 --all 评估全部（--list 查看清单）")

    print("=" * 96)
    print(f"因子评估  {args.start} ~ {args.end}  股票池前 {args.limit or '全部'} 只"
          f"  持有期 {args.periods} 日  分层 {args.quantiles}")
    print("=" * 96)

    panel = load_panel(args.start, args.end, limit=(args.limit or None))
    if not panel:
        raise SystemExit("未取到数据")
    close = adjusted_close(panel)
    print(f"面板: {close.shape[0]} 个交易日 × {close.shape[1]} 只股票")
    fwd = forward_returns(close, periods=args.periods, lag=1)

    rows = []
    reports = {}
    for name in names:
        try:
            fac = get_factor(name)
            rep = evaluate(fac, panel, periods=args.periods,
                           n_quantiles=args.quantiles,
                           min_count=args.min_count,
                           with_decay=not args.no_decay)
        except Exception as e:
            print(f"  !! {name} 失败: {type(e).__name__}: {e}")
            continue
        reports[name] = rep
        d = rep.stats
        row = {"因子": name, "方向": "大→多" if rep.direction > 0 else "小→多",
               "IC均值": d["IC均值"], "ICIR": d["ICIR"], "t值": d["t值"],
               "正IC占比": d["正IC占比"], "样本天数": d["样本天数"]}
        if not rep.quantile.empty:
            n = len([c for c in rep.quantile.columns if c.startswith("Q")])
            row["多空均值"] = rep.quantile["多空"].mean()
            row["单调性"] = monotonicity(rep.quantile, n)
        if not rep.turnover.empty:
            row["分层换手"] = rep.turnover.mean()
        row["均值中位一致"] = "否 ⚠" if rep.mean_median_diverges() else "是"
        rows.append(row)

    if not rows:
        raise SystemExit("没有成功评估的因子")

    df = pd.DataFrame(rows)
    df["_absIC"] = df["IC均值"].abs()
    df = df.sort_values("_absIC", ascending=False).drop(columns="_absIC")

    # 单个因子：打印完整报告
    if len(names) == 1 and names[0] in reports:
        print()
        print(reports[names[0]].summary())

    # 汇总表
    print()
    print("=" * 96)
    print("汇总（按 |IC均值| 降序）")
    print("=" * 96)
    show = df.copy()
    for c in ("IC均值", "正IC占比", "多空均值", "分层换手"):
        if c in show.columns:
            show[c] = show[c].map(lambda x: f"{x:.4f}" if c == "IC均值" else
                                  (f"{x:.2%}" if pd.notna(x) else "-"))
    for c in ("ICIR", "t值", "单调性"):
        if c in show.columns:
            show[c] = show[c].map(lambda x: f"{x:.3f}" if pd.notna(x) else "-")
    print(show.to_string(index=False))

    valid = int((df["IC均值"].abs() > 0.03).sum())
    print(f"\n|IC|>0.03 的因子: {valid}/{len(df)}   "
          f"（IC 是横截面相关系数，>0.03 通常已算有效，>0.05 较强）")
    diverge = df[df["均值中位一致"].str.startswith("否")] if "均值中位一致" in df else df.iloc[0:0]
    if len(diverge):
        print(f"⚠️ 有 {len(diverge)} 个因子的「算术均值分层」与「中位数分层」方向相反: "
              f"{', '.join(diverge['因子'])}")
        print("   原因：个股收益右偏，高波动组右尾更肥，等权算术均值被少数暴涨拉高。")
        print("   这类因子应以秩 IC / 中位数为准，不要直接按均值下结论。")
    print("注意：IC 反映的是预测力，实盘还要看换手率带来的交易成本与策略容量。")

    if args.out:
        df.to_csv(args.out, index=False, encoding="utf-8-sig")
        print(f"已保存: {args.out}")


if __name__ == "__main__":
    main()
