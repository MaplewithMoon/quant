# -*- coding: utf-8 -*-
"""策略参数优化 / Walk-Forward 样本外验证 CLI

【用法】
    # 网格搜索（参数少、范围窄时用；结果含数据窥探偏差）
    python scripts/optimize.py --symbol 000001 --strategy MA \
        --grid fast=3,5,10,20 slow=20,30,60,120 --top 10

    # 随机搜索（参数多、范围大时更有效）
    python scripts/optimize.py --symbol 000001 --strategy MA \
        --random fast=2:20 slow=20:120 --n-iter 60 --objective sharpe

    # ★ 推荐的正确做法：Walk-Forward 样本外验证
    python scripts/optimize.py --symbol 000001 --strategy MA \
        --random fast=2:20 slow=20:120 --walk-forward \
        --train-days 252 --test-days 63 --n-iter 30

    # 多标的批量 walk-forward（更能说明策略是否稳健）
    python scripts/optimize.py --symbols 000001,600519,300750 --strategy MA \
        --random fast=2:20 slow=20:120 --walk-forward

【参数空间写法】
    fast=5,10,20     离散取值
    fast=2:20        整数区间（随机搜索均匀采样 / 网格展开成 2..20）
    k=0.1:1.0        浮点区间（网格展开成 10 档）

【重要提醒】
    不加 --walk-forward 时，结果是"在同一段历史上挑出来的最好看的数字"，
    必然含数据窥探偏差。脚本会在结尾给出明确警告。
"""
import sys
import io
import json
import argparse
sys.path.insert(0, ".")
if __name__ == "__main__":
    # 只在直接运行时切编码：模块顶层替换 sys.stdout 是有副作用的 import，
    # 会破坏 pytest 的输出捕获（详见 utils/console.py）
    from utils.console import force_utf8_stdout
    force_utf8_stdout()

import pandas as pd

from optimizer.search import (grid_search, random_search, overfit_warning,
                              OBJECTIVES)
from optimizer.walkforward import walk_forward
from backtest.engine import FILL_NEXT_OPEN, FILL_SAME_CLOSE
from execution.market_rules import MarketRules
from scripts.backtest import load_dataset, make_strategy, STRATEGIES, DEFAULT_PARAMS


def parse_symbols(text: str):
    """解析股票代码列表，兼容多种分隔符并补回前导零

    为什么需要补零：在 PowerShell 里 `--symbols 000001,600519` 会被当成数组，
    元素被转成整数 → 000001 变成 1，直接查不到数据。这里统一按 6 位补零。
    """
    import re
    out = []
    for p in re.split(r"[,;\s]+", (text or "").strip()):
        p = p.strip()
        if not p:
            continue
        if p.isdigit() and len(p) <= 6:
            p = p.zfill(6)
        out.append(p)
    return out


def parse_space(items):
    """把 ['fast=5,10,20', 'slow=20:120'] 解析成参数空间字典"""
    space = {}
    for it in items or []:
        if "=" not in it:
            raise SystemExit(f"参数空间写法错误: {it!r}（应为 name=1,2,3 或 name=lo:hi）")
        name, val = it.split("=", 1)
        name = name.strip()
        if ":" in val:
            lo, hi = val.split(":", 1)
            lo_f, hi_f = float(lo), float(hi)
            if lo_f.is_integer() and hi_f.is_integer():
                space[name] = (int(lo_f), int(hi_f))
            else:
                space[name] = (lo_f, hi_f)
        else:
            parts = [p.strip() for p in val.split(",") if p.strip()]
            vals = []
            for p in parts:
                try:
                    vals.append(int(p))
                except ValueError:
                    try:
                        vals.append(float(p))
                    except ValueError:
                        vals.append(p)
            space[name] = vals
    return space


def make_factory(strategy_name):
    """参数字典 -> 策略实例"""
    def factory(params):
        return make_strategy(strategy_name, params)
    return factory


def run_one(symbol, args, space, bt_kwargs):
    """对单个标的做优化或 walk-forward"""
    ds = load_dataset(symbol, args.start, args.end, args.adjust)
    factory = make_factory(args.strategy)
    print("=" * 90)
    print(f"{symbol} / {args.strategy} / {len(ds)} 根日线 "
          f"({ds.data.index[0].date()}~{ds.data.index[-1].date()})")
    print("=" * 90)

    if args.walk_forward:
        res = walk_forward(ds, factory, space,
                           train_days=args.train_days, test_days=args.test_days,
                           step_days=args.step_days, mode=args.mode,
                           n_iter=args.n_iter, objective=args.objective,
                           seed=args.seed, bt_kwargs=bt_kwargs,
                           anchored=args.anchored)
        print()
        print(res.summary())
        if not res.windows.empty and args.top:
            print("\n各窗口明细（前 %d 行）:" % args.top)
            print(res.windows.head(args.top).to_string(index=False))
        return {"symbol": symbol, "walk_forward": res}

    # 普通搜索
    if args.mode == "grid":
        df = grid_search(ds, factory, space, objective=args.objective,
                         bt_kwargs=bt_kwargs)
    else:
        df = random_search(ds, factory, space, n_iter=args.n_iter,
                           objective=args.objective, seed=args.seed,
                           bt_kwargs=bt_kwargs)

    cols = list(space.keys()) + ["total_return", "sharpe_ratio", "calmar_ratio",
                                 "max_drawdown", "win_rate", "total_trades"]
    cols = [c for c in cols if c in df.columns]
    print(f"\n结果（按 {args.objective} 降序，前 {args.top} 组）:")
    show = df.head(args.top)[cols].copy()
    for c in ("total_return", "max_drawdown", "win_rate"):
        if c in show.columns:
            show[c] = show[c].map(lambda x: f"{x:.2%}" if pd.notna(x) else "-")
    for c in ("sharpe_ratio", "calmar_ratio"):
        if c in show.columns:
            show[c] = show[c].map(lambda x: f"{x:.3f}" if pd.notna(x) else "-")
    print(show.to_string(index=False))
    print()
    print(overfit_warning(df))
    return {"symbol": symbol, "search": df}


def main():
    p = argparse.ArgumentParser(description="策略参数优化 / Walk-Forward 验证")
    p.add_argument("--symbol", default="000001", help="股票代码")
    p.add_argument("--symbols", default="", help="多标的逗号分隔（每个都跑一遍）")
    p.add_argument("--start", default="2020-01-01")
    p.add_argument("--end", default="2024-12-31")
    p.add_argument("--adjust", default="qfq", choices=["qfq", ""])
    p.add_argument("--strategy", default="MA", choices=list(STRATEGIES))
    p.add_argument("--objective", default="sharpe",
                   choices=list(OBJECTIVES) + ["max_drawdown"],
                   help="选参目标（默认夏普）")

    # 参数空间
    p.add_argument("--grid", nargs="*", default=None,
                   help="网格搜索空间，如 fast=5,10,20 slow=20,30,60")
    p.add_argument("--random", nargs="*", default=None,
                   help="随机搜索空间，如 fast=2:20 slow=20:120")

    # walk-forward
    p.add_argument("--walk-forward", action="store_true", help="执行样本外验证（推荐）")
    p.add_argument("--train-days", type=int, default=252, help="训练段交易日数")
    p.add_argument("--test-days", type=int, default=63, help="测试段交易日数")
    p.add_argument("--step-days", type=int, default=None, help="前进步长（默认=测试段）")
    p.add_argument("--anchored", action="store_true",
                   help="训练段逐窗扩张（默认滚动固定长度）")
    p.add_argument("--n-iter", type=int, default=30, help="随机搜索每轮尝试组数")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--top", type=int, default=10, help="展示前 N 行")

    # 执行/成本（与 backtest 保持一致）
    p.add_argument("--capital", type=float, default=1_000_000)
    # 执行参数由公共层统一提供（原先各脚本各写一遍，见 execution/setup.py）
    from execution.setup import add_execution_args
    add_execution_args(p, fill_default=FILL_NEXT_OPEN)
    p.add_argument("--no-rules", action="store_true", help="关闭 A股制度约束")
    p.add_argument("--out", default="", help="把结果保存为 CSV")

    args = p.parse_args()

    # 参数空间：--grid 与 --random 二选一
    if args.random is not None:
        args.mode = "random"
        space = parse_space(args.random)
    elif args.grid is not None:
        args.mode = "grid"
        space = parse_space(args.grid)
    else:
        # 未指定则用策略的默认参数做一次搜索，方便先跑通
        args.mode = "random"
        base = DEFAULT_PARAMS.get(args.strategy, {})
        space = {k: [v] for k, v in base.items()}
        print(f"[提示] 未指定 --grid/--random，使用 {args.strategy} 的默认参数: {base}")

    if not space:
        raise SystemExit("参数空间为空")

    # 执行参数（成交时点/滑点/冲击模型/费用）统一从公共层取；
    # transfer_fee 与 truncate_data 是单标的引擎特有的，单独补
    from execution.setup import build_execution
    bt_kwargs = dict(
        initial_capital=args.capital,
        transfer_fee=args.transfer_fee,
        market_rules=MarketRules(enabled=not args.no_rules),
        truncate_data=True,
        **build_execution(args),
    )

    symbols = (parse_symbols(args.symbols) if args.symbols
               else parse_symbols(args.symbol))
    if not symbols:
        raise SystemExit("没有有效的股票代码")

    print(f"参数空间: {space}")
    print(f"目标: {args.objective}   模式: {args.mode}"
          f"{'   Walk-Forward' if args.walk_forward else '（同一段数据挑参，含数据窥探偏差）'}")

    all_results = {}
    for sym in symbols:
        try:
            all_results[sym] = run_one(sym, args, space, bt_kwargs)
            print()
        except Exception as e:
            print(f"!! {sym} 失败: {type(e).__name__}: {e}\n")

    # ---- 多标的 walk-forward 汇总 ----
    if args.walk_forward and len(symbols) > 1:
        print("=" * 90)
        print("多标的 Walk-Forward 汇总（样本外）")
        print("=" * 90)
        rows = []
        for sym, r in all_results.items():
            res = r.get("walk_forward")
            if res is None or res.windows.empty:
                continue
            rows.append({
                "代码": sym, "窗口数": len(res.windows),
                "IS平均": res.is_sharpe_mean, "OOS平均": res.oos_sharpe_mean,
                "过拟合落差": res.overfit_gap(),
                "OOS总收益": res.oos_metrics.total_return,
                "OOS夏普": res.oos_metrics.sharpe_ratio,
                "OOS回撤": res.oos_metrics.max_drawdown,
            })
        if rows:
            df = pd.DataFrame(rows).sort_values("OOS夏普", ascending=False)
            print(df.to_string(index=False))
            pos = int((df["OOS夏普"] > 0).sum())
            print(f"\n样本外夏普为正的标的: {pos}/{len(df)}")
            if args.out:
                df.to_csv(args.out, index=False, encoding="utf-8-sig")
                print(f"已保存: {args.out}")
        else:
            print("没有有效窗口")

    elif args.out:
        for sym, r in all_results.items():
            if "search" in r:
                r["search"].to_csv(args.out, index=False, encoding="utf-8-sig")
                print(f"已保存: {args.out}")
                break


if __name__ == "__main__":
    main()
