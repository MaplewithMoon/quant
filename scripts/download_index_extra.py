# -*- coding: utf-8 -*-
"""补齐指数/ETF 原始数据（聚宽策略移植所需的 P0+P1 范围）

【为什么需要】
    docs/聚宽策略移植报告.md 的「近似与差异」里，有 3 条可以用 tushare 直接消除：
        399101 中小综指 成分/点位  -> 原来用 "002/003 开头" 与等权合成重建
        000985 中证全指 成分/点位  -> 原来用 "全部 A 股" 近似
        511880 银华日利 ETF 日线   -> 原来用 "现金等价物" 合成

【落盘位置（沿用现有分区约定）】
    frozen/index_daily/year=YYYY/<ts_code>.parquet    12 列，与既有 000300 等一致
    frozen/index_cons/year=2005/<index_code>.parquet  4 列（trade_date 为 YYYYMMDD 字符串）
                                                      —— 与既有的"非年度数据统一放 year=2005"一致
    frozen/fund_daily/year=YYYY/<ts_code>.parquet     日线（新建数据集，
                                                      与 etf 的月度快照分开，避免粒度混用）

【用法】
    python scripts/download_index_extra.py --scope p0        # 只下 399101 + 511880
    python scripts/download_index_extra.py --scope p0p1      # 加上 000985
    python scripts/download_index_extra.py --scope p0p1 --dry-run
"""
import argparse
import io
import os
import sys
import time
from pathlib import Path

sys.path.insert(0, ".")
if __name__ == "__main__":
    # 只在直接运行时切编码：模块顶层替换 sys.stdout 是有副作用的 import，
    # 会破坏 pytest 的输出捕获（详见 utils/console.py）
    from utils.console import force_utf8_stdout
    force_utf8_stdout()

import pandas as pd

from database.config import frozen_dir
from database.downloader.tushare_client import TushareClient, TushareCallError

# 指数日线：code -> (开始年份, 说明)
INDEX_DAILY = {
    "p0": {"399101.SZ": ("2005", "中小综指")},
    "p0p1": {"399101.SZ": ("2005", "中小综指"),
             "000985.CSI": ("2005", "中证全指"),
             "H00985.CSI": ("2005", "中证全指全收益")},
}
# 指数成分权重：code -> (开始年月, 说明)
INDEX_CONS = {
    "p0": {"399101.SZ": ("2016-01", "中小综指")},
    "p0p1": {"399101.SZ": ("2016-01", "中小综指"),
             "000985.CSI": ("2015-01", "中证全指")},
}
# ETF 日线
FUND_DAILY = {
    "p0": {"511880.SH": ("2005", "银华日利货币ETF")},
    "p0p1": {"511880.SH": ("2005", "银华日利货币ETF")},
}

INDEX_DAILY_COLS = ["ts_code", "trade_date", "close", "open", "high", "low",
                    "pre_close", "change", "pct_chg", "vol", "amount", "index_code"]
INDEX_CONS_COLS = ["index_code", "con_code", "trade_date", "weight"]
FUND_DAILY_COLS = ["ts_code", "trade_date", "open", "high", "low", "close",
                   "pre_close", "change", "pct_chg", "vol", "amount"]

END_DATE = "20261231"


def _atomic_write(df: pd.DataFrame, path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".parquet.tmp")
    df.to_parquet(tmp, index=False)
    os.replace(tmp, path)


def _months(start_ym: str, end_ym: str):
    y0, m0 = map(int, start_ym.split("-"))
    y1, m1 = map(int, end_ym.split("-"))
    out = []
    y, m = y0, m0
    while (y, m) <= (y1, m1):
        out.append((y, m))
        m += 1
        if m > 12:
            y, m = y + 1, 1
    return out


def download_index_daily(client, code, start_year, dry, stat):
    print(f"\n  指数日线 {code}（{start_year} 起）")
    if dry:
        stat["calls"] += 1
        return
    df = client.call("index_daily", ts_code=code,
                     start_date=f"{start_year}0101", end_date=END_DATE)
    if df is None or df.empty:
        print("    [空] 接口无数据")
        return
    df["trade_date"] = pd.to_datetime(df["trade_date"], format="%Y%m%d")
    df["index_code"] = code
    for c in INDEX_DAILY_COLS:
        if c not in df.columns:
            df[c] = None
    df = df[INDEX_DAILY_COLS].sort_values("trade_date")
    base = frozen_dir("index_daily")
    n_files = 0
    for y, g in df.groupby(df["trade_date"].dt.year):
        _atomic_write(g.reset_index(drop=True), base / f"year={y}" / f"{code}.parquet")
        n_files += 1
    stat["calls"] += 1
    stat["rows"] += len(df)
    print(f"    {len(df):>6} 行  {df['trade_date'].min():%Y-%m-%d} ~ "
          f"{df['trade_date'].max():%Y-%m-%d}  -> {n_files} 个年份文件")


def download_index_cons(client, code, start_ym, dry, stat):
    """指数权重：**必须按月分页**（单次调用上限 7000 行，按年取会被截断）"""
    months = _months(start_ym, f"{END_DATE[:4]}-{END_DATE[4:6]}")
    print(f"\n  指数成分权重 {code}（{start_ym} 起，{len(months)} 个月，按月分页）")
    if dry:
        stat["calls"] += 2 + len(months)
        return
    parts, got, empty_months = [], 0, []
    t0 = time.time()
    for i, (y, m) in enumerate(months):
        a = f"{y}{m:02d}01"
        b = f"{y}{m:02d}31"
        try:
            df = client.call("index_weight", index_code=code,
                             start_date=a, end_date=b)
        except TushareCallError as e:
            print(f"    [{y}-{m:02d}] 失败: {str(e)[:60]}")
            continue
        stat["calls"] += 1
        if df is None or df.empty:
            empty_months.append(f"{y}-{m:02d}")
        else:
            parts.append(df)
            got += len(df)
        if (i + 1) % 20 == 0:
            print(f"    ... {i+1}/{len(months)} 个月，累计 {got:,} 行 "
                  f"({time.time()-t0:.0f}s)")
    if not parts:
        print("    [空] 接口无数据")
        return
    all_df = pd.concat(parts, ignore_index=True)
    all_df = all_df[["index_code", "con_code", "trade_date", "weight"]]
    all_df["trade_date"] = all_df["trade_date"].astype(str)
    all_df = all_df.drop_duplicates(["index_code", "con_code", "trade_date"])
    all_df = all_df.sort_values(["trade_date", "con_code"]).reset_index(drop=True)
    # 与既有约定一致：非年度数据统一落在 year=2005
    _atomic_write(all_df, frozen_dir("index_cons") / "year=2005" / f"{code}.parquet")
    stat["rows"] += len(all_df)
    n_dates = all_df["trade_date"].nunique()
    print(f"    {len(all_df):>7,} 行  {n_dates} 个快照日  "
          f"{all_df['trade_date'].min()} ~ {all_df['trade_date'].max()}")
    if empty_months:
        print(f"    无数据的月份 {len(empty_months)} 个：{empty_months[:8]}"
              f"{' ...' if len(empty_months) > 8 else ''}")


def download_fund_daily(client, code, start_year, dry, stat):
    print(f"\n  ETF 日线 {code}（{start_year} 起）")
    if dry:
        stat["calls"] += 1
        return
    df = client.call("fund_daily", ts_code=code,
                     start_date=f"{start_year}0101", end_date=END_DATE)
    if df is None or df.empty:
        print("    [空] 接口无数据")
        return
    df["trade_date"] = pd.to_datetime(df["trade_date"], format="%Y%m%d")
    for c in FUND_DAILY_COLS:
        if c not in df.columns:
            df[c] = None
    df = df[FUND_DAILY_COLS].sort_values("trade_date")
    base = frozen_dir("fund_daily")
    n_files = 0
    for y, g in df.groupby(df["trade_date"].dt.year):
        _atomic_write(g.reset_index(drop=True), base / f"year={y}" / f"{code}.parquet")
        n_files += 1
    stat["calls"] += 1
    stat["rows"] += len(df)
    print(f"    {len(df):>6} 行  {df['trade_date'].min():%Y-%m-%d} ~ "
          f"{df['trade_date'].max():%Y-%m-%d}  -> {n_files} 个年份文件")


def main():
    ap = argparse.ArgumentParser(description="补齐指数/ETF 原始数据")
    ap.add_argument("--scope", default="p0p1", choices=["p0", "p0p1"])
    ap.add_argument("--dry-run", action="store_true", help="只估算调用次数，不下载")
    ap.add_argument("--calls-per-min", type=int, default=200)
    args = ap.parse_args()

    stat = {"calls": 0, "rows": 0}
    print("=" * 90)
    print(f"补齐指数/ETF 数据   范围 {args.scope}   "
          f"{'【DRY RUN 不写盘】' if args.dry_run else ''}")
    print("=" * 90)

    client = None if args.dry_run else TushareClient(args.calls_per_min)
    t0 = time.time()

    for code, (sy, name) in INDEX_DAILY[args.scope].items():
        download_index_daily(client, code, sy, args.dry_run, stat)
    for code, (sy, name) in INDEX_CONS[args.scope].items():
        download_index_cons(client, code, sy, args.dry_run, stat)
    for code, (sy, name) in FUND_DAILY[args.scope].items():
        download_fund_daily(client, code, sy, args.dry_run, stat)

    print("\n" + "=" * 90)
    print(f"{'预计' if args.dry_run else '实际'}调用 {stat['calls']} 次，"
          f"写入 {stat['rows']:,} 行，用时 {time.time()-t0:.0f}s")
    print("=" * 90)
    return 0


if __name__ == "__main__":
    sys.exit(main())
