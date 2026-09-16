# -*- coding: utf-8 -*-
"""从 tushare 重新下载 frozen 层数据（加工过/低可信度数据集）

数据集:
  adjust    tushare adj_factor  (原始复权因子，去掉了派生的qfq_factor)
  st        tushare namechange  (原始名称变更，去掉了派生的is_st)
  valuation tushare daily_basic (原始单位，市值保留万元)
  etf       tushare fund_basic + fund_daily
  options   tushare opt_basic + opt_daily

用法:
    python scripts/download_frozen_tushare.py --only adjust     # 只下某数据集
    python scripts/download_frozen_tushare.py                   # 全量
    python scripts/download_frozen_tushare.py --only adjust --codes 000001,600519
"""
import sys
import io
import os
sys.path.insert(0, ".")

import argparse

from pathlib import Path
import pandas as pd
import tushare as ts
from tqdm import tqdm
from database.token import load_token
from scripts._tushare_common import RateLimiter, api_call, ts_code_of, check_disk

DB = Path("db")
FROZEN = DB / "frozen"
START_YEAR = 2005


def _store(dataset):
    """断点统一走 `database.storage.Storage`（**唯一实现**）

    旧实现在这里自己读写 `_checkpoint.json`，问题有两个：
      1. `write_text` **非原子**（断电会写坏，而 Storage 那边做了原子写 + 损坏容错）
      2. 只写 `{"done": [...]}`，会把 `Storage` 记的 `spans` 等键**整个抹掉**
    两套实现写同一个文件，迟早互相踩。现在统一。
    """
    from database.storage import Storage
    return Storage(dataset, allow_frozen=True)


def done_set(dataset) -> set:
    return set(_store(dataset).load_checkpoint().get("done", []))


def mark(dataset, key, df=None, empty=False):
    """标记完成：能取到日期范围就记下来（B4 靠它发现"数据只到某年"的静默截断）"""
    st = _store(dataset)
    span = None
    if df is not None and len(df) and "trade_date" in df.columns:
        s = pd.to_datetime(df["trade_date"], errors="coerce").dropna()
        if len(s):
            span = (s.min(), s.max())
    st.mark_done(key, span=span, empty=empty if span is None else False)


def save_by_year(df, dataset, code=None):
    """按年分区保存到 frozen/{dataset}/year={Y}/

    ⚠️ **`code` 为 None 时一律写 `data.parquet`，同一年的第二次调用会覆盖第一次**
    —— `etf`/`options` 就是被这个坑掉的：它们按"每天一个 trade_date"下载，
    结果每年只剩最后一次成功日期的数据（实测 2020~2026 每年各 1 天）。
    这类"按日期"的数据集必须传 `code=<日期>` 分开存。
    """
    df["year"] = df["trade_date"].dt.year
    for y, g in df.groupby("year"):
        if y < START_YEAR:
            continue
        part = FROZEN / dataset / f"year={int(y)}"
        part.mkdir(parents=True, exist_ok=True)
        fname = f"{code}.parquet" if code else "data.parquet"
        g.drop(columns="year").to_parquet(part / fname, index=False)


# ============ adjust: tushare adj_factor（原始） ============
def download_adjust(pro, codes):
    limiter = RateLimiter()
    done = done_set("adjust")
    todo = [c for c in codes if c not in done]
    print(f"[adjust] 待下 {len(todo)} 只", flush=True)
    with tqdm(total=len(todo), desc="adjust", ncols=100) as pbar:
        for code in todo:
            check_disk()
            try:
                df = api_call(pro, "adj_factor", limiter,
                              ts_code=ts_code_of(code),
                              start_date=f"{START_YEAR}0101", end_date="20500101")
                if df is None or df.empty:
                    mark("adjust", code, empty=True)
                    pbar.update(1)
                    continue
                df["trade_date"] = pd.to_datetime(df["trade_date"])
                df["code"] = code
                save_by_year(df, "adjust", code)
                mark("adjust", code, df)
            except Exception as e:
                print(f"  {code} 失败: {str(e)[:50]}", flush=True)
            pbar.set_postfix(code=code)
            pbar.update(1)


# ============ st: tushare namechange（原始） ============
def download_st(pro, codes):
    limiter = RateLimiter()
    done = done_set("st")
    todo = [c for c in codes if c not in done]
    print(f"[st] 待下 {len(todo)} 只", flush=True)
    with tqdm(total=len(todo), desc="st", ncols=100) as pbar:
        for code in todo:
            check_disk()
            try:
                df = api_call(pro, "namechange", limiter, ts_code=ts_code_of(code))
                if df is None or df.empty:
                    mark("st", code, empty=True)
                    pbar.update(1)
                    continue
                df["code"] = code
                if "start_date" in df.columns:
                    df["start_date"] = pd.to_datetime(df["start_date"])
                (FROZEN / "st" / "year=2005").mkdir(parents=True, exist_ok=True)
                df.to_parquet(FROZEN / "st" / "year=2005" / f"{code}.parquet", index=False)
                # namechange 没有 trade_date，用 start_date 作为"数据范围"
                _s = (pd.to_datetime(df["start_date"], errors="coerce").dropna()
                      if "start_date" in df.columns else None)
                _store("st").mark_done(
                    code, span=(_s.min(), _s.max()) if _s is not None and len(_s) else None)
            except Exception as e:
                print(f"  {code} 失败: {str(e)[:50]}", flush=True)
            pbar.set_postfix(code=code)
            pbar.update(1)


# ============ valuation: tushare daily_basic（原始单位） ============
def download_valuation(pro, codes):
    limiter = RateLimiter()
    done = done_set("valuation")
    todo = [c for c in codes if c not in done]
    print(f"[valuation] 待下 {len(todo)} 只", flush=True)
    with tqdm(total=len(todo), desc="valuation", ncols=100) as pbar:
        for code in todo:
            check_disk()
            try:
                df = api_call(pro, "daily_basic", limiter,
                              ts_code=ts_code_of(code),
                              start_date=f"{START_YEAR}0101", end_date="20500101",
                              fields="trade_date,pe,pe_ttm,pb,ps,ps_ttm,total_mv,circ_mv,turnover_rate")
                if df is None or df.empty:
                    mark("valuation", code, empty=True)
                    pbar.update(1)
                    continue
                df["trade_date"] = pd.to_datetime(df["trade_date"])
                df["code"] = code
                save_by_year(df, "valuation", code)
                mark("valuation", code, df)
            except Exception as e:
                print(f"  {code} 失败: {str(e)[:50]}", flush=True)
            pbar.set_postfix(code=code)
            pbar.update(1)


# ============ etf / options：按**真实交易日**下全市场快照 ============
def _download_by_trading_days(pro, dataset: str, api: str, desc: str,
                              calls_per_min: int, start_year: int = 2020):
    """按交易日逐日下载全市场快照，**每天一个文件**

    ⚠️ 旧实现有三个错，直接导致 `frozen/etf` 与 `frozen/options` 废掉：
      1. 取数日期用"每月 1 号"，而 1 号经常不是交易日 -> 大量空响应；
      2. 空响应也会 `done.add(d)` -> **永久不再重试**（B4 的真实案例）；
      3. `save_by_year(df, ds)` 不传 code，每年都写同一个 `data.parquet`
         -> 后一天覆盖前一天，实测 2020~2026 每年只剩 1 天数据。
    现在：从 `database.calendar.trading_days()` 取真实交易日；**只有真的写出
    数据才标完成并记范围**；每天写 `year={Y}/{日期}.parquet`，互不覆盖。
    """
    limiter = RateLimiter()
    store = _store(dataset)
    done = done_set(dataset)
    from database.calendar import trading_days
    days = [d for d in trading_days(start=f"{start_year}-01-01")]
    todo = [d for d in days if d.strftime("%Y%m%d") not in done]
    print(f"[{dataset}] 交易日 {len(days)} 个，待下 {len(todo)} 个", flush=True)
    n_empty = 0
    with tqdm(total=len(todo), desc=desc, ncols=100) as pbar:
        for d in todo:
            key = d.strftime("%Y%m%d")
            check_disk()
            try:
                df = api_call(pro, api, limiter, trade_date=key)
                if df is None or df.empty:
                    # **不标完成**：交易日却拿不到数据，下次还要重试
                    n_empty += 1
                else:
                    df = df.copy()
                    df["trade_date"] = pd.to_datetime(d.strftime("%Y-%m-%d"))
                    save_by_year(df, dataset, code=d.strftime("%Y-%m-%d"))
                    store.mark_done(key, span=(d, d))
            except Exception as e:
                print(f"  {key} 失败: {str(e)[:50]}", flush=True)
            pbar.set_postfix(day=key)
            pbar.update(1)
    if n_empty:
        print(f"[{dataset}] 有 {n_empty} 个交易日返回空 —— 未标记完成，"
              f"下次运行会重试（空响应 != 确认无数据）", flush=True)
    print(f"[{dataset}] 完成 {len(done_set(dataset))} 个交易日", flush=True)


def download_etf(pro):
    (FROZEN / "etf").mkdir(parents=True, exist_ok=True)
    try:
        basic = api_call(pro, "fund_basic", RateLimiter(), market="E", status="L")
        basic.to_parquet(FROZEN / "etf" / "fund_basic.parquet", index=False)
        print(f"[etf] 基金列表: {len(basic)} 只", flush=True)
    except Exception as e:
        print(f"[etf] 基金列表失败: {str(e)[:60]}", flush=True)
    _download_by_trading_days(pro, "etf", "fund_daily", "etf日线",
                              calls_per_min=20)


def download_options(pro):
    (FROZEN / "options").mkdir(parents=True, exist_ok=True)
    try:
        basic = api_call(pro, "opt_basic", RateLimiter(), exchange="SSE")
        basic.to_parquet(FROZEN / "options" / "opt_basic.parquet", index=False)
        print(f"[options] 合约列表: {len(basic)} 个", flush=True)
    except Exception as e:
        print(f"[options] 合约列表失败: {str(e)[:60]}", flush=True)
    _download_by_trading_days(pro, "options", "opt_daily", "options日线",
                              calls_per_min=15)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--only", choices=["adjust", "st", "valuation", "etf", "options", "all"], default="all")
    parser.add_argument("--codes", default="", help="指定股票")
    parser.add_argument("--fresh", action="store_true", help="忽略断点，从头重新下载")
    args = parser.parse_args()

    token = load_token()
    if not token:
        print("错误: 未找到 Tushare token")
        return
    ts.set_token(token)
    pro = ts.pro_api()

    codes = [c.strip() for c in args.codes.split(",") if c.strip()]
    if not codes:
        st = pd.read_parquet(FROZEN / "stocks" / "year=2005" / "all.parquet")
        # 跳过退市股（名称含"退"，tushare 无其行情数据）
        st = st[~st["name"].astype(str).str.contains("退", na=False)]
        codes = sorted(st["code"].astype(str).tolist())
        print(f"有效股票(剔除退市): {len(codes)} 只", flush=True)

    if args.fresh:
        for ds in ["adjust", "st", "valuation", "etf", "options"]:
            # 断点文件就在数据集目录里，删目录时一并清掉；不依赖已删除的 ckpt_path()
            d = FROZEN / ds
            if d.exists():
                import shutil as _sh
                _sh.rmtree(d)
        print("已清空断点和数据，从头重新下载", flush=True)

    order = ["adjust", "st", "valuation", "etf", "options"] if args.only == "all" else [args.only]
    for ds in order:
        print(f"\n{'='*60}\n下载: {ds}\n{'='*60}", flush=True)
        if ds == "adjust":
            download_adjust(pro, codes)
        elif ds == "st":
            download_st(pro, codes)
        elif ds == "valuation":
            download_valuation(pro, codes)
        elif ds == "etf":
            download_etf(pro)
        elif ds == "options":
            download_options(pro)

    print("\n全部完成", flush=True)


if __name__ == "__main__":
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")
    main()
