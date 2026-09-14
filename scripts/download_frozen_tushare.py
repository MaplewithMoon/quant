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
import sys, io, os
sys.path.insert(0, ".")

import argparse
import json
from pathlib import Path
import pandas as pd
import tushare as ts
from tqdm import tqdm
from database.token import load_token
from scripts._tushare_common import RateLimiter, api_call, ts_code_of, check_disk

DB = Path("db")
FROZEN = DB / "frozen"
START_YEAR = 2005


def ckpt_path(dataset):
    return FROZEN / dataset / "_checkpoint.json"


def load_ckpt(dataset):
    p = ckpt_path(dataset)
    if p.exists():
        return json.loads(p.read_text(encoding="utf-8"))
    return {"done": []}


def save_ckpt(dataset, done):
    (FROZEN / dataset).mkdir(parents=True, exist_ok=True)
    ckpt_path(dataset).write_text(json.dumps({"done": sorted(done)}), encoding="utf-8")


def save_by_year(df, dataset, code=None):
    """按年分区保存到 frozen/{dataset}/year={Y}/"""
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
    done = set(load_ckpt("adjust")["done"])
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
                    done.add(code); save_ckpt("adjust", done); pbar.update(1); continue
                df["trade_date"] = pd.to_datetime(df["trade_date"])
                df["code"] = code
                save_by_year(df, "adjust", code)
                done.add(code); save_ckpt("adjust", done)
            except Exception as e:
                print(f"  {code} 失败: {str(e)[:50]}", flush=True)
            pbar.set_postfix(code=code); pbar.update(1)


# ============ st: tushare namechange（原始） ============
def download_st(pro, codes):
    limiter = RateLimiter()
    done = set(load_ckpt("st")["done"])
    todo = [c for c in codes if c not in done]
    print(f"[st] 待下 {len(todo)} 只", flush=True)
    with tqdm(total=len(todo), desc="st", ncols=100) as pbar:
        for code in todo:
            check_disk()
            try:
                df = api_call(pro, "namechange", limiter, ts_code=ts_code_of(code))
                if df is None or df.empty:
                    done.add(code); save_ckpt("st", done); pbar.update(1); continue
                df["code"] = code
                if "start_date" in df.columns:
                    df["start_date"] = pd.to_datetime(df["start_date"])
                (FROZEN / "st" / "year=2005").mkdir(parents=True, exist_ok=True)
                df.to_parquet(FROZEN / "st" / "year=2005" / f"{code}.parquet", index=False)
                done.add(code); save_ckpt("st", done)
            except Exception as e:
                print(f"  {code} 失败: {str(e)[:50]}", flush=True)
            pbar.set_postfix(code=code); pbar.update(1)


# ============ valuation: tushare daily_basic（原始单位） ============
def download_valuation(pro, codes):
    limiter = RateLimiter()
    done = set(load_ckpt("valuation")["done"])
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
                    done.add(code); save_ckpt("valuation", done); pbar.update(1); continue
                df["trade_date"] = pd.to_datetime(df["trade_date"])
                df["code"] = code
                save_by_year(df, "valuation", code)
                done.add(code); save_ckpt("valuation", done)
            except Exception as e:
                print(f"  {code} 失败: {str(e)[:50]}", flush=True)
            pbar.set_postfix(code=code); pbar.update(1)


# ============ etf: tushare fund_basic + fund_daily ============
def download_etf(pro):
    limiter = RateLimiter()
    done = set(load_ckpt("etf")["done"])
    (FROZEN / "etf").mkdir(parents=True, exist_ok=True)
    try:
        basic = api_call(pro, "fund_basic", limiter, market="E", status="L")
        basic.to_parquet(FROZEN / "etf" / "fund_basic.parquet", index=False)
        print(f"[etf] 基金列表: {len(basic)} 只", flush=True)
    except Exception as e:
        print(f"[etf] 基金列表失败: {str(e)[:60]}", flush=True)
    # 按交易日下全市场基金日线（fund_daily 按 trade_date 返回全部）
    import datetime
    today = datetime.date.today()
    dates = []
    for y in range(2020, today.year + 1):
        for m in range(1, 13):
            dates.append(f"{y}{m:02d}01")
    with tqdm(total=len(dates), desc="etf日线", ncols=100) as pbar:
        for d in dates:
            if d in done:
                pbar.update(1); continue
            check_disk()
            try:
                df = api_call(pro, "fund_daily", limiter, trade_date=d)
                if df is not None and not df.empty:
                    df["trade_date"] = pd.to_datetime(d)
                    save_by_year(df, "etf")
                done.add(d); save_ckpt("etf", done)
            except Exception as e:
                pass
            pbar.update(1)
    print(f"[etf] 完成 {len(done)} 个交易日", flush=True)


# ============ options: tushare opt_basic + opt_daily ============
def download_options(pro):
    limiter = RateLimiter()
    done = set(load_ckpt("options")["done"])
    (FROZEN / "options").mkdir(parents=True, exist_ok=True)
    try:
        basic = api_call(pro, "opt_basic", limiter, exchange="SSE")
        basic.to_parquet(FROZEN / "options" / "opt_basic.parquet", index=False)
        print(f"[options] 合约列表: {len(basic)} 个", flush=True)
    except Exception as e:
        print(f"[options] 合约列表失败: {str(e)[:60]}", flush=True)
    # 按交易日下全市场期权（opt_daily 按 trade_date）
    import datetime
    today = datetime.date.today()
    dates = [f"{y}{m:02d}01" for y in range(2020, today.year + 1) for m in range(1, 13)]
    with tqdm(total=len(dates), desc="options日线", ncols=100) as pbar:
        for d in dates:
            if d in done:
                pbar.update(1); continue
            check_disk()
            try:
                df = api_call(pro, "opt_daily", limiter, trade_date=d)
                if df is not None and not df.empty:
                    df["trade_date"] = pd.to_datetime(d)
                    save_by_year(df, "options")
                done.add(d); save_ckpt("options", done)
            except Exception as e:
                pass
            pbar.update(1)
    print(f"[options] 完成 {len(done)} 个交易日", flush=True)


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
            p = ckpt_path(ds)
            if p.exists():
                p.unlink()
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
