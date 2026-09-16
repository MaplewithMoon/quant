# -*- coding: utf-8 -*-
"""下载 tushare 不复权原始价到 frozen/daily_raw（可信数据源）

tushare pro.daily 字段: trade_date, open, high, low, close, pre_close,
                         change, pct_chg, vol(手), amount(千元)
单位统一: vol×100→股, amount×1000→元

用法:
    python scripts/download_tushare_raw.py             # 全量（断点续跑）
    python scripts/download_tushare_raw.py --codes 000001,600519
    python scripts/download_tushare_raw.py --fresh     # 忽略断点重下
"""
import sys
import io
import os
sys.path.insert(0, ".")

import argparse
import json
import glob
from pathlib import Path
import pandas as pd
import tushare as ts
from tqdm import tqdm
from database.token import load_token
from scripts._tushare_common import RateLimiter, api_call, ts_code_of, check_disk

DB = Path("db")
OUT = DB / "frozen" / "daily_raw"
CKPT = OUT / "_checkpoint.json"
START_YEAR = 2005


def main():
    parser = argparse.ArgumentParser(description="下载tushare不复权原始价")
    parser.add_argument("--codes", default="", help="指定股票，逗号分隔")
    parser.add_argument("--fresh", action="store_true", help="忽略断点重下")
    args = parser.parse_args()

    token = load_token()
    if not token:
        print("错误: 未找到 Tushare token")
        return
    ts.set_token(token)
    pro = ts.pro_api()

    OUT.mkdir(parents=True, exist_ok=True)
    codes = [c.strip() for c in args.codes.split(",") if c.strip()] or None
    if codes is None:
        st = pd.read_parquet(DB / "stocks" / "year=2005" / "all.parquet")
        codes = sorted(st["code"].astype(str).tolist())

    if args.fresh:
        ckpt = {"done": []}
    else:
        ckpt = json.loads(CKPT.read_text(encoding="utf-8")) if CKPT.exists() else {"done": []}
    done = set(ckpt["done"])
    todo = [c for c in codes if c not in done]
    print(f"待下载 {len(todo)} 只（已完成 {len(done)}）", flush=True)

    limiter = RateLimiter()
    with tqdm(total=len(todo), desc="tushare原始价", unit="股", ncols=100) as pbar:
        for code in todo:
            # 磁盘保护
            try:
                check_disk()
            except RuntimeError as e:
                print(f"\n{e}。断点已保存，可续跑。", flush=True)
                break
            try:
                df = api_call(pro, "daily", limiter, ts_code=ts_code_of(code),
                              start_date=f"{START_YEAR}0101", end_date="20500101")
                if df is None or df.empty:
                    done.add(code)
                    ckpt["done"] = sorted(done)
                    CKPT.write_text(json.dumps(ckpt), encoding="utf-8")
                    pbar.update(1)
                    continue
                df["trade_date"] = pd.to_datetime(df["trade_date"])
                df["code"] = code
                df["volume"] = df["vol"] * 100  # 手 → 股
                df["amount"] = df["amount"] * 1000  # 千元 → 元
                cols = ["code", "trade_date", "open", "high", "low",
                        "close", "pre_close", "change", "pct_chg", "volume", "amount"]
                df = df[[c for c in cols if c in df.columns]]
                df["year"] = df["trade_date"].dt.year
                for y, g in df.groupby("year"):
                    if y < START_YEAR:
                        continue
                    part = OUT / f"year={int(y)}"
                    part.mkdir(parents=True, exist_ok=True)
                    g.drop(columns="year").to_parquet(part / f"{code}.parquet", index=False)
                done.add(code)
                ckpt["done"] = sorted(done)
                CKPT.write_text(json.dumps(ckpt), encoding="utf-8")
            except Exception as e:
                print(f"  {code} 失败: {str(e)[:60]}", flush=True)
            pbar.set_postfix(code=code)
            pbar.update(1)

    print(f"\n下载完成: {len(done)}/{len(codes)} 只", flush=True)


if __name__ == "__main__":
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")
    main()
