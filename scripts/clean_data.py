# -*- coding: utf-8 -*-
"""行情数据逻辑一致性校验与清理

检查项:
  1. OHLC 关系: high>=low, high>=open/close, low<=open/close
  2. 量价非负: 价格/成交量非负；停牌日量为0/NaN
  3. 三角校验: VWAP=成交额/成交量 必须落在 [low, high] 内（可抓单位错误/字段错位）
  4. 复权因子跳变: 因子跳变日必须能对应分红送转事件

用法:
    python scripts/clean_data.py --scan             # 只扫描报告问题（默认）
    python scripts/clean_data.py --clean            # 清理：删除脏行
    python scripts/clean_data.py --codes 000001     # 指定股票
    python scripts/clean_data.py --year 2024        # 指定年份
    python scripts/clean_data.py --factor           # 只做复权因子检查
"""
import sys, io, glob, os
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
sys.path.insert(0, ".")

import argparse
import pandas as pd
import numpy as np
from pathlib import Path

from database.config import FROZEN_ROOT, dir_of

DB = Path("db")
DAILY_DIR = dir_of("daily")            # db/cleaned/daily_basic（清洗后日线）
ADJUST_DIR = FROZEN_ROOT / "adjust"    # db/frozen/adjust（只读）
DIVIDEND_DIR = FROZEN_ROOT / "dividend"
LOG = []


def log(msg):
    LOG.append(str(msg))
    print(msg, flush=True)


# ============================================================
# 1. OHLC 关系 + 2. 非负检查（向量化）
# ============================================================
def check_ohlc_nonneg(df, code, year):
    """返回脏行索引"""
    num = pd.DataFrame({
        "open": pd.to_numeric(df["open"], errors="coerce"),
        "high": pd.to_numeric(df["high"], errors="coerce"),
        "low": pd.to_numeric(df["low"], errors="coerce"),
        "close": pd.to_numeric(df["close"], errors="coerce"),
    }, index=df.index)
    valid = num.notna().all(axis=1)
    bad = pd.Series(False, index=df.index)
    bad |= valid & ~(num["high"] >= num["low"])
    bad |= valid & ~(num["low"] <= num["open"])
    bad |= valid & ~(num["open"] <= num["high"])
    bad |= valid & ~(num["low"] <= num["close"])
    bad |= valid & ~(num["close"] <= num["high"])
    bad |= valid & (num < 0).any(axis=1)
    vol = pd.to_numeric(df["volume"], errors="coerce")
    bad |= valid & (vol < 0)
    return set(bad[bad].index)


# ============================================================
# 3. 三角校验（向量化）: VWAP = amount/volume 落在 [low, high]
# ============================================================
def check_triangle(df, code, year, tol=0.05):
    """VWAP = amount/volume, 应与价格一致且落在 [low, high] 内"""
    amt = pd.to_numeric(df["amount"], errors="coerce")
    vol = pd.to_numeric(df["volume"], errors="coerce")
    lo = pd.to_numeric(df["low"], errors="coerce")
    hi = pd.to_numeric(df["high"], errors="coerce")
    vwap = amt / vol.replace(0, pd.NA)
    valid = vwap.notna() & lo.notna() & hi.notna()
    ok = (vwap >= lo * (1 - tol)) & (vwap <= hi * (1 + tol))
    bad = valid & ~ok
    return set(bad[bad].index)


# ============================================================
# 4. 复权因子跳变检查（frozen 层 tushare adj_factor）
# ============================================================
def check_factor_jump(codes=None):
    """因子跳变日必须有对应分红送转事件（全量遍历）"""
    adjust_files = sorted(ADJUST_DIR.glob("year=*/*.parquet"))
    codes_all = sorted({f.stem for f in adjust_files})
    if codes:
        codes_all = [c for c in codes_all if c in codes]
    print(f"复权因子跳变检查: 全量 {len(codes_all)} 只股票", flush=True)

    issues = 0
    for i, code in enumerate(codes_all):
        af = [f for f in adjust_files if f.stem == code]
        if not af:
            continue
        adj = pd.concat([pd.read_parquet(f) for f in af], ignore_index=True)
        adj["trade_date"] = pd.to_datetime(adj["trade_date"])
        adj = adj.sort_values("trade_date")
        if "adj_factor" not in adj.columns:
            continue
        adj["factor"] = pd.to_numeric(adj["adj_factor"], errors="coerce")
        # 因子发生跳变的日期（相对变化>0.1%，过滤浮点噪声）
        adj["factor_prev"] = adj["factor"].shift(1)
        rel = adj["factor"] / adj["factor_prev"]
        jumps = adj[adj["factor"].notna() & adj["factor_prev"].notna() & (rel.abs() - 1 > 0.001)]

        # 分红送转事件（除权除息日 ex_date）
        div_files = list(DIVIDEND_DIR.glob(f"year=*/{code}.parquet"))
        div_dates = set()
        if div_files:
            dv = pd.concat([pd.read_parquet(f) for f in div_files], ignore_index=True)
            for c in ["ex_date", "dividOperateDate"]:
                if c in dv.columns:
                    dv[c] = pd.to_datetime(dv[c], errors="coerce")
                    div_dates.update(dv[c].dropna().dt.strftime("%Y-%m-%d"))
                    break
        # 跳变日必须有分红事件（±7天容差）
        for _, r in jumps.iterrows():
            d = r["trade_date"]
            near = any(abs((d - pd.Timestamp(x)).days) <= 7 for x in div_dates)
            if not near:
                issues += 1
                if issues <= 20:
                    log(f"  {code} 因子跳变无分红事件: {d.strftime('%Y-%m-%d')} "
                        f"因子 {r['factor_prev']:.4f}→{r['factor']:.4f}")
        if (i + 1) % 500 == 0:
            log(f"  进度 {i+1}/{len(codes_all)}, 累计 {issues} 处")
    log(f"复权因子跳变检查: 发现 {issues} 处无对应分红的事件" if issues else "复权因子跳变检查: 全部有分红事件对应")
    return issues


# ============================================================
# 清理入口
# ============================================================
def process_daily(mode, codes=None, year=None, verbose=50):
    files = sorted(DAILY_DIR.glob("year=*/*.parquet"))  # 清洗层原始价日线（口径一致）
    if year:
        files = [f for f in files if f.parent.name == f"year={year}"]
    if codes:
        files = [f for f in files if f.stem in codes]

    total_bad = 0
    total_files = len(files)
    log(f"扫描 {total_files} 个日线文件 (mode={mode})")
    for i, f in enumerate(files):
        df = pd.read_parquet(f)
        code = f.stem
        y = f.parent.name.split("=")[1]
        if df.empty:
            continue

        bad_ohlc = check_ohlc_nonneg(df, code, y)
        bad_tri = check_triangle(df, code, y)
        bad_all = bad_ohlc | bad_tri

        if bad_all:
            total_bad += len(bad_all)
            dates = ", ".join(str(df.at[i, "trade_date"])[:10] for i in list(bad_all)[:5])
            log(f"  {code}/{y}: {len(bad_all)} 脏行 (OHLC+非负:{len(bad_ohlc)}, 三角:{len(bad_tri)}) 例:{dates}")
            if mode == "clean":
                df = df.drop(index=list(bad_all))
                df.to_parquet(f, index=False)
        if (i + 1) % verbose == 0 or i + 1 == total_files:
            log(f"  进度 {i+1}/{total_files}, 累计脏行 {total_bad}")
    log(f"日线逻辑校验完成: 共发现 {total_bad} 脏行")
    return total_bad


def main():
    parser = argparse.ArgumentParser(description="行情数据逻辑一致性校验与清理")
    parser.add_argument("--scan", action="store_true", help="只扫描（默认）")
    parser.add_argument("--clean", action="store_true", help="删除脏行")
    parser.add_argument("--factor", action="store_true", help="只做复权因子跳变检查")
    parser.add_argument("--codes", default="", help="指定股票")
    parser.add_argument("--year", type=int, default=0, help="指定年份")
    args = parser.parse_args()

    mode = "clean" if args.clean else "scan"
    codes = [c.strip() for c in args.codes.split(",") if c.strip()] or None

    if args.factor:
        check_factor_jump(codes=codes)
    else:
        process_daily(mode, codes=codes, year=args.year)
        check_factor_jump(codes=codes)


if __name__ == "__main__":
    main()
