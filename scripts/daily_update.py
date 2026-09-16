# -*- coding: utf-8 -*-
"""每日收盘后数据更新脚本

流程:
  1. 增量下载最近 N 天数据到 frozen（upsert，可重复运行安全）
     - daily_raw: tushare daily（不复权原始价）
     - valuation: tushare daily_basic（每日估值）
     - adjust:    tushare adj_factor（复权因子）
  2. 重建清洗层 db/cleaned/daily_basic（仅当年，从 frozen 派生 + 清洗）
  3. 重建涨跌停价 db/cleaned/limit_price（仅当年，由清洗层计算）

用法:
    python scripts/daily_update.py                  # 更新全部（默认今天）
    python scripts/daily_update.py --date 2026-08-29
    python scripts/daily_update.py --only valuation # 只更新估值
    python scripts/daily_update.py --days 10        # 回看天数（默认15）
"""
import sys
import io
import os
sys.path.insert(0, ".")

import argparse
import datetime
from pathlib import Path
import pandas as pd
import tushare as ts
from tqdm import tqdm
from database.token import load_token
from database.config import FROZEN_ROOT, dir_of, parquet_glob
from scripts._tushare_common import RateLimiter, api_call, ts_code_of, check_disk

# 路径统一走 database.config，禁止再手写 "db/xxx" 字符串
DB = FROZEN_ROOT.parent                 # db/
FROZEN = FROZEN_ROOT                    # db/frozen             只读原始层
CLEANED_DAILY = dir_of("daily")         # db/cleaned/daily_basic
LIMIT_DIR = dir_of("limit")             # db/cleaned/limit_price
BACKUP_DIR = CLEANED_DAILY.parent / ".daily_backup"
DAILY_GLOB = parquet_glob(CLEANED_DAILY)


def live_codes():
    st = pd.read_parquet(FROZEN / "stocks" / "year=2005" / "all.parquet")
    st = st[~st["name"].astype(str).str.contains("退", na=False)]
    return sorted(st["code"].astype(str).tolist())


def upsert(dataset, code, df, date_col="trade_date"):
    """把 df 按 code 分区 upsert 进 frozen/{dataset}/year={Y}/{code}.parquet"""
    df = df.copy()
    df[date_col] = pd.to_datetime(df[date_col])
    df["year"] = df[date_col].dt.year
    for y, g in df.groupby("year"):
        if y < 2005:
            continue
        part = FROZEN / dataset / f"year={int(y)}"
        part.mkdir(parents=True, exist_ok=True)
        path = part / f"{code}.parquet"
        cols = [c for c in g.columns if c != "year"]
        new = g[cols]
        if path.exists():
            old = pd.read_parquet(path)
            old[date_col] = pd.to_datetime(old[date_col])
            new_dates = set(pd.to_datetime(new[date_col]))
            old = old[~old[date_col].isin(new_dates)]
            merged = pd.concat([old, new], ignore_index=True).sort_values(date_col)
        else:
            merged = new.sort_values(date_col)
        merged.to_parquet(path, index=False)


# ============ 增量下载 ============
def update_daily_raw(pro, codes, start, end):
    limiter = RateLimiter()
    print(f"\n[1/3] 更新 daily_raw（frozen，{len(codes)} 只）", flush=True)
    updated = 0
    for code in tqdm(codes, desc="daily_raw", ncols=100):
        try:
            df = api_call(pro, "daily", limiter, ts_code=ts_code_of(code),
                          start_date=start, end_date=end)
            if df is None or df.empty:
                continue
            df["trade_date"] = pd.to_datetime(df["trade_date"])
            df["code"] = code
            df["volume"] = df["vol"] * 100
            df["amount"] = df["amount"] * 1000
            cols = ["code", "trade_date", "open", "high", "low", "close",
                    "pre_close", "change", "pct_chg", "volume", "amount"]
            upsert("daily_raw", code, df[[c for c in cols if c in df.columns]])
            updated += 1
        except Exception as e:
            print(f"  {code} 失败: {str(e)[:50]}", flush=True)
    print(f"  daily_raw 更新 {updated} 只", flush=True)


def update_valuation(pro, codes, start, end):
    limiter = RateLimiter()
    print(f"\n[2/3] 更新 valuation（frozen，{len(codes)} 只）", flush=True)
    updated = 0
    for code in tqdm(codes, desc="valuation", ncols=100):
        try:
            df = api_call(pro, "daily_basic", limiter, ts_code=ts_code_of(code),
                          start_date=start, end_date=end,
                          fields="trade_date,pe,pe_ttm,pb,ps,ps_ttm,total_mv,circ_mv,turnover_rate")
            if df is None or df.empty:
                continue
            df["trade_date"] = pd.to_datetime(df["trade_date"])
            df["code"] = code
            upsert("valuation", code, df)
            updated += 1
        except Exception as e:
            print(f"  {code} 失败: {str(e)[:50]}", flush=True)
    print(f"  valuation 更新 {updated} 只", flush=True)


def update_adjust(pro, codes, start, end):
    limiter = RateLimiter()
    print(f"\n[3/3] 更新 adjust（frozen，{len(codes)} 只）", flush=True)
    updated = 0
    for code in tqdm(codes, desc="adjust", ncols=100):
        try:
            df = api_call(pro, "adj_factor", limiter, ts_code=ts_code_of(code),
                          start_date=start, end_date=end)
            if df is None or df.empty:
                continue
            df["trade_date"] = pd.to_datetime(df["trade_date"])
            df["code"] = code
            upsert("adjust", code, df)
            updated += 1
        except Exception as e:
            print(f"  {code} 失败: {str(e)[:50]}", flush=True)
    print(f"  adjust 更新 {updated} 只", flush=True)


# ============ 重建清洗层（仅当年） ============
def clean_file(df):
    if df.empty:
        return df
    bad = pd.Series(False, index=df.index)
    vol = pd.to_numeric(df["volume"], errors="coerce")
    bad |= vol.isna() | (vol == 0)
    for c in ["open", "high", "low", "close"]:
        v = pd.to_numeric(df[c], errors="coerce")
        bad |= v.isna() | (v <= 0)
    o, h, lo, cl = (pd.to_numeric(df[c], errors="coerce") for c in ["open", "high", "low", "close"])
    vok = o.notna() & h.notna() & lo.notna() & cl.notna()
    bad |= vok & ~(h >= lo) | vok & ~(lo <= o) | vok & ~(o <= h) | vok & ~(lo <= cl) | vok & ~(cl <= h)
    amt = pd.to_numeric(df["amount"], errors="coerce")
    vwap = amt / vol.replace(0, pd.NA)
    bad |= vwap.notna() & vok & ~((vwap >= lo * 0.95) & (vwap <= h * 1.05))
    return df[~bad]


def rebuild_cleaned_year(year):
    """重建清洗层某一年（从 frozen/daily_raw 派生）"""
    src = FROZEN / "daily_raw" / f"year={year}"
    dst = CLEANED_DAILY / f"year={year}"
    if not src.exists():
        return
    os.makedirs(dst, exist_ok=True)
    files = sorted(src.glob("*.parquet"))
    removed = 0
    for f in files:
        df = pd.read_parquet(f)
        clean = clean_file(df)
        removed += len(df) - len(clean)
        if not clean.empty:
            clean.to_parquet(dst / f.name, index=False)
    print(f"  清洗层 {year}: {len(files)} 文件, 清理 {removed} 行", flush=True)


def backup_daily():
    """重建前备份清洗层（用于校验失败时回滚）"""
    import shutil
    if not CLEANED_DAILY.exists():
        return None
    tmp = BACKUP_DIR
    if tmp.exists():
        shutil.rmtree(tmp)
    shutil.copytree(CLEANED_DAILY, tmp)
    return tmp


def rollback_daily(backup):
    """校验失败：恢复清洗层到更新前状态"""
    import shutil
    if backup is None:
        return
    if CLEANED_DAILY.exists():
        shutil.rmtree(CLEANED_DAILY)
    shutil.move(str(backup), str(CLEANED_DAILY))
    print("已回滚：清洗层恢复为更新前状态")


def run_validation():
    """更新后自动执行数据校验（validate_data + clean_data 核心检查）

    返回: (ok, 问题列表)。ok=False 时由调用方回滚。
    """
    import duckdb
    problems = []
    con = duckdb.connect()

    # 1) 逻辑一致性：OHLC/非负/三角（清洗层全查）
    try:
        r = con.execute(f"""
            WITH t AS (
                SELECT open, high, low, close, volume, amount,
                       amount / NULLIF(volume, 0) AS vwap
                FROM read_parquet('{DAILY_GLOB}')
            )
            SELECT
                sum(CASE WHEN close <= 0 THEN 1 ELSE 0 END),
                sum(CASE WHEN NOT (high>=low AND low<=open AND open<=high
                                   AND low<=close AND close<=high) THEN 1 ELSE 0 END),
                sum(CASE WHEN volume=0 OR volume IS NULL THEN 1 ELSE 0 END),
                sum(CASE WHEN vwap IS NOT NULL AND volume>0
                         AND NOT (vwap >= low*0.95 AND vwap <= high*1.05) THEN 1 ELSE 0 END)
            FROM t
        """).fetchone()
        if r[0] or r[1] or r[2] or r[3]:
            problems.append(f"逻辑校验: 价<=0:{r[0]}, OHLC违规:{r[1]}, 幽灵K线:{r[2]}, 三角违规:{r[3]}")
    except Exception as e:
        problems.append(f"逻辑校验失败: {e}")

    # 2) 主键唯一性（code+trade_date）
    try:
        dup = con.execute(f"""
            SELECT count(*) FROM (
                SELECT code, trade_date, count(*) c
                FROM read_parquet('{DAILY_GLOB}')
                GROUP BY code, trade_date HAVING count(*) > 1
            )
        """).fetchone()[0]
        if dup:
            problems.append(f"主键重复: {dup} 组")
    except Exception as e:
        problems.append(f"唯一性校验失败: {e}")

    # 3) 数据新鲜度：清洗层最新日期不应滞后太多
    try:
        latest = con.execute(f"""
            SELECT max(trade_date) FROM read_parquet('{DAILY_GLOB}')
        """).fetchone()[0]
        gap = (datetime.date.today() - latest.date()).days if latest else 999
        if gap > 10:
            problems.append(f"数据滞后: 清洗层最新 {latest.date()}，距今 {gap} 天")
    except Exception:
        pass

    if problems:
        print("[校验] 发现问题:", *problems, sep="\n  - ")
        return False, problems
    print("[校验] 全部通过：逻辑/唯一性/新鲜度正常")
    return True, problems


# ============ 重建涨跌停价（仅当年） ============
def rebuild_limit_year(year):
    """从清洗层日线计算涨跌停价

    ⚠️ 三处修正（原先这里是错的，且会覆盖 rebuild_limit.py 生成的正確数据）：
      1. **用官方 pre_close**，不是 `close.shift(1)`。后者是未除权的昨收，
         除权除息日算出的涨跌停价会错得离谱（10 送 10 时差一倍）。
      2. **规则统一到 database/limit_rules.py**：补上 ST ±5% 与北交所
         43/83/87/88（原先只认 920），并处理创业板 2020-08-24 的制度切换。
      3. **上市初期特殊规则**（前 N 日不设涨跌幅 -> limit 为空 / 首日 ±44%），
         必须与全量重建用同一份 `listing_windows()`，否则增量会覆盖掉全量的结果。
    """
    from database.limit_rules import (apply_limit_prices, listing_windows,
                                      load_st_intervals)
    src = CLEANED_DAILY / f"year={year}"
    dst = LIMIT_DIR / f"year={year}"
    if not src.exists():
        return
    os.makedirs(dst, exist_ok=True)
    st_map = load_st_intervals()
    windows = listing_windows()
    n = 0
    for f in sorted(src.glob("*.parquet")):
        code = f.stem
        df = pd.read_parquet(f)
        if "pre_close" not in df.columns:
            # 清洗层应带官方 pre_close；缺失就跳过而不是退回 close.shift(1)
            print(f"  [跳过] {code}: 清洗层缺 pre_close，无法正确计算涨跌停")
            continue
        df = df.sort_values("trade_date")
        out = apply_limit_prices(df, code, st_map,
                                 listing_rule=windows.get(code))
        # **不要 dropna**：limit 为空可能是"不设涨跌幅"（新股上市初期）或
        # pre_close 缺失，两种都要保留成空值行 —— 空值在引擎里表示"没有涨跌停
        # 限制"（`_num()` 返回 None 会跳过检查），删掉行会造成增量与全量重建
        # 的口径不一致。
        out = out[["code", "trade_date", "pre_close", "limit_up", "limit_down"]]
        out.to_parquet(dst / f"{code}.parquet", index=False)
        n += 1
    print(f"  涨跌停价 {year}: {n} 文件（官方 pre_close + 统一规则 + 上市初期规则）",
          flush=True)


def main():
    parser = argparse.ArgumentParser(description="每日数据更新")
    parser.add_argument("--date", default="", help="更新到该日(默认今天, YYYY-MM-DD)")
    parser.add_argument("--days", type=int, default=15, help="回看天数(默认15)")
    parser.add_argument("--only", choices=["daily", "valuation", "adjust"], default="all")
    parser.add_argument("--no-verify", action="store_true", help="跳过更新后自动校验")
    args = parser.parse_args()

    token = load_token()
    if not token:
        print("错误: 未找到 Tushare token")
        return
    ts.set_token(token)
    pro = ts.pro_api()

    end_date = args.date.replace("-", "") if args.date else datetime.date.today().strftime("%Y%m%d")
    start = (datetime.date.today() - datetime.timedelta(days=args.days)).strftime("%Y%m%d")

    try:
        check_disk()
    except RuntimeError as e:
        print(e)
        return

    codes = live_codes()
    print(f"更新区间: {start} ~ {end_date}, 股票 {len(codes)} 只")

    # 更新前备份清洗层（用于校验失败回滚）
    backup = backup_daily()
    print("已备份清洗层（校验失败将自动回滚）")

    if args.only in ("daily", "all"):
        update_daily_raw(pro, codes, start, end_date)
    if args.only in ("valuation", "all"):
        update_valuation(pro, codes, start, end_date)
    if args.only in ("adjust", "all"):
        update_adjust(pro, codes, start, end_date)

    # 重建清洗层 + 涨跌停价（当年）
    year = int(end_date[:4])
    print(f"\n重建清洗层/涨跌停价 (year={year})")
    rebuild_cleaned_year(year)
    rebuild_limit_year(year)

    # 自动校验：不通过则回滚
    if not args.no_verify:
        print("\n=== 更新后自动校验 ===")
        ok, problems = run_validation()
        if not ok:
            print("校验未通过，执行回滚...")
            rollback_daily(backup)
            print("已回滚。请检查数据源问题后重试。")
            return
        # 校验通过后删除备份
        import shutil
        if backup and backup.exists():
            shutil.rmtree(backup)
    else:
        print("(已跳过校验)")

    print("\n每日更新完成")


if __name__ == "__main__":
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")
    main()
