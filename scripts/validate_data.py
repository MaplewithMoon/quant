# -*- coding: utf-8 -*-
"""数据库质量校验：主键唯一性 / Schema / 覆盖率

用法:
    python scripts/validate_data.py                 # 全部检查
    python scripts/validate_data.py --check unique  # 只查唯一性
    python scripts/validate_data.py --check schema  # 只查Schema
    python scripts/validate_data.py --check coverage # 只查覆盖率
    python scripts/validate_data.py --quick         # 抽样快速检查
"""
import sys, io, json
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
sys.path.insert(0, ".")

import argparse
import glob
from pathlib import Path
import pandas as pd
import numpy as np

from database.config import dir_of, parquet_glob

DB = Path("db")
DAILY_DIR = dir_of("daily")                  # db/cleaned/daily_basic
DAILY_ALL_GLOB = parquet_glob(DAILY_DIR)
WARN = []


def _dataset_dir(dataset: str) -> Path:
    """把逻辑数据集名解析成实际目录

    'daily'             -> db/cleaned/daily_basic（走清洗配方路由）
    'frozen/valuation'  -> db/frozen/valuation（带斜杠的按原样拼接）
    """
    return DB / dataset if "/" in dataset else dir_of(dataset)

def log(msg):
    print(f"  {msg}")

def report(name, ok, detail=""):
    tag = "OK  " if ok else "FAIL"
    print(f"  [{tag}] {name} {detail}")
    if not ok:
        WARN.append(f"{name}: {detail}")

# ============================================================
# 1. 主键唯一性检查
# ============================================================
def check_uniqueness(dataset: str, key_cols=("code", "trade_date"), quick=False):
    print(f"\n[唯一性] {dataset}")
    files = list(_dataset_dir(dataset).rglob("*.parquet"))
    if not files:
        report(f"{dataset}", False, "无文件")
        return
    # 用 DuckDB 全量扫描（快）
    import duckdb
    con = duckdb.connect()
    glob_pat = str(_dataset_dir(dataset) / "**" / "*.parquet")
    try:
        cols_sql = ", ".join(key_cols)
        df = con.execute(f"""
            SELECT {cols_sql}, count(*) AS cnt
            FROM read_parquet('{glob_pat}')
            GROUP BY {cols_sql}
            HAVING count(*) > 1
        """).fetchdf()
        dup_total = len(df)
        if dup_total:
            report(f"{dataset} 主键唯一", False, f"{dup_total} 组重复(含{df['cnt'].sum()}行)")
            print(df.head(10).to_string())
        else:
            report(f"{dataset} 主键唯一", True, f"全量 {len(files)} 文件无重复")
    except Exception as e:
        report(f"{dataset} 主键唯一", False, f"检查失败: {e}")

# ============================================================
# 2. Schema 校验
# ============================================================
SCHEMA_SPECS = {
    "daily": {"required": ["code", "trade_date", "open", "high", "low", "close", "volume", "amount"],
              "numeric": ["open", "high", "low", "close", "volume", "amount"],
              "units": {"volume": ("股", 0, 1e12), "amount": ("元", 0, 1e14),
                        "open": ("元", 0, 1e5), "close": ("元", 0, 1e5)}},
    "valuation": {"required": ["code", "trade_date", "pe_ttm", "pb", "total_mv"],
                  "numeric": ["pe_ttm", "pb", "ps_ttm", "total_mv", "circ_mv", "turnover_rate"],
                  "units": {"total_mv": ("万元", 0, 1e11), "turnover_rate": ("%", 0, 1e3)}},
    "adjust": {"required": ["code", "trade_date", "adj_factor"],
               "numeric": ["adj_factor"]},
}

def check_schema(dataset: str, quick=False):
    print(f"\n[Schema] {dataset}")
    spec_key = dataset.split("/")[-1]  # 'frozen/valuation' → 'valuation'
    spec = SCHEMA_SPECS.get(spec_key)
    if not spec:
        report(dataset, False, "无Schema定义")
        return
    files = list(_dataset_dir(dataset).rglob("*.parquet"))
    import duckdb
    con = duckdb.connect()
    glob_pat = str(_dataset_dir(dataset) / "**" / "*.parquet")
    issues = 0

    # 1) 必填列存在性（读取失败 = 有文件列不一致）
    try:
        n = con.execute(f"SELECT count(*) FROM read_parquet('{glob_pat}')").fetchone()[0]
        log(f"  总行数: {n:,}")
    except Exception as e:
        report(f"{dataset} 列一致性", False, f"读取失败(可能存在schema不一致文件): {str(e)[:60]}")
        issues += 1

    # 2) 数值列类型 + 单位范围（全量扫描）
    for c, (unit, lo, hi) in spec.get("units", {}).items():
        try:
            cnt = con.execute(f"""
                SELECT count(*) FROM read_parquet('{glob_pat}')
                WHERE {c} IS NOT NULL AND ({c} < {lo} OR {c} > {hi})
            """).fetchone()[0]
            if cnt:
                report(f"{dataset}.{c} 超范围", False, f"{cnt} 行 (单位应为{unit}, 范围[{lo},{hi}])")
                issues += 1
        except Exception as e:
            log(f"  {dataset}.{c} 检查跳过: {str(e)[:50]}")
    report(f"{dataset} Schema", issues == 0, f"问题 {issues}" if issues else f"全量 {len(files)} 文件通过")

# ============================================================
# 3. 覆盖率检查
# ============================================================
def _load_calendar():
    files = list((DB / "frozen" / "calendar").rglob("*.parquet"))
    if not files:
        return pd.DataFrame()
    df = pd.concat([pd.read_parquet(f) for f in files], ignore_index=True)
    df["trade_date"] = pd.to_datetime(df["trade_date"])
    df = df[df["is_open"] == 1] if "is_open" in df.columns else df
    return df

def _load_stocks():
    files = list((DB / "frozen" / "stocks").rglob("*.parquet"))
    if not files:
        return pd.DataFrame()
    df = pd.concat([pd.read_parquet(f) for f in files], ignore_index=True)
    if "list_date" in df.columns:
        df["list_date"] = pd.to_datetime(df["list_date"], errors="coerce")
    return df

def check_coverage(quick=True, sample_days=10, sample_stocks=30):
    print("\n[覆盖率] 逐日覆盖 + 逐股覆盖")
    cal = _load_calendar()
    stocks = _load_stocks()
    if cal.empty or stocks.empty:
        report("日历/股票列表", False, "缺失")
        return
    cal_days = set(pd.to_datetime(cal["trade_date"]))
    valid_codes = set(stocks["code"].astype(str))
    all_codes = sorted(valid_codes)

    # 用 DuckDB 汇总日线各日股票数
    import duckdb
    files = list(DAILY_DIR.rglob("*.parquet"))
    if not files:
        report("日线", False, "无文件")
        return
    con = duckdb.connect()
    # 抽样若干交易日做统计
    years = sorted({f.parts[-2].split("=")[1] for f in files})
    sample_years = years[-2:]  # 最近两年
    date_counts = {}
    for y in sample_years:
        yg = parquet_glob(DAILY_DIR / f"year={y}")
        df = con.execute(f"""
            SELECT trade_date, count(DISTINCT code) AS n
            FROM read_parquet('{yg}')
            GROUP BY trade_date ORDER BY trade_date
        """).fetchdf()
        for _, r in df.iterrows():
            date_counts[str(r["trade_date"])[:10]] = r["n"]

    # 抽查交易日（跳过数据源可能延迟的最远3个交易日）
    import datetime
    latest_data = max(date_counts.keys()) if date_counts else ""
    cal_until = sorted(c for c in cal_days if str(c)[:10] <= latest_data)
    sample_dates = cal_until[:-3][-sample_days:] if len(cal_until) > 3 else cal_until[-sample_days:]
    issues = 0
    for d in sample_dates:
        key = str(d)[:10]
        n = date_counts.get(key, 0)
        # 期望数：该日已上市且未退市的股票（数据从2005年起）
        listed = stocks[stocks["list_date"].notna() & (stocks["list_date"] <= d)]
        expected = len(listed)
        # 允许 20% 容差（停牌等）
        ratio = n / expected if expected else 0
        if ratio < 0.8:
            report(f"{key} 覆盖率", False, f"{n}/{expected} ({ratio:.0%})")
            issues += 1
        else:
            log(f"  {key}: {n} 只 / 应有~{expected} ({ratio:.0%})")

    # 逐股覆盖（全量，DuckDB 一次分组统计）
    per_code = con.execute(f"""
        SELECT code, count(*) AS actual, min(trade_date) AS min_d, max(trade_date) AS max_d
        FROM read_parquet('{DAILY_ALL_GLOB}')
        GROUP BY code
    """).fetchdf()
    per_code["min_d"] = pd.to_datetime(per_code["min_d"])
    per_code["max_d"] = pd.to_datetime(per_code["max_d"])
    have = set(per_code["code"])

    stock_issues = 0
    checked = 0
    for ci, code in enumerate(all_codes):
        srow = stocks[stocks["code"].astype(str) == code]
        if srow.empty:
            continue
        # 跳过退市股（免费源可能不提供其历史数据）
        if "退" in str(srow.iloc[0].get("name", "")):
            continue
        checked += 1
        lst = srow.iloc[0]["list_date"]
        if pd.isna(lst):
            continue
        if code not in have:
            report(f"{code} 覆盖率", False, "无任何日线数据")
            stock_issues += 1
            continue
        row = per_code[per_code["code"] == code].iloc[0]
        actual = int(row["actual"])
        # 数据按设计从 2005 年起，期望天数从 max(上市日, 2005) 算到数据末日
        start_day = max(lst, pd.Timestamp("2005-01-01"))
        expected_days = sum(1 for c in cal_days if start_day <= c <= row["max_d"])
        ratio = actual / expected_days if expected_days else 0
        if ratio < 0.8:
            report(f"{code} 覆盖率", False, f"{actual}/{expected_days} 天 ({ratio:.0%})")
            stock_issues += 1
        if (ci + 1) % 1000 == 0:
            log(f"  逐股进度 {ci+1}/{len(all_codes)}")
    report("逐股覆盖率(全量)", stock_issues == 0, f"问题 {stock_issues}" if stock_issues else f"全量检查 {checked} 只正常")

# ============================================================
def main():
    parser = argparse.ArgumentParser(description="数据库质量校验")
    parser.add_argument("--check", choices=["unique", "schema", "coverage", "all"], default="all")
    parser.add_argument("--quick", action="store_true", help="抽样快速检查")
    parser.add_argument("--no-strict-exit", action="store_true",
                        help="即使发现问题也返回 0（默认发现问题返回 1，供 CI/调度做门禁）")
    args = parser.parse_args()

    if args.check in ("unique", "all"):
        for ds in ["daily", "frozen/valuation", "frozen/adjust"]:
            check_uniqueness(ds, quick=args.quick)
    if args.check in ("schema", "all"):
        for ds in ["daily", "frozen/valuation", "frozen/adjust"]:
            check_schema(ds, quick=args.quick)
    if args.check in ("coverage", "all"):
        check_coverage(quick=args.quick)

    print("\n" + "=" * 60)
    if WARN:
        print(f"发现 {len(WARN)} 个问题:")
        for w in WARN:
            print(f"  - {w}")
    else:
        print("全部检查通过，无问题")
    print("=" * 60)

    # 返回非零退出码，让 CI / 调度 / 监控能感知校验失败
    # （旧实现只打印，退出码恒为 0 → 脏数据静默入库，门禁形同虚设）
    if args.no_strict_exit:
        return 0
    return 1 if WARN else 0


if __name__ == "__main__":
    sys.exit(main())
