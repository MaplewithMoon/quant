# -*- coding: utf-8 -*-
"""数据库质量校验：主键唯一性 / Schema / 覆盖率

用法:
    python scripts/validate_data.py                 # 全部检查
    python scripts/validate_data.py --check unique  # 只查唯一性
    python scripts/validate_data.py --check schema  # 只查Schema
    python scripts/validate_data.py --check coverage # 只查覆盖率
    python scripts/validate_data.py --quick         # 抽样快速检查
"""
import sys
import io
import json
sys.path.insert(0, ".")
if __name__ == "__main__":
    # 只在直接运行时切编码：模块顶层替换 sys.stdout 是有副作用的 import，
    # 会破坏 pytest 的输出捕获（详见 utils/console.py）
    from utils.console import force_utf8_stdout
    force_utf8_stdout()

import argparse
import glob
from pathlib import Path
import pandas as pd
import numpy as np

from database.config import (CLEANED_ROOT, FROZEN_ROOT, connect_duckdb, dir_of,
                             parquet_glob)

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
    "valuation": {"required": ["code", "trade_date", "pe_ttm", "pb", "total_mv",
                               "turnover_rate"],
                  "numeric": ["pe_ttm", "pb", "ps_ttm", "total_mv", "circ_mv", "turnover_rate"],
                  # ⚠️ 只设上界抓不到"亿元污染"（亿元数值**更小**，会照样通过）。
                  # 所以额外给 total_mv 一个**中位数下界**：A 股总市值中位数约
                  # 50 亿元 = 5e5 万元；若被误按亿元写入，中位数会掉到 ~50，
                  # 差 4 个数量级，用中位数判定极稳。
                  "units": {"total_mv": ("万元", 0, 1e11), "turnover_rate": ("%", 0, 1e3)},
                  "median_min": {"total_mv": 1e4, "turnover_rate": 0.01}},
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

    # 2) 必填列**逐列**做一次 count —— `count(*)` 不会碰列，
    #    所以只要不显式引用，某些文件缺列也不报错；显式 count 才会抛
    #    schema mismatch。这是抓"第二个写入方用了不同的列集"的关键。
    for c in spec.get("required", []):
        try:
            con.execute(f"SELECT count({c}) FROM read_parquet('{glob_pat}')").fetchone()
        except Exception as e:
            report(f"{dataset}.{c} 列存在性", False, f"{str(e)[:70]}")
            issues += 1

    # 3) 数值列类型 + 单位范围（全量扫描）
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

    # 4) 中位数下界：抓"单位被整体缩小"这类污染
    #    （上界检查对它无效 —— 亿元数值更小，照样落在 [lo, hi] 内）
    for c, floor in spec.get("median_min", {}).items():
        try:
            med = con.execute(f"""
                SELECT median({c}) FROM read_parquet('{glob_pat}') WHERE {c} IS NOT NULL
            """).fetchone()[0]
            if med is None:
                continue
            ok = float(med) >= floor
            report(f"{dataset}.{c} 中位数合理（单位未错）", ok,
                   f"中位数 {float(med):,.2f}，下界 {floor:,.0f}" +
                   ("" if ok else " —— 偏低，单位可能被整体换算过"))
        except Exception as e:
            log(f"  {dataset}.{c} 中位数检查跳过: {str(e)[:50]}")
    report(f"{dataset} Schema", issues == 0, f"问题 {issues}" if issues else f"全量 {len(files)} 文件通过")


# ============================================================
# 2b. 列一致性（按文件名采样，2026-09 新增）
# ============================================================
# **已声明**的多列集数据集：分裂是设计使然，且读取方都按文件名区分。
# 不在这个白名单里的分裂一律算问题 —— 这样新出现的意外分裂
# （例如某个下载器用不同列集写了同一目录）仍然会被抓到。
MULTI_SCHEMA_DECLARED = {
    "financial": "同一目录下按前缀分三张报表（balance_/profit_/cashflow_），列数本就不同",
    "industry": "stock_industry（个股行业归属）与 sw_l1（申万一级清单）是两张不同的表",
    "stocks": "all（tushare 当前列表）与 data（旧管线产物）是两种 schema",
}


def check_column_consistency():
    """逐数据集检查：**同一目录下的文件是否列名一致**

    为什么单独做这一条
    ------------------
    `count(*)` 不引用任何列，所以**缺列的文件不会让查询失败** —— 于是
    "第二个写入方用了不同的列集"能长期潜伏。本项目已经踩过两次：
      - `frozen/stocks/` 里 `all.parquet`（含 list_date）与 `data.parquet`
        （旧管线，只有 list_status）并存，选 `ts_code` 直接抛 schema mismatch
      - `frozen/industry/` 里 `sw_l1` 与 `stock_industry` 完全不同
    另外 `database/downloader/valuation.py` 曾经把市值换算成亿元、列名改成
    `turnover`，写进**同一个** `frozen/valuation`（已修）。

    只比**列名**、不比类型：DuckDB 能自动统一 INTEGER/DOUBLE，
    `frozen/financial`/`dividend` 逐文件类型不同是正常的，比类型会全是假阳性。
    """
    import os
    import re
    print("\n[列一致性] 同一数据集内不同文件的列名是否一致")
    con = connect_duckdb()
    bad = []
    checked = 0
    for root, label in ((FROZEN_ROOT, "frozen"), (CLEANED_ROOT, "cleaned")):
        if not root.exists():
            continue
        for ds in sorted(os.listdir(root)):
            p = root / ds
            if not p.is_dir():
                continue
            by_name = {}
            for f in p.glob("year=*/*.parquet"):
                by_name.setdefault(f.name, f)
            if not by_name:
                continue
            # 【抽样】逐文件 DESCRIBE 在 daily_basic 上有 5,500 个不同文件名，
            # 全量要十几分钟。这里取：**所有"非 6 位股票代码"的特殊文件名**
            # （第二次写入方产生的正是这类：all/data、sw_l1/stock_industry）
            # + 均匀抽取 30 个股票代码文件。
            names = sorted(by_name)
            special = [n for n in names if not re.fullmatch(r"\d{6}\.parquet", n)]
            plain = [n for n in names if re.fullmatch(r"\d{6}\.parquet", n)]
            step = max(1, len(plain) // 30)
            pick = special + plain[::step][:30]
            sigs = {}
            for name in pick:
                f = by_name[name]
                try:
                    cols = con.execute(
                        f"DESCRIBE SELECT * FROM read_parquet('{f.as_posix()}')").fetchdf()
                    sig = tuple(sorted(cols["column_name"]))
                except Exception as e:
                    sig = (f"<读取失败 {type(e).__name__}>",)
                sigs.setdefault(sig, []).append(name)
            checked += 1
            if len(sigs) > 1:
                detail = " | ".join(
                    f"{len(v)} 文件({', '.join(v[:2])}…) 列数 {len(k)}"
                    for k, v in sorted(sigs.items(), key=lambda kv: -len(kv[1])))
                if ds in MULTI_SCHEMA_DECLARED:
                    log(f"    ℹ {label}/{ds} 有 {len(sigs)} 种列集（已声明："
                        f"{MULTI_SCHEMA_DECLARED[ds]}）")
                else:
                    bad.append(f"{label}/{ds}: {len(sigs)} 种列集 -> {detail}")
    con.close()
    for b in bad:
        log(f"    ⚠ {b}")
    report(f"各数据集列名一致（查了 {checked} 个）", not bad,
           f"{len(bad)} 个数据集出现**未声明**的列集分裂" if bad else
           f"{checked} 个数据集均一致（{len(MULTI_SCHEMA_DECLARED)} 个已声明的分裂）")


# ============================================================
# 2c. 交易日历（B15）
# ============================================================
def check_calendar():
    """交易日历检查：**未来占位日不能被当成"已发生"**

    `frozen/calendar` 来自 tushare `trade_cal`，天然包含未来交易日（实测到
    2027-12-31）。把它当"已发生的交易日"用会安静地毁掉检查结论 ——
    本项目已经真实发生过一次：`check_year_completeness` 用日历最大年份判定
    "仍在上市"，2027 年没有数据，于是 `live` 为空、整条检查**空转通过**。

    这条检查做三件事：
      ① 日历本身干净（无重复、无周末、已排序）
      ② 未来占位日的**边界**是明确的（从明天起连续，不是中间挖洞）
      ③ 全库没有数据集把 `trade_date` 写到未来去（那是真的数据错误）
    """
    from database.calendar import calendar_meta, raw_calendar, trading_days
    print("\n[交易日历] 未来占位日的边界与影响")

    raw = raw_calendar()
    meta = calendar_meta()
    if len(raw) == 0:
        report("交易日历非空", False, "frozen/calendar 无数据")
        return
    log(f"日历 {meta['n']:,} 个交易日：{str(meta['first'])[:10]} ~ {str(meta['last'])[:10]}")
    log(f"  已发生 {meta['n'] - meta['n_future']:,} 天（至 {str(meta['last_past'])[:10]}），"
        f"**未来占位 {meta['n_future']:,} 天**（{str(meta['future_first'])[:10]} 起）")
    report("日历无重复", raw.is_unique, f"{len(raw) - raw.nunique()} 个重复日期")
    n_out = int((raw[1:] <= raw[:-1]).sum())
    report("日历严格递增", n_out == 0, f"{n_out} 处逆序")
    # 周末不可能是交易日（tushare 的日历里确有极少数"周末调休交易日"，
    # 例如春节前的周六补班 —— A 股实际上不交易，但历史上出现过，
    # 所以这里只做提示、不作门禁）
    wk = int((raw.dayofweek >= 5).sum())
    if wk:
        log(f"  ℹ 含 {wk} 个周六/周日（调休补班日，tushare 口径，仅提示）")

    # ① 裁剪后的日历必须以"最新已发生交易日"结尾
    clipped = trading_days()
    report("默认取到的日历已裁剪（不含未来）",
           len(clipped) == 0 or clipped.max() <= pd.Timestamp.now().normalize(),
           f"裁剪后仍到 {clipped.max() if len(clipped) else '?'}")
    # ② 未来占位段的**长度**要合理
    #    ⚠️ 不要检查"未来段是否连续"：春节/国庆本来就有 8~10 天的空档，
    #    那样写会稳定误报（第一版就误报了 3 处，全是长假）。
    #    真正该守的是"不要长得离谱"（tushare 一般只公布到下一年的年底）。
    report("未来占位段长度合理（≤500 个交易日）",
           0 <= meta["n_future"] <= 500,
           f"{meta['n_future']} 个未来交易日（到 {str(meta['last'])[:10]}）—— "
           f"过长的日历多半是脏数据或年份占位")

    # ③ 全库不许有 trade_date 落到未来（那才是真错误，不是日历问题）
    from database.config import parquet_glob
    today = pd.Timestamp.now().normalize()
    con = connect_duckdb()
    bad = []
    for ds in ("daily",):
        g = parquet_glob(dir_of(ds))
        try:
            n = con.execute(f"""SELECT count(*) FROM read_parquet('{g}')
                                WHERE trade_date > ?""", [today]).fetchone()[0]
        except Exception:
            continue
        if n:
            bad.append(f"{ds}: {n} 行")
    con.close()
    for b in bad:
        log(f"    ⚠ {b} 的 trade_date 超过了今天")
    report("行情数据不含未来日期", not bad,
           "；".join(bad) if bad else "daily_basic 无未来行")


# ============================================================
# 3. 覆盖率检查
# ============================================================
def _load_calendar():
    """交易日历 —— 转调唯一实现 `database/calendar.py`

    ⚠️ 必须用**裁剪版**（不含未来占位日）。`frozen/calendar` 覆盖到
    2027-12-31，早期直接拿它当分母/判据，出现过"覆盖率检查空转、
    完整性检查假通过"。要完整日历请显式用 `database.calendar.raw_calendar()`。
    """
    from database.calendar import trading_days
    cal = trading_days()
    if len(cal) == 0:
        return pd.DataFrame(columns=["trade_date", "is_open"])
    return pd.DataFrame({"trade_date": cal, "is_open": 1})

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
        yg = (DAILY_DIR / f"year={y}" / "*.parquet").as_posix()
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
    # 每个 (code, year) 的交易日范围 —— 用于识别**连续停牌段**并把它从分母里剔除。
    # 【为什么】长期停牌股（000757 停 5 年、000670 从 2020-03 停到 2022-08、
    # 000638 在 2016~2017 跨年停牌 …）的覆盖率天然只有 56%~80%，
    # 把它们全报成"问题"会让门禁永久变红。
    # 【判据】与 check_year_completeness 用同一套两类证据：
    #   ① 与"整年没有数据"相连的交易日（跨年中途停牌/复牌就是这个形状）
    #   ② `frozen/suspend` 里 `suspend_type='S'` 覆盖的交易日
    # 整年缺口本身仍由 check_year_completeness 另行证据化把关。
    pc_year = con.execute(f"""
        SELECT code, year(trade_date) y, min(trade_date) a, max(trade_date) b
        FROM read_parquet('{DAILY_ALL_GLOB}') GROUP BY 1, 2
    """).fetchdf()
    yspan = {}
    for r in pc_year.itertuples(index=False):
        yspan.setdefault(r.code, {})[int(r.y)] = (pd.Timestamp(r.a), pd.Timestamp(r.b))
    # 证据②：停牌记录（只取 S）
    sus_by_code = {}
    sus_dir = FROZEN_ROOT / "suspend"
    if sus_dir.exists():
        sg = (sus_dir / "year=2005" / "*.parquet").as_posix()
        sd = con.execute(f"""
            SELECT code, strptime(trade_date, '%Y%m%d')::DATE d
            FROM read_parquet('{sg}') WHERE suspend_type = 'S'
        """).fetchdf()
        for r in sd.itertuples(index=False):
            sus_by_code.setdefault(str(r.code).zfill(6), set()).add(pd.Timestamp(r.d))
    con.close()
    cal_idx = pd.DatetimeIndex(sorted(cal_days))

    def _explained_days(code, lo, hi, full):
        """[lo, hi] 内属于"停牌段"的交易日数（两类证据取并集，不重复计）"""
        exp = set()
        # 证据①：与整年缺失相连的区间
        ys = yspan.get(code)
        if ys:
            for y in sorted(ys):
                if (y + 1) in ys:
                    continue
                end_y = y
                while (end_y + 1) not in ys and (end_y + 1) <= hi.year:
                    end_y += 1
                s = ys[y][1] + pd.Timedelta(days=1)
                nxt = ys.get(end_y + 1)
                e = nxt[0] - pd.Timedelta(days=1) if nxt else hi
                s, e = max(s, lo), min(e, hi)
                if s <= e:
                    i0 = cal_idx.searchsorted(s, "left")
                    i1 = cal_idx.searchsorted(e, "right")
                    exp.update(cal_idx[i0:i1])
        # 证据②：停牌记录
        sd = sus_by_code.get(code)
        if sd:
            exp |= (sd & set(full))
        return len(exp)

    stock_issues = 0
    checked = 0
    skipped_long_halt = []
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
        # 分母 = [max(上市日, 2005), 数据末日] ∩ [该股**实际数据起点**, 末日]。
        # 【为什么从实际起点算】`000638`(*ST万方) 的 list_date 是 1996，
        # 但它在 2005~2009 是**暂停上市**、数据自然从 2009-06-05 才开始；
        # 若从 2005 起算就会凭空多出 971 个"缺失日"。**开头/结尾的整段缺失
        # 属于"整年缺口"**，已由 check_year_completeness 用证据化判据把关；
        # 这里只负责抓**区间内部**的部分缺失。职责分开，避免两处都报同一件事。
        start_day = max(lst, pd.Timestamp("2005-01-01"), row["min_d"])
        full = [c for c in cal_days if start_day <= c <= row["max_d"]]
        exp_gap = _explained_days(code, start_day, row["max_d"], full)
        expected_days = len(full) - exp_gap
        if exp_gap:
            skipped_long_halt.append((code, exp_gap))
        ratio = actual / expected_days if expected_days else 0
        if ratio < 0.8:
            report(f"{code} 覆盖率", False,
                   f"{actual}/{expected_days} 天 ({ratio:.0%}，已扣停牌段 {exp_gap})")
            stock_issues += 1
        if (ci + 1) % 1000 == 0:
            log(f"  逐股进度 {ci+1}/{len(all_codes)}")
    if skipped_long_halt:
        log(f"  有长期停牌历史、已从分母剔除相应交易日的股票: {len(skipped_long_halt)} 只"
            f"（整年缺口由 --check completeness 证据化判定）")
    report("逐股覆盖率(全量)", stock_issues == 0,
           f"问题 {stock_issues}" if stock_issues else f"全量检查 {checked} 只正常")

# ============================================================
# 4. 涨跌停价规则一致性（2026-09 新增）
# ============================================================
def check_limit_rules(sample_codes: int = 300):
    """校验 cleaned/limit_price 是否符合**唯一规则** database/limit_rules.py

    为什么必须做成门禁：这个数据集曾经有三套实现互相覆盖，其中两套用
    `close.shift(1)` 当昨收（除权日必错），且都没有"创业板 2020-08-24 前
    ±10%"这条 —— 实测有 **115 万行**（占 7.2%）涨跌停价偏宽，
    方向是**让回测偏乐观**（涨停价偏高 → 少拦买入）。
    """
    import pandas as pd
    from database.limit_rules import GEM_20PCT_FROM
    print("\n[涨跌停规则] cleaned/limit_price")
    g = f"{dir_of('limit').as_posix()}/year=*/*.parquet"
    con = connect_duckdb()
    try:
        # 【性能】全部汇总指标合并成**一次**全表扫描。1600 万行、7 万个文件，
        # 每多一次扫描就多几十秒；旧实现还逐只股票发查询（300 次扫描），
        # 单次校验超过 10 分钟、跑不完。
        r = con.execute(f"""SELECT
            count(*) n,
            sum(CASE WHEN limit_up IS NULL OR limit_down IS NULL THEN 1 ELSE 0 END) n_null,
            sum(CASE WHEN limit_up < limit_down THEN 1 ELSE 0 END) n_inv,
            sum(CASE WHEN limit_up = limit_down THEN 1 ELSE 0 END) n_eq,
            sum(CASE WHEN limit_up = limit_down AND pre_close > 0.09
                     THEN 1 ELSE 0 END) n_eq_bad,
            sum(CASE WHEN pre_close <= 0 THEN 1 ELSE 0 END) n_badpre,
            sum(CASE WHEN pre_close IS NOT NULL
                      AND abs(pre_close*100 - round(pre_close*100)) > 1e-6
                     THEN 1 ELSE 0 END) n_offgrid,
            sum(CASE WHEN (code LIKE '300%' OR code LIKE '301%')
                      AND trade_date < DATE '{GEM_20PCT_FROM.date()}'
                      AND abs(limit_up/pre_close - 1.20) < 0.005
                     THEN 1 ELSE 0 END) gem_pre20,
            sum(CASE WHEN abs(limit_up/pre_close-1.05) < 0.002 THEN 1 ELSE 0 END) st5,
            sum(CASE WHEN abs(limit_up/pre_close-1.10) < 0.005 THEN 1 ELSE 0 END) p10,
            sum(CASE WHEN abs(limit_up/pre_close-1.20) < 0.005 THEN 1 ELSE 0 END) p20,
            sum(CASE WHEN abs(limit_up/pre_close-1.30) < 0.005 THEN 1 ELSE 0 END) p30
            FROM read_parquet('{g}')""").fetchdf().iloc[0]
    except Exception as e:
        report("涨跌停价读取", False, str(e)[:80])
        con.close()
        return
    log(f"总行数 {int(r['n']):,}")
    # 【空值判据】limit 为空有且只有三种合法来源：
    #   ① 上市初期"不设涨跌幅"窗口（科创板/创业板注册制/主板 2023 起前 5 日、
    #      主板与创业板开板初期首日），2026-09 起建模，约 8,700 行
    #   ② 北交所 920xxx 在 2023 年之前 pre_close 本身就是垃圾值（C1），238 行
    #   ③ 以上都不是 -> 真问题
    # 所以这里逐行核对，而不是简单地要求"为空的行数 = 0"。
    from database.limit_rules import NO_LIMIT, listing_windows
    nulls = con.execute(f"""SELECT code, trade_date FROM read_parquet('{g}')
                            WHERE limit_up IS NULL""").fetchdf()
    win = listing_windows()
    n_window = n_junk = n_bad = 0
    for rr in nulls.itertuples(index=False):
        code, td = rr.code, pd.Timestamp(rr.trade_date)
        w = win.get(code)
        if w and w[0] == NO_LIMIT and w[1] <= td <= w[2]:
            n_window += 1
        elif str(code).startswith("920") and td < pd.Timestamp("2023-01-01"):
            n_junk += 1
        else:
            n_bad += 1
    log(f"涨跌停价为空的行: {len(nulls):,}"
        f"（上市初期不设涨跌幅 {n_window:,} / 北交所垃圾昨收 {n_junk:,}）")
    report("涨跌停价空值均可解释", n_bad == 0,
           f"无法解释的空值 {n_bad:,} 行")
    # 真倒挂（<）一行都不该有；相等（=）在价格粒度粗于涨跌幅时是**必然结果**
    # （0.09 元的 ST 股 ±5% ⇒ 上下限都四舍五入到 0.09），不是数据错误。
    report("limit_up 不小于 limit_down", int(r["n_inv"]) == 0,
           f"真倒挂 {int(r['n_inv']):,} 行")
    report("上下限相等仅出现在极低价", int(r["n_eq_bad"]) == 0,
           f"相等 {int(r['n_eq']):,} 行（均为 ≤0.09 元的粒度塌缩）"
           if int(r["n_eq_bad"]) == 0 else f"异常相等 {int(r['n_eq_bad']):,} 行")
    report("pre_close > 0", int(r["n_badpre"]) == 0, f"非正 {int(r['n_badpre'])}")
    # 分网格：整数分舍入成立的前提（若价格有 0.001 位，舍入粒度就错了）
    report("pre_close 落在 0.01 网格上", int(r["n_offgrid"]) == 0,
           f"偏离 {int(r['n_offgrid']):,} 行")
    report("创业板 2020-08-24 前应为 ±10%", int(r["gem_pre20"]) == 0,
           f"仍按 20% 的行 {int(r['gem_pre20']):,}" if r["gem_pre20"] else "符合")

    # 最强的一条：用唯一实现逐行重算并比对（能同时抓住规则错和舍入错）
    # 只发**一次**查询把所有抽样股票取回来，再在 pandas 里按股票分组重算。
    from database.limit_rules import (apply_limit_prices, listing_windows,
                                      load_st_intervals)
    codes = con.execute(f"""SELECT DISTINCT code FROM read_parquet('{g}')
                            ORDER BY code LIMIT {sample_codes}""").fetchdf()["code"].tolist()
    st_map = load_st_intervals()
    windows = listing_windows()
    diff = n = 0
    if codes:
        ph = ",".join(["?"] * len(codes))
        s = con.execute(f"""SELECT code, trade_date, pre_close, limit_up, limit_down
                            FROM read_parquet('{g}') WHERE code IN ({ph})""",
                        codes).fetchdf()
        for c, sub in s.groupby("code"):
            exp = apply_limit_prices(sub, c, st_map,
                                     listing_rule=windows.get(c))
            n += len(sub)
            diff += int((abs(exp["limit_up"]
                             - pd.to_numeric(sub["limit_up"])) > 0.005).sum())
            diff += int((abs(exp["limit_down"]
                             - pd.to_numeric(sub["limit_down"])) > 0.005).sum())
    report(f"逐行重算一致（抽样 {len(codes)} 只 / {n:,} 行）", diff == 0,
           f"不一致 {diff:,} 处" if diff else "与唯一规则完全一致")
    con.close()

    # 各档位占比（异常占比说明规则跑偏）
    tot = int(r["n"])
    known = sum(int(r[k]) for k in ("st5", "p10", "p20", "p30"))
    log(f"档位分布: 5%={int(r['st5']):,} 10%={int(r['p10']):,} "
        f"20%={int(r['p20']):,} 30%={int(r['p30']):,}  合计 {known/tot:.2%}")
    report("涨跌停档位可识别率 > 99%", known / tot > 0.99,
           f"仅 {known/tot:.2%} 落在 5/10/20/30% 档")


# ============================================================
# 4b. 上市初期规则（2026-09 新增）
# ============================================================
def check_listing_rules():
    """上市初期特殊规则的**实证**校验

    制度条文容易记错，所以这里不掉书袋，直接用**行情数据反证**：
      ① 不设涨跌幅的窗口内，确实存在价格突破常规档位的交易日
         （若一条都没有，说明规则要么没生效、要么根本就不该有）
      ② 全库最高价不得超过涨停价、最低价不得低于跌停价
         —— ±44% 的首日也必须满足这条
      ③ 首日 ±44% 确实"在 44% 处封顶"：存在大量贴着 44% 的涨停行
    """
    print("\n[上市初期规则] 前 N 日不设涨跌幅 / 首日 ±44%")
    from database.config import parquet_glob
    lg = parquet_glob(dir_of("limit"))
    dg = parquet_glob(dir_of("daily"))
    con = connect_duckdb()
    # 北交所 920xxx 在 2023 年之前 pre_close 是垃圾值（C1），
    # 它的涨跌停价本身无意义，必须排除，否则门禁永远红着。
    NOT_JUNK = "NOT (l.code LIKE '920%' AND l.trade_date < DATE '2023-01-01')"
    try:
        r = con.execute(f"""
            WITH l AS (SELECT code, trade_date, pre_close, limit_up, limit_down
                       FROM read_parquet('{lg}')),
                 d AS (SELECT code, trade_date, high, low
                       FROM read_parquet('{dg}'))
            SELECT count(*) n,
              sum(CASE WHEN l.pre_close > 0 AND l.limit_up IS NULL
                       THEN 1 ELSE 0 END) nolimit,
              sum(CASE WHEN l.pre_close > 0 AND l.limit_up IS NULL
                        AND d.high > l.pre_close * 1.31
                       THEN 1 ELSE 0 END) broke_band,
              sum(CASE WHEN l.limit_up IS NOT NULL AND {NOT_JUNK}
                        AND d.high > l.limit_up + 0.011
                       THEN 1 ELSE 0 END) over_up,
              sum(CASE WHEN l.limit_down IS NOT NULL AND {NOT_JUNK}
                        AND d.low < l.limit_down - 0.011
                       THEN 1 ELSE 0 END) under_dn,
              sum(CASE WHEN l.pre_close > 0 AND l.limit_up IS NOT NULL
                        AND abs(l.limit_up / l.pre_close - 1.44) < 0.005
                       THEN 1 ELSE 0 END) n44
            FROM l JOIN d USING (code, trade_date)
        """).fetchdf().iloc[0]
    except Exception as e:
        report("上市初期规则读取", False, str(e)[:80])
        con.close()
        return
    n = int(r["n"])
    log(f"配对行数 {n:,}")
    log(f"  不设涨跌幅的行（limit 为空且 pre_close>0）: {int(r['nolimit']):,}")
    log(f"    其中价格突破 ±31%（任何常规档位都不可能）: {int(r['broke_band']):,}")
    log(f"  首日 ±44% 的行: {int(r['n44']):,}")
    report("不设涨跌幅的窗口确实存在", int(r["nolimit"]) > 0,
           f"{int(r['nolimit']):,} 行")
    # 若窗口内的价格从未突破常规档位，说明这条规则对数据毫无影响 —— 要么规则
    # 没被真正应用，要么适用面判断错了，两种情况都该人工看一眼。
    report("不设涨跌幅窗口内有突破常规档位的行情",
           int(r["broke_band"]) > 0,
           f"{int(r['broke_band']):,} 行（少于此数说明规则可能没生效）")
    report("首日 ±44% 规则生效", int(r["n44"]) > 0, f"{int(r['n44']):,} 行")

    # 价格越界：**用比率门禁 + 归因**，不用绝对 0。
    # 历史上有少量零散的制度性例外（2006-2007 未股改股、2014 的 6 行 +45.2% 等），
    # 要求绝对 0 会让门禁永久变红、失去报警价值。这里给出占比与最大越界者。
    for name, cnt in (("最高价 > 涨停价", int(r["over_up"])),
                      ("最低价 < 跌停价", int(r["under_dn"]))):
        rate = cnt / n if n else 0
        report(f"{name}（占比 < 0.02%）", rate < 0.0002,
               f"{cnt:,} 行 / {rate:.4%}")
    if int(r["over_up"]):
        try:
            top = con.execute(f"""
                WITH l AS (SELECT code, trade_date, pre_close, limit_up
                           FROM read_parquet('{lg}')),
                     d AS (SELECT code, trade_date, high FROM read_parquet('{dg}'))
                SELECT year(l.trade_date) y, count(*) c FROM l JOIN d USING (code, trade_date)
                WHERE l.limit_up IS NOT NULL AND {NOT_JUNK}
                  AND d.high > l.limit_up + 0.011
                GROUP BY 1 ORDER BY 2 DESC LIMIT 6
            """).fetchdf()
            log("    越界最多的年份: " +
                ", ".join(f"{int(x.y)}={int(x.c)}" for x in top.itertuples(index=False)))
        except Exception:
            pass
    con.close()



# ============================================================
# 5. 年份完整性（2026-09 新增）
# ============================================================
def check_year_completeness():
    """逐 code 检查"该有的年份"是否都在（只查**仍在上市**的股票）

    退市股、整年停牌股有缺口是正常的，所以只用 list_status='L' 的股票做判据。
    这条能抓住"下载被静默截断/部分失败被标记完成"这类问题。
    """
    import pandas as pd
    print("\n[年份完整性] frozen/daily_raw vs cleaned/daily_basic")
    # cleaned 与 frozen 的文件数应一一对应（除非清洗层按规则剔除）
    raw, cle = dir_of("daily_raw"), dir_of("daily")
    diffs = []
    for y in range(2005, 2027):
        a, b = raw / f"year={y}", cle / f"year={y}"
        sa = {f.stem for f in a.glob("*.parquet")} if a.exists() else set()
        sb = {f.stem for f in b.glob("*.parquet")} if b.exists() else set()
        if sa - sb:
            diffs.append((y, sorted(sa - sb)[:5], len(sa - sb)))
    if diffs:
        log(f"清洗层比原始层少的文件: {[(y, c) for y, _, c in diffs][:8]}")
        log("提示：少量差异通常是被清洗规则剔除的退化行（如 volume=1 股、"
            "VWAP 越界），属正常；数量大则说明清洗配方有问题")
    report("清洗层与原始层文件数一致（允许少量剔除）",
           sum(c for _, _, c in diffs) < 50, f"共 {sum(c for _, _, c in diffs)} 个")

    # 仍在上市股票的整年缺口
    con = connect_duckdb()
    # ⚠️ frozen/stocks 一个分区里并存两种 schema：
    #     all.parquet  = code/ts_code/name/list_date/industry/market（tushare 当前列表）
    #     data.parquet = code/name/list_status（旧管线产物）
    #   用 `year=*/*.parquet` 一把捞会因缺列抛 schema mismatch，所以按文件名直取。
    stk_f = FROZEN_ROOT / "stocks" / "year=2005" / "all.parquet"
    if not stk_f.exists():
        report("股票列表读取", False, f"{stk_f} 不存在")
        return
    try:
        stk = con.execute(f"SELECT code, list_date FROM "
                          f"read_parquet('{stk_f.as_posix()}')").fetchdf()
    except Exception as e:
        report("股票列表读取", False, str(e)[:80])
        return
    stk["code"] = stk["code"].astype(str).str.zfill(6)
    ld = dict(zip(stk["code"], pd.to_datetime(stk["list_date"], errors="coerce")))
    # 各年开市日数 —— 取自**裁剪版**日历（不含未来占位日），否则 2027 这种
    # 没有数据的年份会混进分母，把整年缺口的判据带偏。
    from database.calendar import trading_days
    _cal = trading_days()
    _s = pd.Series(1, index=_cal).groupby(_cal.year).sum()
    cd = {int(y): int(n) for y, n in _s.items()}

    from collections import defaultdict
    # 用 DuckDB 一次聚合出每个 (code, year) 的交易日范围：
    # 判定"整年缺口"是否**有交易断裂作证据**需要相邻年份的首末交易日。
    dg = parquet_glob(FROZEN_ROOT / "daily_raw")
    agg = con.execute(f"""
        SELECT code, year(trade_date) y, min(trade_date) a, max(trade_date) b
        FROM read_parquet('{dg}') GROUP BY 1, 2
    """).fetchdf()
    con.close()
    if agg.empty:
        report("日线数据读取", False, "frozen/daily_raw 无数据")
        return
    agg["code"] = agg["code"].astype(str).str.zfill(6)
    agg["y"] = agg["y"].astype(int)
    years = defaultdict(set)
    ymin, ymax = {}, {}
    for r in agg.itertuples(index=False):
        years[r.code].add(r.y)
        ymin[(r.code, r.y)] = pd.Timestamp(r.a)
        ymax[(r.code, r.y)] = pd.Timestamp(r.b)

    # "仍在上市" 用**最新年份仍有日线数据**判定，比依赖 list_status 更可靠
    # （list_status 只在旧管线的 data.parquet 里，且不含北交所）。
    # ⚠️ last_year 必须取**日线数据自身**的最大年份：交易日历里带着未来的
    #    占位年份（如 2027），用它会导致 live 为空、检查**假通过**。
    all_years = {y for ys in years.values() for y in ys}
    last_year = max(all_years) if all_years else 0
    live = {c for c, ys in years.items() if last_year in ys}
    log(f"日线数据最新年份 {last_year}，该年仍有数据的股票 {len(live):,} 只")
    report("仍在上市股票集合非空", len(live) > 100,
           f"只识别出 {len(live)} 只 —— 年份判定可能出错（检查会假通过）")

    # 整年缺口分两类：
    #   ① **有断裂证据**：缺口前一年在 12 月附近就停止交易、或缺口后一年
    #      要到 1 月之后才恢复 -> 真实的长期停牌/暂停上市（例如 000757
    #      方向光电 2007-04 停牌、2013-02 才复牌，整整 5 年没有日线）。
    #      注意 `frozen/suspend` **不记录**这种"暂停上市"，不能拿它当依据。
    #   ② **无断裂证据**：前后都在正常交易，却整年没有数据 -> 数据缺失，真问题。
    # 第二类证据：`frozen/suspend` 里的 S 记录覆盖了该年的大部分交易日。
    # 两条证据互补 —— 长期"暂停上市"（如 000757 停 6 年）不在 suspend 里，
    # 而 002015 的 2015 年整年停牌则**要靠** suspend 才能解释。
    sus_year = {}
    sus_dir = FROZEN_ROOT / "suspend"
    if sus_dir.exists():
        try:
            sg = f"{(sus_dir / 'year=2005' / '*.parquet').as_posix()}"
            sdf = connect_duckdb()
            sy = sdf.execute(f"""
                SELECT code, year(strptime(trade_date, '%Y%m%d')) y, count(*) n
                FROM read_parquet('{sg}')
                WHERE suspend_type = 'S' GROUP BY 1, 2
            """).fetchdf()
            sdf.close()
            for r in sy.itertuples(index=False):
                sus_year[(str(r.code).zfill(6), int(r.y))] = int(r.n)
        except Exception as e:
            log(f"（停牌记录读取失败，仅用价格断裂判定: {type(e).__name__}）")

    bad, explained = [], []
    for code in sorted(live):
        ys = years.get(code)
        ldate = ld.get(code)
        if not ys or pd.isna(ldate):
            continue
        y0 = max(ldate.year, 2005)
        exp = {y for y in range(y0, last_year + 1) if cd.get(y, 0) >= 20}
        exp.discard(y0)                       # 上市当年不完整属正常
        # 连续缺失年份要合并成**一段**再用两端之外的数据判定 ——
        # 否则 000156 缺 2007~2010 这种会被逐年后看，次年也缺就判成"无证据"。
        runs = []
        for y in sorted(exp - ys):
            if runs and y == runs[-1][1] + 1:
                runs[-1][1] = y
            else:
                runs.append([y, y])
        for a, b in runs:
            pm, nm = ymax.get((code, a - 1)), ymin.get((code, b + 1))
            if pm is None and nm is None:
                # 两侧都没有数据：只可能是该股数据集的边界（此前未上市/此后已退市）
                broken = (b < min(ys)) or (a > max(ys))
            else:
                broken = ((pm is not None and pm < pd.Timestamp(a - 1, 12, 1))
                          or (nm is not None and nm > pd.Timestamp(b + 1, 1, 31)))
            # 证据二：缺口年份的交易日大多有停牌记录
            cov = sum(1 for y in range(a, b + 1) if cd.get(y, 0) and
                      sus_year.get((code, y), 0) >= 0.8 * cd[y])
            if not broken and cov:
                broken = True
            (explained if broken else bad).append((code, a, b))
    log(f"仍在上市股票 {len(live):,} 只")
    log(f"  整年缺口有交易断裂证据（长期停牌/暂停上市）: {len(explained)} 段")
    for code, a, b in explained[:6]:
        span = f"{a}" if a == b else f"{a}~{b}"
        log(f"    {code} 缺 {span}（缺口前最后交易日 {ymax.get((code, a - 1), '?')}）")
    log(f"  整年缺口**无**断裂证据（疑似数据缺失）: {len(bad)} 段")
    for code, a, b in bad[:10]:
        span = f"{a}" if a == b else f"{a}~{b}"
        log(f"    {code} 缺 {span} 前一年末 {ymax.get((code, a - 1), '?')} "
            f"后一年初 {ymin.get((code, b + 1), '?')}")
    report("整年缺口均有交易断裂证据", not bad,
           f"{len(bad)} 段疑似数据缺失" if bad
           else f"{len(explained)} 段缺口全部有断裂证据（真实长期停牌）")


# ============================================================
# 6. 交易状态面板（2026-09 新增）
# ============================================================
def check_status_panels():
    """校验"停牌 / ST / 涨跌停"三张状态面板**真的能读到数据**

    为什么必须是门禁
    ----------------
    这三张面板都曾经因为**静默吞异常**而变成空表，而空表在回测里的表现是
    "所有股票任何一天都能交易、ST 股正常留在池子里"——不报错，只是结论错：

      停牌：`frozen/suspend` 是非年度数据集（只有占位分区 `year=2005`），
            按回测区间拼 `year=2024/*.parquet` -> 路径不存在 -> DuckDB 抛
            IOException -> 被 `except Exception` 吞掉 -> 面板为空。
            另外 `suspend.trade_date` 是 VARCHAR 'YYYYMMDD'，不是 DATE。
      ST  ：`universe/pool.py` 自己实现了一遍，读取失败时静默返回全 False。

    所以这里逐项断言"非空 + 覆盖合理"。
    """
    import pandas as pd
    from database.status import suspended_wide
    print("\n[交易状态面板] 停牌 / ST / 涨跌停")

    dates = pd.bdate_range("2024-01-02", "2024-12-31")
    try:
        from database.config import connect_duckdb, dir_of
        con = connect_duckdb()
        codes = con.execute(f"""
            SELECT DISTINCT code FROM read_parquet(
                '{dir_of('daily').as_posix()}/year=2024/*.parquet')
        """).fetchdf()["code"].tolist()
        con.close()
    except Exception as e:
        report("状态面板：取 2024 股票池", False, str(e)[:70])
        return
    if not codes:
        report("状态面板：取 2024 股票池", False, "2024 年没有股票")
        return
    sample = codes[:2000]
    log(f"2024 年股票池 {len(codes):,} 只（抽样 {len(sample):,} 只验证面板）")

    # ① 停牌
    try:
        sus = suspended_wide(dates, sample)
        n = int(sus.values.sum())
        report("停牌面板非空", n > 0,
               "全为 False —— 停牌股会被当成可正常交易" if n == 0
               else f"{n:,} 个停牌标记")
    except Exception as e:
        report("停牌面板可加载", False, f"{type(e).__name__}: {str(e)[:70]}")

    # ② ST
    try:
        from universe.pool import st_panel
        sp = st_panel(dates, sample)
        n = int(sp.values.sum())
        report("ST 面板非空", n > 0,
               "全为 False —— ST 股会被当成正常股留在池子里" if n == 0
               else f"{n:,} 个 ST 标记")
    except Exception as e:
        report("ST 面板可加载", False, f"{type(e).__name__}: {str(e)[:70]}")

    # ③ 涨跌停面板能否拼装
    lim_dir = dir_of("limit")
    if lim_dir.exists() and any(lim_dir.glob("year=2024/*.parquet")):
        try:
            from backtest.panel_data import load_status_panels
            st = load_status_panels("2024-01-01", "2024-12-31")
            lu = st.get("limit_up")
            report("涨跌停面板非空", lu is not None and not lu.empty,
                   f"{None if lu is None else lu.shape}")
        except Exception as e:
            report("涨跌停面板可加载", False, f"{type(e).__name__}: {str(e)[:70]}")
    else:
        log("（2024 年无 limit_price 分区，跳过涨跌停面板拼装检查）")


def check_date_columns():
    """校验各数据集里的 VARCHAR 日期列**能被按 YYYYMMDD 解析**

    全库审计发现日期列类型不统一：
        TIMESTAMP  daily_raw / adjust / valuation / limit_price / st.start_date …
        VARCHAR    suspend.trade_date / index_cons.trade_date / dividend.* /
                   financial.ann_date / holders.ann_date / st.end_date …
    VARCHAR 一旦被读取方按另一种格式解析，就会静默变成 NaT
    （`errors="coerce"`），于是"停牌/成分股"这类数据整体失效。
    """
    import os
    from database.config import CLEANED_ROOT, FROZEN_ROOT, connect_duckdb, year_globs
    from database.dates import is_date_column

    print("\n[VARCHAR 日期列可解析性]")
    con = connect_duckdb()
    bad, mixed, checked = [], [], 0
    for root in (FROZEN_ROOT, CLEANED_ROOT):
        if not root.exists():
            continue
        for ds in sorted(os.listdir(root)):
            p = root / ds
            if not p.is_dir():
                continue
            cand = None
            for yd in sorted(p.glob("year=*")):
                fs = list(yd.glob("*.parquet"))
                if fs:
                    cand = fs[0]
                    break
            if cand is None:
                continue
            try:
                cols = con.execute(
                    f"DESCRIBE SELECT * FROM read_parquet('{cand.as_posix()}')").fetchdf()
            except Exception:
                continue
            g = year_globs(p)
            if g == "[]":
                continue
            for r in cols.itertuples():
                nm, ty = r.column_name, r.column_type
                if "VARCHAR" not in ty or not is_date_column(nm):
                    continue
                checked += 1
                q, err = None, None
                # 先按普通方式读（LIMIT 能下推到扫描，内存友好）；
                # 只有同一目录下并存不同 schema 时才需要 union_by_name
                # （`frozen/industry` 的 sw_l1 与 stock_industry），
                # 而 union_by_name 会让 DuckDB 多物化，financial 上直接 OOM。
                for extra in ("", ", union_by_name=true"):
                    try:
                        q = con.execute(f"""
                            SELECT count(*) n,
                                   sum(CASE WHEN try_strptime({nm}, '%Y%m%d') IS NULL
                                            THEN 1 ELSE 0 END) mixed,
                                   sum(CASE WHEN try_strptime({nm}, '%Y%m%d') IS NULL
                                             AND try_cast({nm} AS TIMESTAMP) IS NULL
                                            THEN 1 ELSE 0 END) unparseable
                            FROM (SELECT {nm} FROM read_parquet({g}{extra})
                                  WHERE {nm} IS NOT NULL LIMIT 50000)
                        """).fetchdf().iloc[0]
                        break
                    except Exception as e:
                        err = e
                if q is None:
                    bad.append(f"{ds}.{nm}: 查询失败 {type(err).__name__}")
                    continue
                n = int(q["n"])
                if n and int(q["unparseable"]) / n > 0.01:
                    bad.append(f"{ds}.{nm}: {int(q['unparseable']):,}/{n:,} 无法解析")
                elif int(q["mixed"]):
                    mixed.append(f"{ds}.{nm}: {int(q['mixed']):,}/{n:,} 带时间格式"
                                 f"（容错解析已覆盖）")
    con.close()
    log(f"检查了 {checked} 个 VARCHAR 日期列（每列抽样 ≤5 万行）")
    for b in mixed[:10]:
        log(f"    ℹ {b}")
    for b in bad[:10]:
        log(f"    ✗ {b}")
    report("VARCHAR 日期列均可解析", not bad,
           f"{len(bad)} 列无法解析" if bad else
           f"{checked} 列全部可解析（其中 {len(mixed)} 列为混合格式，已容错兼容）")


# ============================================================
# 6b. 行级完全重复（下载器批量 concat 后没去重）
# ============================================================
# 为什么必须单独查这个
# --------------------
# `--check unique` 查的是**主键**重复，而且只覆盖 daily/valuation/adjust 三个
# 数据集。但真正发生过的事故是：`MarginDownloader` 按季度拉数时把 `end_date`
# 写成了**年末**（`f"{q[:4]}1231"`），于是 Q1 拉了 1~12 月、Q2 又拉 4~12 月……
# 10 月以后的数据被拉了 4 遍，`pd.concat` 之后**没有 drop_duplicates**，
# 整行重复被原样写进 parquet。
#
# 后果很隐蔽：主键是 `(trade_date, exchange_id)`，**每个主键仍然唯一**
# （重复行之间也一样），所以主键唯一性检查完全查不出来；但任何
# `sum(rzye)` 都会拿到 ~2.5 倍的真实值。市场级信号、两融情绪指标全被污染。
#
# 所以这里查的是**整行**（所有列）完全一致，且**逐文件**做 —— 逐文件有两个
# 好处：① 内存可控（`daily_raw`/`financial` 整库 DISTINCT 会 OOM）；② 能直接
# 指出是哪个文件坏的。
#
# 为什么分两类扫
# --------------
# 实测各数据集的文件命名（2026-09 审计）：
#   **批量写入** — 每个文件装**很多标的**或**整个市场**：`margin/data.parquet`
#       （一年一文件）、`etf|options/2020-01-02.parquet`（一天一文件）、
#       `futures/IC.parquet`。这类才会因为"多批 API 结果 concat"产生重复，
#       **全量扫**，文件数少（合计约 3,400 个）。
#   **按代码分文件** — `year=Y/{code}.parquet` 只装一只标的：daily_raw(7.3万)、
#       valuation(7.2万)、adjust(7.3万)、financial(1.7万)、st/suspend/dividend/
#       holders 各数千。文件数太大（20 万+）逐个 `DISTINCT *` 不现实，改成
#       **抽样**：若是系统性的 concat 缺陷，每个文件都会中招，抽几十个必然命中
#       （margin 那个 bug 就是 17 个文件全中）。
PER_CODE_DATASETS = {
    "daily_raw", "valuation", "adjust", "daily_basic", "limit_price",
    "financial", "holders", "dividend", "st", "suspend", "index_daily",
}
# 抽样模式下每个「按代码」数据集查多少个文件
SAMPLE_PER_CODE_FILES = 25


def _dup_count(con, path) -> tuple:
    """一个 parquet 文件里 (总行数, 去重后行数)"""
    p = path.as_posix()
    return con.execute(
        f"SELECT (SELECT count(*) FROM read_parquet('{p}')), "
        f"(SELECT count(*) FROM (SELECT DISTINCT * FROM read_parquet('{p}')))"
    ).fetchone()


def _spread_sample(files: list, n: int) -> list:
    """从文件列表里**跨整个区间均匀**取 n 个

    ⚠️ 不能直接 `files[:n]`：文件是按 `year=YYYY/代码` 排序的，前 25 个全是
    2010 年的，等于只查了最老的一年 —— 而下载器 bug 往往只影响某段时间
    （比如 margin 是"10 月以后被拉 4 遍"），偏采样会直接漏掉。
    """
    if len(files) <= n:
        return files
    step = len(files) / n
    return [files[int(i * step)] for i in range(n)]


def check_row_duplicates(repair: bool = False):
    """逐文件检查「整行完全重复」

    覆盖策略（实测耗时，2026-09）：
      **批量数据集全扫**（margin 17 个文件、etf/options 各 1,627 个日期文件…）
        —— 一次 `SELECT DISTINCT *` 约 0.01 秒，合计 ~13 秒。
      **按代码数据集抽样**（`year=Y/{code}.parquet`，daily_raw/valuation/adjust
        各 7 万余个，合计 20 万+）—— 逐个扫不现实；但若是系统性的 concat 缺陷，
        每个文件都会中招，跨年均匀抽 25 个必然命中（margin 的 bug 就是 17 个
        文件全中）。**这是抽样，不是全量**，报告里会写明。

    repair=True 时把重复行**就地去掉**（原子写回）。去重去掉的是"API 根本没返回过、
    纯由下载器区间重叠造出来的副本"，属于**还原**而不是篡改原始数据；但默认关闭，
    必须显式 `--repair-duplicates` 才动数据。
    """
    from database.config import FROZEN_ROOT, connect_duckdb
    print("\n[整行重复] 逐文件比对 count(*) 与 count(DISTINCT *)"
          f"{'（并就地修复）' if repair else ''}")
    if not FROZEN_ROOT.exists():
        report("frozen 目录存在", False, "不存在")
        return
    bad, checked, full_ds, sampled_ds, repaired, unreadable = [], 0, [], [], [], 0
    for d in sorted(FROZEN_ROOT.iterdir()):
        if not d.is_dir():
            continue
        files = sorted(d.glob("year=*/*.parquet"))
        if not files:
            continue
        if d.name in PER_CODE_DATASETS:
            files = _spread_sample(files, SAMPLE_PER_CODE_FILES)
            sampled_ds.append(d.name)
        else:
            full_ds.append(d.name)
        # 每个数据集单独建连接：几万次查询堆在一个连接上会把 DuckDB 拖到 OOM
        con = connect_duckdb()
        ds_bad, ds_dup, ds_rows = [], 0, 0
        try:
            for f in files:
                try:
                    n, u = _dup_count(con, f)
                except Exception as e:
                    unreadable += 1
                    if unreadable <= 5:
                        log(f"  ⚠ {d.name}/{f.name}: 读取失败 {type(e).__name__}")
                    continue
                checked += 1
                ds_rows += n
                if n != u:
                    ds_dup += n - u
                    ds_bad.append((f.name, n, n - u))
                    if repair:
                        try:
                            _dedupe_file(f)
                            repaired.append((d.name, f.name, n - u))
                        except Exception as e:
                            log(f"  ✗ {d.name}/{f.name} 修复失败: "
                                f"{type(e).__name__}: {e}")
        finally:
            con.close()
        if ds_bad:
            bad.append((d.name, ds_rows, ds_dup, ds_bad))

    for name, rows, dup, detail in bad:
        top = ", ".join(f"{fn}(+{dr:,})" for fn, _, dr in detail[:3])
        log(f"  ✗ {name}: {len(detail)} 个文件有整行重复，多出 {dup:,} 行 "
            f"[{top}{'...' if len(detail) > 3 else ''}]")
    if unreadable:
        log(f"  ⚠ 有 {unreadable} 个文件读不出来（无法判定重复）")
    if repaired:
        log(f"  已修复 {len(repaired)} 个文件，去掉 "
            f"{sum(r[2] for r in repaired):,} 行重复")
    total_dup = sum(b[2] for b in bad)
    detail = (f"全量扫 {len(full_ds)} 个批量数据集 + 抽样 {len(sampled_ds)} 个"
              f"按代码数据集（每个 {SAMPLE_PER_CODE_FILES} 文件），"
              f"共 {checked:,} 个文件")
    if bad:
        report("frozen 无整行重复", False,
               f"{len(bad)} 个数据集有整行重复（多出 {total_dup:,} 行）—— "
               f"下游聚合会被放大"
               + ("；已修复" if repair else "；加 --repair-duplicates 可就地修复"))
    else:
        report("frozen 无整行重复", True,
               f"{detail}，未发现整行重复"
               + ("（已修复）" if repair and repaired else ""))


def _dedupe_file(path):
    """去掉一个 parquet 文件里的整行重复，原子写回

    复用 `Storage._atomic_to_parquet`：直接 `to_parquet(最终路径)` 时若进程被杀，
    会留下读不出来的文件，而 `save()` 的 exists 跳过逻辑会让它永不被重写。
    """
    import pandas as pd

    from database.storage import Storage
    df = pd.read_parquet(path)
    ded = df.drop_duplicates().reset_index(drop=True)
    if len(ded) == len(df):
        return 0
    Storage._atomic_to_parquet(ded, path)
    return len(df) - len(ded)


# ============================================================
# 6c. parquet 文件完整性（半截文件）
# ============================================================
def check_parquet_integrity(all_years: bool = False):
    """检查 parquet 文件是否完整（头尾魔数）

    **为什么要有这个检查**：非原子写（`df.to_parquet(最终路径)`）被中断时，
    文件会只剩一半 —— 头魔数 `PAR1` 还在，尾部变成零字节。实测
    `frozen/daily_raw/year=2026/601089.parquet` 就是 1570 字节的截断文件。
    它比"缺文件"更坏：读取方常见的 `except Exception: continue` 会把它当成
    "这只股票没数据"跳过，于是坏文件**永远不会被重写**，一直烂到某次
    全量重建才炸。

    已修：`scripts/daily_update.py` 与 `database/storage.py` 一律走
    `atomic_to_parquet`（写临时文件 -> 校验 footer -> os.replace）。
    这里只是**事后哨兵**，用于发现别的原因造成的损坏（磁盘故障、别处直接写）。

    ⚠️ 默认**只查每个数据集最新一年的分区** —— 那是日常更新写入的部分。
    实测 1~25 秒（取决于系统文件缓存冷热；冷缓存约 22 秒）；
    全库 25 万文件要 5 分钟以上，因此全量扫描必须显式 `--parquet-all`
    （见 AGENTS.md 第 1 节：全局校验先问）。

    占位年份（`config.NON_ANNUAL_YEAR`，financial/holders/st/suspend 这些
    按代码分文件的静态数据集都塞在 `year=2005`）在默认模式下**跳过**：
    它们有 1.6 万+ 文件却不是"最近更新"的，把它们算进"最新一年"会让这个
    检查从 30 秒变成 200 秒（第一版就是这么写的，实测 201 秒）。
    """
    import time

    from database.config import FROZEN_ROOT, NON_ANNUAL_YEAR
    from database.storage import find_corrupt_parquet
    mode = "全部年份" if all_years else "每个数据集最新一年（跳过占位年份）"
    print(f"\n[parquet 完整性] 检查半截/损坏文件（{mode}）")
    if not FROZEN_ROOT.exists():
        report("frozen 目录存在", False, "不存在")
        return
    t0 = time.time()
    bad, checked, ds_n, skipped = [], 0, 0, []
    for d in sorted(FROZEN_ROOT.iterdir()):
        if not d.is_dir():
            continue
        years = sorted(d.glob("year=*"))
        if not years:
            continue
        if all_years:
            targets = years
        else:
            last = years[-1]
            if last.name == f"year={NON_ANNUAL_YEAR}":
                skipped.append(d.name)
                continue
            targets = [last]
        ds_n += 1
        for y in targets:
            files = list(y.glob("*.parquet"))
            checked += len(files)
            for f in find_corrupt_parquet(y, "*.parquet"):
                bad.append((d.name, y.name, f.name, f.stat().st_size))
    dt = time.time() - t0
    for ds, yr, fn, sz in bad[:20]:
        log(f"  ✗ {ds}/{yr}/{fn}  ({sz} 字节，魔数不对 = 半截文件)")
    if len(bad) > 20:
        log(f"  ... 另有 {len(bad) - 20} 个")
    if skipped:
        log(f"  ℹ 跳过占位年份数据集: {', '.join(skipped)}"
            f"（--parquet-all 可覆盖）")
    report("parquet 文件完整", not bad,
           f"{len(bad)} 个文件损坏（半截文件，读取方会当成'没数据'静默跳过）—— "
           f"用 `python scripts/daily_update.py --repair-corrupt` 隔离并重下整年"
           if bad else
           f"检查 {checked:,} 个文件（{ds_n} 个数据集，{mode}），耗时 {dt:.1f} 秒，"
           f"全部完整")


# ============================================================
# 7. 数据新鲜度与消费方（B12）
# ============================================================
# 每个数据集的"新鲜度预算"：`(类型说明, 提示阈值, 硬失败阈值)`，单位=天
#
# 为什么分两级：
#   **提示阈值** = 期望的更新节奏。落后超过它就说明该更新了 —— 但"数据还没更新"
#                 是**运维状态**，不是数据错误，不该让 `--check all` 永久变红
#                 （那会让人对告警麻木，反而漏掉真问题）。
#   **硬失败阈值** = 结构性损坏。日频数据落后两个月，那不是"没更新"而是
#                 **下载坏了** —— `etf`/`options` 正是如此（每年只剩 1 天数据）。
# 基准是 `database.calendar.latest_trading_day()`，**不是日历最大日期**
# （那个是未来占位日，见 B15 与 `--check calendar`）。
FRESHNESS_BUDGET = {
    # dataset:        (类型,          提示, 硬失败)
    "daily_raw": ("日频", 7, 60), "valuation": ("日频", 7, 60),
    "adjust": ("日频", 7, 60), "index_daily": ("日频", 7, 60),
    "fund_daily": ("日频", 7, 60), "margin": ("日频(全市场)", 7, 60),
    "northbound": ("日频(全市场)", 7, 60), "futures": ("日频(全市场)", 7, 60),
    "etf": ("月度快照", 45, 200), "options": ("月度快照", 45, 200),
    "index_cons": ("月度快照", 45, 200),
    "holders": ("季度", 150, 400), "financial": ("季度", 150, 400),
    # 静态/事件流：没有"新鲜度"概念，不参与
    "stocks": ("静态", None, None), "st": ("静态", None, None),
    "suspend": ("静态", None, None), "industry": ("静态", None, None),
    "dividend": ("事件流", None, None), "calendar": ("日历", None, None),
}

# 数据集 -> 用来判新鲜度的日期列（None = 自动挑第一个以 date 结尾的列）
FRESHNESS_DATE_COL = {
    "adjust": "trade_date", "valuation": "trade_date", "daily_raw": "trade_date",
    "etf": "trade_date", "options": "trade_date", "futures": "trade_date",
    "margin": "trade_date", "northbound": "trade_date",
    "index_daily": "trade_date", "index_cons": "trade_date",
    "fund_daily": "trade_date", "holders": "ann_date", "financial": "ann_date",
    "st": "start_date", "stocks": "list_date", "industry": "list_date",
}


def _max_date_of(ds_dir, col=None, tail_partitions: int = 2):
    """数据集里日期列的最大值

    ⚠️ 两个必须做对的地方（第一版审计脚本两处都踩了）：
      1. `YYYYMMDD` 是 VARCHAR，`try_cast(... AS TIMESTAMP)` 会**全变 NULL**，
         只剩 ISO 格式那几行能显示 —— 于是 `holders` 被误报成"落后 411 天"。
         必须 `try_strptime` + ISO 兜底。
      2. 年度数据集不必全扫：最大值必在末尾分区里。静态数据集（只有一个
         `year=2005`）只能全扫，数据量不大。
    """
    import os
    from database.config import connect_duckdb
    yds = sorted(ds_dir.glob("year=*"))
    if not yds:
        return None, None
    if len(yds) > tail_partitions:
        yds = yds[-tail_partitions:]
    g = "[" + ", ".join(f"'{y.as_posix()}/*.parquet'" for y in yds) + "]"
    cand = None
    for yd in sorted(ds_dir.glob("year=*")):
        fs = list(yd.glob("*.parquet"))
        if fs:
            cand = fs[0]
            break
    if cand is None:
        return None, col
    con = connect_duckdb()
    try:
        cols = con.execute(
            f"DESCRIBE SELECT * FROM read_parquet('{cand.as_posix()}')").fetchdf()
        names = list(cols["column_name"])
        if col is None:
            col = next((c for c in names if c.lower().endswith("date")), None)
        if col is None or col not in names:
            return None, col
        ty = cols[cols["column_name"] == col].iloc[0]["column_type"]
        expr = (f"coalesce(try_strptime({col}, '%Y%m%d'), "
                f"try_cast({col} AS TIMESTAMP))" if "VARCHAR" in ty
                else f"try_cast({col} AS TIMESTAMP)")
        r = con.execute(f"SELECT max({expr}) FROM read_parquet({g})").fetchone()
        return (pd.Timestamp(r[0]) if r and r[0] is not None else None), col
    except Exception:
        return None, col
    finally:
        con.close()


def _dataset_readers(ds: str) -> list:
    """哪些项目源码**真的在读**这个数据集（路径级匹配）

    ⚠️ 不能按裸词匹配：`margin` 会命中 CSS/HTML 里的 `margin:`，`options`
    会命中函数参数名 `options=` —— 第一版就是这么误判的，把 5 个"只下不用"的
    数据集全报成"有消费方"。只认**路径式**写法：
        dir_of('x') / frozen_dir('x') / Storage('x')
        FROZEN_ROOT / "x" / FROZEN / "x"
        "x/year=" / "/x"（拼路径）
    """
    import re
    root = Path(__file__).resolve().parent.parent
    skip = {".cache", ".deps", "db", "__pycache__", ".git", ".venv", "venv",
            "vnpy-4.4.0", "others", "tests", "docs"}
    esc = re.escape(ds)
    pats = [re.compile(p) for p in (
        rf"""(?:dir_of|frozen_dir|recipe_dir)\(\s*['"]{esc}['"]""",
        rf"""Storage\(\s*['"]{esc}['"]""",
        rf"""(?:FROZEN_ROOT|FROZEN|DB)\s*/\s*['"]{esc}['"]""",
        # `DB / "frozen" / "industry"` 这种三段路径（loader.py 就是这么写的）
        rf"""(?:FROZEN_ROOT|FROZEN|DB)\s*/\s*['"]frozen['"]\s*/\s*['"]{esc}['"]""",
        rf"""['"]{esc}/year=""",
        rf"""['"]{esc}/\*""",
        rf"""/['"]?{esc}['"]?\s*/\s*['"]year=""",
    )]
    out = set()
    for p in root.rglob("*.py"):
        parts = set(p.parts)
        if any(x in parts for x in skip) or any(
                str(x).startswith(("方正证券", "node_modules")) for x in p.parts):
            continue
        rel = str(p.relative_to(root))
        # config.py 只是登记表；downloader/ 与 download_*.py / run_download.py
        # 是**写入方**，不算消费方 —— 否则"只下不用"永远查不出来
        if ("config.py" in rel or "downloader" in p.parts
                or "validate_data" in rel or p.name.startswith("download_")
                or p.name == "run_download.py"):
            continue
        try:
            txt = p.read_text(encoding="utf-8", errors="ignore")
        except OSError:
            continue
        if any(pt.search(txt) for pt in pats):
            out.add(Path(rel).name)
    return sorted(out)


def check_freshness():
    """每个 frozen 数据集是否**跟得上最新交易日**，以及**有没有消费方**

    B12：`margin`/`northbound`/`futures`/`options`/`etf` 五个数据集只有下载器、
    没有任何分析或回测在读它们。风险不是占空间，而是"以为有数据在用"。

    **判据分两级**（见 FRESHNESS_BUDGET 的说明）：
      落后 > 提示阈值 -> 记为信息（该更新了，但这是运维状态）
      落后 > 硬失败阈值 -> 记为问题（这个量级只可能是下载坏了）
    "没有消费方"一律只作信息：保留还是删除是**产品决定**，不该由门禁替人拍板。
    """
    from database.calendar import calendar_meta, latest_trading_day
    print("\n[数据新鲜度] 与最新已发生交易日对比")
    ref = latest_trading_day()
    if ref is None:
        report("交易日历可用", False, "无法确定最新交易日（frozen/calendar 为空）")
        return
    m = calendar_meta()
    log(f"最新已发生交易日: {str(ref)[:10]}"
        f"（日历另有 {m['n_future']} 个未来占位日，已排除 —— 见 --check calendar）")

    hard, soft, ok_n, unused = [], [], 0, []
    for ds in sorted(FROZEN_ROOT.iterdir()) if FROZEN_ROOT.exists() else []:
        if not ds.is_dir():
            continue
        name = ds.name
        label, warn, fail = FRESHNESS_BUDGET.get(name, ("未声明", None, None))
        if not any(ds.glob("year=*/*.parquet")):
            continue
        try:
            mx, col = _max_date_of(ds, FRESHNESS_DATE_COL.get(name))
        except Exception as e:
            log(f"  ⚠ {name}: 取最大日期失败 {type(e).__name__}")
            continue
        lag = (ref - mx).days if mx is not None else None
        rd = _dataset_readers(name)
        if not rd:
            unused.append(name)
        tail = f" 读取方: {', '.join(rd[:3]) if rd else '**无**'}"
        if warn is None:
            log(f"  {name:<12} {label:<12} 最大日期 {str(mx)[:10] if mx else '?':<12}{tail}")
            continue
        if lag is None:
            flag = "无日期列"
        elif fail is not None and lag > fail:
            flag = f"❌ 落后 {lag} 天(>{fail})"
            hard.append((name, lag, fail))
        elif lag > warn:
            flag = f"⚠ 落后 {lag} 天"
            soft.append((name, lag))
        else:
            flag = "ok"
            ok_n += 1
        log(f"  {name:<12} {label:<12} 最大日期 {str(mx)[:10] if mx else '?':<12}"
            f" 落后 {lag if lag is not None else '?':>4} 天 / 提示 {warn:<4} "
            f"上限 {fail if fail is not None else '-':<4} {flag:<16}{tail}")

    for name, lag in soft:
        log(f"  ℹ {name} 落后 {lag} 天 —— 到了该更新的时间（运维状态，不计为问题）")
    for name, lag, fail in hard:
        log(f"    ⚠ {name} 落后 {lag} 天，超过硬上限 {fail} —— 这不是'没更新'，"
            f"更像下载坏了，建议重跑该数据集")
    report("各数据集新鲜度未超硬上限", not hard,
           f"{len(hard)} 个数据集结构性过期" if hard
           else f"{ok_n} 个有预算的数据集未超硬上限（{len(soft)} 个提示）")
    if unused:
        # 只提示、不作门禁：保留还是删除这些"下了但没人用"的数据集是产品决定
        log(f"  ℹ 无任何消费方的 frozen 数据集（B12）: {', '.join(unused)}")
        log("    要么补上消费方，要么删掉 —— 留着容易让人误以为在用")


# ============================================================
# 8. 断点与磁盘的一致性（B4）
# ============================================================
# 断点按"股票"而不是"股票×年份"标记，所以**上游半路返回一截历史**时同样算
# "成功"，之后永远不会补 —— 这不是理论风险：`frozen/etf`/`frozen/options`
# 的断点里各有 84 个日期（43 个交易日），磁盘上只有 7 天有数据。
# 这条检查做两件事：① 断点说完成了、磁盘上却没有 -> 列出来并可修复；
# ② 逐 code 比对年份覆盖 vs `daily_raw`（参考真值），抓"整段静默截断"。
CHECKPOINT_CROSS = ("adjust", "valuation")     # 与 daily_raw 做逐 code 年份比对
# 北交所 920xxx 在 2023 年之前的数据本身就是垃圾（C1），不能拿它当"截断"证据
_JUNK_PREFIX = "920"
_JUNK_BEFORE = 2023


def _looks_like_date_key(key: str) -> bool:
    return len(key) == 8 and key.isdigit()


def backfill_checkpoint(datasets=None) -> dict:
    """把磁盘上**已有的数据范围回填进断点**，让旧断点也能被判定

    旧断点只记了"完成了"、没记范围 —— 于是 1,500 多条既不能说它错、也不能说它对
    （`--check checkpoint` 里那批"无法判定"）。这里直接从 parquet 读出每个 key 的
    实际日期范围写回去，未知就消失了，之后能抓"数据只到 2015 年"这种尾部截断。

    ⚠️ 回填记录的是**磁盘现状**，不是"它本该有的范围"。所以它不会把已经发生的
    静默截断洗白 —— 磁盘范围就是短的那一段，覆盖比对照样能看出来。
    返回 {dataset: 回填条数}
    """
    from database.config import connect_duckdb
    from database.storage import Storage
    print("\n[断点回填] 用磁盘上的真实数据范围补齐旧断点")
    targets = datasets or ["daily_raw", "adjust", "valuation", "holders",
                           "suspend", "st", "financial", "dividend"]
    con = connect_duckdb()
    out = {}
    for ds in targets:
        d = FROZEN_ROOT / ds
        if not d.exists() or not (d / "_checkpoint.json").exists():
            continue
        st = Storage(ds, allow_frozen=True)
        done = list(st.load_checkpoint().get("done", []))
        if not done:
            continue
        n = 0
        # 日期键（etf/options）：范围就是它自己，不用查
        for k in [x for x in done if _looks_like_date_key(x)]:
            t = pd.Timestamp(f"{k[:4]}-{k[4:6]}-{k[6:]}")
            if st.span_of(k) is None and _has_data_for(d, ds, k):
                st.mark_done(k, span=(t, t))
                n += 1
        code_keys = {k for k in done if not _looks_like_date_key(k)}
        if not code_keys:
            out[ds] = n
            log(f"  {ds:<12} 回填 {n:>6} 条范围")
            continue
        # code 键：一次 group-by 拿到每只股票的实际范围
        try:
            cols = con.execute(f"DESCRIBE SELECT * FROM read_parquet("
                               f"'{parquet_glob(d)}')").fetchdf()
            names = set(cols["column_name"])
            dcol = next((c for c in ("trade_date", "ann_date", "start_date", "end_date")
                         if c in names), None)
            if dcol is None:
                out[ds] = n
                continue
            ty = cols[cols["column_name"] == dcol].iloc[0]["column_type"]
            expr = (f"coalesce(try_strptime({dcol}, '%Y%m%d'), "
                    f"try_cast({dcol} AS TIMESTAMP))" if "VARCHAR" in ty
                    else f"try_cast({dcol} AS TIMESTAMP)")
            df = con.execute(f"""SELECT code, min({expr}) a, max({expr}) b
                                 FROM read_parquet('{parquet_glob(d)}')
                                 GROUP BY 1""").fetchdf()
            for r in df.itertuples(index=False):
                c = str(r.code).zfill(6)
                if c in code_keys and pd.notna(r.a) and pd.notna(r.b):
                    st.mark_done(c, span=(r.a, r.b))
                    n += 1
        except Exception as e:
            log(f"  ⚠ {ds}: 回填失败 {type(e).__name__}: {str(e)[:60]}")
            continue
        out[ds] = n
        log(f"  {ds:<12} 回填 {n:>6} 条范围")
    con.close()
    total = sum(out.values())
    log(f"  合计回填 {total:,} 条；之后 --check checkpoint 就能判定这些条目了")
    return out


def _has_data_for(ds_dir, dataset: str, key: str) -> bool:
    """这个 key 在磁盘上是否真有数据（**按数据集的文件命名规则判定**）

    ⚠️ 三个坑，每一个都真实误报过：
      1. 不能一律用 `**/{key}.parquet`：`financial` 的文件名是
         `{sheet}_{code}.parquet`（profit_000001.parquet），
         `stocks`/`industry`/`margin` 这类是 `all.parquet`/`data.parquet` 单文件。
         第一版没区分，`financial` 报出 5,889 条假阳性。
      2. **日期键有两种写法**：etf/options 的断点键是 `20200102`（无分隔符），
         而文件是 `2020-01-02.parquet`（带连字符）。只按前者 glob，会把
         **1,627 个完全正常的文件全报成缺失** —— 实跑完就是这么误报 361 条的。
      3. 同一个 key 可能落在不同年份分区，必须递归找。
    """
    if key == "all":
        return any(ds_dir.glob("year=*/*.parquet"))
    if dataset == "financial":
        return any(ds_dir.glob(f"year=*/*_{key}.parquet"))
    if any(ds_dir.glob(f"**/{key}.parquet")):
        return True
    if _looks_like_date_key(key):        # 20200102 -> 2020-01-02.parquet
        dashed = f"{key[:4]}-{key[4:6]}-{key[6:]}"
        return any(ds_dir.glob(f"**/{dashed}.parquet"))
    return False


def check_checkpoint(repair: bool = False, strict: bool = False):
    """断点审计：检测"标记完成但数据缺失"，可选自动修复

    B4 说的是"断点按**股票**记，不按**股票×年份**记，所以上游半路返回一截历史
    时同样算成功、之后永远不补"。**它已经真实发生过**：`frozen/etf` 与
    `frozen/options` 的断点里各有 84 个日期（43 个交易日），磁盘上只有 7 天有数据。

    分四类处理 —— 关键是**只把能证伪的算作问题**：
      ① **日期键**（etf/options 按 trade_date 下）没有文件 -> 真问题。
         一个交易日拿不到数据，永远不是"本来就没有"。
      ② 北交所 `920xxx` -> C1 的上游垃圾数据，不计。
      ③ 股票键 + 断点**记过范围**（新格式）却没有文件 -> 真问题（断点在撒谎）。
      ④ 股票键 + 旧断点（没记范围）-> **无法判定**：既可能是退市股本来就没数据，
         也可能是被静默截断。列为信息；加 `--strict-checkpoint` 可把这类也当问题。
    """
    from database.config import connect_duckdb
    from database.storage import Storage
    print("\n[断点一致性] 断点说完成了，磁盘上真的有数据吗")

    con = connect_duckdb()
    ref_codes = set()
    try:
        rr = con.execute(f"""SELECT DISTINCT code FROM read_parquet(
            '{parquet_glob(FROZEN_ROOT / 'daily_raw')}')""").fetchdf()
        ref_codes = {str(c).zfill(6) for c in rr["code"]}
    except Exception as e:
        log(f"  （daily_raw 扫描失败: {type(e).__name__}，只能按文件存在性判断）")
    if ref_codes:
        log(f"  参考：daily_raw 里有数据的股票 {len(ref_codes):,} 只")

    problems, unknown, n_c1 = {}, {}, 0
    for ds_dir in sorted(FROZEN_ROOT.iterdir()) if FROZEN_ROOT.exists() else []:
        if not ds_dir.is_dir():
            continue
        if not (ds_dir / "_checkpoint.json").exists():
            continue
        name = ds_dir.name
        st = Storage(name, allow_frozen=True)
        info = st.audit_done(
            has_data=lambda k, d=ds_dir, n=name: _has_data_for(d, n, k))
        real, unk = [], []
        for key in info["no_data"]:
            if _looks_like_date_key(key):
                real.append(key)                       # ① 日期键：确证
            elif key.startswith(_JUNK_PREFIX):
                n_c1 += 1                              # ② C1 上游垃圾
            elif st.span_of(key) is None:
                unk.append(key)                        # ④ 旧断点：无法判定
            else:
                real.append(key)                       # ③ 记过范围还缺 -> 确证
        if real:
            problems[name] = real
        if unk:
            unknown[name] = unk
        log(f"  {name:<12} 断点 {info['n_done']:>6} 条 / 确证缺失 "
            f"{len(real):>4} / 无法判定 {len(unk):>5} / "
            f"已有范围 {len(info['span_end']):>5} / "
            f"确认空 {len(info['confirmed_empty']):>4}")

    n_missing = sum(len(v) for v in problems.values())
    for ds, keys in problems.items():
        log(f"    ⚠ {ds}: {len(keys)} 个 key 标记完成但磁盘无数据（样例 {keys[:5]}）")
    n_unknown = sum(len(v) for v in unknown.values())
    if n_unknown:
        top = sorted(unknown.items(), key=lambda kv: -len(kv[1]))[:4]
        log(f"  ℹ {n_unknown} 条**无法判定**（旧断点没记数据范围，无法区分"
            f"「本来就没数据」与「被静默截断」）: "
            + "、".join(f"{k}={len(v)}" for k, v in top))
        log("    新下载会记录数据范围，之后就能判定；"
            "想把这类也当问题请加 --strict-checkpoint")
    if n_c1:
        log(f"  ℹ {n_c1} 条属北交所 920xxx（C1 上游垃圾数据），不计")
    fail = n_missing > 0 or (strict and n_unknown > 0)
    report("断点标记完成的都有数据", not fail,
           f"{n_missing} 条断点确证与磁盘不一致（这些永远不会被重试）"
           + (f"，另有 {n_unknown} 条无法判定" if n_unknown else "")
           if fail else "确证一致")

    # ② 逐 code 年份覆盖 vs daily_raw（抓整段静默截断）
    for ds in CHECKPOINT_CROSS:
        d = FROZEN_ROOT / ds
        if not d.exists():
            continue
        try:
            ref = con.execute(f"""SELECT code, year(trade_date) y FROM read_parquet(
                '{parquet_glob(FROZEN_ROOT / 'daily_raw')}') GROUP BY 1, 2""").fetchdf()
            got = con.execute(f"""SELECT code, year(trade_date) y FROM read_parquet(
                '{parquet_glob(d)}') GROUP BY 1, 2""").fetchdf()
        except Exception as e:
            log(f"  ⚠ {ds}: 扫描失败 {type(e).__name__}")
            continue
        ry, gy = {}, {}
        for r in ref.itertuples(index=False):
            ry.setdefault(str(r.code).zfill(6), set()).add(int(r.y))
        for r in got.itertuples(index=False):
            gy.setdefault(str(r.code).zfill(6), set()).add(int(r.y))
        bad = []
        for code, ys in ry.items():
            if code not in gy:
                continue                      # 该数据集本就没有这只股票
            if code.startswith(_JUNK_PREFIX):
                # 北交所 920xxx 的历史在 daily_raw 里是垃圾（C1），不参与比对
                ys = {y for y in ys if y >= _JUNK_BEFORE}
            miss = sorted(ys - gy[code])
            if len(miss) >= 2:
                bad.append((code, miss))
        log(f"  {ds}: 与 daily_raw 逐 code 比对 {len(ry):,} 只，"
            f"缺 ≥2 个年份的 {len(bad)} 只")
        for code, miss in bad[:6]:
            log(f"    ⚠ {code} 缺 {miss}")
        report(f"{ds} 年份覆盖不落后于 daily_raw", len(bad) == 0,
               f"{len(bad)} 只缺多年（疑似静默截断）" if bad else "与参考一致")

    # ③ 可选修复：把不一致的 key 从断点里摘掉，让下次下载重试
    if repair and problems:
        total = 0
        for ds, keys in problems.items():
            st = Storage(ds, allow_frozen=True)
            total += st.forget_done(keys)
        log(f"  [修复] 已从 {len(problems)} 个数据集的断点里移除 {total} 个 key，"
            f"下次运行会重新下载")
    elif problems:
        log("  （加 --repair-checkpoint 可把这些 key 摘出断点，让它们重新下载）")
    con.close()


# ============================================================
def main():
    parser = argparse.ArgumentParser(description="数据库质量校验")
    parser.add_argument("--check", action="append",
                        choices=["unique", "schema", "coverage", "limit",
                                 "completeness", "status", "dates", "listing",
                                 "calendar", "freshness", "checkpoint",
                                 "duplicates", "parquet",
                                 "all"],
                        help="可重复指定，如 --check limit --check completeness；"
                             "不传等价于 all")
    parser.add_argument("--quick", action="store_true", help="抽样快速检查")
    parser.add_argument("--strict-checkpoint", action="store_true",
                        help="把「旧断点未记范围、无法判定」的条目也算作问题")
    parser.add_argument("--backfill-checkpoint", action="store_true",
                        help="先用磁盘上的真实数据范围补齐旧断点，再做 --check checkpoint")
    parser.add_argument("--repair-checkpoint", action="store_true",
                        help="配合 --check checkpoint：把「标记完成但无数据」的"
                             "key 从断点里摘掉，让下次下载重试")
    parser.add_argument("--repair-duplicates", action="store_true",
                        help="配合 --check duplicates：把整行重复就地去掉"
                             "（下载器区间重叠造成的副本，非原始数据）")
    parser.add_argument("--parquet-all", action="store_true",
                        help="配合 --check parquet：扫描全部年份（25 万文件，"
                             "5 分钟以上）。默认只查每个数据集最新一年，1~25 秒")
    parser.add_argument("--no-strict-exit", action="store_true",
                        help="即使发现问题也返回 0（默认发现问题返回 1，供 CI/调度做门禁）")
    args = parser.parse_args()
    # 旧实现 `--check` 只能取一个值：传 `--check limit --check completeness`
    # 时 argparse 会**静默丢掉前一个**，让人以为两项都查了。
    wanted = set(args.check or ["all"])
    if "all" in wanted:
        wanted = {"unique", "schema", "coverage", "limit", "completeness",
                  "status", "dates", "listing", "calendar",
                  "freshness", "checkpoint", "duplicates", "parquet"}

    if "unique" in wanted:
        for ds in ["daily", "frozen/valuation", "frozen/adjust"]:
            check_uniqueness(ds, quick=args.quick)
    if "schema" in wanted:
        for ds in ["daily", "frozen/valuation", "frozen/adjust"]:
            check_schema(ds, quick=args.quick)
        check_column_consistency()
    if "coverage" in wanted:
        check_coverage(quick=args.quick)
    if "limit" in wanted:
        check_limit_rules()
    if "listing" in wanted:
        check_listing_rules()
    if "calendar" in wanted:
        check_calendar()
    if "freshness" in wanted:
        check_freshness()
    if "checkpoint" in wanted:
        if args.backfill_checkpoint:
            backfill_checkpoint()
        check_checkpoint(repair=args.repair_checkpoint,
                         strict=args.strict_checkpoint)
    if "completeness" in wanted:
        check_year_completeness()
    if "status" in wanted:
        check_status_panels()
    if "dates" in wanted:
        check_date_columns()
    if "duplicates" in wanted:
        check_row_duplicates(repair=args.repair_duplicates)
    if "parquet" in wanted:
        check_parquet_integrity(all_years=args.parquet_all)

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
