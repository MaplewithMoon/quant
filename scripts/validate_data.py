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

from database.config import FROZEN_ROOT, connect_duckdb, dir_of, parquet_glob

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
            sum(CASE WHEN limit_up IS NULL AND NOT (code LIKE '920%'
                     AND trade_date < DATE '2023-01-01')
                     THEN 1 ELSE 0 END) n_null_bad,
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
    # 北交所 920xxx 在 2023 年之前的历史（其实是不存在北交所时期的新三板数据）
    # 里 pre_close 是垃圾值（常为 0.01 或缺失），属**已知上游限制**（C1）。
    # 所以只对"2023 年起 + 非 920"要求非空 —— 否则门禁会永久变红、失去意义。
    report("涨跌停价非空（2023 起 / 非北交所）", int(r["n_null_bad"]) == 0,
           f"空值 {int(r['n_null']):,}，其中非已知限制的 {int(r['n_null_bad']):,}")
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
    from database.limit_rules import apply_limit_prices, load_st_intervals
    codes = con.execute(f"""SELECT DISTINCT code FROM read_parquet('{g}')
                            ORDER BY code LIMIT {sample_codes}""").fetchdf()["code"].tolist()
    st_map = load_st_intervals()
    diff = n = 0
    if codes:
        ph = ",".join(["?"] * len(codes))
        s = con.execute(f"""SELECT code, trade_date, pre_close, limit_up, limit_down
                            FROM read_parquet('{g}') WHERE code IN ({ph})""",
                        codes).fetchdf()
        for c, sub in s.groupby("code"):
            exp = apply_limit_prices(sub, c, st_map)
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
    gcal = f"{(FROZEN_ROOT / 'calendar').as_posix()}/year=*/*.parquet"
    cal = con.execute(f"""SELECT year(trade_date) y,
        sum(CASE WHEN is_open=1 THEN 1 ELSE 0 END) d
        FROM read_parquet('{gcal}') GROUP BY 1""").fetchdf()
    cd = dict(zip(cal["y"].astype(int), cal["d"].astype(int)))

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
            sdf = con2 = connect_duckdb()
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
        l = ld.get(code)
        if not ys or pd.isna(l):
            continue
        y0 = max(l.year, 2005)
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
def main():
    parser = argparse.ArgumentParser(description="数据库质量校验")
    parser.add_argument("--check", action="append",
                        choices=["unique", "schema", "coverage", "limit",
                                 "completeness", "status", "dates", "all"],
                        help="可重复指定，如 --check limit --check completeness；"
                             "不传等价于 all")
    parser.add_argument("--quick", action="store_true", help="抽样快速检查")
    parser.add_argument("--no-strict-exit", action="store_true",
                        help="即使发现问题也返回 0（默认发现问题返回 1，供 CI/调度做门禁）")
    args = parser.parse_args()
    # 旧实现 `--check` 只能取一个值：传 `--check limit --check completeness`
    # 时 argparse 会**静默丢掉前一个**，让人以为两项都查了。
    wanted = set(args.check or ["all"])
    if "all" in wanted:
        wanted = {"unique", "schema", "coverage", "limit", "completeness",
                  "status", "dates"}

    if "unique" in wanted:
        for ds in ["daily", "frozen/valuation", "frozen/adjust"]:
            check_uniqueness(ds, quick=args.quick)
    if "schema" in wanted:
        for ds in ["daily", "frozen/valuation", "frozen/adjust"]:
            check_schema(ds, quick=args.quick)
    if "coverage" in wanted:
        check_coverage(quick=args.quick)
    if "limit" in wanted:
        check_limit_rules()
    if "completeness" in wanted:
        check_year_completeness()
    if "status" in wanted:
        check_status_panels()
    if "dates" in wanted:
        check_date_columns()

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
