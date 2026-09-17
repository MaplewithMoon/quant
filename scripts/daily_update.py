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
from database.storage import atomic_to_parquet, is_valid_parquet
from scripts._tushare_common import RateLimiter, api_call, ts_code_of, check_disk

# 路径统一走 database.config，禁止再手写 "db/xxx" 字符串
DB = FROZEN_ROOT.parent                 # db/
FROZEN = FROZEN_ROOT                    # db/frozen             只读原始层
CLEANED_DAILY = dir_of("daily")         # db/cleaned/daily_basic
LIMIT_DIR = dir_of("limit")             # db/cleaned/limit_price
BACKUP_DIR = CLEANED_DAILY.parent / ".daily_backup"
DAILY_GLOB = parquet_glob(CLEANED_DAILY)
# 损坏的 frozen 分片会被隔离到这里，而不是留在原地反复把任务打挂
QUARANTINE = FROZEN.parent / ".corrupt"
DATASETS = ("daily_raw", "valuation", "adjust")


class CorruptParquetError(RuntimeError):
    """frozen 分片损坏，无法安全 upsert"""


# 本次运行发现的 (数据集, 代码) —— 供结尾集中报警
CORRUPT_FOUND: set = set()


def live_codes():
    """当前仍在上市的代码（供数据更新）

    ⚠️ 以前是 `st[~st["name"].str.contains("退")]` —— 拿**名字里有没有『退』**
    当判据。它既漏（改名/重组/被吸收合并退市的不带"退"）又错
    （退市整理期带"退"但仍在交易）。现在走证券主表的 `delist_date`：
    一个纯粹的日期比较，见 `database/master.py`。

    这里**不按交易所过滤**：数据要下全（北交所的行情也是数据），
    至于哪些进回测池由 `universe.pool.UniverseSpec.exchanges` 决定。
    """
    from database.master import live_codes as _live
    codes = _live()
    if codes:
        return codes
    # 主表不可用时的兜底（老数据只有 name 列）
    st = pd.read_parquet(FROZEN / "stocks" / "year=2005" / "all.parquet")
    st = st[~st["name"].astype(str).str.contains("退", na=False)]
    return sorted(st["code"].astype(str).tolist())


def scan_corrupt(year, datasets=DATASETS):
    """找出某一年分区里损坏的 parquet

    只查头尾魔数，很快（实测 17,078 个文件 22 秒）。
    """
    from database.storage import find_corrupt_parquet
    out = []
    for ds in datasets:
        d = FROZEN / ds / f"year={year}"
        if d.exists():
            out += [(ds, f) for f in find_corrupt_parquet(d, "*.parquet")]
    return out


def quarantine(dataset, path):
    """把损坏文件挪到隔离区（保留证据，不直接删）"""
    dst_dir = QUARANTINE / dataset / path.parent.name
    dst_dir.mkdir(parents=True, exist_ok=True)
    dst = dst_dir / path.name
    n = 1
    while dst.exists():
        dst = dst_dir / f"{path.stem}.{n}{path.suffix}"
        n += 1
    import shutil
    shutil.move(str(path), str(dst))
    return dst


def upsert(dataset, code, df, date_col="trade_date"):
    """把 df 按 code 分区 upsert 进 frozen/{dataset}/year={Y}/{code}.parquet

    ⚠️ 三处必须做对（都踩过坑）：

    1. **原子写**（`atomic_to_parquet`）。原先直接 `merged.to_parquet(path)`，
       中断就留下**半截文件**。实测 `frozen/daily_raw/year=2026/601089.parquet`
       是个 1570 字节的截断文件（头魔数 `PAR1` 还在、尾部成了零字节），
       下载阶段被 `except` 吞掉只打印一行，最后在重建清洗层时把整个更新任务打挂。

    2. **旧文件损坏时绝不能"当作它不存在"**。若走 `else` 分支只写 new，
       就等于把这次 15 天的窗口当成全部历史写进去 —— 这只股票**多年的数据
       被静默删除**，而且看起来一切正常。正确做法是**拒绝写入**并报错，
       交给 `--repair-corrupt` 重下整年。

    3. upsert 语义 = 按"新数据的日期集合"剔除旧行再合并，因此可以重复运行。
    """
    df = df.copy()
    df[date_col] = pd.to_datetime(df[date_col])
    df["year"] = df[date_col].dt.year
    for y, g in df.groupby("year"):
        if y < 2005:
            continue
        part = FROZEN / dataset / f"year={int(y)}"
        path = part / f"{code}.parquet"
        cols = [c for c in g.columns if c != "year"]
        new = g[cols].sort_values(date_col)
        if path.exists():
            if not is_valid_parquet(path):
                CORRUPT_FOUND.add((dataset, code))
                raise CorruptParquetError(
                    f"{dataset}/{code} 的 {path.name} 已损坏（半截文件），"
                    f"拒绝写入以免覆盖历史")
            old = pd.read_parquet(path)
            old[date_col] = pd.to_datetime(old[date_col])
            new_dates = set(pd.to_datetime(new[date_col]))
            old = old[~old[date_col].isin(new_dates)]
            merged = pd.concat([old, new], ignore_index=True).sort_values(date_col)
        else:
            merged = new
        part.mkdir(parents=True, exist_ok=True)
        atomic_to_parquet(merged, path)


# ============ 增量下载 ============
def _on_code_error(code, e):
    """单只失败的处理

    ⚠️ 损坏分片必须与"网络抖动/接口报错"**分开报**。原先两者都只打印一行
    `{code} 失败: ...`，混在 5556 只的进度条里根本看不见 —— 2026-09-17 那次
    `601089` 的损坏文件就是这么被忽略的，直到重建清洗层时才炸。
    """
    if isinstance(e, CorruptParquetError):
        print(f"  ⚠ {code}: frozen 分片已损坏，已跳过（拒绝覆盖历史）。"
              f"用 --repair-corrupt 修复", flush=True)
    else:
        print(f"  {code} 失败: {str(e)[:50]}", flush=True)


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
            _on_code_error(code, e)
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
            _on_code_error(code, e)
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
            _on_code_error(code, e)
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
    """重建清洗层某一年（从 frozen/daily_raw 派生）

    ⚠️ 两处必须做对：

    1. **原子写**。原先 `clean.to_parquet(dst / f.name)` 是直接写最终路径，
       中断就留下半截文件 —— 而清洗层是回测直接读的那一层，坏一个文件就可能
       让回测静默少一只股票。

    2. **单个文件出错不能让整个任务挂掉**。原先没有任何 try/except，遇到一个
       损坏的 frozen 分片直接抛到顶层（2026-09-17 就是这么挂的），而且此时
       **清洗层已经被覆盖了一半**：因为这是"边读边就地覆盖"，按文件名排序，
       排在前面的股票已经是新数据、后面的还是旧数据，目录里文件数看起来却
       完全正常 —— 这种"半新半旧"最危险，不报错就没人发现。现在改为跳过并
       汇总报告，由调用方决定是否回滚。
    """
    src = FROZEN / "daily_raw" / f"year={year}"
    dst = CLEANED_DAILY / f"year={year}"
    if not src.exists():
        return 0
    os.makedirs(dst, exist_ok=True)
    files = sorted(src.glob("*.parquet"))
    removed, failed, written = 0, [], 0
    for f in files:
        try:
            df = pd.read_parquet(f)
        except Exception as e:
            failed.append((f.stem, f"{type(e).__name__}"))
            continue
        clean = clean_file(df)
        removed += len(df) - len(clean)
        if not clean.empty:
            atomic_to_parquet(clean, dst / f.name)
            written += 1
    print(f"  清洗层 {year}: {len(files)} 文件, 写出 {written}, "
          f"清理 {removed} 行", flush=True)
    if failed:
        print(f"  ⚠ 清洗层 {year}: {len(failed)} 个文件读取失败（已跳过，"
              f"清洗层对该股为**旧数据**）："
              + ", ".join(f"{c}({why})" for c, why in failed[:5])
              + ("..." if len(failed) > 5 else ""), flush=True)
    return len(failed)


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
    n, failed = 0, []
    for f in sorted(src.glob("*.parquet")):
        code = f.stem
        try:
            df = pd.read_parquet(f)
        except Exception as e:
            failed.append((code, type(e).__name__))
            continue
        if "pre_close" not in df.columns:
            # 清洗层应带官方 pre_close；缺失就跳过而不是退回 close.shift(1)
            print(f"  [跳过] {code}: 清洗层缺 pre_close，无法正确计算涨跌停")
            continue
        df = df.sort_values("trade_date")
        # resume_no_limit=True：清理层是该股的完整日线序列，
        # 「复牌首日不设涨跌幅」的判定（靠"错过几个交易日"）才成立。
        out = apply_limit_prices(df, code, st_map,
                                 listing_rule=windows.get(code),
                                 resume_no_limit=True)
        # **不要 dropna**：limit 为空可能是"不设涨跌幅"（新股上市初期）或
        # pre_close 缺失，两种都要保留成空值行 —— 空值在引擎里表示"没有涨跌停
        # 限制"（`_num()` 返回 None 会跳过检查），删掉行会造成增量与全量重建
        # 的口径不一致。
        out = out[["code", "trade_date", "pre_close", "limit_up", "limit_down"]]
        atomic_to_parquet(out, dst / f"{code}.parquet")
        n += 1
    print(f"  涨跌停价 {year}: {n} 文件（官方 pre_close + 统一规则 + 上市初期规则）",
          flush=True)
    if failed:
        print(f"  ⚠ 涨跌停价 {year}: {len(failed)} 个文件读取失败（已跳过）："
              + ", ".join(f"{c}({why})" for c, why in failed[:5])
              + ("..." if len(failed) > 5 else ""), flush=True)
    return len(failed)


# ============ 修复损坏分片 ============
def repair_corrupt(year, pro, datasets=DATASETS):
    """隔离损坏的 frozen 分片，并重下这些股票的**整年**数据

    ⚠️ 为什么必须重下**整年**而不是最近 15 天：分片是按 `(代码, 年份)` 存的，
    文件坏了就等于这一年的历史没了。只补最近 15 天会把整年数据写成 15 天 ——
    **比坏文件更糟**，因为它看起来完全正常，回测会静默少掉大半年的行情。

    流程：扫出坏文件 -> 挪到 `db/.corrupt/`（保留证据，不直接删）-> 对涉及的
    代码按整年重新下载 -> 重新 upsert 生成完整分片。
    """
    bad = scan_corrupt(year, datasets)
    if not bad:
        print(f"[修复] {year} 年未发现损坏分片")
        return 0
    print(f"[修复] {year} 年发现 {len(bad)} 个损坏分片：")
    for ds, f in bad:
        print(f"  {ds}/year={year}/{f.name}  ({f.stat().st_size} 字节)")

    need = {ds: [] for ds in datasets}
    for ds, f in bad:
        quarantine(ds, f)
        need[ds].append(f.stem)
    print(f"  已隔离到 {QUARANTINE}")

    start, end = f"{year}0101", f"{year}1231"
    if need.get("daily_raw"):
        update_daily_raw(pro, need["daily_raw"], start, end)
    if need.get("valuation"):
        update_valuation(pro, need["valuation"], start, end)
    if need.get("adjust"):
        update_adjust(pro, need["adjust"], start, end)

    # 复查：重下之后必须真的好了
    left = scan_corrupt(year, datasets)
    if left:
        print(f"  ⚠ 仍有 {len(left)} 个分片损坏，需人工检查")
        return len(left)
    print(f"[修复] 完成：{sum(len(v) for v in need.values())} 个分片已重下，复查通过")
    return 0


def main():
    parser = argparse.ArgumentParser(description="每日数据更新")
    parser.add_argument("--date", default="", help="更新到该日(默认今天, YYYY-MM-DD)")
    parser.add_argument("--days", type=int, default=15, help="回看天数(默认15)")
    parser.add_argument("--only", choices=["daily", "valuation", "adjust", "rebuild"],
                        default="all",
                        help="rebuild = 只重建清洗层/涨跌停价，不下载"
                             "（下载阶段崩溃后不必重跑 1.5 小时）")
    parser.add_argument("--no-verify", action="store_true", help="跳过更新后自动校验")
    parser.add_argument("--repair-corrupt", action="store_true",
                        help="隔离损坏的 frozen 分片并重下整年数据后退出")
    args = parser.parse_args()

    token = load_token()
    if not token:
        print("错误: 未找到 Tushare token")
        return 1
    # ⚠️ 不要用 ts.set_token(token)：它会把 token 写进 C:\Users\<user>\tk.csv，
    # 受限环境直接 PermissionError 让整个脚本起不来。pro_api 支持直接传 token。
    # （同 database/downloader/tushare_client.py 的处理）
    pro = ts.pro_api(token)

    end_date = args.date.replace("-", "") if args.date else datetime.date.today().strftime("%Y%m%d")
    start = (datetime.date.today() - datetime.timedelta(days=args.days)).strftime("%Y%m%d")
    year = int(end_date[:4])

    try:
        check_disk()
    except RuntimeError as e:
        print(e)
        return 1

    if args.repair_corrupt:
        print(f"[修复模式] 检查 frozen year={year}")
        n = repair_corrupt(year, pro)
        if n:
            return 1
        print("\n接着重建清洗层（--only rebuild 也能单独跑）")
    elif args.only != "rebuild":
        codes = live_codes()
        print(f"更新区间: {start} ~ {end_date}, 股票 {len(codes)} 只")

        # 下载前先体检：损坏分片会让 upsert 拒绝写入，早知道早处理
        pre = scan_corrupt(year)
        if pre:
            print(f"⚠ 检测到 {len(pre)} 个损坏分片（这些股票本次会被跳过）：")
            for ds, f in pre[:5]:
                print(f"    {ds}/year={year}/{f.name}")
            print("  修好后再更新：python scripts/daily_update.py --repair-corrupt")
            print("  （本次继续更新其余股票）")

        if args.only in ("daily", "all"):
            update_daily_raw(pro, codes, start, end_date)
        if args.only in ("valuation", "all"):
            update_valuation(pro, codes, start, end_date)
        if args.only in ("adjust", "all"):
            update_adjust(pro, codes, start, end_date)

    if CORRUPT_FOUND:
        print(f"\n⚠ 有 {len(CORRUPT_FOUND)} 只股票因分片损坏被跳过：")
        print("   " + ", ".join(f"{ds}/{c}" for ds, c in sorted(CORRUPT_FOUND)[:10])
              + ("..." if len(CORRUPT_FOUND) > 10 else ""))
        print("   修：python scripts/daily_update.py --repair-corrupt")

    # 重建清洗层 + 涨跌停价（当年）
    # 重建阶段是**就地覆盖**清洗层，所以两种模式都要先备份：
    # 下载可以重跑，被覆盖坏的清洗层只能靠备份恢复。
    backup = backup_daily()
    print("\n已备份清洗层（重建失败或校验不过将自动回滚）")

    print(f"\n重建清洗层/涨跌停价 (year={year})")
    n_bad = rebuild_cleaned_year(year)
    if n_bad:
        # 清洗层此时是**半新半旧**：按文件名排序，排在前面的股票已是新数据、
        # 后面的还是旧数据，而目录里文件数完全正常 —— 不报错就没人会发现。
        # 必须回滚，不能让这种层留在磁盘上。
        print(f"\n⚠ {n_bad} 个 frozen 分片读不出来，清洗层不完整（会半新半旧）。")
        print("  先修数据：python scripts/daily_update.py --repair-corrupt")
        rollback_daily(backup)
        return 1
    rebuild_limit_year(year)

    # 自动校验：不通过则回滚
    if not args.no_verify:
        print("\n=== 更新后自动校验 ===")
        ok, problems = run_validation()
        if not ok:
            print("校验未通过，执行回滚...")
            rollback_daily(backup)
            print("已回滚。请检查数据源问题后重试。")
            return 1
        # 校验通过后删除备份
        import shutil
        if backup and backup.exists():
            shutil.rmtree(backup)
    else:
        print("(已跳过校验)")

    print("\n每日更新完成")
    return 0


if __name__ == "__main__":
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")
    sys.exit(main())
