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
# tushare 单次返回有**行数上限**（实测 opt_daily 恒定 15000 行就不再增长），
# 按 trade_date 一把取会**静默截断**：实测某交易日一把取 15,000 行，
# 按交易所拆开合计 **26,566 行** —— 每天丢掉约 44%。所以必须拆。
OPT_EXCHANGES = ("SSE", "SZSE", "CFFEX", "DCE", "CZCE", "SHFE", "INE", "GFEX")
SINGLE_CALL_LIMIT = 15000        # 触到这个数就怀疑被截断


def _download_by_trading_days(pro, dataset: str, api: str, desc: str,
                              calls_per_min: int, start_year: int = 2020,
                              exchanges=None):
    """按交易日逐日下载全市场快照，**每天一个文件**

    ⚠️ 旧实现有三个错，直接导致 `frozen/etf` 与 `frozen/options` 废掉：
      1. 取数日期用"每月 1 号"，而 1 号经常不是交易日 -> 大量空响应；
      2. 空响应也会 `done.add(d)` -> **永久不再重试**（B4 的真实案例）；
      3. `save_by_year(df, ds)` 不传 code，每年都写同一个 `data.parquet`
         -> 后一天覆盖前一天，实测 2020~2026 每年只剩 1 天数据。
    现在：从 `database.calendar.trading_days()` 取真实交易日；**只有真的写出
    数据才标完成并记范围**；每天写 `year={Y}/{日期}.parquet`，互不覆盖。

    exchanges: 需要按交易所拆分时传入（`opt_daily` **必须**拆，见上）；
               `fund_daily` 不支持 exchange 过滤（传了返回同一批），传 None。
    """
    # ⚠️ 必须把 calls_per_min 传进 RateLimiter：不传就是默认 190/分钟，
    #    远超这些接口的限频，会被拒（拒了不会标完成、下次还能补，但很吵）
    limiter = RateLimiter(calls_per_min)
    store = _store(dataset)
    done = done_set(dataset)
    from database.calendar import trading_days
    days = [d for d in trading_days(start=f"{start_year}-01-01")]
    todo = [d for d in days if d.strftime("%Y%m%d") not in done]
    n_call = len(todo) * len(exchanges or (None,))
    print(f"[{dataset}] 交易日 {len(days)} 个，待下 {len(todo)} 个"
          f"（{n_call:,} 次调用 @ {calls_per_min}/分钟 ≈ {n_call / calls_per_min:.0f} 分钟）",
          flush=True)
    n_empty, n_capped = 0, 0
    with tqdm(total=len(todo), desc=desc, ncols=100) as pbar:
        for d in todo:
            key = d.strftime("%Y%m%d")
            check_disk()
            frames = []
            try:
                for ex in (exchanges or (None,)):
                    params = {"trade_date": key}
                    if ex:
                        params["exchange"] = ex
                    df = api_call(pro, api, limiter, **params)
                    if df is None or df.empty:
                        continue
                    if len(df) >= SINGLE_CALL_LIMIT:
                        # 触顶说明这次很可能被截断了 —— 必须让人看见，
                        # 否则就是"安静地少数据"
                        n_capped += 1
                        print(f"  ⚠ {key}/{ex or '全部'} 返回 {len(df)} 行，"
                              f"疑似触到单次上限，数据可能被截断", flush=True)
                    frames.append(df)
            except Exception as e:
                print(f"  {key} 失败: {str(e)[:60]}", flush=True)
            if frames:
                out = pd.concat(frames, ignore_index=True)
                out["trade_date"] = pd.to_datetime(d.strftime("%Y-%m-%d"))
                save_by_year(out, dataset, code=d.strftime("%Y-%m-%d"))
                store.mark_done(key, span=(d, d))
            else:
                # **不标完成**：交易日却拿不到数据，下次还要重试
                n_empty += 1
            pbar.set_postfix(day=key)
            pbar.update(1)
    if n_empty:
        print(f"[{dataset}] 有 {n_empty} 个交易日返回空 —— 未标记完成，"
              f"下次运行会重试（空响应 != 确认无数据）", flush=True)
    if n_capped:
        print(f"[{dataset}] 有 {n_capped} 次调用触到单次上限 {SINGLE_CALL_LIMIT}，"
              f"数据可能不完整 —— 请检查是否需要再细分（换/加过滤维度）", flush=True)
    print(f"[{dataset}] 完成 {len(done_set(dataset))} 个交易日", flush=True)


def download_etf(pro):
    (FROZEN / "etf").mkdir(parents=True, exist_ok=True)
    try:
        basic = api_call(pro, "fund_basic", RateLimiter(), market="E", status="L")
        basic.to_parquet(FROZEN / "etf" / "fund_basic.parquet", index=False)
        print(f"[etf] 基金列表: {len(basic)} 只", flush=True)
    except Exception as e:
        print(f"[etf] 基金列表失败: {str(e)[:60]}", flush=True)
    # fund_daily **不支持** exchange 过滤（实测传 SSE/SZSE 返回同一批 2,139 行），
    # 好在它没触到单次上限（每日 2,100~2,200 行，逐日浮动 → 是真实行数）
    _download_by_trading_days(pro, "etf", "fund_daily", "etf日线",
                              calls_per_min=60)


def fetch_opt_basic(pro) -> pd.DataFrame:
    """取**全部交易所**的期权合约列表并合并

    ⚠️ 旧实现只写了 `exchange="SSE"`，于是 `opt_basic` 只有 12,000 个上交所
    合约 —— 占期权日线里 226,402 个合约的 **4.6%**。后果是期权数据没法按
    认购/认沽分类，Put/Call Ratio 只能算上交所那一小块
    （`factors/market.py::load_option_pcr` 会在 attrs 里标注覆盖率）。

    补齐只要 8 次调用，**不需要重下 2,300 万行日线** —— 所以单独开一个
    `--only opt_basic` 入口。
    """
    limiter = RateLimiter(60)
    frames = []
    for ex in OPT_EXCHANGES:
        try:
            d = api_call(pro, "opt_basic", limiter, exchange=ex)
        except Exception as e:
            print(f"  [opt_basic] {ex} 失败: {str(e)[:60]}", flush=True)
            continue
        if d is None or d.empty:
            continue
        d = d.copy()
        d["exchange"] = ex
        frames.append(d)
        print(f"  [opt_basic] {ex:<6} {len(d):>7,} 个合约", flush=True)
    if not frames:
        return pd.DataFrame()
    out = pd.concat(frames, ignore_index=True)
    return out.drop_duplicates(subset=["ts_code"])


def fetch_sw_member(pro) -> pd.DataFrame:
    """取申万行业成分（分级）`index_member_all` —— **带进出日期的 PIT 原料**

    ⚠️ 这个接口有个和 `opt_daily` 同款的坑：**单次调用恒定返回 3000 行**。
    直接 `pro.index_member_all()` 实测正好 3000 行、`is_new` 全是 `Y`、
    `out_date` 100% 缺失 —— A 股有 5,400+ 只，也就是说**默认调用是截断的**，
    而且一条历史记录都没拿到。必须用 `limit` + `offset` 翻页。

    两组数据合起来才是完整历史：
        is_new='Y'  5,906 行 / 5,906 只   当前有效（`out_date` 恒空）
        is_new='N'  2,006 行 / 1,646 只   已剔除  （`out_date` 恒非空）
    1,646 只股票有 ≥2 段行业区间（最多 6 段），所以 Y∪N 能拼出逐股区间序列。
    合计仅 7,912 行、约 4 次调用。

    为什么需要它：`stock_basic.industry` 只是**当前快照**，拿它做行业中性化
    等于用"现在的行业归属"去回测历史 —— 对换过行业的 1,646 只股票构成**前视**。
    """
    limiter = RateLimiter(200)
    frames = []
    for is_new in ("Y", "N"):
        off, got = 0, 0
        while True:
            try:
                d = api_call(pro, "index_member_all", limiter,
                             is_new=is_new, limit=3000, offset=off)
            except Exception as e:
                print(f"  [sw_member] is_new={is_new} offset={off} 失败: "
                      f"{str(e)[:60]}", flush=True)
                break
            if d is None or d.empty:
                break
            frames.append(d)
            got += len(d)
            if len(d) < 3000:
                break
            off += 3000
        print(f"  [sw_member] is_new={is_new}: {got:,} 行"
              f"（{'已到末页' if got % 3000 else '⚠ 可能仍被截断，请核对'}）",
              flush=True)
    if not frames:
        return pd.DataFrame()
    out = pd.concat(frames, ignore_index=True)
    return out.drop_duplicates(subset=["ts_code", "l1_code", "in_date"])


def _write_sw_member(pro) -> int:
    """下载并写入 `frozen/industry/year=2005/sw_member.parquet`"""
    from database.config import NON_ANNUAL_YEAR
    d = FROZEN / "industry" / f"year={NON_ANNUAL_YEAR}"
    d.mkdir(parents=True, exist_ok=True)
    df = fetch_sw_member(pro)
    if df.empty:
        print("[sw_member] 未取到数据", flush=True)
        return 0
    from database.storage import atomic_to_parquet
    atomic_to_parquet(df, d / "sw_member.parquet")
    n_y = int((df["is_new"] == "Y").sum())
    n_n = int((df["is_new"] == "N").sum())
    print(f"[sw_member] 已写入 {len(df):,} 行（当前 {n_y:,} / 历史 {n_n:,}），"
          f"{df['ts_code'].nunique():,} 只股票，{df['l1_name'].nunique()} 个 L1 行业",
          flush=True)
    return len(df)


def download_options(pro):
    (FROZEN / "options").mkdir(parents=True, exist_ok=True)
    basic = fetch_opt_basic(pro)
    if not basic.empty:
        basic.to_parquet(FROZEN / "options" / "opt_basic.parquet", index=False)
        print(f"[options] 合约列表: {len(basic):,} 个（全交易所）", flush=True)
    # **必须按交易所拆**：一把取恒定 15,000 行（单次上限），实测拆开合计 26,566 行
    _download_by_trading_days(pro, "options", "opt_daily", "options日线",
                              calls_per_min=100, exchanges=OPT_EXCHANGES)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--only", choices=["adjust", "st", "valuation", "etf",
                                           "options", "opt_basic", "sw_member",
                                           "all"],
                        default="all")
    parser.add_argument("--codes", default="", help="指定股票")
    parser.add_argument("--fresh", action="store_true", help="忽略断点，从头重新下载")
    args = parser.parse_args()

    token = load_token()
    if not token:
        print("错误: 未找到 Tushare token")
        return
    # 不要用 ts.set_token()：它会往家目录写 tk.csv，受限环境 PermissionError
    pro = ts.pro_api(token)

    # 只补期权合约列表：不碰日线、不碰断点（8 次调用，秒级完成）
    if args.only == "opt_basic":
        (FROZEN / "options").mkdir(parents=True, exist_ok=True)
        basic = fetch_opt_basic(pro)
        if basic.empty:
            print("未取到任何合约", flush=True)
            return
        basic.to_parquet(FROZEN / "options" / "opt_basic.parquet", index=False)
        print(f"[opt_basic] 已写入 {len(basic):,} 个合约（全交易所），"
              f"日线与断点未改动", flush=True)
        return

    # 只补申万行业成分（PIT 行业归属的原料）：7,912 行、约 4 次调用
    if args.only == "sw_member":
        _write_sw_member(pro)
        return

    codes = [c.strip() for c in args.codes.split(",") if c.strip()]
    if not codes:
        st = pd.read_parquet(FROZEN / "stocks" / "year=2005" / "all.parquet")
        # 跳过退市股（名称含"退"，tushare 无其行情数据）
        st = st[~st["name"].astype(str).str.contains("退", na=False)]
        codes = sorted(st["code"].astype(str).tolist())
        print(f"有效股票(剔除退市): {len(codes)} 只", flush=True)

    order = (["adjust", "st", "valuation", "etf", "options"]
             if args.only == "all" else [args.only])

    if args.fresh:
        # ⚠️ 必须**只清 --only 指定的那个**：早期写法无条件清空五个数据集，
        # 于是 `--fresh --only etf` 会把 adjust/st/valuation 一起删掉
        # （那是 7 万只股票的数据，重下要几十小时）。
        for ds in order:
            d = FROZEN / ds
            if d.exists():
                import shutil as _sh
                _sh.rmtree(d)
            print(f"已清空 {ds} 的断点和数据", flush=True)
        print("从头重新下载", flush=True)

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
