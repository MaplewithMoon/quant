# -*- coding: utf-8 -*-
"""交易状态（停牌）的**唯一**读取实现

【为什么要有这个模块】
    在 2026-09 的数据层专项里发现同一件事被实现了四遍，而且**每一遍的口径都不同**：

    | 位置                              | 只取 S 记录 | 日期类型处理      | glob         |
    |-----------------------------------|-------------|-------------------|--------------|
    | backtest/panel_data.py            | —           | 当作 DATE（错）   | 按年份区间 ✗ |
    | universe/pool.py                  | ✗ 含 R      | to_datetime ✓     | year=*  ✓    |
    | database/loader.py（load_suspend）| ✗ 含 R      | to_datetime ✓     | year=*  ✓    |
    | joinquant/data.py                 | 依赖 panel  | —                 | —            |

    其中 `panel_data` 那一路最严重：`frozen/suspend` 是**非年度数据集**
    （只有占位分区 `year=2005`），按回测区间会拼出 `year=2024/*.parquet`，
    路径不存在 -> DuckDB 抛 IOException -> 被 `except Exception` 吞掉 ->
    **每一次组合回测的停牌面板都是空的**，停牌股照常买卖。

    另外 `suspend.trade_date` 是 **VARCHAR 'YYYYMMDD'**，不是 DATE。
    与 Timestamp 直接比较本身就是错的（DuckDB 会报类型不匹配）。

【语义（tushare suspend_d）】
    suspend_type = 'S'  停牌：**当天该股处于停牌状态，不可交易**
    suspend_type = 'R'  复牌：**当天恢复交易**，不是停牌日！
                       把 R 也算成停牌，会在复牌当天错误地禁止交易。
    S 记录只罗列**交易日**（不含周末/节假日），所以可以直接当成"每日停牌标记"。

    `suspend_timing` 非空 = **日内停牌**（只在某个时段停，不是全天）。
    这类日子当天仍有可交易时段，**不应**整天禁掉。本模块把它作为独立标记
    暴露出来，让调用方自行决定（默认不改变"是否停牌"的布尔语义）。

【三层"不可成交"来源】
    ① 行情缺失      —— 全天停牌日没有 K 线（数据的自然兜底）
    ② `suspend_d`   —— 权威停复牌记录（本模块）
    ③ 涨跌停封板    —— 一字板买不进/卖不出，由 `execution/market_rules.py` 处理
    三者相互独立，任何一层说"不可成交"就成立。`audit_coverage()` 做交叉验证。
"""
from typing import Optional

import pandas as pd

from .config import FROZEN_ROOT, connect_duckdb, year_globs

SUSPEND_DIR = FROZEN_ROOT / "suspend"
_EMPTY_COLS = ["code", "trade_date", "suspend_type", "suspend_timing"]


def _empty() -> pd.DataFrame:
    return pd.DataFrame(columns=_EMPTY_COLS)


def load_suspensions(start=None, end=None, codes=None,
                     include_intraday: bool = True) -> pd.DataFrame:
    """停牌记录（唯一实现），只含 `suspend_type='S'`

    返回列: code / trade_date(datetime64) / suspend_type / suspend_timing
            `suspend_timing` 非空表示**日内停牌**（该日仍有可交易时段）。

    ⚠️ 必须用 `union_by_name=true` 读：早期写盘时 `suspend_timing` 在全是空值
    的文件里被推断成 **NULL 类型**、在有值的文件里是 VARCHAR，直接 glob 会抛
    `ConversionException`（"failed to cast column suspend_timing from VARCHAR
    to NULL"）。原先没人踩到只是因为查询没投影这一列。
    """
    if not SUSPEND_DIR.exists():
        return _empty()
    con = connect_duckdb()
    try:
        y0 = pd.Timestamp(start).year if start is not None else None
        y1 = pd.Timestamp(end).year if end is not None else None
        g = year_globs(SUSPEND_DIR, y0, y1)
        if g == "[]":
            return _empty()
        where, params = ["suspend_type = 'S'"], []
        if start is not None:
            where.append("strptime(trade_date, '%Y%m%d') >= ?")
            params.append(pd.Timestamp(start))
        if end is not None:
            where.append("strptime(trade_date, '%Y%m%d') <= ?")
            params.append(pd.Timestamp(end))
        if codes is not None:
            codes = list(codes)
            if not codes:
                return _empty()
            where.append(f"code IN ({','.join(['?'] * len(codes))})")
            params += codes
        timing = ("CAST(suspend_timing AS VARCHAR)"
                  if include_intraday else "CAST(NULL AS VARCHAR)")
        df = con.execute(f"""
            SELECT code, strptime(trade_date, '%Y%m%d')::DATE AS trade_date,
                   suspend_type, {timing} AS suspend_timing
            FROM read_parquet({g}, union_by_name = true)
            WHERE {' AND '.join(where)}
        """, params).fetchdf()
    finally:
        con.close()
    if df.empty:
        return _empty()
    df["trade_date"] = pd.to_datetime(df["trade_date"])
    return df.sort_values(["code", "trade_date"]).reset_index(drop=True)


def intraday_suspended_wide(dates: pd.DatetimeIndex, codes) -> pd.DataFrame:
    """**日内停牌**宽表（index=dates, columns=codes）

    与 `suspended_wide`（全天停牌）分开：日内停牌当天仍有可交易时段，
    整天禁掉会误伤；但用于"能否按开盘/收盘价成交"的可行性判断时它又是相关的。
    """
    codes = [str(c).zfill(6) for c in codes]
    dates = pd.DatetimeIndex(dates)
    empty = pd.DataFrame(False, index=dates, columns=codes)
    df = load_suspensions(dates.min(), dates.max(), codes=codes)
    if df.empty:
        return empty
    t = df["suspend_timing"].astype("string").fillna("").str.strip()
    df = df[t.ne("")]
    if df.empty:
        return empty
    w = (df.assign(_v=True).pivot_table(index="trade_date", columns="code",
                                        values="_v", aggfunc="first"))
    return w.reindex(index=dates, columns=codes).fillna(False).astype(bool)


def audit_coverage(start=None, end=None, sample_codes=None) -> dict:
    """交叉验证「行情缺失」与「suspend_d」是否自洽

    三层"不可成交"来源本应互相印证：**全天停牌日不该有 K 线**。
    若某天有 S 记录、却又有正成交量，说明两者矛盾 —— 列出来让人判断，
    而不是静默选一边。

    返回 dict，包含计数与最多 20 条矛盾样例。
    """
    out = {"n_suspend": 0, "n_intraday": 0, "intraday_ratio": 0.0,
           "n_suspended_with_bar": 0, "suspended_with_bar": [],
           "n_missing_bar": 0}
    df = load_suspensions(start, end, codes=sample_codes)
    if df.empty:
        return out
    out["n_suspend"] = int(len(df))
    t = df["suspend_timing"].astype("string").fillna("").str.strip()
    out["n_intraday"] = int(t.ne("").sum())
    out["intraday_ratio"] = out["n_intraday"] / max(1, out["n_suspend"])

    full = df[t.eq("")]          # 只看全天停牌：它们**不该**有 K 线
    if full.empty:
        return out
    y0 = int(pd.Timestamp(full["trade_date"].min()).year)
    y1 = int(pd.Timestamp(full["trade_date"].max()).year)
    con = connect_duckdb()
    try:
        g = year_globs(FROZEN_ROOT / "daily_raw", y0, y1)
        if g == "[]":
            return out
        dr = con.execute(f"""
            SELECT code, CAST(trade_date AS DATE) AS trade_date, volume
            FROM read_parquet({g})
        """).fetchdf()
    finally:
        con.close()
    if dr.empty:
        return out
    dr["trade_date"] = pd.to_datetime(dr["trade_date"])
    m = full.merge(dr, on=["code", "trade_date"], how="left")
    has_bar = m["volume"].notna()
    out["n_missing_bar"] = int((~has_bar).sum())
    bad = m[has_bar & (m["volume"].fillna(0) > 0)]
    out["n_suspended_with_bar"] = int(len(bad))
    out["suspended_with_bar"] = (
        bad[["code", "trade_date", "volume"]].head(20).to_dict("records"))
    return out




def suspension_days(code: str, start=None, end=None) -> set:
    """某只股票的停牌交易日集合（date 对象），供单标的回测使用"""
    df = load_suspensions(start, end, codes=[str(code).zfill(6)])
    return set(df["trade_date"].dt.date) if not df.empty else set()


def suspended_wide(dates: pd.DatetimeIndex, codes) -> pd.DataFrame:
    """停牌宽表 bool（index=dates, columns=codes）

    缺失一律视为"未停牌"（False）—— 只有明确有 S 记录的日子才算停牌。
    """
    codes = [str(c).zfill(6) for c in codes]
    dates = pd.DatetimeIndex(dates)
    empty = pd.DataFrame(False, index=dates, columns=codes)
    if not SUSPEND_DIR.exists() or len(dates) == 0 or not codes:
        return empty
    df = load_suspensions(dates.min(), dates.max(), codes=codes)
    if df.empty:
        return empty
    sus = (df.assign(_v=True)
             .pivot_table(index="trade_date", columns="code", values="_v",
                          aggfunc="first"))
    return sus.reindex(index=dates, columns=codes).fillna(False).astype(bool)


def is_suspended(code: str, date) -> bool:
    """单点查询：某只股票某个交易日是否停牌"""
    d = pd.Timestamp(date).date()
    return d in suspension_days(code, start=date, end=date)
