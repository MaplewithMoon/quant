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
"""
from typing import Optional

import pandas as pd

from .config import FROZEN_ROOT, connect_duckdb, year_globs

SUSPEND_DIR = FROZEN_ROOT / "suspend"


def load_suspensions(start=None, end=None, codes=None) -> pd.DataFrame:
    """停牌记录（唯一实现），只含 `suspend_type='S'`

    返回列: code(VARCHAR 6 位), trade_date(datetime64), suspend_type
    """
    if not SUSPEND_DIR.exists():
        return pd.DataFrame(columns=["code", "trade_date", "suspend_type"])
    con = connect_duckdb()
    try:
        y0 = pd.Timestamp(start).year if start is not None else None
        y1 = pd.Timestamp(end).year if end is not None else None
        g = year_globs(SUSPEND_DIR, y0, y1)
        if g == "[]":
            return pd.DataFrame(columns=["code", "trade_date", "suspend_type"])
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
                return pd.DataFrame(columns=["code", "trade_date", "suspend_type"])
            where.append(f"code IN ({','.join(['?'] * len(codes))})")
            params += codes
        df = con.execute(f"""
            SELECT code, strptime(trade_date, '%Y%m%d')::DATE AS trade_date,
                   suspend_type
            FROM read_parquet({g})
            WHERE {' AND '.join(where)}
        """, params).fetchdf()
    finally:
        con.close()
    if df.empty:
        return pd.DataFrame(columns=["code", "trade_date", "suspend_type"])
    df["trade_date"] = pd.to_datetime(df["trade_date"])
    return df.sort_values(["code", "trade_date"]).reset_index(drop=True)


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
