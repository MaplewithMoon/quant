# -*- coding: utf-8 -*-
"""交易日历的**唯一**读取实现

【为什么要有这个模块】
    `frozen/calendar` 来自 tushare `trade_cal`，它**包含未来的交易日**
    （实测覆盖到 2027-12-31）。这本身没问题（排期、节假日安排确实需要），
    但把它当"**已发生**的交易日"用会出两类错：

    1. **判定"最新交易日"时拿到未来日期** → 覆盖率/完整性检查空转通过。
       本项目已经真实发生过一次：`check_year_completeness` 用日历的最大年份
       判定"仍在上市"，2027 年显然没有数据，于是 `live` 集合为空、
       整条检查形同虚设（假通过）。
    2. **按日历推算调仓日/窗口** → 落到还没有数据的未来日期。
       `joinquant/data.py::trade_days()` 就是因为踩了这个（日历比面板长 486 天，
       策略算出的调仓日引擎跑不到、状态机卡死）才改回用面板日期。

    所以统一从这里取，且**默认裁剪到"今天"**；确实需要完整日历（含未来）
    时显式传 `asof=None`。

【约定】
    - 日历表只存开市日（实测 9,041 行全部 `is_open=1`），但仍按 `is_open` 过滤防御
    - 所有函数返回 `pd.DatetimeIndex`（已排序、去重）
"""
from typing import Optional

import pandas as pd

__all__ = ["raw_calendar", "trading_days", "latest_trading_day",
           "is_trading_day", "calendar_meta", "FUTURE_DAYS_WARNING"]

_CACHE = {}

# 未来占位日超过这个数量就在日志里提醒一次（正常情况下 tushare 只给到下一年）
FUTURE_DAYS_WARNING = 400


def raw_calendar() -> pd.DatetimeIndex:
    """**完整**交易日历（含未来占位日）—— 只在确实需要未来日历时使用"""
    if "raw" in _CACHE:
        return _CACHE["raw"]
    from .config import FROZEN_ROOT, connect_duckdb, year_globs
    cal = pd.DatetimeIndex([])
    d = FROZEN_ROOT / "calendar"
    if d.exists():
        g = year_globs(d)
        if g != "[]":
            con = connect_duckdb()
            try:
                df = con.execute(f"""
                    SELECT DISTINCT trade_date FROM read_parquet({g})
                    WHERE try_cast(is_open AS INTEGER) IS NULL
                       OR try_cast(is_open AS INTEGER) = 1
                    ORDER BY 1
                """).fetchdf()
                cal = pd.DatetimeIndex(pd.to_datetime(df["trade_date"])).normalize()
            finally:
                con.close()
    _CACHE["raw"] = cal
    return cal


def trading_days(start=None, end=None, asof=None,
                 include_future: bool = False) -> pd.DatetimeIndex:
    """交易日列表

    参数:
        start, end:    起止日期（含）
        asof:          裁到这个日期（默认**今天**）。传 None 表示不裁剪
        include_future: True 时等价于 `asof=None`（要完整日历就显式写出来，
                        让调用点在 code review 里一眼可见）

    ⚠️ 默认裁剪，理由见模块文档。需要未来日期时必须显式开这个口子。
    """
    cal = raw_calendar()
    if len(cal) == 0:
        return cal
    if include_future:
        asof = None
    elif asof is None:
        asof = pd.Timestamp.now().normalize()
    if asof is not None:
        cal = cal[cal <= pd.Timestamp(asof)]
    if start is not None:
        cal = cal[cal >= pd.Timestamp(start)]
    if end is not None:
        cal = cal[cal <= pd.Timestamp(end)]
    return cal


def latest_trading_day(asof=None) -> Optional[pd.Timestamp]:
    """最新**已发生**的交易日（默认裁到今天）；无日历数据时返回 None

    这就是"判定最新数据是否落后"时该用的基准 —— 不能用 `raw_calendar().max()`。
    """
    cal = trading_days(asof=asof)
    return cal.max() if len(cal) else None


def is_trading_day(date) -> bool:
    """某天是否交易日（按**已发生**的日历判断）"""
    cal = trading_days()
    return pd.Timestamp(date).normalize() in cal


def calendar_meta() -> dict:
    """日历概况（供校验/日志使用）

    返回: {n, n_future, first, last, last_past, future_first}
    """
    raw = raw_calendar()
    today = pd.Timestamp.now().normalize()
    past = raw[raw <= today]
    future = raw[raw > today]
    return {
        "n": int(len(raw)),
        "n_future": int(len(future)),
        "first": raw.min() if len(raw) else None,
        "last": raw.max() if len(raw) else None,
        "last_past": past.max() if len(past) else None,
        "future_first": future.min() if len(future) else None,
    }
