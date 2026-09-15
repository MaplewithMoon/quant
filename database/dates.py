# -*- coding: utf-8 -*-
"""tushare 日期字符串的**统一容错解析**

【为什么需要单独一个模块】
    全库审计（`validate_data.py --check dates`）发现 tushare 的"日期"列有两种格式
    混在同一列里，而且**列类型也不统一**：

        '20230428'             绝大多数（YYYYMMDD）
        '2025-06-03 16:01:59'  少数记录带上了时分秒（tushare 自身的返回不一致）

    读取方若写 `pd.to_datetime(s, format="%Y%m%d", errors="coerce")`，
    第二种会**静默变成 NaT**，紧接着被 `dropna()` 丢掉 —— 数据无声消失。
    实测 `frozen/holders.ann_date` 有约 **1.75%** 的行是这种带时间的格式。

    所以**所有**解析 tushare 日期字符串的地方都必须走这里。
"""
import pandas as pd

__all__ = ["parse_tushare_date", "DATE_COL_SUFFIXES", "is_date_column"]

# 日期列名判定：只认**以 date 结尾**或以 period 结尾。
# 不能用"包含 date" —— `update_flag` 里含 "date"，会被误判成日期列。
DATE_COL_SUFFIXES = ("date", "period")


def is_date_column(name: str) -> bool:
    """列名是否像日期列（严格后缀匹配，避免 `update_flag` 这类误报）"""
    nm = str(name).lower()
    return nm.endswith(DATE_COL_SUFFIXES)


def parse_tushare_date(s, normalize: bool = True):
    """把 tushare 日期字符串解析成 datetime，**同时兼容 YYYYMMDD 与 ISO 带时间**

    参数:
        s:         Series 或标量
        normalize: True 时把带时间的结果抹掉时分秒（保留日期语义）

    返回:
        与输入同型的 Series / Timestamp；无法解析 -> NaT
    """
    if not isinstance(s, pd.Series):
        return pd.Timestamp(
            parse_tushare_date(pd.Series([s]), normalize=normalize).iloc[0])
    raw = s.astype("string").str.strip()
    out = pd.to_datetime(raw, format="%Y%m%d", errors="coerce")
    # 少数记录是 'YYYY-MM-DD HH:MM:SS' 之类，用通用解析补一次。
    # ⚠️ 必须 format="mixed" 逐元素推断：pandas≥2 会**按第一个元素定格式**，
    # 混着 '2025-06-03 16:01:59' 和 '2024-01-02' 时后者会被判成 NaT。
    miss = out.isna() & raw.notna() & (raw != "") & (raw.str.lower() != "nan")
    if bool(miss.any()):
        sub = raw[miss]
        try:
            alt = pd.to_datetime(sub, format="mixed", errors="coerce")
        except (TypeError, ValueError):        # 老 pandas 无 format="mixed"
            alt = sub.map(lambda x: pd.to_datetime(x, errors="coerce"))
        if normalize:
            alt = alt.dt.normalize() if hasattr(alt, "dt") else alt
        out = out.copy()
        out.loc[miss] = alt
    return out
