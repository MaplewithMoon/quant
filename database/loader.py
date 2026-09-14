"""数据库加载器：读取原始价 + 复权因子，现场计算前复权

数据分层：
  db/cleaned/daily_basic/  = 清洗后日线：原始价 + 原始量额（口径一致，三角校验可过）
  db/frozen/adjust/        = 复权因子（原始）
  前复权价 = 原始价 × factor / factor_latest

路径一律走 database.config，不要在别处手写 "db/xxx" 字符串。
"""
import pandas as pd
from pathlib import Path

from .config import FROZEN_ROOT, dir_of

DB = FROZEN_ROOT.parent            # db/ 根目录
CLEANED_DAILY = dir_of("daily")    # db/cleaned/daily_basic


def load_raw_daily(code: str, start: str = None, end: str = None) -> pd.DataFrame:
    """读取清洗后原始价日线（按年/代码分区）"""
    files = list(CLEANED_DAILY.glob(f"year=*/{code}.parquet"))
    if not files:
        return pd.DataFrame()
    dfs = []
    for f in files:
        dfs.append(pd.read_parquet(f))
    df = pd.concat(dfs, ignore_index=True)
    df["trade_date"] = pd.to_datetime(df["trade_date"])
    df = df.sort_values("trade_date").drop_duplicates("trade_date")
    if start:
        df = df[df["trade_date"] >= pd.to_datetime(start)]
    if end:
        df = df[df["trade_date"] <= pd.to_datetime(end)]
    return df.reset_index(drop=True)


def load_factor(code: str) -> pd.DataFrame:
    """读取复权因子（frozen 层 tushare 原始 adj_factor）
    返回含 'factor' 列（前复权因子, 最新=1）: qfq_factor = adj_factor/adj_factor_latest
    """
    files = list((DB / "frozen" / "adjust").glob(f"year=*/{code}.parquet"))
    if not files:
        return pd.DataFrame()
    dfs = []
    for f in files:
        dfs.append(pd.read_parquet(f))
    df = pd.concat(dfs, ignore_index=True)
    df["trade_date"] = pd.to_datetime(df["trade_date"])
    col = "qfq_factor" if "qfq_factor" in df.columns else "adj_factor"
    df["factor"] = pd.to_numeric(df[col], errors="coerce")
    df = df[["trade_date", "factor"]].dropna().sort_values("trade_date")
    if df.empty:
        return df
    # 前复权因子: 最新=1
    latest = df["factor"].iloc[-1]
    df["factor"] = (df["factor"] / latest).round(6)
    return df.reset_index(drop=True)


def to_qfq(daily: pd.DataFrame, factor: pd.DataFrame = None) -> pd.DataFrame:
    """把原始价日线转换为前复权价（量额不变）
    前复权价 = 原始价 × qfq_factor   (qfq_factor 最新=1)
    """
    df = daily.copy()
    if factor is None or factor.empty:
        return df
    # merge_asof 向后匹配最近的复权因子（即使因子事件在区间外也正确）
    f = factor.sort_values("trade_date")
    merged = pd.merge_asof(
        df.sort_values("trade_date"), f, on="trade_date", direction="backward"
    )
    merged["factor"] = merged["factor"].fillna(1.0)
    for c in ["open", "high", "low", "close"]:
        df[c] = (df[c] * merged["factor"].values).round(4)
    return df


def load_daily(code: str, start: str = None, end: str = None,
               adjust: str = "qfq") -> pd.DataFrame:
    """加载日线，adjust: 'qfq' 前复权 / '' 原始价"""
    df = load_raw_daily(code, start, end)
    if df.empty:
        return df
    if adjust == "qfq":
        factor = load_factor(code)
        df = to_qfq(df, factor)
    return df


def load_daily_qfq(code: str, start: str = None, end: str = None) -> pd.DataFrame:
    """读取 frozen 层 tushare 不复权原始数据（可信源，量额单位已统一: 股/元）"""
    files = list((DB / "frozen" / "daily_raw").glob(f"year=*/{code}.parquet"))
    if not files:
        return pd.DataFrame()
    dfs = [pd.read_parquet(f) for f in files]
    df = pd.concat(dfs, ignore_index=True)
    df["trade_date"] = pd.to_datetime(df["trade_date"])
    df = df.sort_values("trade_date").drop_duplicates("trade_date")
    if start:
        df = df[df["trade_date"] >= pd.to_datetime(start)]
    if end:
        df = df[df["trade_date"] <= pd.to_datetime(end)]
    return df.reset_index(drop=True)


def load_suspend(code: str) -> pd.DataFrame:
    """读取停牌记录（frozen 层 tushare suspend_d）"""
    files = list((DB / "frozen" / "suspend").glob(f"year=*/{code}.parquet"))
    if not files:
        return pd.DataFrame()
    dfs = []
    for f in files:
        dfs.append(pd.read_parquet(f))
    df = pd.concat(dfs, ignore_index=True)
    if "trade_date" in df.columns:
        df["trade_date"] = pd.to_datetime(df["trade_date"])
    return df.sort_values("trade_date").reset_index(drop=True)


def mark_suspended(daily: pd.DataFrame, code: str) -> pd.DataFrame:
    """策略1: 停牌缺失标记 suspended=True
    日线数据停牌日天然缺失（不填充），此处按交易日历补齐停牌日并打标记。
    返回: 含 suspended 列的完整面板（交易日历对齐）
    """
    sus = load_suspend(code)
    sus_days = set(pd.to_datetime(sus["trade_date"]).dt.strftime("%Y-%m-%d")) if not sus.empty else set()

    df = daily.copy()
    df["suspended"] = df["trade_date"].dt.strftime("%Y-%m-%d").isin(sus_days).astype(int)
    return df


def fill_factor_nan(daily: pd.DataFrame, factor_map: dict = None) -> pd.DataFrame:
    """策略3: 因子真缺失填充（不修改原始库，仅加载时处理）
    factor_map: {列名: 填充值}，传入行业中位数则按研报标准填充；
    不传则保留 NaN 让模型自行处理。
    """
    df = daily.copy()
    if factor_map:
        for col, val in factor_map.items():
            if col in df.columns:
                df[col] = df[col].fillna(val)
    return df


def load_valuation(code: str, start: str = None, end: str = None) -> pd.DataFrame:
    """读取每日估值指标（frozen 层 tushare daily_basic，市值单位为万元）"""
    files = list((DB / "frozen" / "valuation").glob(f"year=*/{code}.parquet"))
    if not files:
        return pd.DataFrame()
    dfs = [pd.read_parquet(f) for f in files]
    df = pd.concat(dfs, ignore_index=True)
    df["trade_date"] = pd.to_datetime(df["trade_date"])
    df = df.sort_values("trade_date").drop_duplicates("trade_date")
    if start:
        df = df[df["trade_date"] >= pd.to_datetime(start)]
    if end:
        df = df[df["trade_date"] <= pd.to_datetime(end)]
    return df.reset_index(drop=True)


def industry_median_factor(factor_col: str, date: str, industry_col: str = "industry") -> dict:
    """策略3: 计算某日各行业的因子中位数，用于填充缺失值
    基于 frozen 层 valuation（各股票因子）+ stocks 行业归属
    返回: {行业: 中位数值}
    """
    import numpy as np
    from .downloader.tushare_client import get_client
    # 加载该日全部股票因子
    files = list((DB / "frozen" / "valuation").glob("year=*/*.parquet"))
    if not files:
        return {}
    dfs = []
    for f in files:
        df = pd.read_parquet(f, columns=["code", "trade_date", factor_col])
        df["trade_date"] = pd.to_datetime(df["trade_date"])
        dfs.append(df)
    val = pd.concat(dfs, ignore_index=True)
    val = val[val["trade_date"] == pd.to_datetime(date)]
    if val.empty:
        return {}
    # 行业归属
    ind = load_industry_map()
    if ind.empty or "code" not in ind.columns:
        return {}
    ind_code = "code" if "code" in ind.columns else ind.columns[0]
    merged = val.merge(ind[[ind_code, industry_col]] if industry_col in ind.columns
                       else ind[[ind_code]], left_on="code", right_on=ind_code, how="left")
    if industry_col not in merged.columns:
        return {}
    result = merged.groupby(industry_col)[factor_col].median().dropna().to_dict()
    return result


def load_industry_map() -> pd.DataFrame:
    """读取股票-行业归属映射（frozen 层）"""
    files = list((DB / "frozen" / "industry").glob("year=*/*.parquet"))
    if not files:
        return pd.DataFrame()
    dfs = [pd.read_parquet(f) for f in files]
    return pd.concat(dfs, ignore_index=True)
