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
    """读取停牌记录（**唯一实现**在 database/status.py）

    只返回 `suspend_type='S'`（停牌日）。'R' 是复牌日，当天可交易，
    旧实现把它也算成停牌，会在复牌当天错误地禁止交易。
    """
    from .status import load_suspensions
    df = load_suspensions(codes=[str(code).zfill(6)])
    return df


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


def load_limit_price(code: str, start: str = None, end: str = None) -> pd.DataFrame:
    """读取涨跌停价（cleaned 层配方 limit_price，由清洗层日线派生）

    返回列: trade_date, pre_close, limit_up, limit_down
    """
    files = list(dir_of("limit").glob(f"year=*/{code}.parquet"))
    if not files:
        return pd.DataFrame()
    df = pd.concat([pd.read_parquet(f) for f in files], ignore_index=True)
    df["trade_date"] = pd.to_datetime(df["trade_date"])
    df = df.sort_values("trade_date").drop_duplicates("trade_date")
    if start:
        df = df[df["trade_date"] >= pd.to_datetime(start)]
    if end:
        df = df[df["trade_date"] <= pd.to_datetime(end)]
    return df.reset_index(drop=True)


def load_trading_status(code: str, start: str = None, end: str = None,
                        adjust: str = "qfq") -> pd.DataFrame:
    """读取交易日状态：涨跌停价 + 停牌标记（供回测的制度约束使用）

    返回列: trade_date, limit_up, limit_down, suspended

    参数:
        adjust: 涨跌停价所在的价格空间，必须与回测用的价格一致！
                'qfq' 前复权（默认）/ '' 不复权

    为什么需要 adjust 参数:
        `cleaned/limit_price` 是从**不复权**清洗层派生的，而回测默认用前复权价。
        若不换算就直接比较，等于拿前复权价去比不复权涨停价 —— 有分红送转的
        股票会被误判（例如 002644 在 2024-03-12 前复权开盘 8.2574 会被当成
        "超过不复权涨停 8.25"）。这里用与 to_qfq() 完全相同的因子把涨跌停价
        缩放到同一空间。

    注意:
        - 停牌来自 frozen/suspend
        - **不做前向填充**：涨跌停价必须逐日精确，用前值填充会误拦交易
        - 已知局限：limit_price 用 `pre_close = 昨收` 计算，除权除息日的
          交易所参考价与之不同，因此除权日的涨跌停价本身可能有偏差
    """
    lim = load_limit_price(code, start, end)
    if lim.empty:
        return pd.DataFrame(columns=["trade_date", "limit_up", "limit_down", "suspended"])

    out = lim[["trade_date", "limit_up", "limit_down"]].copy()
    out = out.sort_values("trade_date").reset_index(drop=True)

    # 把涨跌停价换算到与回测价格相同的复权空间
    if adjust == "qfq":
        fac = load_factor(code)
        if not fac.empty:
            merged = pd.merge_asof(out, fac.sort_values("trade_date"),
                                   on="trade_date", direction="backward")
            f = merged["factor"].fillna(1.0).to_numpy()
            out["limit_up"] = (out["limit_up"].to_numpy() * f).round(4)
            out["limit_down"] = (out["limit_down"].to_numpy() * f).round(4)

    sus = load_suspend(code)
    if not sus.empty and "trade_date" in sus.columns:
        sus_days = set(pd.to_datetime(sus["trade_date"]).dt.normalize())
        out["suspended"] = out["trade_date"].dt.normalize().isin(sus_days).astype(int)
    else:
        out["suspended"] = 0
    return out


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
    """读取股票-行业归属映射（frozen 层 stock_industry）

    ⚠️ 只读 `stock_industry.parquet`。`frozen/industry/` 下还躺着 `sw_l1.parquet`
    （申万一级行业清单），两者 schema 完全不同；用 `year=*/*.parquet` 一把捞会得到
    两张表的列并集，全是 NaN —— DuckDB 遇到这种 glob 会直接抛
    `schema mismatch in glob`。
    """
    files = sorted((DB / "frozen" / "industry").glob("year=*/*.parquet"))
    files = [f for f in files if f.name == "stock_industry.parquet"]
    if not files:
        return pd.DataFrame()
    dfs = [pd.read_parquet(f) for f in files]
    return pd.concat(dfs, ignore_index=True)


def load_sw_l1() -> pd.DataFrame:
    """读取申万一级行业清单（frozen 层 sw_l1，31 个行业）

    返回列: index_code（如 801010.SI）/ industry_name / industry_code
    注意：库里**没有**这些行业指数的行情，只有清单。
    """
    files = sorted((DB / "frozen" / "industry").glob("year=*/*.parquet"))
    files = [f for f in files if f.name == "sw_l1.parquet"]
    if not files:
        return pd.DataFrame()
    df = pd.concat([pd.read_parquet(f) for f in files], ignore_index=True)
    return df.drop_duplicates("index_code").reset_index(drop=True)


# ============================================================
# 指数行情（frozen/index_daily）
# ============================================================
INDEX_DAILY = DB / "frozen" / "index_daily"

# 库里实际有行情的指数（宽基）。申万行业指数只有清单、没有行情。
AVAILABLE_INDEXES = {
    "000300.SH": "沪深300",
    "000905.SH": "中证500",
    "000852.SH": "中证1000",
    "932000.CSI": "中证2000",
}


def load_index_daily(code: str, start: str = None, end: str = None) -> pd.DataFrame:
    """读取指数日行情（frozen/index_daily）

    返回列: trade_date / open / high / low / close / pre_close / pct_chg / vol / amount
    这是**真实指数点位**，可直接作为基准；不要拿"成分股加权"去近似。
    """
    files = sorted(INDEX_DAILY.glob("year=*/*.parquet"))
    if not files:
        return pd.DataFrame()
    dfs = []
    for f in files:
        try:
            d = pd.read_parquet(f, columns=["ts_code", "trade_date", "open", "high",
                                            "low", "close", "pre_close", "pct_chg",
                                            "vol", "amount"])
        except Exception:
            d = pd.read_parquet(f)
        d = d[d["ts_code"] == code]
        if not d.empty:
            dfs.append(d)
    if not dfs:
        return pd.DataFrame()
    df = pd.concat(dfs, ignore_index=True)
    df["trade_date"] = pd.to_datetime(df["trade_date"])
    df = df.sort_values("trade_date").drop_duplicates("trade_date")
    if start:
        df = df[df["trade_date"] >= pd.to_datetime(start)]
    if end:
        df = df[df["trade_date"] <= pd.to_datetime(end)]
    return df.reset_index(drop=True)


def load_index_close(code: str, start: str = None, end: str = None) -> pd.Series:
    """指数收盘价序列（index=交易日），基准对齐最常用"""
    df = load_index_daily(code, start, end)
    if df.empty:
        return pd.Series(dtype=float)
    return df.set_index("trade_date")["close"].astype(float)

