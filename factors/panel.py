# -*- coding: utf-8 -*-
"""横截面面板构建

把逐股票、逐年的 parquet 分区，汇成"日期 × 股票"的宽表（panel），
这是因子计算与 IC 分析的基础数据结构。

为什么用 DuckDB 而不是逐个股票 load
------------------------------------
`database.loader` 是"单股票、单数据集"的接口，要拼一个 5000 只股票的面板
得循环 5000 次读文件。这里直接用 DuckDB 对 parquet 分区做一次聚合查询，
只读需要的列，快一到两个数量级。

复权处理
--------
因子（动量/波动率）必须建立在**复权价**上，否则除权日的暴跌会被当成真实收益。
这里 join `frozen/adjust` 的 adj_factor 得到后复权价（close × adj_factor)，
后复权与"收益率"口径一致，适合做因子。
"""
import pandas as pd

from database.config import FROZEN_ROOT, dir_of


def _glob(path, y0: int, y1: int) -> str:
    """生成只覆盖所需年份的 parquet 路径列表（DuckDB 的 read_parquet 接受列表）

    注意：DuckDB 不支持 `{2022,2023}` 这种花括号展开，必须显式给列表。
    """
    parts = [f"'{path.as_posix()}/year={y}/*.parquet'" for y in range(y0, y1 + 1)]
    return "[" + ", ".join(parts) + "]"


def load_panel(start: str, end: str, codes: list = None,
               with_valuation: bool = True, adjust: bool = True,
               limit: int = None) -> dict:
    """加载因子研究用的横截面面板

    参数:
        start, end:     起止日期 'YYYY-MM-DD'
        codes:          限定股票池（None = 全部）
        with_valuation: 是否附带估值字段（市值/PE/PB/换手率）
        adjust:         True=附带后复权价（列名带 _adj 后缀）
        limit:          只保留流动性最好的前 N 只（按区间内日均成交额），
                        用于在探索阶段控制计算量

    返回:
        dict[str, pd.DataFrame]，每个 DataFrame 是 index=交易日, columns=股票代码
        主要键:
            close / high / low / open / pre_close / volume / amount / pct_chg
            close_adj / high_adj / low_adj / open_adj     （adjust=True 时）
            total_mv / circ_mv / pe_ttm / pb / turnover_rate （with_valuation 时）
    """
    import duckdb

    y0, y1 = pd.Timestamp(start).year, pd.Timestamp(end).year
    daily_glob = _glob(dir_of("daily"), y0, y1)
    con = duckdb.connect()
    con.execute("PRAGMA threads=4")

    where = ["trade_date >= ?", "trade_date <= ?"]
    params = [pd.Timestamp(start), pd.Timestamp(end)]
    if codes:
        ph = ",".join(["?"] * len(codes))
        where.append(f"code IN ({ph})")
        params += list(codes)
    wsql = " AND ".join(where)

    base_sql = f"""
        SELECT code, trade_date, open, high, low, close, pre_close,
               pct_chg, volume, amount
        FROM read_parquet({daily_glob})
        WHERE {wsql}
    """
    df = con.execute(base_sql, params).fetchdf()
    if df.empty:
        return {}

    if adjust:
        adj_glob = _glob(FROZEN_ROOT / "adjust", y0, y1)
        adj = con.execute(f"""
            SELECT code, trade_date, adj_factor
            FROM read_parquet({adj_glob})
            WHERE trade_date >= ? AND trade_date <= ?
        """, [pd.Timestamp(start), pd.Timestamp(end)]).fetchdf()
        if not adj.empty:
            df = df.merge(adj[["code", "trade_date", "adj_factor"]],
                          on=["code", "trade_date"], how="left")
        else:
            df["adj_factor"] = 1.0
        df["adj_factor"] = pd.to_numeric(df["adj_factor"], errors="coerce").fillna(1.0)
        for c in ("open", "high", "low", "close"):
            df[f"{c}_adj"] = df[c] * df["adj_factor"]

    val = None
    if with_valuation:
        val_glob = _glob(FROZEN_ROOT / "valuation", y0, y1)
        try:
            val = con.execute(f"""
                SELECT code, trade_date, total_mv, circ_mv,
                       pe_ttm, pb, ps_ttm, turnover_rate
                FROM read_parquet({val_glob})
                WHERE trade_date >= ? AND trade_date <= ?
            """, [pd.Timestamp(start), pd.Timestamp(end)]).fetchdf()
        except Exception:
            val = None
    con.close()

    df["trade_date"] = pd.to_datetime(df["trade_date"])
    df = df.drop_duplicates(["code", "trade_date"])

    # 流动性优先的股票池裁剪
    if limit:
        liq = df.groupby("code")["amount"].mean().sort_values(ascending=False)
        keep = set(liq.head(limit).index)
        df = df[df["code"].isin(keep)]
        if val is not None and not val.empty:
            val = val[val["code"].isin(keep)]

    panel = _to_wide(df, [c for c in df.columns if c not in ("code", "trade_date")])
    if val is not None and not val.empty:
        val["trade_date"] = pd.to_datetime(val["trade_date"])
        val = val.drop_duplicates(["code", "trade_date"])
        vp = _to_wide(val, [c for c in val.columns if c not in ("code", "trade_date")])
        panel.update(vp)
    return panel


def _to_wide(df: pd.DataFrame, fields: list) -> dict:
    """长表 -> {字段: 宽表(index=日期, columns=代码)}"""
    out = {}
    for f in fields:
        w = df.pivot(index="trade_date", columns="code", values=f)
        out[f] = w.sort_index()
    return out


def adjusted_close(panel: dict) -> pd.DataFrame:
    """取复权收盘价（没有则退回原始收盘价）"""
    return panel.get("close_adj", panel.get("close"))


def forward_returns(close: pd.DataFrame, periods: int = 1,
                    lag: int = 1) -> pd.DataFrame:
    """前瞻收益面板：t 日的因子对应 t+lag 起、持有 periods 天的收益

    默认 lag=1：t 日收盘算出的因子，t+1 收盘买入，持有到 t+1+periods。
    这样因子与收益不重叠，避免"用当日收盘因子赚当日收益"的未来函数。

    返回: 宽表，index=因子日, columns=代码, value=未来收益
    """
    if lag < 0:
        raise ValueError("lag 不能为负（会引入未来函数）")
    fwd = close.shift(-(lag + periods)) / close.shift(-lag) - 1
    return fwd


def universe_mask(panel: dict, min_amount: float = 0.0) -> pd.DataFrame:
    """基础股票池掩码：有价格、有成交额（可扩展为剔除 ST / 停牌 / 次新）"""
    close = adjusted_close(panel)
    mask = close.notna()
    if min_amount and "amount" in panel:
        mask &= panel["amount"] >= min_amount
    return mask
