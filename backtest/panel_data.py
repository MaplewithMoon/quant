# -*- coding: utf-8 -*-
"""组合回测所需的市场数据面板

`factors.panel.load_panel` 提供价量面板；组合回测还需要**逐日的交易状态**：
    涨跌停价（cleaned/limit_price）→ MarketRules 判断封板
    停牌（frozen/suspend）        → 停牌不可交易

这里把三者拼成统一的宽表 dict，交给 PortfolioBacktestEngine。
"""
import pandas as pd

from database.config import FROZEN_ROOT, dir_of


def _globs(path, y0: int, y1: int) -> str:
    return "[" + ", ".join(
        f"'{path.as_posix()}/year={y}/*.parquet'" for y in range(y0, y1 + 1)) + "]"


def load_status_panels(start: str, end: str, codes=None) -> dict:
    """加载涨跌停价与停牌标记的宽表

    返回: {'limit_up': DF, 'limit_down': DF, 'suspended': DF}
          index=交易日, columns=股票代码
    """
    import duckdb
    y0, y1 = pd.Timestamp(start).year, pd.Timestamp(end).year
    con = duckdb.connect()
    con.execute("SET enable_progress_bar=false")
    out = {}
    try:
        lim = con.execute(f"""
            SELECT code, trade_date, limit_up, limit_down
            FROM read_parquet({_globs(dir_of('limit'), y0, y1)})
            WHERE trade_date >= ? AND trade_date <= ?
        """, [pd.Timestamp(start), pd.Timestamp(end)]).fetchdf()
        if not lim.empty:
            lim["trade_date"] = pd.to_datetime(lim["trade_date"])
            for col in ("limit_up", "limit_down"):
                out[col] = lim.pivot(index="trade_date", columns="code",
                                     values=col).sort_index()

        try:
            sus = con.execute(f"""
                SELECT code, trade_date FROM read_parquet(
                    {_globs(FROZEN_ROOT / 'suspend', y0, y1)})
                WHERE trade_date >= ? AND trade_date <= ?
            """, [pd.Timestamp(start), pd.Timestamp(end)]).fetchdf()
        except Exception:
            sus = pd.DataFrame()
        if not sus.empty:
            sus["trade_date"] = pd.to_datetime(sus["trade_date"])
            sus["_v"] = True
            out["suspended"] = sus.pivot_table(index="trade_date", columns="code",
                                               values="_v", aggfunc="first")
    finally:
        con.close()
    return out


def align_to(panel: dict, ref_index, ref_columns) -> dict:
    """把状态面板对齐到价量面板的日期与股票轴"""
    for k in ("limit_up", "limit_down", "suspended"):
        if k in panel:
            panel[k] = panel[k].reindex(index=ref_index, columns=ref_columns)
    if "suspended" in panel:
        panel["suspended"] = panel["suspended"].fillna(False).astype(bool)
    return panel


def load_price_panel(start: str, end: str, codes=None, limit: int = None,
                     with_status: bool = True) -> dict:
    """组合回测用的统一数据面板

    返回 dict：
        open / high / low / close / volume / amount
        close_adj / open_adj / high_adj / low_adj   （复权，供因子使用）
        limit_up / limit_down / suspended           （with_status=True 时）
    """
    from factors.panel import load_panel

    panel = load_panel(start, end, codes=codes, with_valuation=True,
                       adjust=True, limit=limit)
    if not panel:
        return {}
    if with_status:
        st = load_status_panels(start, end, codes=codes)
        panel.update(st)
        panel = align_to(panel, panel["close"].index, panel["close"].columns)
    return panel
