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
    from database.config import connect_duckdb
    y0, y1 = pd.Timestamp(start).year, pd.Timestamp(end).year
    con = connect_duckdb()
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
                     with_status: bool = True,
                     cache_dir: str = None, refresh_cache: bool = False) -> dict:
    """组合回测用的统一数据面板

    返回 dict：
        open / high / low / close / volume / amount
        close_adj / open_adj / high_adj / low_adj   （复权，供因子使用）
        limit_up / limit_down / suspended           （with_status=True 时）

    cache_dir: 指定则把加载结果缓存到该目录，下次同区间直接读 parquet。
               十年全市场面板要扫 ~8 分钟，调参时反复加载很难受；缓存后 <10 秒。
               缓存带**完整性校验**（日期范围、股票数、表数量），对不上就重新加载，
               不会静默用过期数据。
    """
    from factors.panel import load_panel

    if cache_dir and not refresh_cache:
        cached = _read_panel_cache(cache_dir, start, end)
        if cached:
            return cached

    panel = load_panel(start, end, codes=codes, with_valuation=True,
                       adjust=True, limit=limit)
    if not panel:
        return {}
    if with_status:
        st = load_status_panels(start, end, codes=codes)
        panel.update(st)
        panel = align_to(panel, panel["close"].index, panel["close"].columns)

    if cache_dir:
        _write_panel_cache(panel, cache_dir, start, end)
    return panel


# ============================================================
# 面板缓存
# ============================================================
def _cache_key(start: str, end: str) -> str:
    return f"{pd.Timestamp(start):%Y%m%d}_{pd.Timestamp(end):%Y%m%d}"


def _read_panel_cache(cache_dir: str, start: str, end: str):
    """读缓存并校验；任何异常都返回 None（宁可重新加载，不可用坏数据）"""
    import os
    import json

    d = os.path.join(cache_dir, _cache_key(start, end))
    meta_f = os.path.join(d, "_meta.json")
    if not os.path.isfile(meta_f):
        return None
    try:
        with open(meta_f, encoding="utf-8") as f:
            meta = json.load(f)
        panel = {}
        for name in meta["tables"]:
            panel[name] = pd.read_parquet(os.path.join(d, f"{name}.parquet"))
        # 完整性校验
        close = panel.get("close")
        if close is None or close.empty:
            return None
        # 完整性校验。注意 start/end 是自然日，而面板首行是**交易日**，
        # 中间隔着周末与节假日，所以必须留容差 —— 一开始用严格比较，
        # 结果 2023-01-01 的缓存因为首个交易日是 01-03 而永远被判失效。
        if (close.index.min() - pd.Timestamp(start)).days > 20:
            return None
        if (pd.Timestamp(end) - close.index.max()).days > 20:
            return None
        if int(meta.get("n_codes", -1)) != int(close.shape[1]):
            return None
        if len(panel) != len(meta["tables"]):
            return None
        # 缓存是"某次运行的快照"，库里的数据可能已经更新。让使用者看得见，
        # 否则会静默拿着旧数据得出结论。
        print(f"  [使用面板缓存] 构建于 {meta.get('built_at', '未知时间')}，"
              f"{meta.get('n_days')} 交易日 × {meta.get('n_codes')} 只股票"
              f"（要强制重载请加 --refresh-cache）")
        return panel
    except Exception as e:                      # 缓存损坏 -> 重新加载
        print(f"  [缓存不可用，重新加载] {type(e).__name__}: {e}")
        return None


def _write_panel_cache(panel: dict, cache_dir: str, start: str, end: str):
    import os
    import json

    d = os.path.join(cache_dir, _cache_key(start, end))
    try:
        os.makedirs(d, exist_ok=True)
        tables = []
        for k, v in panel.items():
            if not isinstance(v, pd.DataFrame) or v.empty:
                continue
            v.to_parquet(os.path.join(d, f"{k}.parquet"))
            tables.append(k)
        with open(os.path.join(d, "_meta.json"), "w", encoding="utf-8") as f:
            json.dump({"tables": tables, "start": start, "end": end,
                       "n_days": int(panel["close"].shape[0]),
                       "n_codes": int(panel["close"].shape[1]),
                       "built_at": pd.Timestamp.now().strftime("%Y-%m-%d %H:%M:%S")},
                      f, ensure_ascii=False, indent=2)
        print(f"  面板已缓存 -> {d}（下次同区间加载 <10 秒）")
    except Exception as e:
        print(f"  [缓存写入失败，不影响本次回测] {type(e).__name__}: {e}")
