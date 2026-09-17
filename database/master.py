# -*- coding: utf-8 -*-
"""证券主表（security master）与时点正确的可交易池

【为什么需要这一层】
股票池是回测的地基。以前这个地基是用**当前快照**拼的：

    - `live_codes()`（daily_update）用「名字里含『退』」猜退市股；
    - C1 的排除用**代码前缀**判断"是不是北交所/新三板"；
    - 股票池从 `close.notna()` 起算，隐含假设"有行情就一定该交易"。

这三条都在拿**今天的状态**去决定**历史某天能不能交易**，方向都是错的：
转板股、已退市股、改名股一律会被误判。

【正确的三层过滤】
    ① exchange ∈ (SSE, SZSE)                        交易所归属（天然排除 NEEQ/BSE）
    ② list_date <= t                                  t 日必须已上市
    ③ delist_date 为空 或 delist_date > t              t 日必须未退市

全部是**日期比较**，不含任何"今天叫什么/今天属于谁"。①用 `exchange` 而不是
代码前缀，是因为前缀是今天的编码约定，而 `exchange` 直接表达"这只证券当时
挂在哪个交易所"。

【幸存者偏差】
`frozen/stocks` 必须同时拉 L（上市）/ D（退市）/ P（暂停上市）三种状态。
只拉 L 的话，"当时存在、后来退市"的股票整个不在池子里，历史组合表现会被
系统性高估。主表现在有 339 只退市股带真实 `delist_date`。

【默认为什么排除北交所】
`exchanges=("SSE","SZSE")` 是默认值。北交所有两个现实问题：
    1. 其前身（新三板精选层）在 2021-11-15 北交所开市前的 `pre_close` 大量
       缺失/异常，涨跌停价不可信（见 `database/defects.py` 的 C1）；
    2. 流动性与主板/创业板差一个量级，组合容量分析会失真。
需要覆盖北交所时显式传 `exchanges=("SSE","SZSE","BSE")`，但请连同 C1 一起读。
"""
from typing import Iterable, Optional, Sequence

import numpy as np
import pandas as pd

from database.config import FROZEN_ROOT

# 三层过滤用到的字段。缺任何一个都无法做时点正确的过滤。
REQUIRED_COLS = ("code", "list_date", "delist_date", "exchange")

DEFAULT_EXCHANGES = ("SSE", "SZSE")


def master_path():
    return FROZEN_ROOT / "stocks" / "year=2005" / "all.parquet"


def load_master() -> pd.DataFrame:
    """读证券主表

    返回列（存在时）：ts_code / code / name / market / exchange /
                      list_date / delist_date / list_status
    `list_date` / `delist_date` 已解析成 datetime（退市为空 -> NaT）。
    """
    p = master_path()
    if not p.exists():
        return pd.DataFrame()
    df = pd.read_parquet(p)
    if df.empty:
        return df
    out = df.copy()
    if "code" not in out.columns:
        if "symbol" in out.columns:
            out["code"] = out["symbol"].astype(str).str.strip()
        elif "ts_code" in out.columns:
            out["code"] = out["ts_code"].astype(str).str.split(".").str[0]
        else:
            return pd.DataFrame()
    out["code"] = out["code"].astype(str).str.zfill(6)
    for c in ("list_date", "delist_date"):
        if c in out.columns:
            out[c] = pd.to_datetime(out[c], errors="coerce")
    return out


def master_capabilities(master: pd.DataFrame = None) -> dict:
    """主表能做哪些过滤 —— 让调用方知道自己是不是在"降级运行"

    ⚠️ 老版本的主表（`all.parquet` 只有 8 列）**没有 exchange / delist_date**，
    那时三层过滤做不了。与其静默退化成"全都放行"（偏宽松、引入幸存者偏差
    与不可交易标的），不如把缺口显式报出来，让上层决定要不要警告用户。
    """
    m = load_master() if master is None else master
    if m is None or m.empty:
        return {"has_master": False, "exchange": False, "delist": False,
                "list_status": False, "n": 0, "pit_ready": False}
    cap = {
        "has_master": True,
        "n": int(len(m)),
        "exchange": "exchange" in m.columns and m["exchange"].notna().any(),
        "delist": "delist_date" in m.columns,
        "list_status": "list_status" in m.columns,
        "list_date": "list_date" in m.columns,
    }
    cap["pit_ready"] = bool(cap["exchange"] and cap["delist"] and cap["list_date"])
    return cap


def tradable_mask(dates, codes, exchanges: Sequence[str] = DEFAULT_EXCHANGES,
                  master: pd.DataFrame = None) -> pd.DataFrame:
    """时点正确的**可交易池**（bool 宽表，index=日期, columns=代码）

    三层过滤见模块文档。查不到主表记录的代码一律 False（宁可不交易）——
    "不在主表里"本身就是一条信息：我们不知道它当时是否在交易。
    """
    idx = pd.DatetimeIndex(pd.to_datetime(dates))
    cols = [str(c).zfill(6) for c in codes]
    mask = pd.DataFrame(False, index=idx, columns=cols)
    m = load_master() if master is None else master
    if m is None or m.empty:
        return mask

    m = m.drop_duplicates(subset=["code"]).set_index("code")
    m = m.reindex(cols)
    if "list_date" not in m.columns:
        return mask

    d = idx.values.astype("datetime64[ns]")[:, None]         # (T,1)
    ld = m["list_date"].values.astype("datetime64[ns]")[None, :]   # (1,N)
    ok = ld <= d

    if "delist_date" in m.columns:
        dd = m["delist_date"].values.astype("datetime64[ns]")[None, :]
        # NaT 表示未退市 -> 恒可交易；否则必须 delist_date > t
        ok &= np.isnat(dd) | (dd > d)

    if "exchange" in m.columns and exchanges:
        ex = m["exchange"].astype(object).values[None, :]
        ok &= np.isin(ex, list(exchanges))

    return pd.DataFrame(ok, index=idx, columns=cols)


def tradable_at(t, codes=None, exchanges: Sequence[str] = DEFAULT_EXCHANGES,
                master: pd.DataFrame = None) -> list:
    """某一日可交易的代码列表"""
    m = load_master() if master is None else master
    if m is None or m.empty:
        return []
    use = list(codes) if codes is not None else m["code"].tolist()
    mask = tradable_mask([t], use, exchanges=exchanges, master=m)
    return [c for c in mask.columns if bool(mask.iloc[0][c])]


def live_codes(exchanges: Optional[Iterable[str]] = None,
               master: pd.DataFrame = None) -> list:
    """**当前仍在上市**的代码（供数据更新用）

    判据是 `delist_date` 为空（或晚于今天），而不是"名字里有没有『退』" ——
    后者既漏（改名/重组退市的不带"退"）又错（退市整理期带"退"但仍可交易）。
    `exchanges=None` 时**不按交易所过滤**：数据要下全，
    至于哪些进回测池由 `tradable_mask` 决定。
    """
    m = load_master() if master is None else master
    if m is None or m.empty:
        return []
    if "delist_date" not in m.columns:
        # 老主表：退回 list_status（若也没有就只能是全部）
        if "list_status" in m.columns:
            return sorted(m.loc[m["list_status"] == "L", "code"].tolist())
        return sorted(m["code"].tolist())
    today = pd.Timestamp.today().normalize()
    live = m[m["delist_date"].isna() | (m["delist_date"] > today)]
    if exchanges is not None and "exchange" in live.columns:
        live = live[live["exchange"].isin(list(exchanges))]
    return sorted(live["code"].astype(str).tolist())


def delisted_codes(master: pd.DataFrame = None) -> list:
    """已退市代码（回测历史时必须能出现在池子里，否则是幸存者偏差）"""
    m = load_master() if master is None else master
    if m is None or m.empty or "delist_date" not in m.columns:
        return []
    today = pd.Timestamp.today().normalize()
    return sorted(m.loc[m["delist_date"].notna() & (m["delist_date"] <= today),
                        "code"].astype(str).tolist())


__all__ = ["REQUIRED_COLS", "DEFAULT_EXCHANGES", "master_path", "load_master",
           "master_capabilities", "tradable_mask", "tradable_at", "live_codes",
           "delisted_codes"]
