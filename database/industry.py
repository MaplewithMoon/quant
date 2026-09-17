# -*- coding: utf-8 -*-
"""PIT（point-in-time）行业归属

【问题】
`frozen/industry/stock_industry.parquet` 来自 `stock_basic.industry`，是
**当前快照** —— 只有"这家公司现在属于哪个行业"，没有历史。拿它做行业中性化，
等于用**今天的行业归属**去回测历史：

    000007 深信泰丰：1992 家电 -> 2006 房地产 -> 2013 有色金属
                    -> 2014 社会服务 -> 2017 综合 -> 2019 商贸零售
    用"商贸零售"去中性化它 2006 年的因子值，就是把二十年后的信息提前用了。

这是**前视**（B7），方向是把行业均值算错 —— 而这种错误不会报错。

【原料】
tushare `index_member_all`（申万行业成分分级）直接带：
    `in_date`  纳入日期
    `out_date` 剔除日期（当前有效的为空）
    `is_new`   'Y'=当前有效 / 'N'=已剔除
    以及 l1/l2/l3 三级行业代码与名称

实测：'Y' 5,906 行 / 5,906 只；'N' 2,006 行 / 1,646 只；
**1,646 只股票有 ≥2 段行业区间**（最多 6 段），`in_date` 零缺失。
所以 Y∪N 能拼出逐股区间序列，这正是构造 PIT 面板所需的全部原料。

【落盘】
`frozen/industry/year=2005/sw_member.parquet`（与 stock_industry 同目录，
静态数据集走 `year=2005` 占位约定）。
下载：`python scripts/download_frozen_tushare.py --only sw_member`
"""
from typing import Dict, Optional

import pandas as pd

from database.config import FROZEN_ROOT, NON_ANNUAL_YEAR

# 取不到时的兜底行业名（与 analytics.attribution.UNKNOWN_INDUSTRY 一致）
UNKNOWN = "未分类"


def sw_member_path():
    return FROZEN_ROOT / "industry" / f"year={NON_ANNUAL_YEAR}" / "sw_member.parquet"


def load_sw_member() -> pd.DataFrame:
    """读原始申万成分表

    返回列: ts_code / code / l1_code / l1_name / l2_name / l3_name /
            in_date / out_date / is_new
    `code` 是去掉交易所后缀的 6 位代码（本项目的股票主键）。
    """
    p = sw_member_path()
    if not p.exists():
        return pd.DataFrame()
    df = pd.read_parquet(p)
    if df.empty:
        return df
    out = df.copy()
    out["code"] = out["ts_code"].astype(str).str.split(".").str[0].str.zfill(6)
    out["_in"] = pd.to_datetime(out["in_date"], format="%Y%m%d", errors="coerce")
    if "out_date" in out.columns:
        out["_out"] = pd.to_datetime(out["out_date"], format="%Y%m%d",
                                     errors="coerce")
    else:
        out["_out"] = pd.NaT
    out = out.dropna(subset=["_in"])
    return out


def industry_intervals(level: str = "l1",
                       carry_forward: bool = True) -> Dict[str, list]:
    """{code: [(in_date, out_date_or_None, 行业名), ...]}，按 in_date 升序

    `carry_forward=True`（默认）时，区间被拼成**首尾相接、不重叠**：
    每段的有效结束日取「下一段的 in_date」，而**不是**原始 `out_date`。

    ⚠️ 为什么必须这样 —— 实测 `000007` 的原始记录是：

        家用电器 [1992-04-13, 2006-08-31]
        房地产   [2006-09-01, 2009-05-27]
        有色金属 [2013-07-01, 2014-06-30]     <- 中间空了 3 年多
        ...

    2009-05-28 ~ 2013-06-30 这段**没有任何记录**。tushare 的 `out_date` 只说
    "退出了房地产"，没说"去了哪儿"。三种处理方式里：

      (a) 沿用上一段（房地产）     <- **本函数的选择**
      (b) 落到当前快照（商贸零售） <- 明确的前视，2010 年不可能知道
      (c) 标成"未分类"             <- 信息丢失，这些股票会被抱团中性化

    (a) 是标准 PIT 口径：**没观测到变更就视为没变更**，也正是当时你数据库里
    真实能看到的东西。所以这里桥接了空档，而不是跳到当前快照。
    """
    df = load_sw_member()
    if df.empty:
        return {}
    col = f"{level}_name"
    if col not in df.columns:
        raise KeyError(f"申万成分表没有 {col} 列；可选 l1_name/l2_name/l3_name")
    out: Dict[str, list] = {}
    for code, g in df.groupby("code"):
        recs = [(r["_in"], (None if pd.isna(r["_out"]) else r["_out"]), r[col])
                for _, r in g.sort_values("_in").iterrows()]
        recs = [r for r in recs if isinstance(r[2], str) and r[2]]
        if not recs:
            continue
        if carry_forward:
            fixed = []
            for i, (s, _e, name) in enumerate(recs):
                nxt = recs[i + 1][0] if i + 1 < len(recs) else None
                fixed.append((s, nxt, name))
            recs = fixed
        out[str(code)] = recs
    return out


def industry_at(code: str, date, intervals: Dict[str, list] = None,
                level: str = "l1") -> Optional[str]:
    """某只股票在某个日期的行业（PIT）

    规则：取 `in_date <= date` 且（`out_date` 为空 或 `out_date >= date`）的那一段。
    取不到返回 None（调用方自行决定落到 `未分类`）。
    """
    if intervals is None:
        intervals = industry_intervals(level)
    recs = intervals.get(str(code).zfill(6))
    if not recs:
        return None
    d = pd.Timestamp(date)
    hit = None
    for s, e, name in recs:
        # 区间已由 industry_intervals 拼成半开 [s, e)，e=None 表示延续至今
        if s <= d and (e is None or d < e):
            hit = name          # 区间不重叠；真有重叠取最后命中的
    return hit


def industry_pit_panel(dates, codes, level: str = "l1",
                       intervals: Dict[str, list] = None,
                       fallback: bool = True) -> pd.DataFrame:
    """PIT 行业面板（index=日期, columns=代码, 值=行业名）

    `fallback=True` 时，PIT 表里**完全没有区间记录**的股票（例如在 tushare
    的申万记录之前就退市的老股票）用当前快照 (`stock_industry`) 补。
    有过区间记录、但某段日期落在记录空档里的，已由 `industry_intervals` 的
    `carry_forward` 桥接成"沿用上一段"，**不会**掉到当前快照 —— 那会造成前视。

    `attrs` 里记录 `pit_coverage`（非"未分类"占比）与 `fallback_ratio`，
    让调用方知道有多少是兜底猜的。

    性能：逐股票按区间切日期，不做逐 (股票,日期) 循环。
    """
    idx = pd.DatetimeIndex(pd.to_datetime(dates))
    cols = [str(c).zfill(6) for c in codes]
    panel = pd.DataFrame(UNKNOWN, index=idx, columns=cols, dtype=object)
    if intervals is None:
        intervals = industry_intervals(level)

    for c in cols:
        recs = intervals.get(c)
        if not recs:
            continue
        # 用区间切日期，而不是逐日判断：区间数远小于日期数
        for s, e, name in recs:
            m = idx >= s
            if e is not None:
                m &= idx < e          # carry_forward 下 e = 下一段 in_date，取半开
            if m.any():
                panel.loc[m, c] = name

    total = len(idx) * len(cols)
    n_unknown_before = int(panel.eq(UNKNOWN).to_numpy().sum())
    if fallback and n_unknown_before:
        try:
            from database.loader import load_industry_map
            snap = load_industry_map()
            if not snap.empty and "code" in snap.columns:
                col = next((x for x in ("industry", "sw_l1")
                            if x in snap.columns), None)
                if col:
                    m = snap.set_index("code")[col]
                    for c in cols:
                        v = m.get(c)
                        if isinstance(v, str) and v:
                            panel.loc[panel[c].eq(UNKNOWN), c] = v
        except Exception:
            pass

    got = int((~panel.eq(UNKNOWN)).to_numpy().sum())
    panel.attrs["pit_coverage"] = got / total if total else 0.0
    panel.attrs["unknown_ratio"] = 1.0 - (got / total if total else 0.0)
    panel.attrs["snapshot_fallback_ratio"] = (
        (n_unknown_before - int(panel.eq(UNKNOWN).to_numpy().sum())) / total
        if total else 0.0)
    panel.attrs["source"] = ("index_member_all（PIT，带 in_date/out_date，"
                             "空档沿用上一段）"
                             if intervals else "无 sw_member，全部落未分类")
    return panel


def industry_series_at(date, codes, level: str = "l1",
                       intervals: Dict[str, list] = None) -> pd.Series:
    """某个日期的行业截面（index=代码）—— 给"当天中性化"用"""
    if intervals is None:
        intervals = industry_intervals(level)
    vals = [industry_at(c, date, intervals) or UNKNOWN for c in codes]
    return pd.Series(vals, index=[str(c).zfill(6) for c in codes])


def has_pit_data() -> bool:
    return sw_member_path().exists()


__all__ = ["UNKNOWN", "sw_member_path", "load_sw_member", "industry_intervals",
           "industry_at", "industry_pit_panel", "industry_series_at",
           "has_pit_data"]
