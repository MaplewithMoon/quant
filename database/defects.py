# -*- coding: utf-8 -*-
"""已知数据缺陷注册表

【为什么需要这个】
----------------
项目里有若干"数据本身就不对、但代码没错"的区间。它们的共同危险是：
**回测跑过去不会报错，结果却是错的**。而这些约束此前只写在
`docs/已知问题与待办汇总.md` 里 —— 跑回测的人根本看不到。

这个模块把它们变成**代码里可查询的注册表**，于是：
    1. 回测可以**自动附注**："你这段区间触及了 N 项已知缺陷"
    2. 股票池可以**主动排除**靠不住的数据
    3. 哪些缺陷"再买点数据就能修"一目了然（`needs_data=True`）

【与文档的分工】
    文档讲**为什么**（来龙去脉、实测证据），这里只讲**是什么、影响哪段**。
    每个条目都带 `doc_ref` 指回文档，避免两处漂移。

【诚实原则】
    `severity` 与 `needs_data` 必须如实填。把"需要额外数据才能根治"的写成
    已解决，或者把"保守排除了一个可能没问题的区间"说成"数据错误"，
    都会让后面的人做错决策。
"""
from dataclasses import dataclass
from typing import List, Optional

import pandas as pd

# 北交所开市日：此前不存在北交所，`920xxx` 是**新三板**遗留代码，
# ±30% 的北交所涨跌幅规则不适用，且实测 `pre_close` 大量缺失/异常。
BSE_FROM = pd.Timestamp("2021-11-15")
# 精选层设立日（±30% 由此开始）。留作参考：我们对 2020-07-27 ~ 2021-11-14
# 采取**保守排除**，宁可少回测一段，也不拿不可靠的涨跌停价出结论。
BSE_SELECT_TIER_FROM = pd.Timestamp("2020-07-27")

BSE_PREFIXES = ("920", "43", "83", "87", "88")

# 「重新上市」股：重组/重新上市，**不适用**新股上市首日规则。
#
# 公开记录确认：A 股按「重新上市」通道回来的就是三家 ——
#   招商南油（原长航油运）、国机重装、汇绿生态（退市新规以来首家、A股第三家）。
# 此后注册制下该通道实质冻结，名单大概率稳定。
#
# ⚠️ **必须记公司主体，不能只记代码**：重新上市通常换代码换名。
# ⚠️ **必须与「恢复上市」区分**：盐湖股份、皇台酒业这类是"暂停上市后恢复上市"，
#    代码未变、始终在证券主表里，其中断期应由停复牌数据兜住。
#    把它们加进这张名单会制造**新的误拦**（本该受限的日子被放行）。
RELISTED_CODES = ("601399", "001267", "601155")
# 公司主体（现名 / 曾用名），供换代码时对照
RELISTED_ENTITIES = {
    "601399": ("国机重装", "二重重装"),
    "001267": ("汇绿生态", "六渡桥"),
    "601155": ("新城控股",),
}
# 明确**不属于**重新上市的"恢复上市"案例（写在这里防误加）
RESUMED_NOT_RELISTED = {
    "000792": "盐湖股份（暂停上市后恢复上市，代码未变）",
    "600543": "莫高股份",
    "000995": "皇台酒业（暂停上市后恢复上市，代码未变）",
}


@dataclass
class KnownDefect:
    """一条已知数据缺陷

    key:        短标识（与文档编号对应，如 "C1"）
    title:      一句话说明
    scope:      适用范围的人类可读描述
    date_range: (start, end) 或 None（None=不限日期）
    codes:      具体代码列表（None=不限）
    prefixes:   代码前缀（None=不限）
    boards:     板块（None=不限）
    impact:     对回测的影响
    severity:   "高" / "中" / "低"
    needs_data: True = 需要**额外数据源**才能根治（代码层面只能缓解）
    mitigation: 代码层面已经做了什么
    doc_ref:    文档章节
    """
    key: str
    title: str
    scope: str
    impact: str
    severity: str
    needs_data: bool
    mitigation: str = ""
    date_range: Optional[tuple] = None
    codes: Optional[List[str]] = None
    prefixes: Optional[tuple] = None
    boards: Optional[tuple] = None
    doc_ref: str = ""

    def covers(self, code=None, date=None, board=None) -> bool:
        """这条缺陷是否覆盖给定的 (代码, 日期, 板块)"""
        if self.codes is not None:
            if code is None or str(code).zfill(6) not in self.codes:
                return False
        if self.prefixes is not None:
            if code is None or not str(code).zfill(6).startswith(self.prefixes):
                return False
        if self.boards is not None:
            if board is None or board not in self.boards:
                return False
        if self.date_range is not None:
            if date is None:
                return False
            d = pd.Timestamp(date)
            s, e = self.date_range
            if s is not None and d < pd.Timestamp(s):
                return False
            if e is not None and d > pd.Timestamp(e):
                return False
        return True


# ============================================================
# 注册表
# ============================================================
DEFECTS: List[KnownDefect] = [
    KnownDefect(
        key="C1",
        title="北交所 920xxx 在 2021-11-15 前的涨跌停规则不可用",
        scope="代码前缀 920/43/83/87/88，日期 < 2021-11-15",
        impact="这些代码在数据里是**新三板**遗留记录（北交所 2021-11-15 才开市）。"
               "套用北交所 ±30% 规则算出的涨跌停价没有意义；实测 `pre_close` "
               "大量缺失（238 行）或异常（约 40 行价格 < 0.5 元），集中在 2008–2022 年。"
               "更危险的方向是**偏宽松**：涨跌停价算错会让引擎拦不住本该封板的成交。",
        severity="低",
        needs_data=True,
        mitigation="**主池已从根上排除**：`universe.pool` 默认走证券主表三层过滤，"
                   "`exchange in ('SSE','SZSE')` 天然不含北交所/新三板"
                   "（见 `database/master.py`）。涨跌停价本身仍置 NaN 作为诚实标记；"
                   "`unreliable_limit_mask` 保留为第二道保险（显式纳入 BSE 时生效）。"
                   "根治（让 BSE 也能用）仍需可靠的新三板历史行情/规则数据。",
        date_range=(None, "2021-11-14"),
        prefixes=BSE_PREFIXES,
        doc_ref="2.18",
    ),
    KnownDefect(
        key="B7",
        title="行业分类曾是「当前快照」—— 已用 index_member_all 建成 PIT 表",
        scope="全部日期（历史部分已由 frozen/industry/sw_member 覆盖）",
        impact="用**现在**的行业归属去回测历史，等于知道这家公司后来被划到哪个"
               "行业，对行业中性化类策略构成**前视**。全库有 1,646 只股票换过"
               "行业（最多 6 段），影响不是边角。",
        severity="低",
        needs_data=False,
        mitigation="已引入 `index_member_all`（带 in_date/out_date/is_new）"
                   "建 PIT 行业面板 `database/industry.py`：5,906 只 / 31 个 L1、"
                   "7,912 条区间。记录空档按「未观测到变更 ⇒ 视为未变更」沿用"
                   "上一段（而不是跳到当前快照，那才是前视）。"
                   "`neutralized_score` 与多因子回测已改用 PIT 面板，"
                   "并在报告里打印覆盖率与快照兜底比例。",
        doc_ref="一·B7 / 2.17",
    ),
    KnownDefect(
        key="B9",
        title="指数成分权重覆盖不全",
        scope="399101 从 2016-05-31 起；000985 从 2015-01 起",
        impact="更早区间没有成分快照，只能用规则重建，或干脆不可用。"
               "用这些指数做股票池时，早于起点会取不到成分。",
        severity="低",
        needs_data=False,
        mitigation="区间外的日期由 `year_globs` + 空集自然退化；"
                   "不产生错数据，只是覆盖不到。",
        date_range=(None, "2016-05-31"),
        doc_ref="一·B9",
    ),
    KnownDefect(
        key="B10",
        title="指数成分只有**月末**权重快照，且无进出日期",
        scope="全部使用 index_cons 的股票池（沪深300/中证500/中证1000/399101/000985）",
        impact="月中调整的成分无法精确还原。月末快照之间用前向填充，"
               "会产生最多约 1 个月的**成分滞后**（轻微幸存者偏差方向）。",
        severity="低",
        needs_data=True,
        mitigation="`universe/pool.py` 只用 `<= t` 的快照（不偷看未来），"
                   "滞后方向是「晚了才纳入」而非「提前知道」。"
                   "**已核实 tushare 没有**提供中证指数带 in_date/out_date 的接口："
                   "`index_member` 是**申万行业成分**（返回 801xxx.SI 那套），"
                   "`index_member_all` 是它的分级版，`index_weight` 只有月度快照。"
                   "免费且精确的两条路：① 聚宽 `get_index_stocks(symbol, date)` "
                   "按调整日批量导出；② 中证指数官网历次样本调整公告（Excel/PDF），"
                   "沪深300/中证500/中证1000 各约 40 次调整，一次性建设。",
        doc_ref="2.19",
    ),
    KnownDefect(
        key="B11",
        title="`suspend_d` 不记录「暂停上市」级长期停牌",
        scope="全部日期",
        impact="`000757` 停牌近 6 年却 0 条记录，同类共 43 段。"
               "影响有限（长期停牌期该股无日线、进不了价格面板，自然不会成交），"
               "但**不能**把 `frozen/suspend` 当作停牌的完整来源。",
        severity="低",
        needs_data=True,
        mitigation="价格面板缺失 -> 无法成交，实际已被兜住；"
                   "根治需要另一路停牌数据源。",
        doc_ref="一·B11",
    ),
    KnownDefect(
        key="C2b",
        title="「重新上市」股被误套新股首日 ±44% 规则",
        scope="按**公司主体**识别，不只按代码（重新上市通常换代码换名）",
        impact="它们是重组/重新上市，不适用新股首日规则。"
               "实测首日涨幅分别 +188.9% / +122.5% / +235.9%，"
               "被规则误拦 -> 该日**本该能成交却成交不了**（偏悲观）。",
        severity="低",
        needs_data=True,
        mitigation="公开记录确认 A 股按「重新上市」通道回来的就是三家，"
                   "已加入 `RELISTED_CODES` 由 `listing_windows()` 跳过。"
                   "⚠️ 必须与「**恢复上市**」区分：盐湖股份、皇台酒业这类是"
                   "暂停上市后恢复上市，**代码未变**、一直在证券主表里，"
                   "其中断期由停复牌数据兜住；把它们加进排除名单会制造**新的误拦**。"
                   "完整名单需每年核对交易所公告（低速变更集合）。",
        codes=["601399", "001267", "601155"],
        doc_ref="2.19",
    ),
    KnownDefect(
        key="C4b",
        title="复牌首日不设涨跌幅：阈值是经验性的（原描述「未股改 S 股无限制」是错的）",
        scope="长期停牌（错过 ≥11 个交易日）后复牌的首个交易日",
        impact="原文档写作「2006–2007 未股改 S 股无涨跌幅限制」，**用行情数据反证后"
               "不成立**：纯 S 股 2006-2007 的 68,552 行里 |涨跌幅|>10.5% 只有 "
               "97 行（0.14%），同期全市场 0.157% —— 几乎一样，纯 S 股照样受 ±10% 约束。"
               "真实机制是**停牌复牌首日不设涨跌幅**：960 行越界里能判定间隔的 "
               "771 行中有 770 行（99.87%）是复牌首日，|涨跌幅|>20% 的 444 行全部是。",
        severity="低",
        needs_data=False,
        mitigation="`limit_rules.resumption_windows()` 用「市场开市而该股无 K 线」的"
                   "交易日数判定复牌首日（≥11 个），这些日子涨跌停置 NaN。"
                   "阈值是**经验性**的（错过 6-10 日时越界率 3.47%、≥11 日时 45%），"
                   "不是查到的交易所条文 —— 这是本条仍标为缺陷的原因。"
                   "另：`frozen/suspend` 的 R 记录**完全不覆盖**这些长期停牌"
                   "（771 个复牌日中 0 个有 R 记录），所以判定只能靠 K 线间隔。",
        date_range=("1990-01-01", None),
        doc_ref="2.19",
    ),
]

_BY_KEY = {d.key: d for d in DEFECTS}


def get(key: str) -> Optional[KnownDefect]:
    return _BY_KEY.get(key)


def defects_for(code=None, date=None, board=None,
                extra_dates=None) -> List[KnownDefect]:
    """给定 (代码, 日期, 板块) 命中的已知缺陷

    extra_dates: 额外要一并检查的日期（例如整段回测的首尾），
                 用于回答"这段区间有没有触及某条缺陷"。
    """
    dates = [d for d in ([date] if date is not None else []) +
             list(extra_dates or [])]
    hits = []
    for dfc in DEFECTS:
        if dfc.covers(code=code, date=date, board=board):
            hits.append(dfc)
            continue
        # 日期区间类缺陷：任一日期落在区间内即算命中
        if dfc.date_range is not None and code is None and board is None:
            for d in dates:
                if dfc.covers(date=d):
                    hits.append(dfc)
                    break
    return hits


def defects_in_window(start, end, board=None, codes=None) -> List[KnownDefect]:
    """回测窗口 [start, end] 触及的已知缺陷

    判断口径：缺陷区间与回测窗口**有交集**即算触及（宁可多标注）。
    这是给回测报告自动附注用的 —— 读者有权知道"这段结论踩着哪些已知问题"。
    """
    s, e = pd.Timestamp(start), pd.Timestamp(end)
    out = []
    for dfc in DEFECTS:
        if dfc.date_range is not None:
            ds, de = dfc.date_range
            ds = pd.Timestamp(ds) if ds is not None else pd.Timestamp("1900-01-01")
            de = pd.Timestamp(de) if de is not None else pd.Timestamp("2100-01-01")
            if de < s or ds > e:
                continue
        out.append(dfc)
    return out


def unreliable_limit_mask(codes, dates) -> "pd.DataFrame":
    """哪些 (代码, 日期) 的涨跌停价**不可信** —— 供股票池排除

    返回 bool 宽表（index=日期, columns=代码），True = 不可信。
    目前只有 C1（北交所前缀在 2021-11-15 之前）会命中。
    """
    idx = pd.DatetimeIndex(pd.to_datetime(dates))
    cols = [str(c).zfill(6) for c in codes]
    mask = pd.DataFrame(False, index=idx, columns=cols)
    for dfc in DEFECTS:
        if dfc.prefixes is None or dfc.date_range is None:
            continue
        ds, de = dfc.date_range
        hit_codes = [c for c in cols if c.startswith(dfc.prefixes)]
        if not hit_codes:
            continue
        cond = pd.Series(True, index=idx)
        if ds is not None:
            cond &= idx >= pd.Timestamp(ds)
        if de is not None:
            cond &= idx <= pd.Timestamp(de)
        for c in hit_codes:
            mask.loc[cond, c] = True
    return mask


def format_banner(defects, indent: str = "  ") -> str:
    """把命中的缺陷渲染成回测报告里的一段附注"""
    if not defects:
        return f"{indent}未触及已知数据缺陷。"
    L = [f"{indent}⚠ 本区间触及 {len(defects)} 项**已知数据缺陷**"
         f"（结论需带此前提解读）："]
    for d in sorted(defects, key=lambda x: x.severity):
        flag = "【需额外数据】" if d.needs_data else ""
        L.append(f"{indent}  · [{d.key}] {d.title}  (严重度 {d.severity}){flag}")
        L.append(f"{indent}      {d.impact}")
        if d.mitigation:
            L.append(f"{indent}      已做: {d.mitigation}")
    return "\n".join(L)


def data_todo() -> List[KnownDefect]:
    """需要**额外数据**才能根治的缺陷 —— 这份清单交给人来决定怎么处理"""
    return [d for d in DEFECTS if d.needs_data]


__all__ = ["KnownDefect", "DEFECTS", "BSE_FROM", "BSE_SELECT_TIER_FROM",
           "BSE_PREFIXES", "get", "defects_for", "defects_in_window",
           "unreliable_limit_mask", "format_banner", "data_todo"]
