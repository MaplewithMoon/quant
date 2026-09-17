# -*- coding: utf-8 -*-
"""回测报告的**集中渲染层**

【为什么要有这一层】
以前每个回测脚本各自 print 自己的报告：涨跌停拦截、账目差额、前视自检、
已知缺陷附注……于是**接 N 次、漏 N-1 次**：

| 项 | 接上的脚本 |
|---|---|
| 已知缺陷附注 | 只有 `multifactor_backtest.py` |
| 前视自检 | 引擎会算，但只有部分脚本打印 |
| 风控触发 | 引擎会记，但没有脚本打印 |

本模块收敛成**一处**：回测脚本只管拿到标准结果对象（`MultiBacktestResult`），
调 `render_result()` 出报告。新脚本只要走标准对象，就自动带全部附注。

【报告结构（顺序固定）】
    1. 头部声明   —— 口径、门禁状态、缺陷计数。**先声明再给数字**，
                     否则读者会先入为主地拿上界的数字当结论
    2. 绩效摘要
    3. 已知缺陷附注（自动匹配，见 database/defects.py::match_defects）
    4. 前视自检
    5. 风控触发
    6. 被制度约束拦下的委托（涨跌停/停牌/整手等）

【与 T1 的关系】
    - C5（排队/部分成交）：拦下的委托会在这里汇总，度量先行
    - C7（成交时点）：第 1 节固定口径声明（same_close=上界 / next_open=可实现）
    - A8（因子门禁）：第 1 节显示门禁状态，不合格因子在结果里**标红不剔除**
    - ④（风格分解）：调用方可传 `extra_sections` 插入
"""
from typing import List, Optional

# 成交时点的固定口径声明 —— **写死**，不让调用方改
FILL_CAVEAT = {
    "next_open": "next_open（T日收盘决策 → T+1开盘成交）：**可实现口径，主结果**",
    "same_close": ("same_close（当日收盘决策且按当日收盘成交）：**上界，实盘不可复现**，"
                   "只能作为对照"),
}


def _sec(title: str, ch: str = "-", w: int = 78) -> List[str]:
    return ["", ch * w, title, ch * w]


def render_header(res, title: Optional[str] = None,
                  universe_desc: Optional[str] = None) -> List[str]:
    """报告头部：口径 + 门禁 + 缺陷计数。**放在所有数字之前。**"""
    L = ["=" * 78]
    L.append(title or "回测结果")
    L.append("=" * 78)

    ft = getattr(res, "fill_timing", "") or "next_open"
    L.append(f"  成交时点 : {FILL_CAVEAT.get(ft, ft)}")

    # 门禁状态（A8）：没跑过就要**明说**没跑，不能留空让人以为是干净的
    gate = getattr(res, "gate", None)
    if not gate:
        L.append("  ⚠ 因子门禁 : **未执行**（结果未经样本外有效性检查）")
    else:
        st = str(gate.get("status", "?")).upper()
        mark = {"PASS": "✅", "WARN": "⚠", "FAIL": "⛔"}.get(st, "?")
        extra = gate.get("summary") or ""
        L.append(f"  {mark} 因子门禁 : {st} {extra}".rstrip())
        for r in (gate.get("reasons") or [])[:5]:
            L.append(f"       · {r}")

    defects = getattr(res, "defects", None) or []
    if defects:
        L.append(f"  ⚠ 已知缺陷 : 本次触及 {len(defects)} 项"
                 f"（详见下方附注，结论需带此前提解读）")
    else:
        L.append("  已知缺陷 : 未触及已登记的缺陷")
    return L


def render_metrics(res) -> List[str]:
    L = _sec("一、绩效")
    m = getattr(res, "metrics", None)
    if m is None:
        L.append("  （无绩效指标）")
        return L
    rows = [("总收益率", "{:.2%}"), ("年化收益率", "{:.2%}"),
            ("年化波动", "{:.2%}"), ("夏普比率", "{:.2f}"),
            ("最大回撤", "{:.2%}"), ("交易笔数", "{:d}"),
            ("胜率", "{:.2%}")]
    for name, fmt in rows:
        v = getattr(m, _metric_attr(name), None)
        if v is None:
            continue
        try:
            L.append(f"  {name:<10}: {fmt.format(v):>12}")
        except (ValueError, TypeError):
            L.append(f"  {name:<10}: {v}")
    gap = getattr(res, "ledger_gap", None)
    if gap is not None:
        L.append(f"  {'账目差额':<10}: {gap:>12.2e}  (应=0)")
    return L


def _metric_attr(cn: str) -> str:
    return {"总收益率": "total_return", "年化收益率": "annual_return",
            "年化波动": "annual_volatility", "夏普比率": "sharpe_ratio",
            "最大回撤": "max_drawdown", "交易笔数": "total_trades",
            "胜率": "win_rate"}.get(cn, cn)


def render_defects(res) -> List[str]:
    defects = getattr(res, "defects", None) or []
    L = _sec("二、已知数据缺陷附注（自动匹配，见 database/defects.py）")
    if not defects:
        L.append("  未触及已登记的缺陷。")
        return L
    from database.defects import format_banner
    L.append(format_banner(defects))
    return L


def render_lookahead(res) -> List[str]:
    L = _sec("三、前视自检（引擎能保证的那部分 PIT）")
    rep = getattr(res, "lookahead_report", "")
    n = getattr(res, "lookahead_violations", 0) or 0
    if not rep:
        L.append("  （未执行；可用 audit_lookahead=True 开启）")
        return L
    if n:
        L.append(f"  ⛔ 发现 {n} 笔成交早于其决策日 —— **本次结果不可信**")
    L.append(rep if rep.startswith("  ") else "  " + rep.replace("\n", "\n  "))
    sd = getattr(res, "same_day_fills", 0) or 0
    if sd:
        L.append(f"  ⚠ 同日成交 {sd} 笔（fill=same_close）：温和前视，见头部口径声明")
    L.append("  注：策略**实际用了哪些数据**引擎无从得知，那一层要用")
    L.append("      backtest.lookahead.verify_point_in_time 逐决策校验。")
    return L


def render_risk(res) -> List[str]:
    ev = getattr(res, "risk_events", None) or []
    if not ev:
        return []
    L = _sec("四、风控触发（这些减仓是风控砍的，不是策略决定的）")
    for e in ev[:15]:
        L.append(f"  {str(e.get('date'))[:10]}  [{e.get('rule')}]  {e.get('reason')}")
    if len(ev) > 15:
        L.append(f"  ... 另有 {len(ev) - 15} 次")
    return L


def render_rejections(res) -> List[str]:
    """被制度约束拦下的委托 —— 也是 C5（排队/部分成交）的**度量入口**

    这里回答的是：「封板判完全不可成交」这个假设**影响多大**。
    度量结论决定要不要建排队模型：

        < 1%  -> 声明即可，不用建模（C5 到此为止）
        1~5%  -> 报告里注明方向（偏悲观），暂不建模
        > 5%  -> 值得建"非一字封板按比例成交"的近似（日线天花板就在那）
    """
    rej = getattr(res, "rejections", None) or {}
    detail = getattr(res, "rejection_detail", None) or []
    attempted = getattr(res, "attempted_turnover", 0.0) or 0.0
    if not rej and not detail:
        return []
    L = _sec("五、被制度约束拦下的委托（涨跌停/停牌/整手…）")
    total = sum(rej.values())
    for k, v in sorted(rej.items(), key=lambda kv: -kv[1])[:15]:
        L.append(f"  {v:>7,} 次  ({v / max(1, total):>5.1%})  {k}")
    if len(rej) > 15:
        L.append(f"  ... 另有 {len(rej) - 15} 类")
    L.append(f"  —— 合计 {total:,} 次")

    # ---- C5 度量：被拦金额占比 ----
    if detail:
        amts = [r.get("amount") for r in detail if r.get("amount")]
        blocked = float(sum(amts)) if amts else 0.0
        ow = [r for r in detail if r.get("one_word") is True]
        n_ow = len(ow)
        ratio = (blocked / attempted) if attempted > 0 else float("nan")
        L.append("")
        L.append("  --- C5 度量：假设「封板完全不可成交」的影响有多大 ---")
        L.append(f"  被拦 {len(detail):,} 笔，金额 {blocked:,.0f} 元；"
                 f"同期想成交总额 {attempted:,.0f} 元")
        if ratio == ratio:      # not NaN
            L.append(f"  **被拦金额占比 {ratio:.2%}**"
                     f"（一字板 {n_ow:,} 笔占 {n_ow / max(1, len(detail)):.0%}）")
            if ratio < 0.01:
                L.append("  判读：占比 <1% —— 该假设**不构成主要不确定性**，"
                         "报告声明即可，无需建排队模型")
            elif ratio < 0.05:
                L.append("  判读：占比 1%~5% —— 方向已知（**偏悲观**），"
                         "报告注明即可；暂不值得建模")
            else:
                L.append("  判读：占比 >5% —— 值得考虑「非一字封板按比例成交」"
                         "的近似（日线天花板就在那；再往上需要分钟数据）")
        # 一字板 vs 非一字：只有非一字才可能排队成交一部分
        L.append(f"  其中一字板 {n_ow:,} 笔（open==high==low==close==板价，"
                 f"全天无打开 -> 维持不可成交是对的）")
        L.append(f"       非一字 {len(detail) - n_ow:,} 笔"
                 f"（盘中打开过、有真实成交量 -> 排队**可能**成交一部分）")
    L.append("")
    L.append("  主结果基于「完全不可成交」，故是收益**下界**（见 T1·C5）。")
    return L


def render_result(res, title: Optional[str] = None,
                  extra_sections: Optional[List[str]] = None,
                  footer: Optional[str] = None) -> str:
    """把标准结果对象渲染成完整报告

    extra_sections: 调用方追加的段落（如 ④ 的风格暴露表），插在末节之前
    """
    L = []
    L += render_header(res, title=title)
    L += render_metrics(res)
    L += render_defects(res)
    L += render_lookahead(res)
    L += render_risk(res)
    L += render_rejections(res)
    if extra_sections:
        L += extra_sections
    if footer:
        L += ["", footer]
    L += ["", "=" * 78]
    return "\n".join(L)


__all__ = ["render_result", "render_header", "render_metrics", "render_defects",
           "render_lookahead", "render_risk", "render_rejections",
           "FILL_CAVEAT"]
