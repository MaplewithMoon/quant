# -*- coding: utf-8 -*-
"""标准结果对象 + 集中渲染层测试（T1·⑤）

背景（为什么要有这一层）
-----------------------
以前每个回测脚本各自 print 报告，于是**接 N 次、漏 N-1 次**：

    已知缺陷附注：只有 multifactor_backtest.py 接了
    前视自检    ：引擎会算，但只有部分脚本打印
    风控触发    ：引擎会记，但没有脚本打印

收敛成「标准结果对象 + 一处渲染」后，新脚本只要走标准对象就自动带全部附注。

本文件盯住三件事：
  1. 缺陷**自动匹配**（不靠脚本作者记得挂）—— 尤其"用了当前快照行业要报 B7"
  2. 报告**头部先声明口径**（成交时点 / 门禁状态），不能先给数字
  3. 被拦委托被汇总（C5 的度量入口）
"""
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))


def _panel(n_days=60, codes=("600000", "000001", "300750"), seed=0):
    rng = np.random.default_rng(seed)
    idx = pd.bdate_range("2024-01-01", periods=n_days)
    close = pd.DataFrame(
        10 * np.exp(np.cumsum(rng.normal(0, 0.01, (n_days, len(codes))), axis=0)),
        index=idx, columns=list(codes))
    return {"close": close,
            "open": close.shift(1).fillna(close.iloc[0])}


def _weights(panel, every=10):
    idx = panel["close"].index
    tw = pd.DataFrame(np.nan, index=idx, columns=panel["close"].columns)
    for d in idx[::every]:
        tw.loc[d] = 1.0 / panel["close"].shape[1]
    return tw


# ============================================================
# 一、缺陷自动匹配
# ============================================================
def test_match_defects_uses_backtest_inputs():
    """**核心**：按回测输入自动匹配，而不是靠脚本作者手工挂

    这解决的是"接 N 次、漏 N-1 次"：新增脚本只要把 ctx 传进引擎，
    附注就自动出现。
    """
    from database.defects import match_defects

    base = dict(start="2017-01-01", end="2025-12-31",
                exchanges=("SSE", "SZSE"))
    pit = match_defects({**base, "industry_source": "pit"})
    snap = match_defects({**base, "industry_source": "snapshot"})
    keys_pit = {d.key for d in pit}
    keys_snap = {d.key for d in snap}

    assert "B7" not in keys_pit, "用 PIT 行业面板不该报 B7（前视）"
    assert "B7" in keys_snap, \
        "用了当前快照行业却没报 B7 —— 这正是前视，必须自动标注"

    # 用了指数成分 -> 报 B10（月末快照滞后）
    with_index = {d.key for d in match_defects({**base, "index_code": "000300.SH"})}
    assert "B10" in with_index, "用指数成分却没报 B10"

    # 池子含北交所 -> 报 C1
    with_bse = {d.key for d in match_defects(
        {**base, "exchanges": ("SSE", "SZSE", "BSE"), "codes": ["920001"]})}
    assert "C1" in with_bse, "池子含北交所却没报 C1"

    # 区间很短 -> 不报噪声（B11/C4b 只在长区间提示）
    short = match_defects({**base, "start": "2024-01-01", "end": "2024-06-30",
                           "industry_source": "pit"})
    assert {d.key for d in short} == set(), f"短区间不该报噪声: {[d.key for d in short]}"
    print("[OK] 缺陷自动匹配：PIT 不报 B7 / 快照报 B7 / 指数报 B10 / "
          "北交所报 C1 / 短区间无噪声")


def test_match_defects_returns_sorted_and_safe():
    """按严重度排序；trigger 自身出错也不能让回测挂掉"""
    from database.defects import KnownDefect, match_defects

    hits = match_defects({"start": "2010-01-01", "end": "2025-12-31",
                          "industry_source": "snapshot",
                          "index_code": "000300.SH"})
    sev = [d.severity for d in hits]
    order = {"高": 0, "中": 1, "低": 2}
    assert sev == sorted(sev, key=lambda s: order[s]), f"未按严重度排序: {sev}"

    # 坏 trigger 不允许把回测拖垮
    bad = KnownDefect(key="TST", title="坏的", scope="-", impact="-",
                      severity="低", needs_data=False,
                      trigger=lambda ctx: 1 / 0)
    import database.defects as dm
    dm.DEFECTS.append(bad)
    try:
        out = match_defects({"start": "2020-01-01", "end": "2021-01-01"})
        assert any(d.key == "TST" for d in out), \
            "trigger 出错时应记为命中（不能静默忽略）"
    finally:
        dm.DEFECTS.remove(bad)
    print("[OK] 自动匹配按严重度排序；坏 trigger 被记为命中而非静默忽略")


def test_defects_without_trigger_are_not_matched():
    """**设计约束**：没写 trigger 的条目不会被匹配到（宁可漏标不误标）

    这是刻意的 —— 误标一片噪声会让人对附注麻木，反而漏掉真问题。
    """
    from database.defects import KnownDefect, match_defects

    nt = KnownDefect(key="NOTRIG", title="没触发条件", scope="-", impact="-",
                     severity="低", needs_data=False, trigger=None)
    import database.defects as dm
    dm.DEFECTS.append(nt)
    try:
        out = match_defects({"start": "2010-01-01", "end": "2025-12-31",
                             "industry_source": "snapshot"})
        assert not any(d.key == "NOTRIG" for d in out), \
            "没有 trigger 的条目不该被自动匹配"
    finally:
        dm.DEFECTS.remove(nt)
    print("[OK] 无 trigger 的条目不参与自动匹配（避免误标噪声）")


# ============================================================
# 二、集中渲染层
# ============================================================
def test_render_header_declares_before_numbers():
    """**核心**：报告头部必须先声明口径与门禁状态，再给数字

    否则读者会先入为主地拿上界（same_close）当结论；
    没跑门禁时也不能留空，必须**明说没跑**。
    """
    from analytics.result_report import render_header

    class R:
        fill_timing = "same_close"
        gate = {}
        defects = []

    txt = "\n".join(render_header(R()))
    assert "上界" in txt and "不可复现" in txt, \
        "same_close 没被声明为上界 —— 读者会拿它当结论"
    assert "未执行" in txt, "没跑门禁时必须明说，不能留空"

    class R2:
        fill_timing = "next_open"
        gate = {"status": "FAIL", "summary": "2 个因子不合格",
                "reasons": ["mom_20 样本外符号相反"]}
        defects = [1, 2]

    t2 = "\n".join(render_header(R2()))
    assert "主结果" in t2, "next_open 应声明为可实现的主结果"
    assert "FAIL" in t2 and "样本外符号相反" in t2, "门禁失败原因应显示"
    assert "2 项" in t2, "缺陷计数应显示"
    print("[OK] 报告头部：先声明口径与门禁，再给数字（same_close 标为上界）")


def test_render_result_sections():
    """统一报告包含全部固定段落"""
    from backtest.multi_engine import PortfolioBacktestEngine

    panel = _panel()
    res = PortfolioBacktestEngine().run(
        panel, _weights(panel),
        context={"exchanges": ("SSE", "SZSE"), "industry_source": "snapshot",
                 "index_code": "000300.SH"})
    txt = res.render(title="自检")
    for sec in ("成交时点", "因子门禁", "已知缺陷", "绩效", "前视自检"):
        assert sec in txt, f"统一报告缺少「{sec}」段落"
    assert "[B7]" in txt, "传了 industry_source=snapshot 却没报 B7"
    assert "[B10]" in txt, "传了 index_code 却没报 B10"
    print("[OK] 统一报告：头部 + 绩效 + 缺陷附注 + 前视自检 全部到齐")


def test_engine_attaches_defects_to_result():
    """缺陷挂在**结果对象**上（不是脚本自己 print）—— 这是 ⑤ 的关键"""
    from backtest.multi_engine import PortfolioBacktestEngine

    panel = _panel()
    res = PortfolioBacktestEngine().run(
        panel, _weights(panel),
        context={"exchanges": ("SSE", "SZSE"), "industry_source": "snapshot"})
    assert hasattr(res, "defects") and res.defects, "结果对象没有 defects 字段"
    assert hasattr(res, "context") and res.context.get("start") is not None, \
        "结果对象没有回测上下文"
    assert res.fill_timing, "结果对象没有记录成交时点口径"
    # 区间由引擎从面板自动填
    assert str(res.context["start"])[:4] == "2024"
    print(f"[OK] 结果对象自带 defects({len(res.defects)}) / context / fill_timing")


def test_render_rejections_is_the_c5_measurement_hook():
    """被拦委托的汇总就是 C5 的**度量入口**（先度量，再决定建不建模）"""
    from analytics.result_report import render_rejections

    class R:
        rejections = {"涨停封板无法买入": 120, "停牌不可交易": 30}

    txt = "\n".join(render_rejections(R()))
    assert "涨停封板无法买入" in txt and "120" in txt
    assert "下界" in txt, "必须声明主结果是下界（真实排队可能成交一部分）"
    assert "150" in txt, "没有汇总总数"
    # 无拦截时不输出空段
    class R2:
        rejections = {}
    assert render_rejections(R2()) == []
    print("[OK] 被拦委托汇总（C5 度量入口）：分类 + 占比 + 总数 + 下界声明")


if __name__ == "__main__":
    test_match_defects_uses_backtest_inputs()
    test_match_defects_returns_sorted_and_safe()
    test_defects_without_trigger_are_not_matched()
    test_render_header_declares_before_numbers()
    test_render_result_sections()
    test_engine_attaches_defects_to_result()
    test_render_rejections_is_the_c5_measurement_hook()
    print("\n全部标准结果对象 / 渲染层测试通过")
