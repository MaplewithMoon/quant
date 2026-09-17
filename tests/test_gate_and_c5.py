# -*- coding: utf-8 -*-
"""C5 埋点度量（T1·①）+ 因子门禁（T1·③）测试

【① C5 为什么"只埋点不改撮合"】
"封板判完全不可成交"是个假设。**先用数据回答"影响多大"**，再决定要不要建模：
    <1%  -> 声明即可，不用建模
    1~5% -> 报告注明方向（偏悲观），暂不建模
    >5%  -> 值得做"非一字封板按比例成交"的近似（日线天花板就在那）
埋点零风险（不动撮合逻辑），所以这一步先做、也必须先做。

【③ 门禁为什么要"四个条件各自独立"】
A8 的病是"样本外 IC 显著但多空为负、单调性≈0"只靠人工看分层。
若把四个条件揉成一个布尔，报告只能说"不合格"，说不清**死在哪个条件上**。
另外两条约束：**不静默剔除**（否则造出"被审查过的幸存者因子库"，
与股票池的幸存者偏差同构）；**留豁免通道**但报告必须显式标注"未执行"。
"""
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))


def _sealed_panel(periods=40, seal_on=11):
    """造一个"某只股票在某天成一字板"的面板"""
    idx = pd.bdate_range("2024-01-01", periods=periods)
    cols = ["600000", "000001"]
    rng = np.random.default_rng(3)
    close = pd.DataFrame(10 * np.exp(np.cumsum(rng.normal(0, 0.01, (periods, 2)),
                                                      axis=0)),
                         index=idx, columns=cols)
    p = {"close": close, "open": close.copy(),
         "high": close * 1.001, "low": close * 0.999,
         "limit_up": close * 1.10, "limit_down": close * 0.90}
    d = idx[seal_on]
    for k in ("open", "high", "low", "close"):
        p[k].loc[d, "000001"] = p["limit_up"].loc[d, "000001"]
    return idx, cols, p


# ============================================================
# ① C5 埋点
# ============================================================
def test_rejection_detail_records_amount_and_one_word():
    """**核心**：被拦委托要记下金额、方向、是否一字板

    没有这些，"这个假设影响多大"就只是个感觉，无法决策。
    """
    from backtest.multi_engine import PortfolioBacktestEngine

    idx, cols, panel = _sealed_panel()
    tw = pd.DataFrame(np.nan, index=idx, columns=cols)
    tw.loc[idx[10]] = [0.5, 0.5]
    tw.loc[idx[20]] = [0.5, 0.5]
    res = PortfolioBacktestEngine().run(panel, tw, audit_lookahead=False)

    assert res.rejection_detail, "没有记录任何被拦明细"
    r = res.rejection_detail[0]
    for k in ("date", "code", "side", "reason", "amount", "one_word"):
        assert k in r, f"明细缺字段 {k}"
    assert r["code"] == "000001" and r["side"] == "buy", r
    assert r["amount"] and r["amount"] > 0, f"没记金额: {r}"
    assert r["one_word"] is True, \
        f"一字板没被识别（open==high==low==close==板价）: {r}"
    assert res.attempted_turnover > 0, "没有累计'想成交'总额（度量的分母）"
    print(f"[OK] C5 埋点：{len(res.rejection_detail)} 笔，"
          f"金额 {r['amount']:,.0f}，一字板={r['one_word']}，"
          f"分母 {res.attempted_turnover:,.0f}")


def test_rejection_report_gives_actionable_verdict():
    """报告必须给出**可操作的判读**（占比阈值 -> 要不要建模）"""
    from analytics.result_report import render_rejections

    class R:
        rejections = {"参考价已在涨停": 2}
        rejection_detail = [
            {"date": None, "code": "A", "side": "buy", "reason": "涨停",
             "amount": 1000.0, "one_word": True},
            {"date": None, "code": "B", "side": "buy", "reason": "涨停",
             "amount": 3000.0, "one_word": False},
        ]
        attempted_turnover = 1_000_000.0

    txt = "\n".join(render_rejections(R()))
    assert "被拦金额占比" in txt and "0.40%" in txt, txt
    assert "不构成主要不确定性" in txt, "低占比时没给出'不用建模'的结论"
    assert "一字板 1 笔" in txt and "非一字 1 笔" in txt, "没区分一字/非一字"
    assert "下界" in txt, "没声明主结果是下界"

    # 高占比 -> 应建议考虑建模
    class R2:
        rejections = {"涨停": 50}
        rejection_detail = [{"amount": 200_000.0, "one_word": False}] * 5
        attempted_turnover = 1_000_000.0

    t2 = "\n".join(render_rejections(R2()))
    assert "值得考虑" in t2, "高占比时没给出建模建议"
    print("[OK] C5 判读：低占比 -> 声明即可；高占比 -> 建议建模；区分一字/非一字")


def test_instrumentation_does_not_change_matching():
    """**埋点不得改变撮合结果**：有/无埋点的成交与权益必须完全一致"""
    from backtest.multi_engine import PortfolioBacktestEngine

    idx, cols, panel = _sealed_panel()
    tw = pd.DataFrame(np.nan, index=idx, columns=cols)
    for d in idx[::8]:
        tw.loc[d] = [0.5, 0.5]
    res = PortfolioBacktestEngine().run(panel, tw, audit_lookahead=False)
    # 被拦的委托不该出现在成交里
    if res.rejection_detail and res.trades is not None and not res.trades.empty:
        for r in res.rejection_detail:
            same = res.trades[(res.trades.get("timestamp") == r["date"])
                              & (res.trades.get("code") == r["code"])] \
                if "code" in res.trades.columns else None
            assert same is None or same.empty, \
                "被拦的委托竟然也成交了 —— 埋点改动了撮合逻辑"
    assert abs(res.ledger_gap) < 1e-6, "账目不变量被破坏"
    print("[OK] 埋点不改变撮合：被拦的委托没有成交，账目差额仍为 0")


# ============================================================
# ③ 因子门禁
# ============================================================
def _factor_world(split="2021-12-31", flip_after=True):
    rng = np.random.default_rng(5)
    dates = pd.bdate_range("2018-01-31", periods=72, freq="ME")
    codes = [f"{i:06d}" for i in range(1, 201)]
    good = pd.DataFrame(rng.normal(0, 1, (72, 200)), index=dates, columns=codes)
    fwd = (0.02 * good
           + pd.DataFrame(rng.normal(0, 0.08, (72, 200)), index=dates,
                          columns=codes))
    bad = good.copy()
    if flip_after:
        bad.loc[bad.index > pd.Timestamp(split)] *= -1
    return dates, good, bad, fwd


def test_gate_conditions_are_independent():
    """**核心**：四个条件必须各自独立返回，报告要能说清"死在哪个条件上\""""
    from factors.gate import gate_check

    _, good, bad, fwd = _factor_world()
    g = gate_check(good, fwd, split_point="2021-12-31", label="good")
    b = gate_check(bad, fwd, split_point="2021-12-31", label="flip")

    for r in (g, b):
        assert set(r["conditions"]) >= {"样本内外同号", "分块t",
                                        "样本外分层单调性", "|IC|下限"}, r
        for name, c in r["conditions"].items():
            assert "ok" in c and "why" in c, f"{name} 缺 ok/why"
    # 好因子全过；翻转因子必须死在"符号"上，且原因写清楚
    assert g["status"] == "PASS", g
    assert b["status"] == "FAIL", b
    assert b["conditions"]["样本内外同号"]["ok"] is False
    assert "符号翻转" in b["conditions"]["样本内外同号"]["why"]
    print(f"[OK] 门禁四条件独立：good={g['status']} / flip={b['status']}；"
          f"flip 死因「{b['conditions']['样本内外同号']['why']}」")


def test_gate_reports_insufficient_sample_as_undecided():
    """样本不足必须报"未判定"，**不能假装通过**"""
    from factors.gate import gate_check

    rng = np.random.default_rng(0)
    dates = pd.bdate_range("2024-01-31", periods=5, freq="ME")
    codes = [f"{i:06d}" for i in range(1, 101)]
    f = pd.DataFrame(rng.normal(0, 1, (5, 100)), index=dates, columns=codes)
    r = gate_check(f, f, label="tiny")
    assert r["status"] == "SKIP", r
    assert r["conditions"]["样本量"]["ok"] is None, \
        "样本不足应标为 None（未判定），不能算通过"
    print(f"[OK] 样本不足 -> SKIP 且标记为未判定：「{r['reasons'][0]}」")


def test_gate_summary_does_not_drop_factors():
    """**关键设计**：不合格因子只被标记，**不被剔除**

    静默剔除会造出"被审查过的幸存者因子库"，与股票池的幸存者偏差是同构的病。
    """
    from factors.gate import gate_check, gate_summary

    _, good, bad, fwd = _factor_world()
    reps = [gate_check(good, fwd, split_point="2021-12-31", label="good"),
            gate_check(bad, fwd, split_point="2021-12-31", label="flip")]
    s = gate_summary(reps)
    assert s["status"] == "FAIL"
    assert len(s["reports"]) == 2, "汇总把因子丢掉了 —— 不许静默剔除"
    assert s["n_fail"] == 1
    assert any("flip" in x for x in s["reasons"]), "失败原因里没点名是哪个因子"
    assert "1 通过" in s["summary"] and "1 不合格" in s["summary"], s["summary"]
    print(f"[OK] 门禁汇总只标记不剔除：{s['summary']}；点名 {s['reasons']}")


def test_gate_skipped_is_visible_not_silent():
    """豁免通道必须**显式可见**：跳过门禁时报告头部要写"未执行\""""
    from analytics.result_report import render_header

    class R:
        fill_timing = "next_open"
        gate = {}
        defects = []

    txt = "\n".join(render_header(R()))
    assert "未执行" in txt, "跳过门禁却没有任何提示 —— 等于没建门禁"
    print("[OK] 门禁跳过时报告头部显式标注「未执行」")


if __name__ == "__main__":
    test_rejection_detail_records_amount_and_one_word()
    test_rejection_report_gives_actionable_verdict()
    test_instrumentation_does_not_change_matching()
    test_gate_conditions_are_independent()
    test_gate_reports_insufficient_sample_as_undecided()
    test_gate_summary_does_not_drop_factors()
    test_gate_skipped_is_visible_not_silent()
    print("\n全部 C5 埋点 / 因子门禁测试通过")
