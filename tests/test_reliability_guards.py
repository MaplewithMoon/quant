# -*- coding: utf-8 -*-
"""回测可靠性护栏测试：前视自检 + 组合层风控

背景（为什么这些必须测）
------------------------
本项目有两条回测路径：

    backtest/engine.py      单标的  —— 三道护栏齐全（冲击成本/风控/前视自检）
    backtest/multi_engine.py 组合   —— 三道护栏**一道都没有**

而所有真实策略（多因子、板块轮动、聚宽移植）走的都是组合路径。这不是三个
独立的小疏漏，是同一个结构病。这个文件盯住其中两道：**前视自检**与**风控**。

前视（look-ahead）是回测里破坏性最大的错误类别：它不会报错，只会让收益曲线
变好看。而组合路径尤其容易踩 —— 决策日和成交日共用同一个日期索引，
"用当天收盘算权重、又按当天收盘成交"只需要写错一行。
"""
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))


# ============================================================
# 一、成交时点前视自检（引擎能自己保证的那部分 PIT）
# ============================================================
def test_fill_timing_detects_same_day_fill_under_next_open():
    """**核心回归**：next_open 模式下，成交不得发生在决策日当天

    这就是"用当日收盘算权重、按当日价格成交"的前视。引擎默认 next_open
    （T 日决策、T+1 开盘成交），一旦成交日等于决策日，说明这条保证被破坏了，
    回测结果不再是可信的。
    """
    from backtest.lookahead import check_fill_timing

    reb = [pd.Timestamp("2024-01-02"), pd.Timestamp("2024-01-15")]
    # ① 正常：决策日之后的交易日成交 -> 通过
    good = pd.DataFrame({"timestamp": [pd.Timestamp("2024-01-03"),
                                       pd.Timestamp("2024-01-16")]})
    r = check_fill_timing(reb, good, fill_timing="next_open")
    assert not r["violations"], f"正常的 T+1 成交被误报违规: {r['violations']}"

    # ② 前视：成交日 == 决策日 -> 必须报违规
    bad = pd.DataFrame({"timestamp": [pd.Timestamp("2024-01-02"),
                                      pd.Timestamp("2024-01-16")]})
    r = check_fill_timing(reb, bad, fill_timing="next_open")
    assert len(r["violations"]) == 1, \
        f"决策日当天成交（前视）未被检出: {r['violations']}"

    # ③ 成交早于任何决策 -> 也必须报
    early = pd.DataFrame({"timestamp": [pd.Timestamp("2024-01-01")]})
    r = check_fill_timing(reb, early, fill_timing="next_open")
    assert len(r["violations"]) == 1, "成交早于首次决策未被检出"
    print("[OK] 成交时点自检：next_open 下同日成交/早于决策 都能检出")


def test_fill_timing_flags_same_close_as_declared_mild_lookahead():
    """same_close 不是错误，但**必须被显式声明**为温和前视

    用当日收盘价决策、又按当日收盘价成交 —— 收盘价要到收盘后才知道，
    所以这是温和前视。它不算 bug（很多简化回测都这么做），但读者有权知道。
    """
    from backtest.lookahead import check_fill_timing

    reb = [pd.Timestamp("2024-01-02")]
    t = pd.DataFrame({"timestamp": [pd.Timestamp("2024-01-02")]})
    r = check_fill_timing(reb, t, fill_timing="same_close")
    assert r["same_day_fills"] == 1, "同日成交笔数没统计出来"
    assert not r["violations"], "same_close 不应报成违规"
    assert "温和前视" in r["report"], f"没有声明温和前视: {r['report']}"
    # next_open 下同样的数据就是违规
    r2 = check_fill_timing(reb, t, fill_timing="next_open")
    assert r2["violations"], "next_open 下同日成交必须报违规"
    print("[OK] same_close 被显式声明为温和前视（而不是静默放过）")


def test_check_lookahead_rejects_full_panel():
    """**回归**：把全量面板当 used_data 传时必须**报错**，而不是产出垃圾结论

    `check_lookahead` 对每个决策日在整个 DataFrame 里找 `_available > 决策日`
    的行。传全量面板时，除最后一个决策日外**每一天**都会命中，于是报出成千上万
    条假违规 —— 检查彻底失效。宁可报错，也不要静默给出垃圾。
    """
    from backtest.lookahead import assert_no_lookahead

    panel = pd.DataFrame({
        "code": ["A"] * 100,
        "trade_date": pd.bdate_range("2024-01-01", periods=100),
    })
    panel["_available"] = panel["trade_date"]          # 全量面板
    reb = [pd.Timestamp("2024-01-02"), pd.Timestamp("2024-02-01")]
    try:
        assert_no_lookahead(reb, {"close": panel})
        raise AssertionError("传全量面板竟然没报错 —— 会产出成千上万条假违规")
    except ValueError as e:
        assert "全量面板" in str(e), f"报错信息没说清原因: {e}"

    # 逐决策切片（正确用法）不应报错
    slice_ = panel[panel["trade_date"] <= pd.Timestamp("2024-01-02")]
    n = assert_no_lookahead(reb[:1], {"close": slice_})
    assert n == 0, "正确的逐决策切片被误报"
    print("[OK] check_lookahead：全量面板被拒绝，逐决策切片正常通过")


# ============================================================
# 二、组合层风控
# ============================================================
def _ctx(dd=-0.0, prev=None, total=1_000_000.0):
    from risk import RiskContext
    return RiskContext(date=pd.Timestamp("2024-06-03"), total_value=total,
                       peak_value=total / (1 + dd) if dd else total,
                       drawdown_pct=dd, cash=0.0, prev_weights=prev)


def test_derisk_scales_down_not_liquidate():
    """回撤降仓：按比例压低总仓位，且**不清仓**（避免阈值附近来回抖动）"""
    from risk import MaxDrawdownDeRisk

    rule = MaxDrawdownDeRisk(max_drawdown_pct=0.20, start_pct=0.10,
                            min_exposure=0.2)
    w = pd.Series({"A": 0.5, "B": 0.5})

    # 未到起点：不动
    out, why = rule.adjust(w, _ctx(-0.05))
    assert why is None and abs(out.sum() - 1.0) < 1e-12, "未触发却改了仓位"

    # 刚到起点：几乎不动（连续过渡）
    out, why = rule.adjust(w, _ctx(-0.1001))
    assert why is not None and out.sum() > 0.95, \
        f"在起点处应该几乎不动，实得 {out.sum():.3f}"

    # 到达下限：压到 min_exposure
    out, why = rule.adjust(w, _ctx(-0.25))
    assert abs(out.sum() - 0.2) < 1e-9, f"应压到 20%，实得 {out.sum():.1%}"

    # 中间：线性，且随回撤加深单调下降
    a = rule.adjust(w, _ctx(-0.12))[0].sum()
    b = rule.adjust(w, _ctx(-0.16))[0].sum()
    assert a > b, f"回撤更深仓位却没更低: {a:.3f} vs {b:.3f}"
    print(f"[OK] 回撤降仓：未触发不动 / 起点连续 / 下限 20% / 中间单调"
          f"（-12%->{a:.0%}, -16%->{b:.0%}）")


def test_max_weight_caps_and_does_not_redistribute():
    """单票上限：削掉超出部分，**不做再分配**（再分配会越控越集中）"""
    from risk import MaxWeightRule

    rule = MaxWeightRule(max_weight=0.10)
    w = pd.Series({"A": 0.30, "B": 0.30, "C": 0.40})
    out, why = rule.adjust(w, _ctx())
    assert why is not None and "2 只" in why or "3 只" in why, why
    assert out.max() <= 0.10 + 1e-12, f"仍有超限权重 {out.max()}"
    assert abs(out.sum() - 0.30) < 1e-12, \
        f"削掉的部分被再分配了（应全削不补），实得合计 {out.sum():.3f}"

    # 未超限：不动
    ok = pd.Series({"A": 0.05, "B": 0.05})
    out2, why2 = rule.adjust(ok, _ctx())
    assert why2 is None
    print("[OK] 单票上限：超出部分削掉且不再分配（总仓位变小 = 留现金，方向保守）")


def test_max_turnover_blends_toward_current_holdings():
    """换手上限：超过上限时向现有持仓靠拢，不超过上限时原样返回"""
    from risk import MaxTurnoverRule

    rule = MaxTurnoverRule(max_turnover=0.5)
    prev = pd.Series({"A": 1.0, "B": 0.0})
    target = pd.Series({"A": 0.0, "B": 1.0})       # 全换手 = 2.0

    out, why = rule.adjust(target, _ctx(prev=prev))
    assert why is not None and "换手" in why, why
    turn = float((out - prev).abs().sum())
    assert turn <= 0.5 + 1e-9, f"换手仍超上限: {turn:.3f}"
    assert 0 < out["B"] < 1.0, "应该是一个混合仓位"

    # 没有上期权重 -> 不触发（宁可不动，也不凭空假设持仓）
    out2, why2 = rule.adjust(target, _ctx(prev=None))
    assert why2 is None, "无 prev_weights 时不应触发换手规则"
    print("[OK] 换手上限：超额时向现有持仓靠拢；无上期权重时不触发")


def test_risk_manager_only_reduces_gross():
    """**方向约束**：任何规则组合都不得把总仓位放大（风控不能变杠杆）"""
    from risk import (MaxDrawdownDeRisk, MaxWeightRule, PortfolioRiskManager)

    # 构造一条会"放大"的假规则，验证兜底 clip 生效。
    # 注意倍数要够大：MaxWeightRule 会先把合计削到 0.2，×5 只有 1.0，不算越界。
    class EvilRule:
        name = "evil"

        def adjust(self, target, ctx):
            return target * 20.0, "恶意放大 20 倍"

    rm = PortfolioRiskManager([MaxWeightRule(0.10), EvilRule()], max_gross=1.0)
    out, events = rm.adjust(pd.Series({"A": 0.5, "B": 0.5}))
    assert out.sum() <= 1.0 + 1e-9, f"总仓位被放大到 {out.sum():.2f}，兜底失效"
    assert any(e["rule"] == "max_gross" for e in events), \
        "总仓位超额没有被记录（事后看不出是风控砍的）"

    # 正常的降仓规则组合
    rm2 = PortfolioRiskManager([MaxDrawdownDeRisk(0.20, 0.10, 0.0),
                                MaxWeightRule(0.10)])
    out2, ev2 = rm2.adjust(pd.Series({"A": 0.5, "B": 0.5}))
    assert out2.sum() <= 1.0 + 1e-9
    assert len(ev2) >= 1, "风控该有触发记录"
    print(f"[OK] 风控方向约束：总仓位不会被放大；触发记录 {len(ev2)} 条（可归因）")


def test_build_risk_manager_from_config():
    """配置化构建：全空 -> None（不改变原行为）；有值 -> 对应规则"""
    from risk import build_risk_manager

    assert build_risk_manager(None) is None
    assert build_risk_manager({}) is None
    assert build_risk_manager({"max_weight": 0}) is None
    rm = build_risk_manager({"max_drawdown_pct": 0.2, "max_weight": 0.1,
                             "max_turnover": 0.6})
    names = [r.name for r in rm.rules]
    assert names == ["max_drawdown_derisk", "max_weight", "max_turnover"], names
    print(f"[OK] 配置化构建：空配置 -> None；有值 -> {names}")


# ============================================================
# 三、引擎端到端：护栏真的接上了
# ============================================================
def _panel(n_days=60, n_codes=4, seed=0):
    rng = np.random.default_rng(seed)
    idx = pd.bdate_range("2024-01-01", periods=n_days)
    cols = [f"{i:06d}" for i in range(1, n_codes + 1)]
    close = pd.DataFrame(
        10 * np.exp(np.cumsum(rng.normal(0, 0.01, (n_days, n_codes)), axis=0)),
        index=idx, columns=cols)
    return {"close": close, "open": close.shift(1).fillna(close.iloc[0])}


def test_engine_reports_lookahead_audit_and_no_violation_by_default():
    """引擎默认做前视自检；next_open 的正常路径应当 0 违规"""
    from backtest.multi_engine import PortfolioBacktestEngine

    panel = _panel()
    idx = panel["close"].index
    # 每 10 天调一次的等权权重
    tw = pd.DataFrame(np.nan, index=idx, columns=panel["close"].columns)
    for d in idx[::10]:
        tw.loc[d] = 1.0 / panel["close"].shape[1]
    eng = PortfolioBacktestEngine(initial_capital=1_000_000)
    res = eng.run(panel, tw)
    assert res.lookahead_report, "引擎没有产出前视自检报告（护栏没接上）"
    assert res.lookahead_violations == 0, \
        f"正常 next_open 路径报了 {res.lookahead_violations} 笔违规: {res.lookahead_report}"
    assert "成交时点校验通过" in res.lookahead_report
    print("[OK] 引擎端到端：默认产出前视自检报告，正常路径 0 违规")


def test_engine_accepts_portfolio_risk_manager():
    """引擎真的接了组合风控：给它一个激进上限，结果应被压下来"""
    from backtest.multi_engine import PortfolioBacktestEngine
    from risk import MaxWeightRule, PortfolioRiskManager

    panel = _panel()
    idx = panel["close"].index
    tw = pd.DataFrame(np.nan, index=idx, columns=panel["close"].columns)
    for d in idx[::10]:
        tw.loc[d] = 1.0 / panel["close"].shape[1]      # 每只 25%

    eng = PortfolioBacktestEngine(initial_capital=1_000_000)
    res_off = eng.run(panel, tw)
    rm = PortfolioRiskManager([MaxWeightRule(0.10)])
    res_on = eng.run(panel, tw, risk_manager=rm)

    assert res_on.risk_events, "风控没有任何触发记录 —— 等于没接上"
    # 单票上限 10%，四只股票最多用 40% 仓位 -> 收益波动应明显更小
    v_off = res_off.equity.pct_change().std()
    v_on = res_on.equity.pct_change().std()
    assert v_on < v_off, f"风控生效后波动没有下降: {v_on:.5f} vs {v_off:.5f}"
    assert res_on.risk_events[0]["rule"] == "max_weight"
    print(f"[OK] 引擎端到端：组合风控生效（波动 {v_off:.4f} -> {v_on:.4f}，"
          f"触发 {len(res_on.risk_events)} 次）")


def test_engine_audit_can_be_disabled():
    """audit_lookahead=False 时不产出报告（保持可关闭，便于对照）"""
    from backtest.multi_engine import PortfolioBacktestEngine

    panel = _panel()
    idx = panel["close"].index
    tw = pd.DataFrame(np.nan, index=idx, columns=panel["close"].columns)
    tw.loc[idx[0]] = 1.0 / panel["close"].shape[1]
    res = PortfolioBacktestEngine().run(panel, tw, audit_lookahead=False)
    assert res.lookahead_report == ""
    assert res.lookahead_violations == 0
    print("[OK] 前视自检可关闭（audit_lookahead=False）")


if __name__ == "__main__":
    test_fill_timing_detects_same_day_fill_under_next_open()
    test_fill_timing_flags_same_close_as_declared_mild_lookahead()
    test_check_lookahead_rejects_full_panel()
    test_derisk_scales_down_not_liquidate()
    test_max_weight_caps_and_does_not_redistribute()
    test_max_turnover_blends_toward_current_holdings()
    test_risk_manager_only_reduces_gross()
    test_build_risk_manager_from_config()
    test_engine_reports_lookahead_audit_and_no_violation_by_default()
    test_engine_accepts_portfolio_risk_manager()
    test_engine_audit_can_be_disabled()
    print("\n全部回测可靠性护栏测试通过")
