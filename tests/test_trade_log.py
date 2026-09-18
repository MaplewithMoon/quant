# -*- coding: utf-8 -*-
"""交易记录 / 收益分析导出测试（对齐聚宽平台的两份导出文本）

盯住四件事：
  1. **两种成交表都能归一**：组合引擎用 `fee`、聚宽兼容层用 `fees`，
     还允许把聚宽导出的中文列回灌做对照 —— 口径不统一就没法并排 diff
  2. **成交回合是 FIFO 且能拆加减仓**（聚宽导出的「平仓盈亏」是按均价算的，
     逐笔复核要 FIFO）
  3. **对账必须能抓出硬错误**：现金变负、卖超、非整手、价格非法
  4. **文本格式与聚宽同构**：列名与顺序一致，且"委托失败明细"是**独立一份**
     （混进成交里会让成交笔数/成交额统计变味）
"""
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from analytics.trade_log import (RECORD_COLS, audit_line, audit_trades,  # noqa: E402
                                 export_trade_log, normalize_trades,
                                 perf_analysis_text, reconcile_equity,
                                 rejection_text, round_trips,
                                 trade_records_text)


def _engine_trades():
    """组合引擎口径：索引 timestamp，列 fee"""
    idx = pd.to_datetime(["2024-01-05", "2024-01-12", "2024-01-20"])
    return pd.DataFrame({
        "code": ["600000", "600000", "000001"],
        "action": ["buy", "sell", "buy"],
        "size": [1000.0, 1000.0, 500.0],
        "price": [10.0, 12.0, 20.0],
        "fee": [5.0, 11.0, 5.0],
        "pnl": [0.0, 1994.0, 0.0],
    }, index=idx)


def test_normalize_accepts_both_engines():
    a = normalize_trades(_engine_trades())
    assert list(a.columns)[:4] == ["date", "time", "code", "name"]
    assert len(a) == 3
    assert a["action"].tolist() == ["buy", "sell", "buy"]
    # 金额：买正卖负（聚宽口径）
    assert abs(a.loc[0, "amount"] - 10_000.0) < 1e-6
    assert abs(a.loc[1, "amount"] + 12_000.0) < 1e-6

    # 聚宽兼容层口径：列名是 fees
    jq = pd.DataFrame({"timestamp": pd.to_datetime(["2024-02-05"]),
                       "code": ["002342"], "action": ["buy"], "size": [100.0],
                       "price": [3.5], "fees": [5.0], "pnl": [0.0]})
    b = normalize_trades(jq)
    assert abs(b.loc[0, "fee"] - 5.0) < 1e-9, "fees 没映射到 fee"

    # 聚宽导出的中文列回灌
    zh = pd.DataFrame({"日期": ["2024-01-03"], "委托时间": ["10:00:00"],
                       "标的": ["银华日利(511880.XSHG)"], "交易类型": ["买"],
                       "下单类型": ["市价单"], "成交数量": ["900股"],
                       "成交价": ["100.115"], "成交额": ["90,103.50"],
                       "平仓盈亏": ["0.00"], "手续费": ["0.00"]})
    c = normalize_trades(zh)
    assert c.loc[0, "code"] == "511880.XSHG"
    assert c.loc[0, "name"] == "银华日利"
    assert abs(c.loc[0, "shares"] - 900) < 1e-9
    assert abs(c.loc[0, "amount"] - 90103.50) < 1e-6
    print("[OK] 归一化：组合引擎(fee) / 聚宽兼容层(fees) / 聚宽导出中文列 三种口径")


def test_round_trips_fifo_splits_partial():
    t = normalize_trades(pd.DataFrame({
        "timestamp": pd.to_datetime(["2024-01-02", "2024-01-03", "2024-01-10"]),
        "code": ["600000"] * 3, "action": ["buy", "buy", "sell"],
        "size": [100.0, 100.0, 150.0], "price": [10.0, 12.0, 15.0],
        "fee": [0.0, 0.0, 0.0], "pnl": [0.0, 0.0, 0.0]}))
    rt = round_trips(t)
    assert len(rt) == 2, f"FIFO 应拆成 2 个回合，实得 {len(rt)}"
    assert abs(rt.iloc[0]["毛利"] - 500.0) < 1e-9        # (15-10)*100
    assert abs(rt.iloc[1]["毛利"] - 150.0) < 1e-9        # (15-12)*50
    assert rt.iloc[0]["持仓天数"] == 8
    print("[OK] 成交回合：FIFO 配对，部分平仓会拆成多个回合")


def test_audit_catches_hard_errors():
    good = normalize_trades(_engine_trades())
    a = audit_trades(good, initial_cash=100_000.0)
    assert a["ok"], f"正常成交不该报错：{a['issues']}"
    assert a["final_positions"] == {"000001": 500.0}

    # 卖超（成交表不自洽）
    bad = normalize_trades(pd.DataFrame({
        "timestamp": pd.to_datetime(["2024-01-02"]), "code": ["600000"],
        "action": ["sell"], "size": [-100.0], "price": [10.0],
        "fee": [0.0], "pnl": [0.0]}))
    a2 = audit_trades(bad)
    assert not a2["ok"] and any("持仓为负" in x for x in a2["issues"])

    # 现金变负
    a3 = audit_trades(good, initial_cash=100.0)
    assert not a3["ok"] and any("现金为负" in x for x in a3["issues"])

    # 非整手
    odd = normalize_trades(pd.DataFrame({
        "timestamp": pd.to_datetime(["2024-01-02"]), "code": ["600000"],
        "action": ["buy"], "size": [150.0], "price": [10.0],
        "fee": [0.0], "pnl": [0.0]}))
    a4 = audit_trades(odd)
    assert not a4["ok"] and any("非整手" in x for x in a4["issues"])
    assert "✗" in audit_line(a4) and "✓" in audit_line(a)
    print("[OK] 对账：现金为负 / 卖超 / 非整手 都能抓出来，并通过/不通过在行内声明")


def test_text_is_layout_compatible_with_joinquant():
    t = normalize_trades(_engine_trades(), fill_time="09:30")
    txt = trade_records_text(t)
    head = txt.splitlines()[0].split("\t")
    assert head == RECORD_COLS, f"列名/顺序与聚宽不一致：{head}"
    row = txt.splitlines()[1].split("\t")
    assert row[0] == "2024-01-05" and row[3] == "买" and row[5] == "1000股"
    assert row[4] == "市价单"

    eq = pd.Series([100_000, 101_000, 99_000, 103_000],
                   index=pd.bdate_range("2024-01-02", periods=4))
    bench = pd.Series([3000, 3010, 2990, 3020], index=eq.index)
    per = perf_analysis_text(eq, bench, t, label="测试策略")
    for k in ("策略收益", "策略年化收益", "夏普比率", "最大回撤", "基准收益",
              "超额收益", "阿尔法", "贝塔", "信息比率", "日均超额收益",
              "超额收益最大回撤", "超额收益夏普比率", "日胜率", "盈利次数",
              "胜率", "盈亏比"):
        assert k in per, f"收益分析缺指标：{k}"
    print("[OK] 交易记录列名与聚宽同构；收益分析含聚宽的 16 项指标名")


def test_rejection_detail_is_separate_from_fills():
    """聚宽把未成交委托混在成交记录里；我们单独一份，避免污染成交统计"""
    rejects = [{"date": pd.Timestamp("2024-03-01"), "code": "600000",
                "side": "buy", "qty": 1000, "px": 10.0,
                "reason": "涨停封板无法买入"}]
    txt = rejection_text(rejects)
    assert "涨停封板无法买入" in txt and "600000" in txt
    assert rejection_text([]).strip().endswith("价格\t原因") or "原因" in rejection_text([])
    # 成交记录里不得出现拒单
    fills = trade_records_text(normalize_trades(_engine_trades()))
    assert "涨停" not in fills
    print("[OK] 委托失败明细独立成表（聚宽混在成交里，我们分开）")


def test_reconcile_flags_off_calendar_fills():
    eq = pd.Series([100.0, 101.0], index=pd.to_datetime(["2024-01-02", "2024-01-03"]))
    t = normalize_trades(pd.DataFrame({
        "timestamp": pd.to_datetime(["2024-01-06"]),   # 周六，不在权益曲线上
        "code": ["600000"], "action": ["buy"], "size": [100.0],
        "price": [10.0], "fee": [0.0], "pnl": [0.0]}))
    r = reconcile_equity(eq, t)
    assert r["trades_off_calendar"] == ["2024-01-06"], r
    print("[OK] 对账：成交日不在权益曲线交易日上会被点名")


def test_export_writes_all_five_files():
    t = _engine_trades()
    eq = pd.Series(np.linspace(100_000, 108_000, 40),
                   index=pd.bdate_range("2024-01-02", periods=40))
    from tests.test_data_provenance import _tmpdir
    out = _tmpdir()
    r = export_trade_log(t, eq, None, outdir=str(out), tag="_t",
                         rejects=[{"date": "2024-01-06", "code": "600000",
                                   "side": "buy", "qty": 100, "px": 10.0,
                                   "reason": "涨停封板无法买入"}],
                         initial_cash=100_000.0, label="单元测试")
    for k, p in r["files"].items():
        assert os.path.isfile(p) and os.path.getsize(p) > 0, f"{k} 没写出来：{p}"
    assert "委托失败明细" in os.path.basename(r["files"]["委托失败明细"])
    print(f"[OK] 一键导出 5 个文件（交易记录/委托失败明细/收益分析/2 个 CSV）-> {out}")


if __name__ == "__main__":
    test_normalize_accepts_both_engines()
    test_round_trips_fifo_splits_partial()
    test_audit_catches_hard_errors()
    test_text_is_layout_compatible_with_joinquant()
    test_rejection_detail_is_separate_from_fills()
    test_reconcile_flags_off_calendar_fills()
    test_export_writes_all_five_files()
    print("\n全部交易记录 / 收益分析导出测试通过")
