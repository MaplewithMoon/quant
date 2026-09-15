# -*- coding: utf-8 -*-
"""基本面因子测试：PIT 对齐（不许提前泄露）、单季还原、同比、中性化、选因子规则"""
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from factors.fundamental import (FACTOR_META, FundamentalData, _safe_div,
                                 _single_quarter, combine_by_rule,
                                 neutralized_score, valuation_factors)


# ============================================================
# 单季还原
# ============================================================
def test_single_quarter_restores_quarterly():
    """A 股利润表是年内累计：Q2 报的是 H1、Q3 报的是 9 个月、Q4 报的是全年"""
    end = pd.to_datetime(["2023-03-31", "2023-06-30", "2023-09-30", "2023-12-31",
                          "2024-03-31", "2024-06-30"])
    ytd = pd.Series([10.0, 25.0, 45.0, 70.0, 12.0, 30.0])
    code = pd.Series(["A"] * 6)
    q = _single_quarter(ytd, pd.Series(end), code)
    expect = [10.0, 15.0, 20.0, 25.0, 12.0, 18.0]
    assert np.allclose(q.values, expect), f"单季还原错误: {q.tolist()} != {expect}"
    print("[OK] 单季还原：累计口径 -> 单季（跨年正确重置，Q1 不被减）")


def test_single_quarter_handles_multiple_codes():
    end = pd.to_datetime(["2023-03-31", "2023-06-30", "2023-03-31", "2023-06-30"])
    ytd = pd.Series([10.0, 25.0, 100.0, 260.0])
    code = pd.Series(["A", "A", "B", "B"])
    q = _single_quarter(ytd, pd.Series(end), code)
    assert np.allclose(q.values, [10.0, 15.0, 100.0, 160.0]), q.tolist()
    print("[OK] 单季还原不会跨股票串行（按 code+年份分组）")


# ============================================================
# 安全除法
# ============================================================
def test_safe_div_combinations():
    """分母 0 / NaN 必须给 NaN 而不是 inf；标量不能被当成 Series"""
    assert _safe_div(1.0, pd.Series([2.0, 0.0, np.nan])).tolist()[:1] == [0.5]
    assert np.isnan(_safe_div(1.0, pd.Series([2.0, 0.0, np.nan])).iloc[1])
    assert _safe_div(pd.Series([2.0, 0.0]), 4.0).tolist() == [0.5, 0.0]
    assert np.isnan(_safe_div(1.0, 0.0))
    assert _safe_div(1.0, 2.0) == 0.5
    df = _safe_div(1.0, pd.DataFrame({"a": [10.0, 0.0]}))
    assert abs(df.iloc[0, 0] - 0.1) < 1e-12 and np.isnan(df.iloc[1, 0])
    assert not np.isinf(_safe_div(pd.Series([1.0]), pd.Series([0.0])).iloc[0])
    print("[OK] 安全除法：Series/DataFrame/标量任意组合，0 与 NaN 都退化为 NaN")


# ============================================================
# PIT 对齐（核心）
# ============================================================
def _fake_fd(recs):
    fd = FundamentalData()
    fd.reports = pd.DataFrame(recs)
    return fd


def test_pit_does_not_leak_before_announcement():
    """**最关键的一条**：公告日之前绝对不能看到该期财报

    如果按 end_date 对齐（3/31）而不是 ann_date（4/28），
    就等于 3 月 31 日已经知道 4 月 28 日才公布的数据 ——
    这是最典型的前视偏差，会把因子 IC 抬得虚高。
    """
    fd = _fake_fd([
        {"code": "A", "ann_date": pd.Timestamp("2023-04-28"),
         "end_date": pd.Timestamp("2023-03-31"), "rev_yoy": 0.5},
        {"code": "A", "ann_date": pd.Timestamp("2023-08-30"),
         "end_date": pd.Timestamp("2023-06-30"), "rev_yoy": 0.9},
    ])
    dates = pd.DatetimeIndex(pd.bdate_range("2023-03-01", "2023-12-29"))
    w = fd.align("rev_yoy", dates, ["A"])

    assert w.loc[:"2023-04-27", "A"].isna().all(), \
        "公告日(4/28)之前不该有任何值 —— 出现了前视"
    assert pd.isna(w.loc["2023-03-31", "A"]), "报告期当天也不能有值"
    # 生效日 = 公告日 + lag(1 天)
    assert abs(float(w.loc["2023-05-02", "A"]) - 0.5) < 1e-12
    assert abs(float(w.loc["2023-09-01", "A"]) - 0.9) < 1e-12
    assert abs(float(w.loc["2023-12-29", "A"]) - 0.9) < 1e-12, "应一直前向保持"
    print("[OK] PIT 对齐：公告日前无值、公告后 lag 天生效、之后前向保持")


def test_pit_dedupe_keeps_first_announcement():
    """同一 (code, end_date) 多次公告时，只认最早那次（as-reported）"""
    fd = _fake_fd([
        {"code": "A", "ann_date": pd.Timestamp("2023-04-28"),
         "end_date": pd.Timestamp("2023-03-31"), "rev_yoy": 0.1},
        {"code": "A", "ann_date": pd.Timestamp("2023-06-15"),
         "end_date": pd.Timestamp("2023-03-31"), "rev_yoy": 0.7},   # 事后修正
    ])
    dates = pd.DatetimeIndex(pd.bdate_range("2023-05-01", "2023-07-31"))
    w = fd.align("rev_yoy", dates, ["A"])
    assert abs(float(w.loc["2023-05-02", "A"]) - 0.1) < 1e-12, \
        "5 月时应该只知道首次公告的 0.1，不能提前用到 6 月才修正的 0.7"
    assert abs(float(w.loc["2023-07-31", "A"]) - 0.1) < 1e-12
    print("[OK] 重复公告只认最早一次（as-reported），不会用到事后的修正值")


def test_pit_align_covers_all_codes():
    """对齐结果必须是指定日期 × 指定代码的完整宽表（缺的填 NaN）"""
    fd = _fake_fd([{"code": "A", "ann_date": pd.Timestamp("2023-04-28"),
                    "end_date": pd.Timestamp("2023-03-31"), "rev_yoy": 0.1}])
    dates = pd.DatetimeIndex(pd.bdate_range("2023-05-01", "2023-05-31"))
    w = fd.align("rev_yoy", dates, ["A", "B", "C"])
    assert list(w.columns) == ["A", "B", "C"]
    assert len(w) == len(dates)
    assert w["B"].isna().all() and w["C"].isna().all()
    print("[OK] PIT 对齐输出完整宽表，缺失代码填 NaN（不静默丢列）")


def test_pit_missing_factor_is_safe():
    fd = _fake_fd([{"code": "A", "ann_date": pd.Timestamp("2023-04-28"),
                    "end_date": pd.Timestamp("2023-03-31"), "rev_yoy": 0.1}])
    dates = pd.DatetimeIndex(pd.bdate_range("2023-05-01", "2023-05-31"))
    w = fd.align("不存在的因子", dates, ["A"])
    assert w.isna().all().all() and len(w) == len(dates)
    print("[OK] 请求不存在的因子 -> 全 NaN 宽表，不抛异常")


# ============================================================
# 估值因子 / 中性化 / 规则
# ============================================================
def test_valuation_factors_direction():
    panel = {"pe_ttm": pd.DataFrame({"a": [10.0, 0.0]}),
             "pb": pd.DataFrame({"a": [2.0, np.nan]}),
             "turnover_rate": pd.DataFrame({"a": [1.0, 3.0]})}
    v = valuation_factors(panel, window=2)
    assert abs(v["ep_ttm"].iloc[0, 0] - 0.1) < 1e-12     # 1/10
    assert np.isnan(v["ep_ttm"].iloc[1, 0])              # 1/0 -> NaN
    assert abs(v["bp"].iloc[0, 0] - 0.5) < 1e-12         # 1/2
    assert abs(v["turnover_20"].iloc[1, 0] - 2.0) < 1e-12
    print("[OK] 估值/换手因子：倒数关系正确，PE=0 退化为 NaN")


def test_neutralized_score_has_no_direction_bias():
    """中性化后，各行业与各市值组的打分均值都应接近 0"""
    dates = pd.bdate_range("2023-01-02", periods=40)
    codes = [f"{i:06d}" for i in range(1, 61)]
    rng = np.random.default_rng(0)
    # 行业属性极强的因子：A 行业天然高一档
    ind = pd.Series(["行业1"] * 20 + ["行业2"] * 20 + ["行业3"] * 20, index=codes)
    base = pd.DataFrame(rng.normal(size=(40, 60)), index=dates, columns=codes)
    base.loc[:, codes[:20]] += 5.0                      # 行业1 系统性更高
    mv = pd.DataFrame(rng.lognormal(3, 1, (40, 60)), index=dates, columns=codes)
    factors = {"bp": base}
    mask = pd.DataFrame(True, index=dates, columns=codes)

    raw = neutralized_score(factors, ["bp"], mask, None, None)
    neu = neutralized_score(factors, ["bp"], mask, ind, mv)
    # 未中性化：行业1 的均值明显偏高
    assert raw.loc[:, codes[:20]].mean().mean() > raw.loc[:, codes[20:]].mean().mean() + 0.3
    # 中性化后：各行业均值都贴近 0（秩的总体均值约 0.5，组内去均值后约 0）
    g = neu.loc[:, codes].T.groupby(ind).mean().mean(axis=1)
    assert g.abs().max() < 0.05, f"中性化后行业均值应≈0，实得 {g.round(4).to_dict()}"
    print(f"[OK] 秩中性化：行业均值 {g.round(4).to_dict()}（未中性化时行业1 显著偏高）")


def test_neutralized_score_respects_direction():
    """方向为 -1 的因子必须取负后再合成（否则会反向选股）"""
    dates = pd.bdate_range("2023-01-02", periods=20)
    codes = [f"{i:06d}" for i in range(1, 31)]
    rng = np.random.default_rng(1)
    hi = pd.DataFrame(rng.normal(size=(20, 30)), index=dates, columns=codes)
    mask = pd.DataFrame(True, index=dates, columns=codes)
    # turnover_20 方向是 -1：值越大分应越低
    assert FACTOR_META["turnover_20"][1] == -1
    # 用"先取秩再 Pearson"代替 spearman（scipy 不是本项目依赖）
    s = neutralized_score({"turnover_20": hi}, ["turnover_20"], mask, None, None)
    c1 = s.iloc[0].rank().corr(hi.iloc[0].rank())
    assert c1 < -0.9, f"方向 -1 的因子合成后应与原值显著负相关，实得 {c1:+.3f}"
    # 方向 +1 的因子则应正相关
    s2 = neutralized_score({"bp": hi}, ["bp"], mask, None, None)
    c2 = s2.iloc[0].rank().corr(hi.iloc[0].rank())
    assert c2 > 0.9, f"方向 +1 的因子合成后应正相关，实得 {c2:+.3f}"
    print(f"[OK] 因子方向：direction=-1 秩相关 {c1:+.3f}，direction=+1 秩相关 {c2:+.3f}")


def test_combine_by_rule_only_uses_in_sample_columns():
    """选因子规则只能读样本内列；样本外列即使相反也不影响选择"""
    tab = pd.DataFrame([
        {"因子": "good", "内p": 0.01, "内多空年化": 0.10, "内单调性": 0.8,
         "外p": 0.9, "外多空年化": -0.5},      # 样本外很差，但仍应入选
        {"因子": "bad", "内p": 0.30, "内多空年化": 0.20, "内单调性": 0.9,
         "外p": 0.001, "外多空年化": 0.30},     # 样本外很好，但不该入选
    ])
    picked = combine_by_rule(tab)
    assert picked == ["good"], f"规则应只看样本内列，实得 {picked}"
    print("[OK] 预注册规则只读样本内列（样本外再好也不影响选择）")


def test_combine_by_rule_requires_columns():
    try:
        combine_by_rule(pd.DataFrame([{"因子": "x"}]))
        raise AssertionError("缺列时应报错")
    except KeyError:
        pass
    print("[OK] 规则缺列时报错，不静默返回空")


if __name__ == "__main__":
    test_single_quarter_restores_quarterly()
    test_single_quarter_handles_multiple_codes()
    test_safe_div_combinations()
    test_pit_does_not_leak_before_announcement()
    test_pit_dedupe_keeps_first_announcement()
    test_pit_align_covers_all_codes()
    test_pit_missing_factor_is_safe()
    test_valuation_factors_direction()
    test_neutralized_score_has_no_direction_bias()
    test_neutralized_score_respects_direction()
    test_combine_by_rule_only_uses_in_sample_columns()
    test_combine_by_rule_requires_columns()
    print("\n全部基本面因子测试通过")
