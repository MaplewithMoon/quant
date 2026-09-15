# -*- coding: utf-8 -*-
"""涨跌停规则测试：唯一实现、板块档位、ST、创业板制度切换、禁用 close.shift(1)"""
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from database.limit_rules import (GEM_20PCT_FROM, apply_limit_prices, base_pct,
                                  compute, is_st_on, limit_pct, st_mask)


def test_board_tiers():
    """各板块基础涨跌幅"""
    cases = [
        ("600519", 0.10),   # 主板
        ("000001", 0.10),
        ("002001", 0.10),   # 中小板并入主板
        ("300750", 0.20),   # 创业板（2024 年）
        ("688981", 0.20),   # 科创板
        ("689009", 0.20),   # 科创板 CDR
        ("920001", 0.30),   # 北交所
        ("830799", 0.30),
        ("871981", 0.30),
        ("430047", 0.30),
    ]
    for code, exp in cases:
        got = base_pct(code, "2024-06-03")
        assert abs(got - exp) < 1e-12, f"{code} 应为 {exp}，实得 {got}"
    print(f"[OK] 板块档位：主板/中小板 10%、创业板/科创板 20%、北交所 30%")


def test_gem_regime_switch():
    """创业板 2020-08-24 注册制改革：之前 ±10%，之后 ±20%"""
    before = base_pct("300750", "2020-08-21")
    after = base_pct("300750", "2020-08-24")
    assert abs(before - 0.10) < 1e-12, f"改革前应为 10%，实得 {before}"
    assert abs(after - 0.20) < 1e-12, f"改革后应为 20%，实得 {after}"
    # 边界当天按新规
    assert abs(base_pct("300750", GEM_20PCT_FROM) - 0.20) < 1e-12
    # 其它板块不受影响
    assert abs(base_pct("600519", "2010-01-01") - 0.10) < 1e-12
    print("[OK] 创业板制度切换：2020-08-24 之前 10%、之后 20%，主板不受影响")


def test_st_takes_precedence():
    """ST 优先于板块档，一律 ±5%"""
    assert abs(limit_pct("600519", None, True) - 0.05) < 1e-12
    assert abs(limit_pct("300750", None, True) - 0.05) < 1e-12, "创业板 ST 也是 5%"
    up, dn = compute(10.0, "600519", None, True)
    assert (up, dn) == (10.5, 9.5), (up, dn)
    print("[OK] ST 优先于板块档：一律 ±5%（含创业板/科创板）")


def test_compute_rounds_to_cent():
    """四舍五入到分，与交易所口径一致"""
    up, dn = compute(10.0, "600519")          # 主板 ±10%
    assert (up, dn) == (11.0, 9.0)
    up, dn = compute(3.33, "600519")
    assert (up, dn) == (3.66, 3.00), (up, dn)
    # 非法昨收 -> NaN，不抛异常
    for bad in (0.0, -1.0, np.nan, None):
        u, d = compute(bad, "600519")
        assert np.isnan(u) and np.isnan(d), f"{bad} 应给 NaN"
    print(f"[OK] 计算与取整：10.00 -> {compute(10.0, '600519')}；非法昨收退化为 NaN")


def test_rounding_is_half_up_not_bankers():
    """**回归**：必须是四舍五入，不能是 Python/numpy 默认的银行家舍入

    银行家舍入把恰好 .5 的情况往偶数方向靠：round(12.155, 2) -> 12.15，
    而交易所口径是 12.16。实测旧数据库里涨停价 2.7%、跌停价 4.4% 的行因此差 1 分。
    """
    # 10% 档：昨收末位为 5 分时必然出现整半分（pc_t*11 mod 10 == 5）
    assert compute(11.05, "600519") == (12.16, 9.95), compute(11.05, "600519")
    assert compute(10.05, "600519") == (11.06, 9.05), compute(10.05, "600519")
    # 对照：银行家舍入给出的错误答案
    assert round(12.155, 2) in (12.15, 12.16)   # 记录浮点不确定性，但不作为期望值
    # 5% 档（ST）：pc_t ≡ 10 (mod 20) 时出现整半分
    assert compute(10.10, "600519", None, True) == (10.61, 9.60), \
        compute(10.10, "600519", None, True)
    # 20% 档与 30% 档：pc_t*12 / pc_t*13 的末位是偶数/3，不产生整半分，但也要对
    assert compute(10.05, "300750") == (12.06, 8.04), compute(10.05, "300750")
    assert compute(10.10, "920001") == (13.13, 7.07), compute(10.10, "920001")
    # 向量化实现必须与标量 compute 完全一致
    df = pd.DataFrame({
        "code": ["600519"] * 5,
        "trade_date": pd.to_datetime(["2024-06-03"] * 5),
        "pre_close": [11.05, 10.05, 3.33, 7.77, 0.55],
    })
    out = apply_limit_prices(df, "600519")
    for i, pc in enumerate(df["pre_close"]):
        assert out.loc[i, "limit_up"] == compute(pc, "600519")[0], f"{pc} 向量化与标量不一致"
        assert out.loc[i, "limit_down"] == compute(pc, "600519")[1], f"{pc} 向量化与标量不一致"
    print("[OK] 四舍五入（非银行家舍入）：11.05×1.1 -> 12.16；向量化与标量一致")


def test_apply_uses_official_pre_close_not_shift():
    """**回归**：必须用官方 pre_close，不能用 close.shift(1)

    除权日两者差异巨大（10 送 10 时差一倍），
    而 daily_update.py 原先正是用 close.shift(1)，会覆盖掉正确数据。
    """
    df = pd.DataFrame({
        "code": ["600519"] * 3,
        "trade_date": pd.to_datetime(["2024-06-03", "2024-06-04", "2024-06-05"]),
        # 6/4 除权：close 从 20 掉到 10，官方 pre_close 仍是 10（已调整）
        "close": [20.0, 10.0, 10.5],
        "pre_close": [19.8, 10.0, 10.0],
    })
    out = apply_limit_prices(df, "600519")
    # 6/4 的涨停应以官方 pre_close=10.0 为基准 -> 11.0，
    # 而不是 close.shift(1)=20.0 -> 22.0
    assert out.loc[1, "limit_up"] == 11.0, \
        f"除权日应以官方 pre_close 为基准，实得 {out.loc[1, 'limit_up']}"
    assert out.loc[1, "limit_down"] == 9.0
    # 缺 pre_close 列 -> 直接报错，不退化成 shift
    try:
        apply_limit_prices(df.drop(columns=["pre_close"]), "600519")
        raise AssertionError("缺 pre_close 应报错")
    except KeyError:
        pass
    print("[OK] 用官方 pre_close（除权日不再算错）；缺 pre_close 时直接报错")


def test_apply_applies_st_intervals():
    """ST 区间内的日子按 ±5%"""
    df = pd.DataFrame({
        "code": ["600519"] * 3,
        "trade_date": pd.to_datetime(["2024-01-02", "2024-06-03", "2024-12-02"]),
        "close": [10.0] * 3,
        "pre_close": [10.0] * 3,
    })
    st_map = {"600519": [(pd.Timestamp("2024-05-01"), pd.Timestamp("2024-08-31"))]}
    out = apply_limit_prices(df, "600519", st_map)
    assert out.loc[0, "limit_up"] == 11.0, "区间外应 10%"
    assert out.loc[1, "limit_up"] == 10.5, "区间内应 5%"
    assert out.loc[2, "limit_up"] == 11.0
    assert is_st_on("600519", "2024-06-03", st_map)
    assert not is_st_on("600519", "2024-01-02", st_map)
    print("[OK] ST 区间生效：区间内 ±5%、区间外按板块")


def test_apply_gem_switch_within_one_series():
    """同一只创业板股票，跨 2020-08-24 的档位切换"""
    df = pd.DataFrame({
        "code": ["300750"] * 2,
        "trade_date": pd.to_datetime(["2020-08-21", "2020-08-24"]),
        "close": [10.0, 10.0],
        "pre_close": [10.0, 10.0],
    })
    out = apply_limit_prices(df, "300750")
    assert out.loc[0, "limit_up"] == 11.0, "改革前 10%"
    assert out.loc[1, "limit_up"] == 12.0, "改革后 20%"
    print("[OK] 同一序列内跨制度切换：11.00 -> 12.00")


def test_three_callers_share_one_rule():
    """三处调用方都指向唯一实现（不再各自写一份规则）"""
    import inspect
    from database.downloader import meta
    from scripts import rebuild_limit as rb

    src_meta = inspect.getsource(meta.MetaDownloader.download_code)
    assert "limit_rules" in src_meta, "meta.py 应转调 database/limit_rules"
    assert "close\", axis" not in src_meta and 'df["close"].shift(1)' not in src_meta, \
        "meta.py 不应再自己算 pre_close"
    src_rb = inspect.getsource(rb.rebuild)
    assert "apply_limit_prices" in src_rb, "rebuild_limit.py 应转调 apply_limit_prices"
    import scripts.daily_update as du
    src_du = inspect.getsource(du.rebuild_limit_year)
    assert "limit_rules" in src_du, "daily_update.py 应转调 database/limit_rules"
    assert 'df["close"].shift(1)' not in src_du, "daily_update.py 不应再用 close.shift(1)"
    print("[OK] 三处调用方共用唯一规则；不再各自实现、不再用 close.shift(1)")


if __name__ == "__main__":
    test_board_tiers()
    test_gem_regime_switch()
    test_st_takes_precedence()
    test_compute_rounds_to_cent()
    test_rounding_is_half_up_not_bankers()
    test_apply_uses_official_pre_close_not_shift()
    test_apply_applies_st_intervals()
    test_apply_gem_switch_within_one_series()
    test_three_callers_share_one_rule()
    print("\n全部涨跌停规则测试通过")
