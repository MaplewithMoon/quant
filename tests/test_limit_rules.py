# -*- coding: utf-8 -*-
"""涨跌停规则测试：唯一实现、板块档位、ST、创业板制度切换、禁用 close.shift(1)"""
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from database.limit_rules import (FIRST_44, GEM_20PCT_FROM, NO_LIMIT,
                                  apply_limit_prices, base_pct, board_of,
                                  compute, is_st_on, limit_pct, listing_rule,
                                  st_mask)


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
    """ST 的 ±5% **只对主板**生效

    创业板 2020-08-24 注册制改革后，创业板/科创板的风险警示股仍按 ±20%；
    给它们套 5% 会让涨停价偏低一半以上（实测约 5,000 行价格越界）。
    """
    assert abs(limit_pct("600519", None, True) - 0.05) < 1e-12
    assert abs(limit_pct("000001", None, True) - 0.05) < 1e-12, "主板 ST = 5%"
    assert abs(limit_pct("300750", None, True) - 0.20) < 1e-12, \
        "创业板 ST 仍是 20%，不是 5%"
    assert abs(limit_pct("688981", None, True) - 0.20) < 1e-12, \
        "科创板 ST 仍是 20%，不是 5%"
    assert abs(limit_pct("920001", None, True) - 0.30) < 1e-12, \
        "北交所 ST 走 30% 档位"
    # 主板 ST 2026-07-06 起放宽到 ±10%（日期由行情数据反推，见 limit_rules 注释）
    assert abs(limit_pct("600519", "2026-07-03", True) - 0.05) < 1e-12
    assert abs(limit_pct("600519", "2026-07-06", True) - 0.10) < 1e-12
    up, dn = compute(10.0, "600519", None, True)
    assert (up, dn) == (10.5, 9.5), (up, dn)
    up, dn = compute(10.0, "300750", None, True)
    assert (up, dn) == (12.0, 8.0), (up, dn)
    print("[OK] ST 档位：**仅主板** ±5%（2026-07-06 起 ±10%）；"
          "创业板/科创板/北交所 ST 走板块档位")


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
    """ST 区间内的日子按 ±5%（主板）"""
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
    # 同一 ST 区间套在创业板股票上：仍按 20%
    df2 = df.assign(code=["300750"] * 3)
    out2 = apply_limit_prices(df2, "300750", st_map)
    assert out2.loc[1, "limit_up"] == 12.0, \
        f"创业板 ST 应为 20%（12.0），实得 {out2.loc[1, 'limit_up']}"
    print("[OK] ST 区间生效：主板区间内 ±5%、区间外按板块；创业板不套 5%")


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


def test_listing_regimes_by_board_and_date():
    """上市初期规则：按"板块 × 上市日"选制度

    这是 A 股最容易被记错的一组规则，逐条钉住：
      主板   1990~2013-12-12 首日不设限 / 2013-12-13~2023-04-09 首日 ±44% /
             2023-04-10 起前 5 日不设限
      创业板 2009-10-30 起首日不设限 / 2020-08-24 起前 5 日不设限
      科创板 2019-07-22 开板起前 5 日不设限
      北交所 2020-07-27 起首日不设限
    """
    cases = [
        # (代码, 上市日, 期望 kind, 期望天数)
        ("600519", "2001-08-27", NO_LIMIT, 1),      # 主板（老规则）
        ("600519", "2013-12-12", NO_LIMIT, 1),      # 改革前一日仍是不设限
        ("600519", "2013-12-13", FIRST_44, 0),      # 改革生效日
        ("601816", "2020-01-16", FIRST_44, 0),      # 京沪高铁（实测首日 +43.2%）
        ("002973", "2020-01-06", FIRST_44, 0),      # 侨银股份（实测首日 +44.1%）
        ("600519", "2023-04-09", FIRST_44, 0),      # 全面注册制前一日
        ("600519", "2023-04-10", NO_LIMIT, 5),      # 全面注册制首批
        ("300750", "2018-06-11", NO_LIMIT, 1),      # 创业板（改革前）
        ("300750", "2020-08-24", NO_LIMIT, 5),      # 创业板注册制改革当日
        ("301001", "2021-05-06", NO_LIMIT, 5),
        ("688981", "2020-07-16", NO_LIMIT, 5),      # 科创板
        ("689009", "2020-10-29", NO_LIMIT, 5),
        ("920001", "2022-12-27", NO_LIMIT, 1),      # 北交所首日
        ("830799", "2021-11-15", NO_LIMIT, 1),
    ]
    for code, ld, kind, n in cases:
        got = listing_rule(code, ld)
        assert got == (kind, n), f"{code} 上市 {ld} 应得 {(kind, n)}，实得 {got}"
    # 上市日缺失 -> 不做特殊处理
    assert listing_rule("600519", None) == (None, 0)
    assert listing_rule("600519", pd.NaT) == (None, 0)
    print("[OK] 上市初期规则：主板/创业板/科创板/北交所 的制度切换日期逐条正确")


def test_listing_window_makes_limit_nan():
    """不设涨跌幅 -> limit 置 NaN（NaN 在引擎里表示"无限制"）

    `MarketRules._num()` 对 NaN 返回 None 会**跳过**涨跌停检查，
    所以 NaN 是正确的表达；若按常规档位算，次新股首日的合法大涨会被误判成封板。
    """
    df = pd.DataFrame({
        "code": ["688981"] * 3,
        "trade_date": pd.to_datetime(["2020-07-16", "2020-07-21", "2020-07-23"]),
        "pre_close": [10.0] * 3,
    })
    # 中芯国际 2020-07-16 上市，前 5 个交易日为 7/16,17,20,21,22
    rule = (NO_LIMIT, pd.Timestamp("2020-07-16"), pd.Timestamp("2020-07-22"), 5)
    out = apply_limit_prices(df, "688981", listing_rule=rule)
    assert np.isnan(out.loc[0, "limit_up"]), "上市首日应无涨跌幅限制"
    assert np.isnan(out.loc[1, "limit_up"]), "第 4 个交易日仍在窗口内"
    assert out.loc[2, "limit_up"] == 12.0, "窗口外应回到科创板 ±20%"
    assert out.loc[2, "limit_down"] == 8.0
    # 不传规则时不生效（回测区间不含次新股时结果一样，但校验会报不一致）
    out2 = apply_limit_prices(df, "688981")
    assert out2.loc[0, "limit_up"] == 12.0
    print("[OK] 不设涨跌幅窗口：窗口内置 NaN、窗口外回到板块档位")


def test_first_day_44_rule():
    """首日 ±44%：基准是**发行价**（tushare 首日 pre_close 即发行价）"""
    df = pd.DataFrame({
        "code": ["002973"] * 2,
        "trade_date": pd.to_datetime(["2020-01-06", "2020-01-07"]),
        "pre_close": [5.74, 8.27],        # 5.74 = 发行价
    })
    rule = (FIRST_44, pd.Timestamp("2020-01-06"), pd.Timestamp("2020-01-06"), 0)
    out = apply_limit_prices(df, "002973", listing_rule=rule)
    # 5.74 × 1.44 = 8.2656 -> 8.27（四舍五入）；× 0.64 = 3.6736 -> 3.67
    assert out.loc[0, "limit_up"] == 8.27, out.loc[0, "limit_up"]
    assert out.loc[0, "limit_down"] == 3.67, out.loc[0, "limit_down"]
    # 次日回到主板 ±10%
    assert out.loc[1, "limit_up"] == 9.10, out.loc[1, "limit_up"]
    assert out.loc[1, "limit_down"] == 7.44, out.loc[1, "limit_down"]
    # 未知规则要报错，不能静默按常规档位算
    try:
        apply_limit_prices(df, "002973",
                           listing_rule=("bogus", pd.Timestamp("2020-01-06"),
                                         pd.Timestamp("2020-01-06"), 0))
        raise AssertionError("未知规则应报错")
    except ValueError:
        pass
    # **回归**：窗口只判上界会让"数据起点早于 list_date"的股票整段历史变 NaN
    df2 = pd.DataFrame({
        "code": ["601399"] * 3,
        "trade_date": pd.to_datetime(["2015-05-04", "2020-06-08", "2020-06-09"]),
        "pre_close": [10.0] * 3,
    })
    # 国机重装 list_date=2020-06-08（重上市），但日线从 2015 年就有
    r2 = (FIRST_44, pd.Timestamp("2020-06-08"), pd.Timestamp("2020-06-08"), 0)
    o2 = apply_limit_prices(df2, "601399", listing_rule=r2)
    assert o2.loc[0, "limit_up"] == 11.0, \
        f"窗口之前的历史行被误改成了 {o2.loc[0, 'limit_up']}（只判上界的 bug）"
    assert o2.loc[1, "limit_up"] == 14.4, "重上市首日应适用 ±44%"
    print("[OK] 首日 ±44%：5.74 × 1.44 -> 8.27；次日回到 ±10%；"
          "窗口之前的历史行不受影响；未知规则报错")


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
    test_listing_regimes_by_board_and_date()
    test_listing_window_makes_limit_nan()
    test_first_day_44_rule()
    test_three_callers_share_one_rule()
    print("\n全部涨跌停规则测试通过")
