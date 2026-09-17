# -*- coding: utf-8 -*-
"""复牌首日不设涨跌幅（C4b 的真实机制）+ 停牌状态（B11）

【C4b 的原始描述是错的，这里是被数据推翻的过程】
文档写「2006–2007 未股改 S 股无涨跌幅限制」，据此的修法是"给纯 S 股放行"。
**用行情数据反证后不成立**：

    纯 S 股 2006-2007 日线 68,552 行，|涨跌幅| > 10.5% 仅 97 行（0.14%）
    同期全市场              611,411 行，|涨跌幅| > 10.5% 有 960 行（0.157%）
    -> 几乎一样。纯 S 股照样受 ±10% 约束。

真实机制是**停牌复牌首日不设涨跌幅**：

    960 行越界里能判定"距上一根 K 线间隔"的 771 行，770 行（99.87%）是复牌首日
    |涨跌幅| > 20% 的 444 行 **全部**是复牌首日
    案例：S石炼化→长江证券(+310%)、S京化二→国元证券(+256%)、S锦六陆→东北证券(+216%)
          都是停牌数月后复牌当日改名，次日立刻恢复 ±10%

若按原描述实现（给纯 S 股整个区间放行），会**错误地取消约 6.8 万行的涨跌幅限制**。

【B11 的三层"不可成交"来源】
    ① 行情缺失（全天停牌日的自然兜底）
    ② `suspend_d`（权威记录）
    ③ 涨跌停封板（由 execution/market_rules.py 处理）
实测三者自洽：2020-2024 的 13,963 条 S 记录中，13,269 条无 K 线（全天停牌），
694 条有 K 线（**正好等于日内停牌数**），矛盾 0 条。
"""
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))


def _series(dates, code="000001", pre_close=9.9):
    return pd.DataFrame({"code": [code] * len(dates),
                         "trade_date": pd.to_datetime(dates),
                         "pre_close": [pre_close] * len(dates)})


# ============================================================
# 一、复牌首日检测（纯合成）
# ============================================================
def _cal(start="2024-01-01", end="2024-04-01"):
    """**合成交易日历**（注入用）

    ⚠️ 必须注入：`resumption_windows` 靠"错过几个交易日"判定停牌，而交易日历
    来自 `frozen/calendar`。**CI 里没有 `db/`，日历为空 -> 恒返回全 False**，
    这些合成断言就全部失效 —— 本地有 db 所以过得去，CI 一跑就红。

    这类"测试隐式依赖 db"和直接读数据的测试不同：它看上去是纯合成用例，
    完全不该依赖本机数据，所以更容易漏。
    """
    return pd.bdate_range(start, end)


def test_resumption_detected_after_long_halt():
    """停牌超过阈值 -> 复牌首日被标记，涨跌停置 NaN"""
    from database.limit_rules import (RESUME_NO_LIMIT_MIN_MISSED,
                                      apply_limit_prices, resumption_windows)

    # 前 3 天连续，然后空约 40 个交易日，再 3 天
    before = pd.bdate_range("2024-01-02", periods=3)
    after = pd.bdate_range("2024-03-01", periods=3)
    dates = list(before) + list(after)
    df = _series(dates)
    cal = _cal()
    w = resumption_windows(df, cal=cal)
    assert not w[:3].any(), "连续交易日不该被标成复牌首日"
    assert w[3], "长期停牌后的第一根 K 线应标成复牌首日"
    assert not w[4:].any(), "复牌后第二天不该再标"

    out = apply_limit_prices(df, "000001", resume_no_limit=True, cal=cal)
    assert pd.isna(out.loc[3, "limit_up"]), "复牌首日应不设涨跌停"
    assert pd.isna(out.loc[3, "limit_down"])
    assert not pd.isna(out.loc[4, "limit_up"]), "复牌次日应恢复正常涨跌停"
    assert bool(out.loc[3, "is_resumption"]) is True
    print(f"[OK] 复牌首日：停牌 {RESUME_NO_LIMIT_MIN_MISSED}+ 交易日后复牌 -> "
          f"涨跌停置 NaN，次日恢复")
    # **默认必须关闭**：传入日期子集（如只取月度调仓日）会把每个采样点
    # 都误判成复牌首日。这个坑在实现时真的踩到了。
    d2 = apply_limit_prices(df, "000001", cal=cal)
    assert d2["limit_up"].notna().all(), \
        "默认不该启用复牌规则 —— 日期子集会被整片误判成复牌首日"
    print("[OK] 复牌规则默认关闭（防止日期子集被误判）")


def test_short_halt_keeps_limit():
    """**关键反例**：短停牌（如停一天开会）**不能**免涨跌停

    这正是原描述会踩的坑：若按"S 股整个区间放行"，会把大量正常受 ±10%
    约束的日子误放行。实测错过 1-5 个交易日的 7,750 行里，越界 **0** 行
    —— 说明限制确实生效。
    """
    from database.limit_rules import apply_limit_prices, resumption_windows

    d = list(pd.bdate_range("2024-01-02", periods=2))
    d.append(pd.Timestamp("2024-01-08"))       # 跳过几天（短停牌）
    d += list(pd.bdate_range("2024-01-09", periods=2))
    df = _series(d)
    cal = _cal()
    w = resumption_windows(df, cal=cal)
    assert not w.any(), f"短停牌不该被标成复牌首日: {w}"
    out = apply_limit_prices(df, "000001", resume_no_limit=True, cal=cal)
    assert out["limit_up"].notna().all(), "短停牌后仍应有涨跌停限制"
    print("[OK] 短停牌不豁免涨跌停（避免把 6.8 万行正常约束误放行）")


def test_resumption_uses_trading_days_not_calendar_days():
    """阈值按**错过交易日数**，不是自然日 —— 长假不能算成停牌"""
    from database.calendar import trading_days
    from database.limit_rules import resumption_windows

    cal = trading_days()
    if len(cal) < 300:
        print("[SKIP] 无交易日历")
        return
    # 取一对相隔约 10 个自然日、但中间没有交易日的日期（长假）
    idx = cal.searchsorted(pd.Timestamp("2024-02-01"))
    holiday_gap = [cal[idx - 1], cal[idx + 1]]     # 春节前后
    df = _series(holiday_gap)
    w = resumption_windows(df)
    assert not w.any(), "长假造成的间隔被误判成复牌首日（应该按交易日算）"
    print("[OK] 阈值按交易日算：长假间隔不触发复牌豁免")


# ============================================================
# 二、停牌状态（B11）
# ============================================================
def test_suspend_timing_is_readable():
    """**回归**：`suspend_timing` 必须能读出来（schema 混合曾导致硬报错）

    早期写盘时该列在"全空值"的文件里被推断成 NULL 类型、有值时是 VARCHAR，
    直接 glob 会抛 ConversionException。原先没暴露只是因为没人投影这一列。
    读侧已改用 union_by_name=true。
    """
    from database.status import load_suspensions

    df = load_suspensions()
    if df.empty:
        print("[SKIP] 无 frozen/suspend")
        return
    assert "suspend_timing" in df.columns, "读不到 suspend_timing（schema 问题复发）"
    assert set(df["suspend_type"].unique()) == {"S"}, "只应含 S 记录"
    t = df["suspend_timing"].astype("string").fillna("").str.strip()
    # 容忍全空（某些数据切片确实没有日内停牌），但格式若有值应像时段
    nz = t[t.ne("")]
    if len(nz):
        assert nz.str.contains(":").all(), \
            f"日内停牌取值不像时间区间: {list(nz.unique()[:5])}"
    print(f"[OK] suspend_timing 可读：{len(df):,} 条 S 记录，"
          f"其中日内停牌 {int(nz.ne('').sum()):,} 条")


def test_intraday_and_fullday_are_separate():
    """日内停牌与全天停牌必须分开：日内停牌当天仍有可交易时段"""
    from database.status import load_suspensions

    df = load_suspensions("2024-01-01", "2024-12-31")
    if df.empty:
        print("[SKIP] 无冻结/suspend")
        return
    t = df["suspend_timing"].astype("string").fillna("").str.strip()
    assert t.ne("").any(), "2024 年应有日内停牌记录（实测 38 条）"
    full = df[t.eq("")]
    intra = df[t.ne("")]
    assert len(intra) > 0 and len(full) > 0
    # 日内停牌不应出现在全天停牌集合里
    assert not set(map(tuple, intra[["code", "trade_date"]].values)) & \
        set(map(tuple, full[["code", "trade_date"]].values))
    print(f"[OK] 全天停牌 {len(full):,} 条 / 日内停牌 {len(intra):,} 条，互不重叠")


def test_three_source_cross_validation():
    """**核心**：全天停牌日应当没有 K 线；日内停牌日应当有 K 线"""
    from database.status import audit_coverage

    r = audit_coverage("2020-01-01", "2024-12-31")
    if not r["n_suspend"]:
        print("[SKIP] 无停牌数据")
        return
    # 矛盾数必须为 0（实测 13,963 条里 0 条矛盾）
    assert r["n_suspended_with_bar"] == 0, \
        f"有 {r['n_suspended_with_bar']} 条全天停牌却有成交量: " \
        f"{r['suspended_with_bar'][:3]}"
    # 有 K 线的那些应当**正好**是日内停牌数
    with_bar = r["n_suspend"] - r["n_missing_bar"]
    assert with_bar == r["n_intraday"], \
        f"有 K 线的 {with_bar} 条 != 日内停牌 {r['n_intraday']} 条 —— " \
        f"三层来源不自洽（这是发现数据问题的信号）"
    print(f"[OK] 三层交叉验证：{r['n_suspend']:,} 条 S 记录 = "
          f"{r['n_missing_bar']:,} 无K线(全天) + {r['n_intraday']:,} 有K线(日内)，"
          f"矛盾 0")


# ============================================================
# 三、C2b 名单维护（主体 vs 代码）
# ============================================================
def test_relisted_entities_and_resumed_are_distinct():
    """**关键**：重新上市名单要记公司主体，且**不能**混入"恢复上市"股

    恢复上市（盐湖股份、皇台酒业）代码未变、一直在证券主表里，其中断期由
    停复牌数据兜住。把它们加进排除名单会制造**新的误拦**。
    """
    from database.defects import (RELISTED_CODES, RELISTED_ENTITIES,
                                  RESUMED_NOT_RELISTED)

    assert set(RELISTED_ENTITIES) == set(RELISTED_CODES), \
        "主体映射与代码列表不一致（重新上市通常换代码换名，必须能对上主体）"
    for code, names in RELISTED_ENTITIES.items():
        assert names and all(isinstance(n, str) and n for n in names), \
            f"{code} 缺少公司主体名"
    overlap = set(RELISTED_CODES) & set(RESUMED_NOT_RELISTED)
    assert not overlap, f"恢复上市股被误列进重新上市名单: {overlap}"
    print(f"[OK] 重新上市 {len(RELISTED_CODES)} 家（均带主体名），"
          f"恢复上市 {len(RESUMED_NOT_RELISTED)} 家不在名单内")


if __name__ == "__main__":
    test_resumption_detected_after_long_halt()
    test_short_halt_keeps_limit()
    test_resumption_uses_trading_days_not_calendar_days()
    test_suspend_timing_is_readable()
    test_intraday_and_fullday_are_separate()
    test_three_source_cross_validation()
    test_relisted_entities_and_resumed_are_distinct()
    print("\n全部复牌/停牌/C2b 测试通过")
