# -*- coding: utf-8 -*-
"""聚宽兼容层测试：调度 / 下单 / T+1 / 数据窗口 / query DSL / 指数代理 / PIT"""
import os
import sys
import types

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import joinquant as jq
from joinquant.api import JQEngine
from joinquant.data import JQData


# ============================================================
# 合成面板
# ============================================================
def _panel(n_days=120, codes=("000001", "000002", "600000", "600519"),
           start="2024-01-02", price=10.0):
    dates = pd.bdate_range(start, periods=n_days)
    rng = np.random.default_rng(0)
    close = pd.DataFrame(
        100.0 * np.cumprod(1 + rng.normal(0.0005, 0.01, (n_days, len(codes))), axis=0),
        index=dates, columns=list(codes))
    op = close.shift(1).fillna(close.iloc[0])
    p = {"close": close, "open": op, "high": close * 1.02, "low": close * 0.98,
         "pre_close": close.shift(1).fillna(close.iloc[0]),
         "volume": pd.DataFrame(1e7, index=dates, columns=list(codes)),
         "amount": pd.DataFrame(1e8, index=dates, columns=list(codes)),
         "total_mv": pd.DataFrame(5e5, index=dates, columns=list(codes)),  # 万元 -> 50 亿
         "limit_up": close * 1.1, "limit_down": close * 0.9,
         "suspended": pd.DataFrame(False, index=dates, columns=list(codes))}
    return p, dates


def _module(funcs: dict, init=None):
    """把函数字典包成一个"策略模块"（模拟 import 进来的 .py）"""
    m = types.ModuleType("fake_strategy")
    for k, v in funcs.items():
        setattr(m, k, v)
    if init is not None:
        m.initialize = init
    return m


# ============================================================
# 调度
# ============================================================
def test_run_daily_order_and_weekly_nth_trading_day():
    """run_daily 按时间顺序执行；run_weekly(N) 落在每周第 N 个交易日"""
    p, dates = _panel(n_days=30)
    calls = []

    def initialize(context):
        jq.run_daily(lambda c: calls.append(("open", c.current_dt)), "9:05")
        jq.run_daily(lambda c: calls.append(("late", c.current_dt)), "14:50")
        jq.run_weekly(lambda c: calls.append(("weekly2", c.current_dt)), 2, "10:00")

    m = _module({}, initialize)
    eng = JQEngine(JQData(p, dates[0], dates[-1]), 100000, fill="close")
    eng.run(m.initialize, dates)
    daily = [c for c in calls if c[0] == "open"]
    assert len(daily) == len(dates), f"run_daily 应每天一次，实得 {len(daily)}"
    # 同一天内 9:05 必须在 14:50 之前
    per_day = {}
    for kind, ts in calls:
        per_day.setdefault(ts.date(), []).append((ts.hour, kind))
    for d, lst in per_day.items():
        assert lst == sorted(lst), f"{d} 任务未按时间排序: {lst}"
    weekly = [pd.Timestamp(c[1].date()) for c in calls if c[0] == "weekly2"]
    # 每周第 2 个交易日（不是"周二"：遇到节假日会顺延，如 2024-01-01 元旦那周是周三）
    expect = []
    for w in sorted({d.isocalendar()[:2] for d in dates}):
        same = [d for d in dates if d.isocalendar()[:2] == w]
        if len(same) >= 2:
            expect.append(same[1])
    assert weekly == expect, f"周任务应落在每周第 2 个交易日\n实得 {weekly[:5]}\n期望 {expect[:5]}"
    print(f"[OK] 调度：每日任务 {len(daily)} 次、周任务 {len(weekly)} 次"
          f"（落在每周第 2 个交易日，遇节假日自动顺延），同日按时间排序")


# ============================================================
# 下单 / T+1 / 手数
# ============================================================
def test_order_target_value_lot_and_cash():
    p, dates = _panel(n_days=10)
    state = {}

    def initialize(context):
        jq.run_daily(_buy, "10:00")

    def _buy(context):
        if context.current_dt.date() == dates[1].date():
            o = jq.order_target_value("000001", 30000)
            state["order"] = o
            state["pos"] = context.portfolio.positions.get("000001")

    m = _module({"_buy": _buy}, initialize)
    eng = JQEngine(JQData(p, dates[0], dates[-1]), 100000, fill="open")
    eng.run(m.initialize, dates)
    o = state["order"]
    assert o is not None and o.filled > 0, "应成交"
    assert o.filled % 100 == 0, f"必须是整手，实得 {o.filled}"
    assert eng.pf.cash >= 0, "现金不能为负"
    # 成交金额应接近目标 30000（受手数与滑点影响）
    px = float(p["open"].loc[dates[1], "000001"])
    assert abs(o.filled * px - 30000) <= px * 100 + 1e-6, \
        f"目标 30000 元，实得 {o.filled * px:.0f}"
    print(f"[OK] order_target_value：成交 {o.filled:.0f} 股（整手），"
          f"金额 {o.filled*px:,.0f}，剩余现金 {eng.pf.cash:,.0f}")


def test_t_plus_1_blocks_same_day_sell():
    p, dates = _panel(n_days=6)
    seen = {}

    def initialize(context):
        jq.run_daily(_act, "10:00")

    def _act(context):
        d = context.current_dt.date()
        if d == dates[1].date():
            jq.order_target_value("000001", 30000)
        if d == dates[1].date():
            # 同一天立刻反向卖出 -> 应被 T+1 拦下
            o = jq.order_target_value("000001", 0)
            seen["same_day_sell"] = o

    m = _module({"_act": _act}, initialize)
    eng = JQEngine(JQData(p, dates[0], dates[-1]), 100000, fill="open")
    eng.run(m.initialize, dates)
    o = seen.get("same_day_sell")
    assert o is not None
    assert o.filled == 0, "当日买入当日不可卖（T+1）"
    assert any("T+1" in k for k in eng.rejections), f"应记录 T+1 拒绝：{eng.rejections}"
    print(f"[OK] T+1：同日反向卖出被拦下（拒绝原因 {list(eng.rejections)}）")


def test_limit_up_blocks_buy():
    """开盘即涨停（open == high_limit）时买不进"""
    p, dates = _panel(n_days=6)
    d1 = dates[1]
    p["limit_up"].loc[d1, "000001"] = float(p["open"].loc[d1, "000001"])   # 开盘即涨停
    p["low"].loc[d1, "000001"] = float(p["open"].loc[d1, "000001"])        # 一字板
    seen = {}

    def initialize(context):
        jq.run_daily(_buy, "10:00")

    def _buy(context):
        if context.current_dt.date() == d1.date():
            seen["o"] = jq.order_target_value("000001", 30000)

    m = _module({"_buy": _buy}, initialize)
    eng = JQEngine(JQData(p, dates[0], dates[-1]), 100000, fill="open")
    eng.run(m.initialize, dates)
    assert seen["o"].filled == 0, "涨停封板应买不进"
    assert any("涨停" in k for k in eng.rejections), eng.rejections
    print(f"[OK] 涨停封板买不进：{list(eng.rejections)}")


# ============================================================
# 数据窗口（不许前视）
# ============================================================
def test_history_daily_excludes_today():
    """盘中调用的 history('1d') 只能拿到**已完成**的日线（到昨日为止）"""
    p, dates = _panel(n_days=10)
    got = {}

    def initialize(context):
        jq.run_daily(_snap, "10:00")

    def _snap(context):
        d = context.current_dt.date()
        if d == dates[5].date():
            got["h"] = jq.history(1, "1d", "close", ["000001"])
            got["today"] = dates[5]
            got["prev"] = context.previous_date

    m = _module({"_snap": _snap}, initialize)
    eng = JQEngine(JQData(p, dates[0], dates[-1]), 100000, fill="open")
    eng.run(m.initialize, dates)
    v = float(got["h"]["000001"].iloc[-1])
    expect = float(p["close"].loc[dates[4], "000001"])   # 昨日收盘
    assert abs(v - expect) < 1e-9, f"history 应返回昨日收盘 {expect}，实得 {v}"
    assert pd.Timestamp(got["h"].index[-1]) < pd.Timestamp(got["today"]), "不能包含当日"
    assert pd.Timestamp(got["h"].index[-1]).date() == got["prev"], "最后一根应是昨日"
    print("[OK] history('1d') 只到昨日（盘中日线未完成），无前视")


def test_get_price_respects_end_date():
    p, dates = _panel(n_days=10)
    data = JQData(p, dates[0], dates[-1])
    s = data.history("close", ["000001"], dates[4], 3)
    assert len(s) == 3 and pd.Timestamp(s.index[-1]) == dates[4]
    assert float(s["000001"].iloc[-1]) == float(p["close"].loc[dates[4], "000001"])
    print("[OK] get_price/history 按 end_date 截断，返回最后 count 根")


def test_current_data_fields():
    p, dates = _panel(n_days=6)
    got = {}

    def initialize(context):
        jq.run_daily(_snap, "10:00")

    def _snap(context):
        if context.current_dt.date() == dates[2].date():
            cd = jq.get_current_data()
            got["b"] = cd["000001"]

    m = _module({"_snap": _snap}, initialize)
    eng = JQEngine(JQData(p, dates[0], dates[-1]), 100000, fill="open")
    eng.run(m.initialize, dates)
    b = got["b"]
    assert not b.paused and not b.is_st
    assert abs(b.day_open - float(p["open"].loc[dates[2], "000001"])) < 1e-9
    assert abs(b.high_limit - float(p["limit_up"].loc[dates[2], "000001"])) < 1e-9
    print("[OK] get_current_data 的 paused/is_st/day_open/high_limit 均可用")


# ============================================================
# query DSL
# ============================================================
def test_query_dsl_filters_and_sort():
    p, dates = _panel(n_days=5)
    mgr = JQData(p, dates[0], dates[-1])
    # 手工塞一份财务数据。⚠️ 公告日必须**早于**查询日，否则 PIT 会正确地
    # 把它全部过滤掉（第一次写成了 2024-04-01 > 面板日期，结果返回空集，
    # 而"空集"会让下面所有断言假通过）。
    mgr._fin = pd.DataFrame({
        "code": ["000001", "000002", "600000", "600519"],
        "ann_date": pd.to_datetime(["2023-04-01"] * 4),
        "end_date": pd.to_datetime(["2022-12-31"] * 4),
        "revenue": [2e8, 5e7, 3e8, 1e9],          # 000002 营收 < 1 亿
        "n_income": [1e7, 1e7, -1e7, 5e8],        # 600000 净利为负
        "n_income_attr_p": [1e7, 1e7, -1e7, 5e8],
    })
    mgr._mktcap_yi = pd.DataFrame(20.0, index=dates,
                                  columns=["000001", "000002", "600000", "600519"])
    mgr._mktcap_yi["600519"] = 2000.0
    eng = JQEngine(mgr, 100000)
    eng.current_date = dates[-1]
    eng.previous_date = dates[-1].date()

    q = (jq.query(jq.valuation.code, jq.valuation.market_cap)
         .filter(jq.valuation.market_cap.between(10, 1e8),
                 jq.income.np_parent_company_owners > 0,
                 jq.income.net_profit > 0,
                 jq.income.operating_revenue > 1e8)
         .order_by(jq.valuation.market_cap.asc()).limit(10))
    df = jq.get_fundamentals(q)
    codes = list(df["code"])
    assert codes, "不应为空集 —— 空集会让下面的断言假通过"
    assert "000002" not in codes, "营收 < 1 亿应被过滤"
    assert "600000" not in codes, "净利为负应被过滤"
    assert set(codes) == {"000001", "600519"}, f"应剩两只，实得 {codes}"
    assert codes == ["000001", "600519"], f"应按市值升序，实得 {codes}"
    assert "market_cap" in df.columns
    print(f"[OK] query DSL：between/>/order_by/limit 全部生效 -> {codes}")


def test_query_in_filter():
    p, dates = _panel(n_days=5)
    mgr = JQData(p, dates[0], dates[-1])
    eng = JQEngine(mgr, 100000)
    eng.current_date = dates[-1]
    eng.previous_date = dates[-1].date()
    df = jq.get_fundamentals(
        jq.query(jq.valuation.code).filter(jq.valuation.code.in_(["000001", "600519"])))
    assert sorted(df["code"]) == ["000001", "600519"], df
    print("[OK] query DSL：code.in_ 生效")


# ============================================================
# 指数代理 / PIT
# ============================================================
def test_index_proxy_rules():
    """指数成分：**优先真实月度快照**，拿不到时才退回规则重建"""
    codes = ["000001", "000002", "002001", "002002", "003001", "600000", "300001"]
    p, dates = _panel(n_days=5, codes=codes)
    mgr = JQData(p, dates[0], dates[-1])
    mgr._list_date = pd.Series(pd.Timestamp("2000-01-01"), index=codes)

    # 有真实成分时（库里已下载 399101.SZ / 000985.CSI）必须用真实的
    if mgr.has_real_members("399101.XSHE"):
        real = mgr.index_stocks("399101.XSHE", dates[-1])
        assert len(real) > 100, f"应返回真实中小综指成分，实得 {len(real)} 只"
        assert all(str(c).startswith(("002", "003", "001", "000")) for c in real)
        print(f"[OK] 指数成分优先用真实快照：399101.XSHE -> {len(real)} 只")
    else:
        print("[SKIP] 库里没有 399101 真实成分，跳过真实优先检查")

    # 清掉真实成分，验证规则重建的兜底路径
    mgr._real_members = {}
    mgr._index_cache = {}
    zx = mgr.index_stocks("399101.XSHE", dates[-1])
    assert set(zx) == {"002001", "002002", "003001"}, \
        f"中小综指代理应为 002/003，实得 {zx}"
    full = mgr.index_stocks("000985.XSHG", dates[-1])
    assert set(full) == set(codes), "中证全指代理应为全部 A 股"
    idx = mgr.index_close("399101.XSHE", dates[-1], 5)
    assert len(idx) > 0 and (idx > 0).all(), "重建指数应为正的等权净值"
    print(f"[OK] 无真实成分时退回规则重建：399101 -> {sorted(zx)}；"
          f"000985 -> 全部 {len(full)} 只；重建指数 {len(idx)} 根")


def test_index_proxy_excludes_unlisted():
    codes = ["002001", "002002"]
    p, dates = _panel(n_days=5, codes=codes)
    mgr = JQData(p, dates[0], dates[-1])
    mgr._real_members = {}          # 强制走规则重建路径（库里有真实成分）
    mgr._index_cache = {}
    mgr._list_date = pd.Series({"002001": pd.Timestamp("2000-01-01"),
                                "002002": dates[-1] + pd.Timedelta(days=30)})
    got = mgr.index_stocks("399101.XSHE", dates[-1])
    assert got == ["002001"], f"未上市的不该入选，实得 {got}"
    print("[OK] 指数代理按 list_date 剔除未上市股票（规则重建兜底路径）")


def test_unknown_history_field_is_nan_not_close():
    """`history()`/`get_price()` 取未知字段必须给 NaN，**不能退回 close**

    回归测试。原先 `_frame()` 写的是 `.get(field, self._close)`，于是
    `fields=['close','high_limit']` 里 `high_limit` 变成收盘价本身：
    `df[df['close'] == df['high_limit']]` 恒成立 -> `g.yesterday_HL_list`
    等于全部持仓 -> 14:00 的 `check_limit_up()` 拿真实涨停价比一次就几乎必然
    `close < high_limit` -> **每个持仓每天都被卖掉**（实测 v2 五个月触发 173 次
    "涨停打开卖出"、只有 3 次"继续持有"）。真需要兜底时给 NaN，让调用方看得见。
    """
    p, dates = _panel(n_days=8)
    mgr = JQData(p, dates[0], dates[-1])
    hl = mgr.history("high_limit", ["000001"], dates[-1], 1)
    assert abs(float(hl["000001"].iloc[-1])
               - float(p["limit_up"].at[dates[-1], "000001"])) < 1e-9, \
        "high_limit 取到的不是真实涨停价"
    junk = mgr.history("not_a_field", ["000001"], dates[-1], 1)
    assert junk["000001"].isna().all(), "未知字段应返回 NaN，而不是退回 close"
    print("[OK] 未知行情字段返回 NaN（不退回 close）；high_limit 取真实涨停价")


def test_get_price_intraday_fields_are_not_all_nan():
    """`get_price(frequency='1m')` 的盘中字段**不能整片 NaN**

    回归测试。`current_bar()` 返回的是 index=代码 / columns=open/high/low/close，
    而这里曾经按相反方向取列，于是每个字段都落到 `else np.nan`：
    `check_limit_up()` 里 `close < high_limit` 恒为 False（NaN 比较），
    "涨停打开就卖、次日回补"这条路径完全死掉 —— 本地 v1 只有 29 个个股成交日，
    聚宽有 89 天，差的主要就是它。

    同时盯住：`high_limit` 必须是**真实涨跌停价**，不能拿开盘价顶替。
    """
    p, dates = _panel(n_days=6)
    eng = JQEngine(JQData(p, dates[0], dates[-1]), 100_000.0)
    from joinquant.api import get_price
    for i, d in enumerate(dates[1:4], start=1):
        eng.current_date = d
        eng.current_dt = d + pd.Timedelta(hours=14)
        x = get_price("000001", end_date=d, frequency="1m",
                      fields=["close", "high_limit", "low_limit"], count=1,
                      panel=False)
        close = float(x.iloc[0, 0])
        hl, ll = float(x.iloc[0, 1]), float(x.iloc[0, 2])
        assert np.isfinite(close), f"{d} close 是 NaN"
        assert np.isfinite(hl) and np.isfinite(ll), f"{d} 涨跌停价是 NaN"
        assert abs(close - float(p["close"].at[d, "000001"])) < 1e-9
        assert abs(hl - float(p["limit_up"].at[d, "000001"])) < 1e-9, \
            "high_limit 不是真实涨停价（很可能又退回开盘价了）"
        assert hl > ll and ll > 0
    print("[OK] get_price(1m) 盘中字段有值，且 high_limit/low_limit 取真实涨跌停价")


def test_financials_asof_is_point_in_time():
    p, dates = _panel(n_days=5)
    mgr = JQData(p, dates[0], dates[-1])
    mgr._fin = pd.DataFrame({
        "code": ["000001", "000001"],
        "ann_date": pd.to_datetime(["2024-04-20", "2024-08-20"]),
        "end_date": pd.to_datetime(["2024-03-31", "2024-06-30"]),
        "revenue": [1e8, 3e8], "n_income": [1e7, 3e7], "n_income_attr_p": [1e7, 3e7],
    })
    mgr._fin = mgr._fin.sort_values("ann_date").reset_index(drop=True)
    mgr._fin_dates = mgr._fin["ann_date"].values
    a = mgr.financials_asof("2024-05-01")
    assert float(a.loc["000001", "revenue"]) == 1e8, "5 月只能看到一季报"
    b = mgr.financials_asof("2024-09-01")
    assert float(b.loc["000001", "revenue"]) == 3e8, "9 月应看到中报"
    c = mgr.financials_asof("2024-04-01")
    assert c.empty, "公告前不应有任何数据（前视）"
    print("[OK] financials_asof 按 ann_date 对齐：公告前为空、公告后取最新一期")


def test_profit_glob_is_limited_to_profit_sheet():
    """财务 glob **必须**限定 profit_*

    回归测试。曾经写成 `year=*/*.parquet`（为了在无 db 的 CI 里不抛异常），
    于是把 balance/cashflow/profit 三张同目录报表一起读进来、`union_by_name`
    补齐列，同一份报告变成三行；再按 (code, end_date) 去重 keep="first" 就
    **系统性留下资产负债表那一行** -> `get_fundamentals` 的 income 字段全 NaN
    -> 两个移植策略的选股结果恒为空 -> 全程空仓买货币 ETF（本地 +2.36% vs
    聚宽 +185.52% 的真正原因）。

    这个测试断言的是**模式串**，不是"跑一遍看结果"，所以没有 db 也能拦住回归。
    """
    from joinquant.data import profit_parquet_glob
    g = profit_parquet_glob()
    assert g.endswith("/year=*/profit_*.parquet"), f"glob 没限定利润表：{g}"
    assert "balance_" not in g and "cashflow_" not in g
    print(f"[OK] 财务 glob 限定在利润表：{g}")


def test_financials_asof_has_values_when_db_present():
    """有 db 时利润表字段**不能整列为空**（上面那个 bug 的直接症状）"""
    from database.config import FROZEN_ROOT
    if not any((FROZEN_ROOT / "financial").glob("year=*/profit_*.parquet")):
        print("[SKIP] 无 frozen/financial，跳过")
        return
    p, dates = _panel(n_days=5)
    mgr = JQData(p, dates[0], dates[-1])
    if mgr._fin.empty:
        print("[SKIP] 利润表读出来为空，跳过")
        return
    asof = mgr.financials_asof("2024-06-30")
    assert not asof.empty, "2024-06-30 应该能看到已公告财报"
    for col in ("revenue", "n_income", "n_income_attr_p"):
        n_ok = int(asof[col].notna().sum())
        assert n_ok > 0.5 * len(asof), \
            f"{col} 非空仅 {n_ok}/{len(asof)} —— 大概率又读到了非利润表的行"
    print(f"[OK] 利润表字段非空（{len(asof)} 只，revenue 非空 "
          f"{int(asof['revenue'].notna().sum())}）")


def test_etf_close_prefers_real_then_synthetic():
    """ETF：**优先真实日线**，没有真实数据时才退回"现金等价物"合成"""
    p, dates = _panel(n_days=252)
    mgr = JQData(p, dates[0], dates[-1], etf_yield=0.02)

    # 库里没有的代码 -> 合成（年化 2%）
    syn = mgr.etf_close("999999.XSHG", dates[-1], 252)
    assert len(syn) == 252
    assert abs(syn.iloc[-1] / syn.iloc[0] - 1.02) < 5e-3, "合成 ETF 应按年化 2% 增长"

    # 511880 已有真实日线（frozen/fund_daily）-> 必须用真实的
    if mgr.has_real_fund("511880.XSHG"):
        real = mgr.etf_close("511880.XSHG", dates[-1], 100)
        assert len(real) > 0
        assert not np.allclose(real.values, syn.values[-len(real):]), \
            "有真实日线时不该再用合成序列"
        print(f"[OK] ETF：无真实数据时代码合成（年化 2%）；"
              f"511880 用真实日线 {len(real)} 根")
    else:
        print("[OK] ETF：无真实数据时按年化 2% 合成（库里暂无 511880 日线）")


# ============================================================
# 端到端迷你策略
# ============================================================
def test_positions_returns_zero_for_unheld():
    """回归：`context.portfolio.positions[未持仓代码]` 必须返回 0 持仓而不是 KeyError

    聚宽的 positions 是"访问即返回对象"的字典，策略里写的是
        if context.portfolio.positions[stock].total_amount == 0:
    用普通 dict 会直接 KeyError —— 实测 506 次拒绝委托全是这个。
    """
    p, dates = _panel(n_days=8)
    got = {}

    def initialize(context):
        jq.run_daily(_check, "10:00")

    def _check(context):
        if context.current_dt.date() == dates[3].date():
            pos = context.portfolio.positions
            got["len_before"] = len(pos)
            z = pos["000001"]                      # 未持仓
            got["amount"] = z.total_amount
            got["len_after"] = len(pos)            # 访问不该写入字典
            got["iter"] = list(pos.keys())

    m = _module({"_check": _check}, initialize)
    eng = JQEngine(JQData(p, dates[0], dates[-1]), 100000, fill="open")
    eng.run(m.initialize, dates)
    assert got["amount"] == 0.0, "未持仓应返回 0"
    assert got["len_before"] == got["len_after"] == 0, "访问未持仓不应改变字典长度"
    assert got["iter"] == [], "不应被写进 keys()"
    print("[OK] positions[未持仓] 返回 0 持仓且不污染字典（聚宽语义）")


def test_get_price_supports_index_and_etf():
    """回归：get_price 必须能查指数（399101）与 ETF（511880），它们不在股票面板里"""
    # ⚠️ 面板里必须真有 002/003 开头的股票，否则 399101 代理成分为空
    p, dates = _panel(n_days=60, codes=("000001", "002001", "600519"))
    mgr = JQData(p, dates[0], dates[-1])
    mgr._list_date = pd.Series({c: pd.Timestamp("2000-01-01")
                                for c in ("000001", "002001", "600519")})
    eng = JQEngine(mgr, 100000)
    eng.current_date = dates[-1]
    eng.previous_date = dates[-2].date()
    # 指数：重建的等权指数
    idx = jq.get_price("399101.XSHE", end_date=dates[-1], count=10, fields=["close"])
    assert len(idx) == 10 and (idx["close"] > 0).all(), "指数应返回正的重建净值"
    # ETF：合成的现金等价物
    etf = jq.get_price("511880.XSHG", end_date=dates[-1], count=5, fields=["close"])
    assert len(etf) == 5 and (etf["close"] > 0).all(), "ETF 应返回合成序列"
    # 混合查询：股票 + ETF，缺失字段补 NaN 而不是 KeyError
    mix = jq.get_price(["000001", "511880.XSHG"], end_date=dates[-1], count=3,
                       fields=["close", "high_limit"], panel=False)
    assert set(mix["code"].unique()) == {"000001", "511880.XSHG"}
    assert "high_limit" in mix.columns
    print("[OK] get_price 支持指数/ETF/混合查询，缺失字段补 NaN")


def test_legacy_frame_pandas1_semantics():
    """回归：还原 pandas 1.x 的两个语义（聚宽平台跑的就是 1.x）

    1. `series[-1]` 是**位置**索引 —— pandas 3 把整数键当标签，直接 KeyError
    2. `df.loc[datetime.date(...)]` 能匹配 DatetimeIndex —— pandas 3 不再自动转换
    这两条分别在 v1（84 次调仓）与 v2（每天）上真实崩过。
    """
    import datetime as _dt
    from joinquant.data import LegacyFrame
    df = LegacyFrame({"close": [1.0, 2.0, 3.0]},
                     index=pd.DatetimeIndex(["2024-01-02", "2024-01-03", "2024-01-04"]))
    s = df["close"]
    assert s[-1] == 3.0, f"series[-1] 应是最后一个值，实得 {s[-1]}"
    assert s[-2] == 2.0
    assert df.loc[_dt.date(2024, 1, 3), "close"] == 2.0, "loc[date, col] 应能匹配"
    assert df.loc[_dt.date(2024, 1, 4)].to_dict() == {"close": 3.0}
    # 正常功能不受影响
    assert df["close"].mean() == 2.0
    assert df.iloc[-1]["close"] == 3.0
    print("[OK] LegacyFrame：series[-1] 位置回退、loc[date] 归一化，其余功能正常")


def test_end_to_end_mini_strategy():
    """一个极简策略跑通：选市值最小的 2 只，等权买入，每周调仓"""
    p, dates = _panel(n_days=60)
    mgr = JQData(p, dates[0], dates[-1])
    mgr._mktcap_yi = pd.DataFrame(
        {"000001": 30.0, "000002": 20.0, "600000": 50.0, "600519": 2000.0},
        index=dates)
    eng = JQEngine(mgr, 100000, fill="open")

    def initialize(context):
        jq.run_daily(_rebalance, "10:00")

    def _rebalance(context):
        # 每周第 1 个交易日调仓
        idx = mgr.dates
        pos = idx.get_loc(pd.Timestamp(context.current_dt.date()))
        if pos == 0 or idx[pos - 1].isocalendar()[:2] != idx[pos].isocalendar()[:2]:
            hold = ["000002", "000001"]
            for c, w in zip(hold, (0.5, 0.5)):
                jq.order_target_value(c, context.portfolio.total_value * w)

    m = _module({"_rebalance": _rebalance}, initialize)
    res = eng.run(m.initialize, dates)
    eq = res["equity"]
    assert len(eq) == len(dates)
    assert eq.iloc[-1] > 0
    assert eng.pf.cash >= 0
    # 账目：现金 + 持仓市值 = 权益
    gap = abs(eng.pf.total_value - eq.iloc[-1])
    assert gap < 1e-6, f"账目差额 {gap}"
    assert len(eng.pf.positions) == 2, f"应持有 2 只，实得 {list(eng.pf.positions)}"
    print(f"[OK] 端到端：{len(eq)} 交易日、最终权益 {eq.iloc[-1]:,.0f}、"
          f"持仓 {len(eng.pf.positions)} 只、账目差额 {gap:.1e}")


def test_engine_survives_strategy_exception():
    """单个任务抛异常不应终止整个回测（与聚宽一致）"""
    p, dates = _panel(n_days=20)
    hits = []

    def initialize(context):
        jq.run_daily(_boom, "10:00")

    def _boom(context):
        hits.append(1)
        raise RuntimeError("故意抛错")

    m = _module({"_boom": _boom}, initialize)
    eng = JQEngine(JQData(p, dates[0], dates[-1]), 100000)
    res = eng.run(m.initialize, dates)
    assert len(hits) == len(dates), "每天都应被调用"
    assert any("策略异常" in k for k in res["rejections"]), res["rejections"]
    print(f"[OK] 策略异常不终止回测（{len(hits)} 天全部执行，异常被记录）")


if __name__ == "__main__":
    test_run_daily_order_and_weekly_nth_trading_day()
    test_order_target_value_lot_and_cash()
    test_t_plus_1_blocks_same_day_sell()
    test_limit_up_blocks_buy()
    test_history_daily_excludes_today()
    test_get_price_respects_end_date()
    test_get_price_intraday_fields_are_not_all_nan()
    test_unknown_history_field_is_nan_not_close()
    test_current_data_fields()
    test_query_dsl_filters_and_sort()
    test_query_in_filter()
    test_index_proxy_rules()
    test_index_proxy_excludes_unlisted()
    test_financials_asof_is_point_in_time()
    test_profit_glob_is_limited_to_profit_sheet()
    test_financials_asof_has_values_when_db_present()
    test_etf_close_prefers_real_then_synthetic()
    test_positions_returns_zero_for_unheld()
    test_get_price_supports_index_and_etf()
    test_legacy_frame_pandas1_semantics()
    test_end_to_end_mini_strategy()
    test_engine_survives_strategy_exception()
    print("\n全部聚宽兼容层测试通过")
