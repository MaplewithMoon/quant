# -*- coding: utf-8 -*-
"""股票池测试：历史成分正确性（幸存者偏差）、成分变动、多维过滤"""
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from universe import (UniverseSpec, build_universe, universe_size,
                      index_member_panel, index_weight_panel, load_index_members)


def _panel(n_days=120, n_codes=50, seed=0):
    rng = np.random.default_rng(seed)
    dates = pd.bdate_range("2023-01-03", periods=n_days)
    codes = [f"{i:06d}" for i in range(1, n_codes + 1)]
    close = pd.DataFrame(100 * np.cumprod(1 + rng.normal(0, 0.02, (n_days, n_codes)), axis=0),
                         index=dates, columns=codes)
    amount = pd.DataFrame(rng.uniform(1e6, 1e9, (n_days, n_codes)), index=dates, columns=codes)
    return {"close": close, "close_adj": close, "amount": amount, "volume": amount / 10}


# ============================================================
# 指数历史成分（核心：消除幸存者偏差）
# ============================================================
def test_index_member_count_is_correct():
    """沪深300 每日应恰好 300 只、中证500 恰好 500 只

    回归测试：早期实现用 pivot + ffill 逐列前向填充，会把已调出指数的股票
    永远保留，实测让沪深300 的成分数从 300 虚增到 865。
    """
    members = load_index_members("000300.SH")
    if members.empty:
        print("[SKIP] 无指数成分数据")
        return
    dates = pd.bdate_range("2023-01-03", periods=200)
    for code, n in (("000300.SH", 300), ("000905.SH", 500)):
        # ⚠️ 每个指数必须用**它自己**的成分全集做列空间；
        # 拿沪深300 的 code 全集去 reindex 中证500，会把不属于沪深300 的
        # 成员全部丢掉（实测 500 → 268），这不是数据问题而是掩码问题。
        mem = load_index_members(code)
        if mem.empty:
            continue
        codes = sorted(mem["code"].unique())
        m = index_member_panel(code, dates, codes)
        s = m.sum(axis=1)
        s = s[s > 0]
        if s.empty:
            continue
        assert s.max() == n, f"{code} 每日成分数应恰为 {n}，实际最大 {s.max()}"
        assert s.min() == n, f"{code} 每日成分数应恰为 {n}，实际最小 {s.min()}"
        print(f"[OK] {code} 每日成分数恒为 {n}")


def test_index_members_actually_change():
    """成分必须真的会变 —— 若被 ffill 粘住，就不会有人被调出"""
    members = load_index_members("000300.SH")
    if members.empty:
        print("[SKIP] 无指数成分数据")
        return
    dates = pd.bdate_range("2020-01-02", "2024-12-31")
    codes = sorted(members["code"].unique())
    m = index_member_panel("000300.SH", dates, codes)
    left = int((m.iloc[0] & ~m.iloc[-1]).sum())
    joined = int((~m.iloc[0] & m.iloc[-1]).sum())
    assert left > 0 and joined > 0, f"5 年间成分应发生变动，实际调出 {left} 调入 {joined}"
    print(f"[OK] 成分确实在换：5 年调出 {left} 只、调入 {joined} 只（未被前向填充粘住）")


def test_member_panel_uses_past_snapshot_only():
    """t 日只能使用 <= t 的快照，不能使用未来快照"""
    members = load_index_members("000300.SH")
    if members.empty:
        print("[SKIP] 无指数成分数据")
        return
    snaps = sorted(members["trade_date"].unique())
    if len(snaps) < 3:
        print("[SKIP] 快照太少")
        return
    s0, s1, s2 = snaps[-3], snaps[-2], snaps[-1]
    codes = sorted(members["code"].unique())
    # 取一个位于 s0 与 s1 之间（若没有则取 s0 当天）的交易日
    mid = s0 if s0 == pd.Timestamp(s0) else s0
    m = index_member_panel("000300.SH", pd.DatetimeIndex([s0]), codes)
    ref = set(members[(members["trade_date"] == s0)]["code"])
    got = set(np.array(codes)[m.iloc[0].values])
    assert got == ref, "s0 当日的成分应等于 s0 快照本身"
    # s0 当日的成分不应包含"只在 s2 才进入"的股票
    later = set(members[members["trade_date"] == s2]["code"]) - set(
        members[members["trade_date"] == s0]["code"])
    assert not (got & later), "使用了未来快照的成分"
    print(f"[OK] 只使用 <=t 的快照：{s0.date()} 成分 {len(got)} 只，未混入后续新进成分")


# ============================================================
# 过滤
# ============================================================
def test_universe_filters_reduce_mask():
    p = _panel(n_days=120, n_codes=50)
    base = build_universe(p, UniverseSpec())
    assert base.shape == p["close"].shape
    assert base.dtypes.iloc[0] == bool

    listed = build_universe(p, UniverseSpec(min_listed_days=60))
    assert listed.sum().sum() <= base.sum().sum(), "上市天数过滤不应放宽股票池"

    liquid = build_universe(p, UniverseSpec(min_amount=5e8))
    assert liquid.sum().sum() <= base.sum().sum()

    top = build_universe(p, UniverseSpec(top_n=10))
    s = universe_size(top)
    s = s[s > 0]
    assert s.max() <= 10, f"top_n=10 时每日最多 10 只，实际 {s.max()}"

    both = build_universe(p, UniverseSpec(min_listed_days=60, min_amount=5e8, top_n=10))
    assert both.sum().sum() <= top.sum().sum()
    print(f"[OK] 股票池过滤生效: 基础 {base.sum().sum():,} -> 上市/流动性/前N "
          f"{both.sum().sum():,} 个(日×股)")


def test_universe_first_day_is_empty():
    """第 0 天没有前一日收盘，无法计算收益，应全部排除"""
    p = _panel(n_days=20, n_codes=10)
    # 关掉 min_listed_days，隔离出"首日无前收"这一条规则
    # （默认 min_listed_days=60，在 20 天的合成数据里前 59 天都会因"上市不足"被剔除，
    #   两个规则混在一起就测不出到底是哪条在起作用）
    m = build_universe(p, UniverseSpec(min_listed_days=0))
    assert m.iloc[0].sum() == 0, "首日不应有可选股票（没有前收）"
    assert m.iloc[1].sum() > 0
    print("[OK] 首日无前收 -> 股票池为空（不会用未来数据凑）")

    # 反向确认：默认 min_listed_days=60 时，20 天样本内确实一只都不该选出来
    m60 = build_universe(p, UniverseSpec())
    assert m60.sum().sum() == 0, "上市不足 60 日应全部剔除"
    print("[OK] min_listed_days=60 对短样本全剔除（规则确实生效）")


def test_index_weight_panel_normalized():
    w = index_weight_panel("000300.SH", pd.bdate_range("2023-01-03", periods=20),
                           None) if False else None
    members = load_index_members("000300.SH")
    if members.empty:
        print("[SKIP] 无指数成分数据")
        return
    codes = sorted(members["code"].unique())
    w = index_weight_panel("000300.SH", pd.bdate_range("2023-01-03", periods=20), codes)
    tot = w.sum(axis=1).dropna()
    assert len(tot) > 0
    assert np.allclose(tot.values, 1.0), f"基准权重应归一化到 1，实际 {tot.values[:3]}"
    print(f"[OK] 基准权重面板归一化: {len(tot)} 个交易日合计均为 1")


if __name__ == "__main__":
    test_index_member_count_is_correct()
    test_index_members_actually_change()
    test_member_panel_uses_past_snapshot_only()
    test_universe_filters_reduce_mask()
    test_universe_first_day_is_empty()
    test_index_weight_panel_normalized()
    print("\n全部股票池测试通过")
