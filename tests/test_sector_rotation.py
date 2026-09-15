# -*- coding: utf-8 -*-
"""板块轮动策略测试：板块合成、动量、选板块、目标权重、无未来函数"""
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from strategy.sector_rotation import (SectorData, SectorRotationSpec,
                                      build_rotation_weights, load_sector_map,
                                      sector_holding_history, sector_index,
                                      sector_momentum, sector_return_panel,
                                      select_sectors)
from portfolio.construction import rebalance_dates


def _panel(n_days=300, n_codes=40, n_sectors=4, seed=0, drift=None):
    """合成面板：n_codes 只股票均分到 n_sectors 个板块

    drift: {板块序号: 日漂移}，用于制造"某板块明显更强"
    """
    rng = np.random.default_rng(seed)
    dates = pd.bdate_range("2022-01-03", periods=n_days)
    codes = [f"{i:06d}" for i in range(1, n_codes + 1)]
    sectors = [f"S{i}" for i in range(n_sectors)]
    sec_of = {c: sectors[i % n_sectors] for i, c in enumerate(codes)}
    sector_map = pd.Series(sec_of)

    ret = rng.normal(0.0, 0.01, (n_days, n_codes))
    if drift:
        for s_idx, mu in drift.items():
            cols = [i for i, c in enumerate(codes) if sec_of[c] == sectors[s_idx]]
            ret[:, cols] += mu
    close = pd.DataFrame(100 * np.cumprod(1 + ret, axis=0),
                         index=dates, columns=codes)
    amount = pd.DataFrame(rng.uniform(1e7, 1e9, (n_days, n_codes)),
                          index=dates, columns=codes)
    return ({"close": close, "close_adj": close, "amount": amount,
             "open": close, "high": close * 1.01, "low": close * 0.99,
             "volume": amount / 10},
            sector_map, codes, sectors)


# ============================================================
# 板块合成
# ============================================================
def test_sector_return_is_equal_weight_mean():
    """板块收益 = 成分股等权平均"""
    p, smap, codes, sectors = _panel(n_days=50, n_codes=8, n_sectors=2)
    sret = sector_return_panel(p, smap)
    assert list(sret.columns) == ["S0", "S1"]
    ret = p["close_adj"].pct_change(fill_method=None).iloc[1]
    for sec in sectors:
        members = [c for c in codes if smap[c] == sec]
        expect = ret[members].mean()
        got = sret[sec].iloc[1]
        assert abs(got - expect) < 1e-12, f"{sec} 应为等权均值"
    print("[OK] 板块收益 = 成分股等权平均（逐票核对）")


def test_sector_min_stocks_filter():
    """板块内有效股票不足时该日置 NaN"""
    p, smap, codes, sectors = _panel(n_days=30, n_codes=8, n_sectors=2)
    # 把 S0 的股票在最后 5 天全部置 NaN
    members = [c for c in codes if smap[c] == "S0"]
    for c in members:
        p["close_adj"].loc[p["close_adj"].index[-5:], c] = np.nan
    sret = sector_return_panel(p, smap, min_stocks=3)
    assert sret["S0"].iloc[-3:].isna().all(), "成分股不足应置 NaN"
    assert sret["S1"].iloc[-3:].notna().all(), "S1 不受影响"
    print("[OK] 板块成分股不足 -> 该板块当日不可用（不会被 3 只票代表）")


def test_sector_index_compounds():
    p, smap, codes, sectors = _panel(n_days=60, n_codes=8, n_sectors=2)
    sret = sector_return_panel(p, smap)
    idx = sector_index(sret)
    r = sret["S0"].fillna(0.0)
    expect = (1 + r).cumprod()
    assert np.allclose(idx["S0"].values, expect.values)
    print("[OK] 板块指数 = 日收益复利累乘")


# ============================================================
# 动量
# ============================================================
def test_momentum_ranks_strong_sector_first():
    """给 S2 加正漂移，动量应把它排到第一"""
    p, smap, codes, sectors = _panel(n_days=300, n_codes=40, n_sectors=4,
                                     seed=3, drift={2: 0.002})
    mom = sector_momentum(p, smap, lookback=60, skip=5, vol_adjust=False)
    last = mom.iloc[-1].dropna()
    assert last.idxmax() == "S2", f"强势板块应排第一，实得 {last.idxmax()} ({last.to_dict()})"
    print(f"[OK] 动量能识别强势板块: {last.round(4).to_dict()}")


def test_momentum_skip_window():
    """skip 的作用：最后 skip 天的暴跌不应影响打分"""
    p, smap, codes, sectors = _panel(n_days=200, n_codes=20, n_sectors=2, seed=5)
    base = sector_momentum(p, smap, lookback=60, skip=5, vol_adjust=False)
    # 把最后 3 天（< skip）的 S0 打成暴跌
    mem = [c for c in codes if smap[c] == "S0"]
    p2 = {k: (v.copy() if isinstance(v, pd.DataFrame) else v) for k, v in p.items()}
    for c in mem:
        p2["close_adj"].loc[p2["close_adj"].index[-3:], c] *= 0.5
    after = sector_momentum(p2, smap, lookback=60, skip=5, vol_adjust=False)
    a, b = base["S0"].iloc[-1], after["S0"].iloc[-1]
    assert abs(a - b) < 1e-9, f"skip=5 时最后 3 天不应影响打分: {a} vs {b}"
    # skip=0 时就会受影响
    after0 = sector_momentum(p2, smap, lookback=60, skip=0, vol_adjust=False)
    assert abs(after0["S0"].iloc[-1] - a) > 1e-6, "skip=0 时应被最近的暴跌影响"
    print("[OK] skip 窗口生效：最近 3 天的暴跌不影响 skip=5 的打分")


def test_momentum_has_no_lookahead():
    """核心回归：篡改未来价格后，历史的动量值必须一字不变"""
    p, smap, codes, sectors = _panel(n_days=250, n_codes=20, n_sectors=3, seed=9)
    mom = sector_momentum(p, smap, lookback=40, skip=3, vol_adjust=True)
    cut = 150
    p2 = {k: (v.copy() if isinstance(v, pd.DataFrame) else v) for k, v in p.items()}
    for k in ("close", "close_adj"):
        p2[k].iloc[cut:] = p2[k].iloc[cut:] * 3.0     # 未来价格 ×3
    mom2 = sector_momentum(p2, smap, lookback=40, skip=3, vol_adjust=True)
    hist_a = mom.iloc[:cut].fillna(-999)
    hist_b = mom2.iloc[:cut].fillna(-999)
    assert np.allclose(hist_a.values, hist_b.values), \
        "第 %d 天之前的动量被未来的价格改变了 —— 存在未来函数" % cut
    # 而切点之后确实变了（否则这个测试没有意义）
    assert not np.allclose(mom.iloc[cut:].fillna(-999).values,
                           mom2.iloc[cut:].fillna(-999).values)
    print(f"[OK] 无未来函数：篡改第 {cut} 天后的价格，之前 {cut} 天的动量完全不变")


def test_sector_data_cache_matches_direct():
    """SectorData 预计算必须与直接计算完全一致（否则缓存会引入偏差）"""
    p, smap, codes, sectors = _panel(n_days=200, n_codes=20, n_sectors=3, seed=4)
    sd = SectorData.build(p, smap, min_stocks=5)
    for lb, sk, va in ((20, 0, False), (60, 5, True), (120, 10, True)):
        direct = sector_momentum(p, smap, lookback=lb, skip=sk, vol_adjust=va)
        cached = sd.momentum(lookback=lb, skip=sk, vol_adjust=va, min_stocks=5)
        assert np.allclose(direct.fillna(-999).values, cached.fillna(-999).values), \
            f"缓存与直算不一致 (lookback={lb}, skip={sk}, vol_adjust={va})"
    print("[OK] SectorData 缓存与直接计算逐值一致")


# ============================================================
# 选板块 / 目标权重
# ============================================================
def test_select_sectors_topk_and_guard():
    p, smap, codes, sectors = _panel(n_days=200, n_codes=40, n_sectors=6,
                                     seed=2, drift={1: 0.001, 4: 0.0008})
    mom = sector_momentum(p, smap, lookback=60, skip=5, vol_adjust=False)
    sel = select_sectors(mom, top_k=2, min_sectors=1)
    assert sel.sum(axis=1).max() <= 2, "最多只能选 2 个板块"
    last = mom.iloc[-1].dropna().sort_values(ascending=False)
    assert bool(sel.iloc[-1][last.index[0]]), "最强板块应被选中"
    assert bool(sel.iloc[-1][last.index[1]]), "次强板块应被选中"
    # min_sectors 保护：有效板块数不足时不开仓
    sel_guard = select_sectors(mom, top_k=2, min_sectors=99)
    assert sel_guard.sum().sum() == 0, "有效板块不足 min_sectors 时应全部不选"
    print("[OK] 选板块：只取前 K、且有效板块不足时不开仓")


def test_rotation_weights_structure():
    """目标权重：调仓日和为 1、非调仓日为 NaN、只买掩码内的票"""
    p, smap, codes, sectors = _panel(n_days=200, n_codes=40, n_sectors=4, seed=6)
    mask = pd.DataFrame(True, index=p["close"].index, columns=codes)
    mask.iloc[:, :5] = False                     # 前 5 只永久剔除
    spec = SectorRotationSpec(lookback=60, skip=5, top_sectors=2,
                              stocks_per_sector=3, rebalance="M", min_sectors=1)
    plan = build_rotation_weights(p, mask, smap, spec)
    w = plan.weights
    reb = w.dropna(how="all")
    assert len(reb) > 0, "应产生调仓"
    assert np.allclose(reb.sum(axis=1).values, 1.0), "调仓日权重和必须为 1"
    assert w.notna().sum().sum() == reb.notna().sum().sum(), "非调仓日必须全 NaN"
    assert reb[codes[:5]].fillna(0).values.sum() == 0, "不能买掩码外的股票"
    n = reb.notna().sum(axis=1)
    assert n.max() <= 6, f"最多 2 板块 × 3 只 = 6 只，实得 {n.max()}"
    # 调仓日必须落在月初
    assert all(pd.Timestamp(d).day <= 7 for d in reb.index)
    print(f"[OK] 目标权重结构正确：{len(reb)} 次调仓、持股 {int(n.min())}~{int(n.max())} 只、"
          f"权重和=1、非调仓日全 NaN、掩码外 0 仓位")


def test_rotation_only_holds_selected_sectors():
    """持股必须全部落在当期选中的板块里"""
    p, smap, codes, sectors = _panel(n_days=200, n_codes=40, n_sectors=4, seed=8)
    mask = pd.DataFrame(True, index=p["close"].index, columns=codes)
    spec = SectorRotationSpec(lookback=60, skip=5, top_sectors=1,
                              stocks_per_sector=4, rebalance="M", min_sectors=1)
    plan = build_rotation_weights(p, mask, smap, spec)
    for d in plan.weights.dropna(how="all").index:
        held = plan.weights.loc[d].dropna().index
        selected = set(plan.momentum.loc[d].dropna()
                       .sort_values(ascending=False).index[:1])
        held_sectors = set(smap[c] for c in held)
        assert held_sectors <= selected, \
            f"{str(d)[:10]} 持有了未选中的板块 {held_sectors - selected}"
    print("[OK] 持股板块 ⊆ 当期选中板块（轮动逻辑闭合）")


def test_sector_weighting_momentum_normalized():
    """板块按动量加权：权重和为 1，强弱有别但不退化成单板块押注"""
    p, smap, codes, sectors = _panel(n_days=200, n_codes=40, n_sectors=4,
                                     seed=1, drift={0: 0.0025})
    mask = pd.DataFrame(True, index=p["close"].index, columns=codes)
    spec = SectorRotationSpec(lookback=60, skip=5, top_sectors=3,
                              stocks_per_sector=5, rebalance="M",
                              sector_weighting="momentum", min_sectors=1)
    plan = build_rotation_weights(p, mask, smap, spec)
    reb = plan.weights.dropna(how="all")
    assert np.allclose(reb.sum(axis=1).values, 1.0), "动量加权后仍须归一化"

    # ⚠️ 必须 dropna：reb 是完整宽表，未持仓的票是 NaN，
    # 直接 groupby(...).sum() 会把这些板块算成 0 权重（那是 NaN 求和，不是真 0）
    last = reb.iloc[-1].dropna()
    sec_w = last.groupby(smap.reindex(last.index)).sum().sort_values(ascending=False)
    assert abs(sec_w.sum() - 1.0) < 1e-9
    # 关键回归：最弱板块不能被压成 ~0（早期实现把它打到 1e-9，等于单板块押注）
    assert sec_w.min() > 0.05, f"最弱板块权重过低 {sec_w.min():.4f}，策略退化成单板块押注"
    assert sec_w.max() / sec_w.min() < 10, \
        f"板块权重极差过大 {sec_w.max()/sec_w.min():.1f} 倍，应被抬底约束"
    assert sec_w.iloc[0] > sec_w.iloc[-1], "强板块权重应更高"
    print(f"[OK] 板块动量加权：权重和=1，极差 {sec_w.max()/sec_w.min():.2f} 倍 "
          f"（最弱 {sec_w.min():.3f}，未退化为单板块）")


def test_rotation_survives_empty_mask():
    """股票池全空时不应崩溃，只应空仓"""
    p, smap, codes, sectors = _panel(n_days=200, n_codes=20, n_sectors=2, seed=0)
    mask = pd.DataFrame(False, index=p["close"].index, columns=codes)
    plan = build_rotation_weights(p, mask, smap,
                                  SectorRotationSpec(lookback=60, skip=5,
                                                     rebalance="M", min_sectors=1))
    assert plan.weights.notna().sum().sum() == 0, "全空掩码应产出空仓"
    print("[OK] 股票池全空 -> 空仓且不抛异常")


def test_sector_momentum_ic_sign():
    """板块动量 IC：正 IC 表示动量有效，负 IC 表示反转有效

    直接合成**板块收益序列**（不走股票面板），精确控制两种世界：

      动量世界：每个板块有一个固定的"质量"，日收益 = 质量 + 噪声。
                质量长期不变 -> 过去涨的未来继续涨 -> IC 应显著为正。
      反转世界：板块强弱**每月翻转**一次，且板块间错开相位。
                过去一个月的强势板块下个月转弱 -> IC 应显著为负。

    ⚠️ 两个踩过的坑：
      1. 不能用"把日收益取反"构造反转 —— 复利指数不是原指数的倒数，IC 不会变号。
      2. 不能用日频 AR(1) 构造动量 —— 打分窗口与预测窗口之间隔了 skip=5 天，
         自相关 0.3^5≈0.002，几乎无关，实测 IC≈−0.03，测不出方向。
    """
    from strategy.sector_rotation import sector_momentum_ic

    dates = pd.bdate_range("2015-01-05", periods=2600)
    n_sec = 10
    cols = [f"S{i}" for i in range(n_sec)]

    # ---- 动量世界：质量长期不变 ----
    rng = np.random.default_rng(5)
    q = rng.normal(0, 1, n_sec)

    def quality_model(seed: int) -> pd.DataFrame:
        r = np.random.default_rng(seed).normal(0, 0.003, (len(dates), n_sec))
        return pd.DataFrame(0.003 * q[None, :] + r, index=dates, columns=cols)

    # ---- 反转世界：强弱每月翻转，板块间错相位 ----
    month_no = pd.factorize(dates.year * 12 + dates.month)[0]
    stagger = np.arange(n_sec)
    sign = np.where(((month_no[:, None] + stagger[None, :]) % 2) == 0, 1.0, -1.0)

    def reversal_model(seed: int) -> pd.DataFrame:
        r = np.random.default_rng(seed).normal(0, 0.004, sign.shape)
        return pd.DataFrame(0.004 * sign + r, index=dates, columns=cols)

    mom = sector_momentum_ic(quality_model(11), lookback=60, skip=5,
                             rebalance="M", min_sectors=4)
    rev = sector_momentum_ic(reversal_model(12), lookback=60, skip=5,
                             rebalance="M", min_sectors=4)
    assert len(mom) > 40 and len(rev) > 40
    assert mom.mean() > 0.5, f"动量世界应给出强正 IC，实得 {mom.mean():+.4f}"
    assert rev.mean() < -0.5, f"反转世界应给出强负 IC，实得 {rev.mean():+.4f}"
    print(f"[OK] 板块动量 IC 方向正确: 动量世界 {mom.mean():+.4f} / "
          f"反转世界 {rev.mean():+.4f}（各 {len(mom)} 期）")


def test_sector_momentum_ic_no_lookahead():
    """IC 计算也不能用未来数据：篡改后半段不应改变前半段的 IC"""
    from strategy.sector_rotation import sector_momentum_ic
    p, smap, codes, sectors = _panel(n_days=500, n_codes=30, n_sectors=5, seed=13)
    sret = sector_return_panel(p, smap, min_stocks=1)
    ic = sector_momentum_ic(sret, lookback=60, skip=5, rebalance="M", min_sectors=4)
    cut = sret.index[300]
    sret2 = sret.copy()
    sret2.loc[sret2.index >= cut] *= 5.0
    ic2 = sector_momentum_ic(sret2, lookback=60, skip=5, rebalance="M", min_sectors=4)
    a = ic[ic.index < cut].dropna()
    b = ic2[ic2.index < cut].dropna()
    assert len(a) > 3
    assert np.allclose(a.values, b.reindex(a.index).values), \
        "篡改未来收益后，历史 IC 发生了变化 —— 存在未来函数"
    print(f"[OK] 板块动量 IC 无未来函数（前 {len(a)} 期不受未来数据影响）")


def test_holding_history_shape():
    p, smap, codes, sectors = _panel(n_days=200, n_codes=40, n_sectors=4, seed=2)
    mask = pd.DataFrame(True, index=p["close"].index, columns=codes)
    plan = build_rotation_weights(p, mask, smap,
                                  SectorRotationSpec(lookback=60, skip=5,
                                                     top_sectors=2, rebalance="M",
                                                     min_sectors=1))
    hist = sector_holding_history(plan)
    assert not hist.empty
    assert hist.values.sum() > 0
    print(f"[OK] 轮动轨迹表: {hist.shape[0]} 期 × {hist.shape[1]} 个被选中的板块")


if __name__ == "__main__":
    test_sector_return_is_equal_weight_mean()
    test_sector_min_stocks_filter()
    test_sector_index_compounds()
    test_momentum_ranks_strong_sector_first()
    test_momentum_skip_window()
    test_momentum_has_no_lookahead()
    test_sector_data_cache_matches_direct()
    test_select_sectors_topk_and_guard()
    test_rotation_weights_structure()
    test_rotation_only_holds_selected_sectors()
    test_sector_weighting_momentum_normalized()
    test_rotation_survives_empty_mask()
    test_sector_momentum_ic_sign()
    test_sector_momentum_ic_no_lookahead()
    test_holding_history_shape()
    print("\n全部板块轮动策略测试通过")
