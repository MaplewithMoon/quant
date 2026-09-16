# -*- coding: utf-8 -*-
"""参数优化与 Walk-Forward 验证测试"""
import os
import sys

import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from optimizer.search import (ParamSpace, expand_space, sample_space,
                              grid_search, random_search, evaluate, overfit_warning)
from optimizer.walkforward import make_windows, walk_forward, slice_ds
from scripts.optimize import parse_space
from scripts.backtest import make_strategy
from mocks import make_dataset


def _factory(name="MA"):
    def f(params):
        return make_strategy(name, params)
    return f


# ============================================================
# 参数空间
# ============================================================
def test_parse_space():
    assert parse_space(["fast=5,10,20"]) == {"fast": [5, 10, 20]}
    assert parse_space(["fast=2:20"]) == {"fast": (2, 20)}
    assert parse_space(["k=0.1:1.0"]) == {"k": (0.1, 1.0)}
    assert parse_space(["fast=5,10", "slow=20:60"]) == {"fast": [5, 10], "slow": (20, 60)}
    print("[OK] 参数空间解析: 离散值 / 整数区间 / 浮点区间 / 多参数")


def test_parse_symbols_fixes_powershell_mangling():
    """PowerShell 会把 000001,600519 当数组并转成整数，前导零丢失"""
    from scripts.optimize import parse_symbols
    assert parse_symbols("1,600519,300750") == ["000001", "600519", "300750"]
    assert parse_symbols("000001,600519") == ["000001", "600519"]
    assert parse_symbols("000001 600519;300750") == ["000001", "600519", "300750"]
    assert parse_symbols("600519") == ["600519"]
    assert parse_symbols("") == []
    print("[OK] 股票代码解析：补回前导零，兼容逗号/空白/分号分隔")


def test_expand_space_is_cartesian():
    combos = expand_space({"fast": [5, 10], "slow": [20, 60]})
    assert len(combos) == 4
    assert {"fast": 5, "slow": 20} in combos
    assert {"fast": 10, "slow": 60} in combos
    # 整数区间展开
    combos2 = expand_space({"fast": (5, 7)})
    assert [c["fast"] for c in combos2] == [5, 6, 7]
    assert ParamSpace({"fast": [1, 2], "slow": (10, 12)}).size() == 6
    print("[OK] 笛卡尔积展开: 4 组 / 整数区间 3 档 / size=6")


def test_sample_space_dedup_and_count():
    s = sample_space({"fast": (2, 20), "slow": (20, 120)}, n_iter=25, seed=1)
    assert len(s) == 25
    assert len({tuple(sorted(p.items())) for p in s}) == 25, "不应有重复组合"
    s2 = sample_space({"fast": (2, 20), "slow": (20, 120)}, n_iter=25, seed=1)
    assert s == s2, "同一 seed 应可复现"
    # 离散空间采样不会越界
    s3 = sample_space({"fast": [5, 10]}, n_iter=3, seed=2)
    assert all(p["fast"] in (5, 10) for p in s3)
    print("[OK] 随机采样: 数量正确、无重复、同 seed 可复现、取值不越界")


# ============================================================
# 窗口切分
# ============================================================
def test_make_windows_tiles_without_overlap():
    idx = make_dataset(n=300, seed=1).data.index
    ws = make_windows(idx, train_days=100, test_days=50)
    assert len(ws) > 0
    for w in ws:
        assert w.train_end < w.test_start, "训练段必须早于测试段"
    for a, b in zip(ws, ws[1:]):
        assert a.test_end < b.test_start, "测试段之间不应重叠"
        assert b.test_start > a.test_end, "窗口必须向前推进"
    # 默认步长 = test_days，测试段应首尾相接
    for a, b in zip(ws, ws[1:]):
        gap = idx.get_loc(b.test_start) - idx.get_loc(a.test_end)
        assert gap == 1, f"测试段应相邻，实际间隔 {gap}"
    print(f"[OK] 窗口切分: {len(ws)} 个窗口，训练早于测试、测试段不重叠且首尾相接")


def test_make_windows_anchored_expands():
    idx = make_dataset(n=300, seed=1).data.index
    roll = make_windows(idx, train_days=100, test_days=50, anchored=False)
    anch = make_windows(idx, train_days=100, test_days=50, anchored=True)
    assert roll[0].train_start == anch[0].train_start
    assert anch[-1].train_start == idx[0], "anchored 模式下训练段起点固定"
    assert roll[-1].train_start > idx[0], "rolling 模式下训练段起点前移"
    print(f"[OK] anchored 模式训练段扩张: 起点保持 {anch[-1].train_start.date()}，"
          f"rolling 已前移到 {roll[-1].train_start.date()}")


def test_make_windows_insufficient_data():
    idx = make_dataset(n=50, seed=1).data.index
    assert make_windows(idx, train_days=100, test_days=50) == []
    print("[OK] 数据不足时返回空窗口列表（不报错）")


# ============================================================
# 搜索
# ============================================================
def test_grid_search_sorted_and_annotated():
    ds = make_dataset(n=400, seed=3)
    df = grid_search(ds, _factory(), {"fast": [5, 10], "slow": [20, 60]},
                     objective="sharpe", verbose=False)
    assert len(df) == 4
    assert list(df["_score"]) == sorted(df["_score"], reverse=True), "应按目标降序"
    assert df.attrs["n_trials"] == 4
    assert "⚠️" in overfit_warning(df)
    print("[OK] 网格搜索: 4 组结果按夏普降序，附数据窥探提示")


def test_random_search_count():
    ds = make_dataset(n=300, seed=4)
    df = random_search(ds, _factory(), {"fast": (2, 10), "slow": (20, 60)},
                       n_iter=8, objective="total_return", verbose=False)
    assert len(df) == 8
    assert df.attrs["n_trials"] == 8
    print("[OK] 随机搜索返回指定组数")


def test_evaluate_handles_bad_params():
    """非法参数组合不应让搜索崩掉"""
    ds = make_dataset(n=100, seed=5)

    class Boom:
        def __init__(self, **kw):
            raise ValueError("参数非法")

    r = evaluate(ds, lambda p: Boom(**p), {"x": 1}, objective="sharpe")
    assert r["_score"] == float("-inf")
    assert r["_error"], "应记录错误原因而不是抛出"
    print("[OK] 非法参数被记录为错误行，搜索不会中断")


# ============================================================
# Walk-Forward
# ============================================================
def test_walk_forward_structure():
    ds = make_dataset(n=500, seed=7)
    res = walk_forward(ds, _factory(), {"fast": [5, 10], "slow": [20, 40]},
                       train_days=150, test_days=50, mode="grid",
                       objective="sharpe", verbose=False)
    assert not res.windows.empty
    assert res.n_trials_per_window == 4
    # 拼接的样本外净值
    eq = res.oos_equity
    assert len(eq) > 0
    assert abs(eq.iloc[0] - 1.0) < 1e-9, "拼接净值应从 1.0 开始"
    assert eq.index.is_monotonic_increasing, "净值时间轴应单调递增"
    assert not eq.index.has_duplicates, "测试段不应重叠导致重复日期"
    # IS/OOS 都有统计
    assert res.is_sharpe_mean != 0 or res.oos_sharpe_mean != 0
    # 每窗口都应用了训练段选出的参数
    assert "最佳参数" in res.windows.columns
    assert res.windows["最佳参数"].str.len().gt(0).all()
    print(f"[OK] Walk-Forward: {len(res.windows)} 窗, 净值 {len(eq)} 点从 1.0 起，"
          f"IS={res.is_sharpe_mean:.3f} OOS={res.oos_sharpe_mean:.3f} 落差={res.overfit_gap():.3f}")


def test_walk_forward_no_train_test_leakage():
    """每个窗口的测试段日期必须晚于训练段，且不重叠"""
    ds = make_dataset(n=500, seed=8)
    res = walk_forward(ds, _factory(), {"fast": [5], "slow": [20]},
                       train_days=150, test_days=50, mode="grid", verbose=False)
    for _, r in res.windows.iterrows():
        tr_end = pd.Timestamp(r["训练区间"].split("~")[1])
        te_start = pd.Timestamp(r["测试区间"].split("~")[0])
        assert tr_end < te_start, f"窗口 {r['窗口']} 训练段未早于测试段"
    oos_dates = pd.to_datetime(
        [d for s in res.windows["测试区间"] for d in [s.split("~")[0]]])
    assert len(set(oos_dates)) == len(oos_dates), "测试段起点不应重复"
    print("[OK] 无训练/测试泄漏：训练段均早于测试段，测试区间互不重叠")


def test_walk_forward_summary_renders():
    ds = make_dataset(n=400, seed=9)
    res = walk_forward(ds, _factory(), {"fast": [5, 10], "slow": [20]},
                       train_days=120, test_days=40, mode="grid", verbose=False)
    s = res.summary()
    assert "样本内(IS)" in s and "样本外(OOS)" in s
    assert "过拟合落差" in s
    assert "拼接后的样本外业绩" in s
    print("[OK] summary 渲染完整（IS/OOS/落差/样本外业绩/参数稳定性）")


def test_slice_ds_bounds():
    ds = make_dataset(n=200, seed=10)
    lo, hi = ds.data.index[50], ds.data.index[99]
    sub = slice_ds(ds, lo, hi)
    assert len(sub) == 50
    assert sub.data.index[0] == lo and sub.data.index[-1] == hi
    assert sub.symbol == ds.symbol
    print("[OK] slice_ds 按日期闭区间切分正确")


if __name__ == "__main__":
    test_parse_space()
    test_parse_symbols_fixes_powershell_mangling()
    test_expand_space_is_cartesian()
    test_sample_space_dedup_and_count()
    test_make_windows_tiles_without_overlap()
    test_make_windows_anchored_expands()
    test_make_windows_insufficient_data()
    test_grid_search_sorted_and_annotated()
    test_random_search_count()
    test_evaluate_handles_bad_params()
    test_walk_forward_structure()
    test_walk_forward_no_train_test_leakage()
    test_walk_forward_summary_renders()
    test_slice_ds_bounds()
    print("\n全部参数优化/样本外验证测试通过")
