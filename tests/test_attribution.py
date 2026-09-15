# -*- coding: utf-8 -*-
"""业绩归因测试：Brinson 恒等式、行业暴露、因子暴露；以及校验脚本的退出码"""
import os
import subprocess
import sys

import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from analytics.attribution import (brinson, factor_exposure, industry_exposure,
                                   multi_factor_exposure, _industry_returns)


def _setup(n_days=60, n_codes=40, seed=0):
    rng = np.random.default_rng(seed)
    dates = pd.bdate_range("2023-01-03", periods=n_days)
    codes = [f"{i:06d}" for i in range(1, n_codes + 1)]
    ret = pd.DataFrame(rng.normal(0.001, 0.02, (n_days, n_codes)),
                       index=dates, columns=codes)
    ind = pd.DataFrame({"code": codes,
                        "industry": ["A" if i < 20 else "B" for i in range(n_codes)]})
    return dates, codes, ret, ind


# ============================================================
# 行业暴露
# ============================================================
def test_industry_exposure_sums_to_one():
    dates, codes, ret, ind = _setup()
    hold = pd.DataFrame(np.nan, index=dates, columns=codes)
    for c in codes[:10]:
        hold[c] = 0.1
    exp = industry_exposure(hold, ind)
    assert not exp.empty
    tot = exp.sum(axis=1)
    assert np.allclose(tot.values, 1.0), f"行业权重应合计为 1，实际 {tot.values[:3]}"
    assert list(exp.columns) == ["A"]
    print(f"[OK] 行业暴露：{len(exp)} 天、合计均为 1，全部落在 A 行业")


def test_industry_exposure_two_industries():
    dates, codes, ret, ind = _setup()
    hold = pd.DataFrame(np.nan, index=dates, columns=codes)
    for c in codes[:5]:
        hold[c] = 0.1
    for c in codes[20:25]:
        hold[c] = 0.1
    exp = industry_exposure(hold, ind)
    assert set(exp.columns) == {"A", "B"}
    assert np.allclose(exp["A"].values, 0.5)
    assert np.allclose(exp["B"].values, 0.5)
    print("[OK] 行业暴露：两行业各 50%")


# ============================================================
# 因子暴露
# ============================================================
def test_factor_exposure_sign():
    dates, codes, ret, ind = _setup()
    hold = pd.DataFrame(np.nan, index=dates, columns=codes)
    # 只持有因子值最高的 5 只
    z = pd.DataFrame(np.tile(np.arange(len(codes)), (len(dates), 1)).astype(float),
                     index=dates, columns=codes)
    for c in codes[-5:]:
        hold[c] = 0.2
    exp = factor_exposure(hold, z)
    assert (exp > 0).all(), "持有高因子组时暴露应为正"
    hold2 = pd.DataFrame(np.nan, index=dates, columns=codes)
    for c in codes[:5]:
        hold2[c] = 0.2
    exp2 = factor_exposure(hold2, z)
    assert (exp2 < 0).all(), "持有低因子组时暴露应为负"
    multi = multi_factor_exposure(hold, {"z": z, "z2": z * 2})
    assert list(multi.columns) == ["z", "z2"]
    print(f"[OK] 因子暴露符号正确：高分组 {exp.iloc[0]:+.1f} / 低分组 {exp2.iloc[0]:+.1f}")


# ============================================================
# Brinson 恒等式（核心正确性）
# ============================================================
def test_brinson_identity():
    """Σ(配置 + 选股 + 交互) 必须等于 组合收益 − 基准收益

    这是 Brinson-Fachler 分解的定义性质，任何实现都必须满足。
    """
    dates, codes, ret, ind = _setup(n_days=80, n_codes=40, seed=1)
    hold = pd.DataFrame(np.nan, index=dates, columns=codes)
    for c in codes[:6]:                 # 组合：A 行业 6 只
        hold[c] = 1 / 6
    bw = pd.DataFrame(1.0 / len(codes), index=dates, columns=codes)   # 等权基准

    br = brinson(hold, bw, ret, ind)
    assert not br.empty, "应产出归因表"

    # 复现同一区间的 rp / rb（brinson 内部用 (t0, t1] 的收益）
    t0, t1 = dates[0], dates[-1]
    seg = ret.loc[(ret.index > t0) & (ret.index <= t1)]
    asset_ret = (1 + seg.fillna(0)).prod() - 1
    wp = hold.loc[t0].dropna()
    wb = bw.loc[t0].dropna()
    rp = float((wp * asset_ret.reindex(wp.index).fillna(0)).sum())
    rb = float((wb * asset_ret.reindex(wb.index).fillna(0)).sum())

    total = float(br.loc["总计", "合计"])
    assert abs(total - (rp - rb)) < 1e-9, (
        f"Brinson 恒等式不成立: Σ效应={total:.6%} vs 超额={rp - rb:.6%}")
    # 对账口径必须由 brinson 自己带出来，别让调用方去猜该跟什么比
    assert "sum_excess" in br.attrs and "n_periods" in br.attrs
    assert abs(float(br.attrs["sum_excess"]) - (rp - rb)) < 1e-9
    assert int(br.attrs["n_periods"]) >= 1
    alloc = float(br.loc["总计", "配置效应"])
    sel = float(br.loc["总计", "选股效应"])
    inter = float(br.loc["总计", "交互效应"])
    print(f"[OK] Brinson 恒等式成立: 配置 {alloc:+.4%} + 选股 {sel:+.4%} + "
          f"交互 {inter:+.4%} = {total:+.4%} == 超额 {rp - rb:+.4%}")


def test_brinson_pure_allocation():
    """组合与基准持有完全相同的股票、只有行业权重不同 -> 选股效应应为 0"""
    dates, codes, ret, ind = _setup(n_days=60, n_codes=40, seed=2)
    # 组合在 A 行业内部与基准同比例，只是 A/B 之间权重不同
    wA, wB = 0.8, 0.2
    hold = pd.DataFrame(np.nan, index=dates, columns=codes)
    for c in codes[:20]:
        hold[c] = wA / 20
    for c in codes[20:]:
        hold[c] = wB / 20
    bw = pd.DataFrame(1.0 / len(codes), index=dates, columns=codes)   # A/B 各 50%

    br = brinson(hold, bw, ret, ind)
    assert not br.empty
    sel = float(br.loc["总计", "选股效应"])
    assert abs(sel) < 1e-12, f"同比例持有下选股效应应为 0，实际 {sel:.3e}"
    alloc = float(br.loc["总计", "配置效应"])
    assert abs(alloc) > 0, "行业权重不同，配置效应不应为 0"
    print(f"[OK] 纯配置情形：选股效应 = {sel:.2e}，配置效应 = {alloc:+.4%}")


def test_brinson_with_unmapped_stocks_keeps_identity():
    """回归：持仓里有"行业表里查不到"的股票时，恒等式仍须成立

    真实数据里这是常态 —— `industry_map` 是**当前**快照，已退市的股票不在里面。
    早期实现让这些股票的行业为 NaN，`groupby` 直接把它们丢掉，
    于是拆解只覆盖了一部分权重：实测全市场组合总计只有 −131%，
    而真实超额是 −108%，恒等式崩掉。
    """
    dates, codes, ret, ind = _setup(n_days=60, n_codes=40, seed=3)
    # 只保留一半股票的行业映射，另一半查不到
    ind_partial = ind[ind["code"].isin(codes[:20])]
    hold = pd.DataFrame(np.nan, index=dates, columns=codes)
    for c in codes[:12]:
        hold[c] = 1 / 12
    bw = pd.DataFrame(1.0 / len(codes), index=dates, columns=codes)

    br = brinson(hold, bw, ret, ind_partial)
    assert not br.empty
    assert "未分类" in br.index, "查不到行业的持仓必须落到'未分类'，不能凭空消失"

    t0, t1 = dates[0], dates[-1]
    seg = ret.loc[(ret.index > t0) & (ret.index <= t1)]
    asset_ret = (1 + seg.fillna(0)).prod() - 1
    wp = hold.loc[t0].dropna()
    wb = bw.loc[t0].dropna()
    rp = float((wp * asset_ret.reindex(wp.index).fillna(0)).sum())
    rb = float((wb * asset_ret.reindex(wb.index).fillna(0)).sum())

    total = float(br.loc["总计", "合计"])
    assert abs(total - (rp - rb)) < 1e-9, (
        f"存在未分类持仓时恒等式被破坏: Σ效应={total:.6%} vs 超额={rp - rb:.6%}")
    print(f"[OK] 含未分类持仓时 Brinson 恒等式仍成立: {total:+.4%} == {rp - rb:+.4%}")


def test_brinson_holdings_columns_subset_of_benchmark():
    """回归：组合只持有少数股票（列空间远小于基准）时恒等式仍需成立

    真实场景：`MultiBacktestResult.holdings` 只含**实际持有过**的代码（几百只），
    而基准有 300 只成分股。早先只用 holdings.columns 建行业映射，
    基准里不在组合持仓中的股票行业为 NaN 被丢掉，基准侧 Σw < 1，
    恒等式破坏（实测逐期残差 1.6e-2）。
    """
    dates, codes, ret, ind = _setup(n_days=60, n_codes=60, seed=6)
    # 组合只持 6 只（列空间 = 6），基准是全 60 只
    hold = pd.DataFrame(np.nan, index=dates, columns=codes[:6])
    for c in codes[:6]:
        hold[c] = 1 / 6
    bw = pd.DataFrame(1.0 / len(codes), index=dates, columns=codes)

    br = brinson(hold, bw, ret, ind)
    assert not br.empty
    assert float(br.attrs["max_period_gap"]) < 1e-12, \
        f"逐期残差应≈0，实得 {br.attrs['max_period_gap']:.3e}"
    total = float(br.loc["总计", "合计"])
    assert abs(total - float(br.attrs["sum_excess"])) < 1e-12
    print(f"[OK] 组合列空间远小于基准时恒等式成立（组合 6 只 / 基准 60 只，"
          f"逐期残差 {br.attrs['max_period_gap']:.1e}）")


def test_industry_exposure_covers_all_weight():
    """行业暴露必须覆盖全部持仓权重（未分类也要算进去）"""
    dates, codes, ret, ind = _setup(n_days=30, n_codes=20, seed=4)
    hold = pd.DataFrame(np.nan, index=dates, columns=codes)
    for c in codes[:10]:
        hold[c] = 0.1
    ind_partial = ind[ind["code"].isin(codes[:3])]      # 只认识 3 只
    exp = industry_exposure(hold, ind_partial)
    assert not exp.empty
    tot = exp.sum(axis=1)
    assert np.allclose(tot.values, 1.0), f"暴露合计应为 1，实际 {tot.values[:3]}"
    assert "未分类" in exp.columns
    print(f"[OK] 行业暴露覆盖全部权重（含未分类 {exp['未分类'].iloc[0]:.0%}）")


def test_brinson_empty_inputs_are_safe():
    assert brinson(pd.DataFrame(), pd.DataFrame(), pd.DataFrame(),
                   pd.DataFrame()).empty
    dates, codes, ret, ind = _setup()
    br = brinson(pd.DataFrame(), pd.DataFrame(), ret, ind)
    assert br.empty
    print("[OK] Brinson 对空输入安全返回空表（不抛异常）")


# ============================================================
# validate_data 退出码（③）
# ============================================================
def test_validate_data_returns_nonzero_on_failure():
    """数据校验发现问题必须返回非零退出码，否则 CI/调度无法感知

    ⚠️ 这个测试**不能依赖真实 db**：CI 在干净环境里跑，没有 db/，
    早先的写法直接 `subprocess.run(validate_data.py)` 再断言退出码 0，
    在没有数据的机器上必然失败（"无文件"会被记为问题）。
    这里改成把三个检查函数打桩，只验证**退出码契约**本身：
        无问题 -> 0 ／ 有问题 -> 1 ／ --no-strict-exit -> 永远是 0
    """
    py = sys.executable

    def run(body: str):
        code = ("import sys; sys.path.insert(0,'.');"
                "import scripts.validate_data as v;"
                + body +
                "sys.exit(v.main())")
        return subprocess.run([py, "-c", code], cwd=ROOT, capture_output=True,
                              text=True, encoding="utf-8", errors="replace")

    stub_clean = ("v.check_uniqueness=lambda *a,**k: None;"
                  "v.check_schema=lambda *a,**k: None;"
                  "v.check_coverage=lambda *a,**k: None;"
                  "sys.argv=['validate_data.py','--check','all'];")

    r0 = run(stub_clean)
    assert r0.returncode == 0, f"无问题时应返回 0，实际 {r0.returncode}\n{r0.stderr[-400:]}"

    r1 = run(stub_clean + "v.WARN.append('injected');")
    assert r1.returncode == 1, f"发现问题时应返回 1，实际 {r1.returncode}\n{r1.stderr[-400:]}"

    r2 = run(stub_clean + "v.WARN.append('injected');"
             "sys.argv=['validate_data.py','--no-strict-exit'];")
    assert r2.returncode == 0, f"--no-strict-exit 应强制返回 0，实际 {r2.returncode}"

    print("[OK] validate_data 退出码契约：无问题 0 / 有问题 1 / --no-strict-exit 恒 0")


if __name__ == "__main__":
    test_industry_exposure_sums_to_one()
    test_industry_exposure_two_industries()
    test_factor_exposure_sign()
    test_brinson_identity()
    test_brinson_pure_allocation()
    test_brinson_with_unmapped_stocks_keeps_identity()
    test_brinson_holdings_columns_subset_of_benchmark()
    test_industry_exposure_covers_all_weight()
    test_brinson_empty_inputs_are_safe()
    test_validate_data_returns_nonzero_on_failure()
    print("\n全部归因测试通过")
