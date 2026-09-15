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
    """数据校验失败必须返回非零退出码，否则 CI/调度无法感知"""
    py = sys.executable
    r = subprocess.run([py, "scripts/validate_data.py", "--check", "unique"],
                       cwd=ROOT, capture_output=True, text=True,
                       encoding="utf-8", errors="replace")
    assert r.returncode == 0, f"数据正常时应返回 0，实际 {r.returncode}\n{r.stderr[-400:]}"

    # 人为注入一个问题，验证确实返回 1
    code = (
        "import sys; sys.path.insert(0,'.');"
        "import scripts.validate_data as v;"
        "v.WARN.append('injected');"
        "sys.argv=['validate_data.py','--check','unique'];"
        "sys.exit(v.main())"
    )
    r2 = subprocess.run([py, "-c", code], cwd=ROOT, capture_output=True, text=True,
                        encoding="utf-8", errors="replace")
    assert r2.returncode == 1, f"发现问题时应返回 1，实际 {r2.returncode}\n{r2.stderr[-400:]}"
    print("[OK] validate_data 退出码：正常 0 / 发现问题 1（可被 CI 与调度感知）")


if __name__ == "__main__":
    test_industry_exposure_sums_to_one()
    test_industry_exposure_two_industries()
    test_factor_exposure_sign()
    test_brinson_identity()
    test_brinson_pure_allocation()
    test_brinson_empty_inputs_are_safe()
    test_validate_data_returns_nonzero_on_failure()
    print("\n全部归因测试通过")
