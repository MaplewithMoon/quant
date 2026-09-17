# -*- coding: utf-8 -*-
"""风格分解与残差 alpha 测试（T1·④）

【为什么这条重要】
A3 的结论是「Alpha +2.40%，p=0.646，Beta 0.886，池子偏中小盘」——
**这个 alpha 混着风格暴露**，"选股能力"和"小盘 beta"在那个数字里分不开。
把风格解释掉、看**残差 alpha**，才是对选股能力的检验。

本文件用一个"**只有小盘 beta、没有任何真 alpha**"的合成组合来验证：
混口径 alpha 必须显著为正，残差 alpha 必须不显著。做不到这一点，
说明风格分解是装饰性的。
"""
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))


def _world(n=300, k=200, beta=-0.0012, seed=7):
    """收益只由 size 因子驱动，真 alpha = 0"""
    rng = np.random.default_rng(seed)
    idx = pd.bdate_range("2022-01-03", periods=n)
    codes = [f"{i:06d}" for i in range(1, k + 1)]
    size = pd.DataFrame(rng.normal(10, 1, (n, k)), index=idx, columns=codes)
    bp = pd.DataFrame(rng.normal(0.5, 0.2, (n, k)), index=idx, columns=codes)
    z = size.sub(size.mean(axis=1), axis=0).div(size.std(axis=1), axis=0)
    ret = beta * z + pd.DataFrame(rng.normal(0, 0.008, (n, k)),
                                  index=idx, columns=codes)
    return idx, codes, size, bp, z, ret


def test_cross_section_factor_returns_recovers_loading():
    """横截面回归应还原出已知的因子载荷"""
    from analytics.style import cross_section_factor_returns

    _, _, size, bp, _, ret = _world(beta=-0.0012)
    B = cross_section_factor_returns(ret, {"size": size, "value": bp},
                                     min_stocks=50)
    assert B.shape[1] == 2 and B["size"].notna().sum() > 200
    got = float(B["size"].mean())
    assert abs(got - (-0.0012)) < 0.0006, \
        f"size 因子收益应≈-0.0012，实得 {got:.4f}"
    # 真实因子是 size，value 的斜率应接近 0
    assert abs(float(B["value"].mean())) < 0.0006, \
        f"value 不该有载荷，实得 {B['value'].mean():.4f}"
    print(f"[OK] 横截面回归：size 载荷 {got:+.4f}（真值 -0.0012），"
          f"value 载荷 {float(B['value'].mean()):+.4f}（真值 0）")


def test_residual_alpha_strips_style():
    """**核心**：只有小盘 beta 的组合，混口径显著、残差必须不显著"""
    from analytics.style import cross_section_factor_returns, residual_alpha

    _, _, size, bp, z, ret = _world(beta=-0.0012)
    rank = size.rank(axis=1)
    w = (rank <= 20).astype(float)
    w = w.div(w.sum(axis=1), axis=0)          # 每天只持最小的 20 只
    sret = (w * ret).sum(axis=1)
    expo = pd.DataFrame({"size": (w * z).sum(axis=1), "value": 0.0})
    B = cross_section_factor_returns(ret, {"size": size, "value": bp},
                                     min_stocks=50)

    out = residual_alpha(sret, expo, B)
    mix, rsl = out["mixed"], out["residual"]
    assert mix["alpha_annual"] > 0.10, \
        f"混口径应显著为正（构造如此），实得 {mix['alpha_annual']:.2%}"
    assert abs(mix["t"]) > 5, f"混口径 t 应很大，实得 {mix['t']:.2f}"
    assert abs(rsl["alpha_annual"]) < abs(mix["alpha_annual"]) / 3, \
        f"残差没有剥掉风格：混 {mix['alpha_annual']:.2%} vs 残差 {rsl['alpha_annual']:.2%}"
    assert abs(rsl["t"]) < 3, \
        f"残差 alpha 不该显著（本就无真 alpha），实得 t={rsl['t']:.2f}"
    # 风格暴露应为负（偏小盘），贡献应能解释掉大部分收益
    assert out["exposure_mean"]["size"] < -1.0, out["exposure_mean"]
    assert out["contrib"]["size"] > 0.1, out["contrib"]
    print(f"[OK] 残差剥掉风格：混口径 {mix['alpha_annual']:+.1%}(t={mix['t']:.1f}) "
          f"-> 残差 {rsl['alpha_annual']:+.1%}(t={rsl['t']:.2f})；"
          f"size 暴露 {out['exposure_mean']['size']:+.2f}，"
          f"贡献 {out['contrib']['size']:+.1%}")


def test_build_style_panels_is_point_in_time():
    """风格面板必须 PIT：`value` 只能前向填充，不能拿未来财报回填"""
    from analytics.style import build_style_panels

    idx = pd.bdate_range("2024-01-01", periods=10)
    codes = ["000001", "600000"]
    panel = {"total_mv": pd.DataFrame(1e10, index=idx, columns=codes)}
    # bp 只在第 5 天有值（模拟财报公告），之前应为 NaN、之后保持
    bp = pd.DataFrame(np.nan, index=idx, columns=codes)
    bp.iloc[5] = [1.0, 2.0]
    panels = build_style_panels(panel, idx, codes, fundamentals={"bp": bp})
    v = panels["value"]
    assert v.iloc[:5].isna().all().all(), \
        "财报公告之前就有值 —— 那是拿未来数据回填（前视）"
    assert np.allclose(v.iloc[5:].to_numpy(), [[1.0, 2.0]] * 5), "前向填充失败"
    # size 直接来自面板
    assert np.allclose(panels["size"].iloc[0],
                       np.log([1e10, 1e10])), "size 应为 log(total_mv)"
    print("[OK] 风格面板 PIT：value 公告前为 NaN、之后前向填充；size 取 log(市值)")


def test_style_section_declares_main_metric():
    """报告必须说清**哪个是主口径**，否则读者会拿混口径当结论"""
    from analytics.style import style_section

    txt = "\n".join(style_section({
        "mixed": {"alpha_annual": 0.024, "t": 0.46, "p": 0.646, "n": 48},
        "residual": {"alpha_annual": 0.003, "t": 0.11, "p": 0.913, "n": 48},
        "exposure_mean": {"size": -0.85, "value": 0.12},
        "contrib": {"size": 0.021, "value": 0.002}}))
    assert "主口径" in txt and "参考" in txt, "没标出主口径/参考口径"
    assert "分不开" in txt, "没解释为什么要做风格分解"
    assert "市值" in txt and "价值" in txt, "没显示风格暴露"
    print("[OK] 风格段落：标明主口径=残差、混口径降为参考，并列暴露与贡献")


if __name__ == "__main__":
    test_cross_section_factor_returns_recovers_loading()
    test_residual_alpha_strips_style()
    test_build_style_panels_is_point_in_time()
    test_style_section_declares_main_metric()
    print("\n全部风格分解测试通过")
