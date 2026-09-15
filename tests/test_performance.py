# -*- coding: utf-8 -*-
"""绩效指标测试：alpha/beta 回归、t 分布 p 值、回撤、分年度收益"""
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from analytics.performance import (alpha_beta, align, annual_return,
                                   annual_returns_table, annual_volatility,
                                   capture_ratios, cum_return, drawdown_info,
                                   drawdown_series, excess_return,
                                   information_ratio, monthly_returns_table,
                                   performance_summary, sharpe_ratio,
                                   sortino_ratio, t_pvalue, to_returns,
                                   total_return, tracking_error, _betainc)


def _series(vals, start="2020-01-01"):
    return pd.Series(vals, index=pd.bdate_range(start, periods=len(vals)),
                     dtype=float)


# ============================================================
# 不完全贝塔函数 / t 分布 p 值
# ============================================================
def test_betainc_known_values():
    """I_x(a,b) 对照解析解

    早期实现把 -lnB(a,b) 当成 lnB(a,b) 用（指数里符号写反），
    I_0.5(2,3) 算出 0.9978（正确 0.6875），所有 p 值全错。
    """
    cases = [(1, 1, 0.5, 0.5), (2, 3, 0.5, 0.6875), (2, 2, 0.3, 0.216),
             (1, 1, 0.25, 0.25)]
    for a, b, x, exp in cases:
        got = _betainc(a, b, x)
        assert abs(got - exp) < 1e-9, f"I_{x}({a},{b}) 应={exp}，实得 {got}"
    print("[OK] 不完全贝塔函数与解析解一致")


def test_t_pvalue_matches_table():
    """t 分布双尾 p 值对照统计表"""
    cases = [
        (1.960, 1000000, 0.0500),   # 大样本 -> 正态
        (2.576, 1000000, 0.0100),
        (2.228, 10, 0.0500),        # df=10 的 95% 分位点
        (3.169, 10, 0.0100),        # df=10 的 99% 分位点
        (1.000, 10, 0.3410),
    ]
    for t, df, exp in cases:
        p = t_pvalue(t, df)
        assert abs(p - exp) < 2e-3, f"t={t},df={df} 的 p 应≈{exp}，实得 {p:.4f}"
    # 分位点自洽：t 的 95% 分位 -> p 恰为 0.05
    assert abs(t_pvalue(2.228, 10) - 0.05) < 1e-3
    print("[OK] t 分布 p 值与统计表一致（含 df=10 的 95%/99% 分位点）")


# ============================================================
# alpha / beta
# ============================================================
def test_alpha_beta_recovers_known_values():
    """构造 r_p = 0.0005 + 1.3 * r_b + 噪声，回归应还原出来"""
    rng = np.random.default_rng(7)
    n = 3000
    rb = pd.Series(rng.normal(0.0003, 0.012, n))
    rp = 0.0005 + 1.3 * rb + pd.Series(rng.normal(0, 0.003, n))
    ab = alpha_beta(rp, rb)
    assert abs(ab.beta - 1.3) < 0.03, f"beta 应≈1.3，实得 {ab.beta:.4f}"
    assert abs(ab.alpha_annual - 0.0005 * 252) < 0.03, \
        f"年化 alpha 应≈{0.0005*252:.4f}，实得 {ab.alpha_annual:.4f}"
    assert ab.r_squared > 0.9
    assert ab.alpha_pvalue < 0.01, "真 alpha 非零，应显著"
    print(f"[OK] alpha/beta 回归还原真值: beta={ab.beta:.3f}(真1.3) "
          f"alpha_ann={ab.alpha_annual:+.2%}(真{0.0005*252:+.2%}) R²={ab.r_squared:.3f}")


def test_alpha_beta_perfect_replication():
    """完全复制基准：beta=1、alpha=0、R²=1"""
    rng = np.random.default_rng(0)
    rb = pd.Series(rng.normal(0.0004, 0.01, 800))
    ab = alpha_beta(rb, rb)
    assert abs(ab.beta - 1.0) < 1e-9
    assert abs(ab.alpha_annual) < 1e-12
    assert abs(ab.r_squared - 1.0) < 1e-9
    assert abs(ab.alpha_tstat) < 1e-6
    print("[OK] 完全复制基准 -> beta=1, alpha=0, R²=1（回归实现自洽）")


def test_alpha_beta_zero_beta():
    """与基准无关的组合：beta≈0，alpha≈自身均值"""
    rng = np.random.default_rng(1)
    rb = pd.Series(rng.normal(0.0003, 0.012, 2000))
    rp = pd.Series(rng.normal(0.0008, 0.010, 2000))
    ab = alpha_beta(rp, rb)
    assert abs(ab.beta) < 0.06, f"独立序列 beta 应≈0，实得 {ab.beta:.4f}"
    assert abs(ab.alpha_annual - 0.0008 * 252) < 0.05
    print(f"[OK] 独立序列 beta≈0 ({ab.beta:+.4f})，alpha 反映自身收益")


# ============================================================
# 收益 / 风险
# ============================================================
def test_total_and_annual_return_compound():
    r = _series([0.10, -0.10])
    assert abs(total_return(r) - (1.1 * 0.9 - 1)) < 1e-12
    # 一年（252 天）恒定日收益 -> 年化就等于复利
    r2 = _series([0.001] * 252)
    assert abs(annual_return(r2) - (1.001 ** 252 - 1)) < 1e-9
    print("[OK] 累计/年化收益按复利计算")


def test_sharpe_and_sortino_scale():
    """恒定收益序列的波动为 0，夏普应退化为 0（不是 inf）"""
    flat = _series([0.001] * 100)
    assert annual_volatility(flat) < 1e-12, "常数序列的波动应≈0"
    assert sharpe_ratio(flat) == 0.0, "零波动时夏普应退化为 0，不能是天文数字"
    assert sortino_ratio(flat) == 0.0, "零下行波动时索提诺同样应退化"
    # 只有下行波动时索提诺分母非零
    rng = np.random.default_rng(3)
    r = pd.Series(rng.normal(0.001, 0.01, 1000))
    assert sortino_ratio(r) > sharpe_ratio(r), "正偏收益下索提诺应高于夏普"
    print("[OK] 夏普/索提诺：零波动退化安全，索提诺对右偏更友好")


def test_drawdown_info_exact():
    """手工构造已知回撤：100 -> 150 -> 75 -> 160"""
    eq = _series([100, 150, 75, 160])
    dd = drawdown_info(eq)
    assert abs(dd.max_drawdown - (-0.5)) < 1e-12, f"应 -50%，实得 {dd.max_drawdown}"
    assert str(dd.peak_date)[:10] == str(eq.index[1])[:10]
    assert str(dd.trough_date)[:10] == str(eq.index[2])[:10]
    assert str(dd.recover_date)[:10] == str(eq.index[3])[:10]
    assert dd.duration_days == 1
    assert dd.recover_days == 2
    # 未收复的情形
    eq2 = _series([100, 150, 75, 80])
    dl = drawdown_series(eq2)
    assert abs(dl.iloc[-1] - (80 / 150 - 1)) < 1e-12
    print("[OK] 回撤明细：幅度/峰谷/收复日期/持续期全部精确")


def test_annual_returns_table():
    """跨年数据分年度复利，超额 = 组合 - 基准"""
    n = 252 * 2
    rp = _series([0.001] * n, "2023-01-02")
    rb = _series([0.0005] * n, "2023-01-02")
    tbl = annual_returns_table(rp, rb)
    assert list(tbl.index) == [2023, 2024]
    assert "超额" in tbl.columns
    for y in tbl.index:
        assert abs(tbl.loc[y, "超额"] - (tbl.loc[y, "组合"] - tbl.loc[y, "基准"])) < 1e-12
    mon = monthly_returns_table(rp)
    assert mon.shape[1] == 12
    print("[OK] 分年度/分月收益表：复利正确、超额可加")


def test_relative_metrics():
    """跟踪误差/信息比率/捕获率的定义自洽"""
    rng = np.random.default_rng(11)
    rb = pd.Series(rng.normal(0.0002, 0.01, 1500))
    # 组合 = 基准 + 恒定超额 -> 跟踪误差仅来自放大 1.1 倍的基准波动
    rp = rb * 1.1 + 0.0003
    te = tracking_error(rp, rb)
    assert te > 0
    ir = information_ratio(rp, rb)
    assert ir > 0, "恒定正超额应给出正信息比率"
    uc, dc = capture_ratios(rp, rb)
    assert uc > 1.05 and dc > 1.05, "1.1 倍放大应同时抬高上下行捕获"
    assert abs(excess_return(rb, rb)) < 1e-12, "自己对自己的超额应为 0"
    print(f"[OK] 相对指标: TE={te:.2%} IR={ir:+.3f} 上行捕获={uc:.3f} 下行捕获={dc:.3f}")


def test_align_uses_intersection():
    """对齐取交集，不拿 0 填充制造假相关"""
    a = _series([100, 101, 102, 103], "2020-01-01")
    b = _series([50, 51, 52], "2020-01-02")       # 少一天
    rets = align(a, b)
    assert len(rets) == 2, f"共同交易日应为 3 天 -> 2 个收益，实得 {len(rets)}"
    assert set(rets.columns) == {"port", "bench"}
    print("[OK] 组合/基准按共同交易日对齐（交集，不填充）")


def test_performance_summary_complete():
    """performance_summary 必须一次性给出用户要的全部指标"""
    rng = np.random.default_rng(5)
    rb = _series(rng.normal(0.0003, 0.011, 1200))
    rp = 1.05 * rb + _series(rng.normal(0.0004, 0.004, 1200))
    summ = performance_summary(cum_return(rp) * 1e6, cum_return(rb) * 4000)
    need = ["sharpe_ratio", "max_drawdown", "alpha_annual", "beta",
            "information_ratio", "tracking_error", "sortino_ratio",
            "calmar_ratio", "excess_return", "up_capture", "down_capture"]
    missing = [k for k in need if k not in summ]
    assert not missing, f"缺少指标 {missing}"
    assert abs(summ["beta"] - 1.05) < 0.05
    assert summ["alpha_tstat"] == summ["alpha_tstat"]     # 不是 NaN
    print(f"[OK] performance_summary 齐全: 夏普={summ['sharpe_ratio']:.3f} "
          f"beta={summ['beta']:.3f} alpha={summ['alpha_annual']:+.2%} "
          f"IR={summ['information_ratio']:.3f}")


if __name__ == "__main__":
    test_betainc_known_values()
    test_t_pvalue_matches_table()
    test_alpha_beta_recovers_known_values()
    test_alpha_beta_perfect_replication()
    test_alpha_beta_zero_beta()
    test_total_and_annual_return_compound()
    test_sharpe_and_sortino_scale()
    test_drawdown_info_exact()
    test_annual_returns_table()
    test_relative_metrics()
    test_align_uses_intersection()
    test_performance_summary_complete()
    print("\n全部绩效指标测试通过")
