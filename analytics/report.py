# -*- coding: utf-8 -*-
"""回测报告生成：指标表 + 四联图 + 落盘

被 `scripts/sector_rotation_backtest.py` 调用，也可复用于任何"组合净值 vs 基准"。

图表用 Agg 后端（无 GUI），中文字体优先 Microsoft YaHei / SimHei，
取不到就退回英文标题 —— 绝不让画图把整个回测流程搞崩。
"""
import os
from typing import Dict, Optional, Tuple

import numpy as np
import pandas as pd

from .performance import (align, annual_returns_table, drawdown_series,
                          drawdown_info, monthly_returns_table,
                          performance_summary, to_returns)

# 指标 -> (中文名, 格式)
METRIC_SPEC = [
    ("total_return",            "累计收益",        "pct"),
    ("annual_return",           "年化收益",        "pct"),
    ("annual_volatility",       "年化波动",        "pct"),
    ("sharpe_ratio",            "夏普比率",        "num"),
    ("sortino_ratio",           "索提诺比率",      "num"),
    ("calmar_ratio",            "卡玛比率",        "num"),
    ("max_drawdown",            "最大回撤",        "pct"),
    ("max_drawdown_peak",       "回撤起点",        "str"),
    ("max_drawdown_trough",     "回撤低点",        "str"),
    ("max_drawdown_recover",    "收复日期",        "str"),
    ("max_drawdown_duration",   "最长未创新高(日)", "int"),
    ("skewness",                "偏度",            "num"),
    ("kurtosis",                "超额峰度",        "num"),
    # ---- 相对基准 ----
    ("bench_total_return",      "基准累计收益",    "pct"),
    ("bench_annual_return",     "基准年化",        "pct"),
    ("bench_max_drawdown",      "基准最大回撤",    "pct"),
    ("excess_return",           "累计超额收益",    "pct"),
    ("alpha_annual",            "年化 Alpha",      "pct"),
    ("alpha_tstat",             "Alpha t 值",      "num"),
    ("alpha_pvalue",            "Alpha p 值",      "num"),
    ("beta",                    "Beta",            "num"),
    ("r_squared",               "R²",              "num"),
    ("tracking_error",          "跟踪误差",        "pct"),
    ("information_ratio",       "信息比率",        "num"),
    ("residual_vol_annual",     "残差波动(年化)",   "pct"),
    ("win_rate_vs_bench",       "跑赢基准天数占比", "pct"),
    ("up_capture",              "上行捕获",        "num"),
    ("down_capture",            "下行捕获",        "num"),
]


def fmt_value(v, kind: str) -> str:
    if v is None:
        return "-"
    if kind == "pct":
        try:
            return f"{float(v):+.2%}"
        except (TypeError, ValueError):
            return str(v)
    if kind == "num":
        try:
            f = float(v)
            return "nan" if not np.isfinite(f) else f"{f:+.3f}"
        except (TypeError, ValueError):
            return str(v)
    if kind == "int":
        try:
            return f"{int(v)}"
        except (TypeError, ValueError):
            return str(v)
    return str(v)


def metrics_table(summary: Dict, benchmark_name: str = "沪深300") -> str:
    """把 performance_summary 的 dict 排成一张可读的表"""
    L = ["=" * 78,
         f"{'指标':<20}{'数值':>18}      说明",
         "-" * 78]
    notes = {
        "alpha_annual": "CAPM 日频回归截距 × 252（年化）",
        "beta": "对基准的日频回归斜率",
        "alpha_pvalue": "双尾 p 值；< 0.05 才算显著",
        "max_drawdown_recover": "未收复 = 期末仍在坑里",
        "up_capture": "基准上涨日的收益捕获（>1 进攻）",
        "down_capture": "基准下跌日的收益捕获（<1 防守）",
        "information_ratio": "年化超额 / 跟踪误差",
        "win_rate_vs_bench": "组合日收益 > 基准日收益的天数占比",
    }
    for key, name, kind in METRIC_SPEC:
        if key not in summary:
            continue
        v = summary[key]
        L.append(f"{name:<20}{fmt_value(v, kind):>18}      {notes.get(key, '')}")
    L.append("=" * 78)
    return "\n".join(L)


# ============================================================
# 绘图
# ============================================================
def setup_font() -> bool:
    """中文字体；返回是否成功（失败就退回英文标签）"""
    import matplotlib
    matplotlib.use("Agg")
    from matplotlib import pyplot as plt
    from matplotlib import font_manager

    have = {f.name for f in font_manager.fontManager.ttflist}
    for cand in ("Microsoft YaHei", "SimHei", "Noto Sans CJK SC",
                 "WenQuanYi Zen Hei", "PingFang SC"):
        if cand in have:
            plt.rcParams["font.sans-serif"] = [cand]
            plt.rcParams["axes.unicode_minus"] = False
            return True
    return False


def plot_performance(equity: pd.Series, benchmark: pd.Series,
                     out_path: str, title: str = "板块轮动策略",
                     split_date: str = None, zh: bool = True) -> str:
    """四联图：累计净值 / 超额净值 / 回撤 / 分年度收益

    返回保存的文件路径。
    """
    import matplotlib
    matplotlib.use("Agg")
    from matplotlib import pyplot as plt

    def T(zh_s, en_s):
        """按语言选择中文/英文标签（原先写成 lambda 赋值）"""
        return zh_s if zh else en_s

    rets = align(equity, benchmark)
    if rets.empty:
        raise ValueError("组合与基准没有重叠交易日，无法绘图")
    rp, rb = rets["port"], rets["bench"]
    nav_p, nav_b = (1 + rp).cumprod(), (1 + rb).cumprod()
    excess = nav_p / nav_b

    fig, axes = plt.subplots(4, 1, figsize=(14, 17),
                             gridspec_kw={"height_ratios": [3, 1.6, 1.6, 1.8]})
    ax1, ax2, ax3, ax4 = axes

    # ---- 1. 累计净值 ----
    ax1.plot(nav_p.index, nav_p.values, lw=1.8, color="#c0392b",
             label=T("板块轮动策略", "Sector Rotation"))
    ax1.plot(nav_b.index, nav_b.values, lw=1.5, color="#2c3e50", alpha=0.85,
             label=T("沪深300", "CSI 300"))
    if split_date is not None:
        sd = pd.Timestamp(split_date)
        if nav_p.index[0] <= sd <= nav_p.index[-1]:
            ax1.axvline(sd, color="gray", ls="--", lw=1.2, alpha=0.8)
            ax1.text(sd, ax1.get_ylim()[1], " 样本外起点", color="gray",
                     fontsize=9, va="top")
    ax1.set_title(title, fontsize=14, fontweight="bold")
    ax1.set_ylabel(T("累计净值（起点=1）", "Cumulative NAV (start=1)"))
    ax1.legend(loc="upper left", fontsize=11)
    ax1.grid(True, alpha=0.3)

    # ---- 2. 超额净值 ----
    ax2.plot(excess.index, excess.values, lw=1.6, color="#8e44ad")
    ax2.axhline(1.0, color="black", lw=0.9, ls="-")
    ax2.fill_between(excess.index, 1.0, excess.values,
                     where=(excess.values >= 1.0), color="#27ae60", alpha=0.25)
    ax2.fill_between(excess.index, 1.0, excess.values,
                     where=(excess.values < 1.0), color="#e74c3c", alpha=0.25)
    ax2.set_ylabel(T("相对沪深300超额净值", "Excess NAV vs CSI300"))
    ax2.grid(True, alpha=0.3)

    # ---- 3. 回撤 ----
    dd_p = drawdown_series(equity).reindex(nav_p.index) * 100
    dd_b = drawdown_series(benchmark).reindex(nav_b.index) * 100
    ax3.fill_between(dd_p.index, 0, dd_p.values, color="#c0392b", alpha=0.55,
                     label=T("策略回撤", "Strategy DD"))
    ax3.plot(dd_b.index, dd_b.values, color="#2c3e50", lw=1.1, alpha=0.85,
             label=T("沪深300回撤", "CSI300 DD"))
    ax3.set_ylabel(T("回撤 (%)", "Drawdown (%)"))
    ax3.legend(loc="lower left", fontsize=10)
    ax3.grid(True, alpha=0.3)

    # ---- 4. 分年度收益 ----
    tbl = annual_returns_table(rp, rb)
    if not tbl.empty:
        x = np.arange(len(tbl))
        w = 0.38
        ax4.bar(x - w / 2, tbl["组合"] * 100, w, color="#c0392b",
                label=T("策略", "Strategy"))
        ax4.bar(x + w / 2, tbl["基准"] * 100, w, color="#2c3e50",
                label=T("沪深300", "CSI300"))
        for i, v in enumerate(tbl["组合"] * 100):
            ax4.text(i - w / 2, v, f"{v:.1f}", ha="center",
                     va="bottom" if v >= 0 else "top", fontsize=8)
        for i, v in enumerate(tbl["基准"] * 100):
            ax4.text(i + w / 2, v, f"{v:.1f}", ha="center",
                     va="bottom" if v >= 0 else "top", fontsize=8)
        ax4.set_xticks(x)
        ax4.set_xticklabels([str(i) for i in tbl.index], fontsize=10)
        ax4.axhline(0, color="black", lw=0.9)
        ax4.set_ylabel(T("年度收益 (%)", "Annual Return (%)"))
        ax4.legend(loc="upper left", fontsize=10)
        ax4.grid(True, alpha=0.3, axis="y")

    plt.tight_layout()
    os.makedirs(os.path.dirname(os.path.abspath(out_path)), exist_ok=True)
    fig.savefig(out_path, dpi=110, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    return out_path


def plot_rolling(equity: pd.Series, benchmark: pd.Series, out_path: str,
                 window: int = 252, title: str = None) -> str:
    """滚动 1 年夏普 + 滚动 1 年超额净值，看策略是否"一直有效" """
    import matplotlib
    matplotlib.use("Agg")
    from matplotlib import pyplot as plt

    rets = align(equity, benchmark)
    if len(rets) < window:
        return ""
    rp, rb = rets["port"], rets["bench"]

    def roll_sharpe(r, w):
        m = r.rolling(w).mean()
        s = r.rolling(w).std(ddof=1)
        return m / s * np.sqrt(252)

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 8), sharex=True)
    ax1.plot(rp.index, roll_sharpe(rp, window).values, lw=1.6, color="#c0392b",
             label="策略")
    ax1.plot(rb.index, roll_sharpe(rb, window).values, lw=1.3, color="#2c3e50",
             alpha=0.85, label="沪深300")
    ax1.axhline(0, color="black", lw=0.9)
    ax1.set_ylabel(f"滚动 {window} 日夏普")
    ax1.set_title(title or "滚动绩效", fontsize=13, fontweight="bold")
    ax1.legend(fontsize=10)
    ax1.grid(True, alpha=0.3)

    ex = ((1 + rp).cumprod() / (1 + rb).cumprod())
    ax2.plot(ex.index, ex.values, lw=1.6, color="#8e44ad")
    ax2.axhline(1.0, color="black", lw=0.9)
    ax2.set_ylabel("累计超额净值")
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    os.makedirs(os.path.dirname(os.path.abspath(out_path)), exist_ok=True)
    fig.savefig(out_path, dpi=110, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    return out_path


# ============================================================
# 文本报告
# ============================================================
def returns_tables(equity: pd.Series, benchmark: pd.Series = None) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """(分年度表, 分月表)"""
    rp = to_returns(equity)
    rb = to_returns(benchmark) if benchmark is not None and not benchmark.empty else None
    return annual_returns_table(rp, rb), monthly_returns_table(rp)


def format_returns_table(tbl: pd.DataFrame, title: str = "分年度收益") -> str:
    if tbl is None or tbl.empty:
        return ""
    show = tbl.copy()
    for c in show.columns:
        show[c] = show[c].map(lambda x: f"{x:+.2%}" if pd.notna(x) else "-")
    L = ["", "=" * 78, title, "-" * 78]
    L.append(show.to_string())
    L.append("=" * 78)
    return "\n".join(L)


def save_csv(df: pd.DataFrame, path: str, index_label: str = "trade_date"):
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    df.to_csv(path, encoding="utf-8-sig", index_label=index_label)
    return path
