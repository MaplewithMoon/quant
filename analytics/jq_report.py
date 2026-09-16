# -*- coding: utf-8 -*-
"""聚宽风格的 HTML 回测报告

对着聚宽「回测结果」页的指标口径来组织，一眼能和我们/聚宽的结果对照：

    顶部三卡：策略收益 / 基准收益 / 超额收益
    指标区  ：年化收益、最大回撤、夏普、索提诺、信息比率、Alpha、Beta、
              波动率、下行风险、收益/风险、胜率、盈亏比、换手率
    图表区  ：收益曲线（策略 vs 基准）、回撤、分年度收益
    表格区  ：分年度收益、分月收益、关键指标对照

图片用 base64 内嵌，产出单文件 HTML，不依赖外部资源，直接双击就能看。
"""
import base64
import io
from typing import Dict, Optional

import numpy as np
import pandas as pd

from .performance import (align, annual_returns_table,
                          drawdown_series, monthly_returns_table,
                          performance_summary, to_returns)


# ============================================================
# 指标口径（补齐聚宽有、我们原先没有的几项）
# ============================================================
def downside_risk(returns: pd.Series, mar: float = 0.0,
                  periods: int = 252) -> float:
    """下行风险：只统计低于 mar 的收益，年化标准差（聚宽同名指标）"""
    r = pd.Series(returns).dropna()
    ex = r - mar / periods
    neg = ex[ex < 0]
    if len(neg) == 0:
        return 0.0
    return float(np.sqrt((neg ** 2).mean()) * np.sqrt(periods))


def return_risk_ratio(ann_return: float, ann_vol: float) -> float:
    """收益/风险（聚宽口径：年化收益 ÷ 年化波动）"""
    return float(ann_return / ann_vol) if ann_vol and ann_vol > 0 else 0.0


def trade_stats(trades: pd.DataFrame) -> Dict[str, float]:
    """交易层面的统计：胜率、盈亏比、盈利/亏损次数"""
    out = {"交易笔数": 0, "盈利次数": 0, "亏损次数": 0,
           "胜率": 0.0, "盈亏比": 0.0, "平均盈利": 0.0, "平均亏损": 0.0}
    if trades is None or trades.empty or "action" not in trades.columns:
        return out
    closed = trades[trades["action"] == "sell"]
    if closed.empty or "pnl" not in closed.columns:
        return out
    pnl = closed["pnl"].dropna()
    if pnl.empty:
        return out
    wins, losses = pnl[pnl > 0], pnl[pnl < 0]
    out["交易笔数"] = int(len(closed))
    out["盈利次数"] = int(len(wins))
    out["亏损次数"] = int(len(losses))
    out["胜率"] = float(len(wins) / len(pnl))
    out["平均盈利"] = float(wins.mean()) if len(wins) else 0.0
    out["平均亏损"] = float(losses.mean()) if len(losses) else 0.0
    out["盈亏比"] = (float(abs(wins.mean() / losses.mean()))
                   if len(wins) and len(losses) and losses.mean() != 0 else 0.0)
    return out


# ============================================================
# 图表（base64 内嵌）
# ============================================================
def _fig_to_b64(fig) -> str:
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=100, bbox_inches="tight",
                facecolor="white")
    buf.seek(0)
    return base64.b64encode(buf.read()).decode("ascii")


def _charts(equity: pd.Series, benchmark: pd.Series, split_date=None) -> str:
    import matplotlib
    matplotlib.use("Agg")
    from matplotlib import pyplot as plt
    from matplotlib import font_manager
    have = {f.name for f in font_manager.fontManager.ttflist}
    for cand in ("Microsoft YaHei", "SimHei", "Noto Sans CJK SC"):
        if cand in have:
            plt.rcParams["font.sans-serif"] = [cand]
            plt.rcParams["axes.unicode_minus"] = False
            break

    rets = align(equity, benchmark)
    if rets.empty:
        return ""
    rp, rb = rets["port"], rets["bench"]
    nav_p, nav_b = (1 + rp).cumprod(), (1 + rb).cumprod()
    imgs = []

    # 1) 收益曲线
    fig, ax = plt.subplots(figsize=(11, 4.2))
    ax.plot(nav_p.index, nav_p.values, lw=1.7, color="#c0392b", label="策略")
    ax.plot(nav_b.index, nav_b.values, lw=1.4, color="#2c3e50", alpha=.85,
            label="基准")
    if split_date is not None:
        sd = pd.Timestamp(split_date)
        if nav_p.index[0] <= sd <= nav_p.index[-1]:
            ax.axvline(sd, color="gray", ls="--", lw=1)
    ax.set_title("收益曲线", fontsize=12, fontweight="bold")
    ax.set_ylabel("净值（起点=1）")
    ax.legend(loc="upper left")
    ax.grid(alpha=.3)
    imgs.append(_fig_to_b64(fig))
    plt.close(fig)

    # 2) 回撤
    fig, ax = plt.subplots(figsize=(11, 2.6))
    dd_p = drawdown_series(equity).reindex(nav_p.index) * 100
    dd_b = drawdown_series(benchmark).reindex(nav_b.index) * 100
    ax.fill_between(dd_p.index, 0, dd_p.values, color="#c0392b", alpha=.55,
                    label="策略回撤")
    ax.plot(dd_b.index, dd_b.values, color="#2c3e50", lw=1, alpha=.85,
            label="基准回撤")
    ax.set_ylabel("回撤 (%)")
    ax.legend(loc="lower left", fontsize=9)
    ax.grid(alpha=.3)
    imgs.append(_fig_to_b64(fig))
    plt.close(fig)

    # 3) 分年度
    tbl = annual_returns_table(rp, rb)
    if not tbl.empty:
        fig, ax = plt.subplots(figsize=(11, 2.8))
        x = np.arange(len(tbl))
        w = .38
        ax.bar(x - w/2, tbl["组合"] * 100, w, color="#c0392b", label="策略")
        ax.bar(x + w/2, tbl["基准"] * 100, w, color="#2c3e50", label="基准")
        for i, v in enumerate(tbl["组合"] * 100):
            ax.text(i - w/2, v, f"{v:.1f}", ha="center",
                    va="bottom" if v >= 0 else "top", fontsize=8)
        for i, v in enumerate(tbl["基准"] * 100):
            ax.text(i + w/2, v, f"{v:.1f}", ha="center",
                    va="bottom" if v >= 0 else "top", fontsize=8)
        ax.set_xticks(x)
        ax.set_xticklabels([str(i) for i in tbl.index])
        ax.axhline(0, color="black", lw=.9)
        ax.set_ylabel("年度收益 (%)")
        ax.legend(loc="upper left", fontsize=9)
        ax.grid(alpha=.3, axis="y")
        imgs.append(_fig_to_b64(fig))
        plt.close(fig)

    return "".join(f'<img src="data:image/png;base64,{b}"/>' for b in imgs)


# ============================================================
# HTML
# ============================================================
CSS = """
body{font-family:'Microsoft YaHei',SimHei,Segoe UI,sans-serif;margin:0;
     background:#f5f6f8;color:#222}
.wrap{max-width:1180px;margin:0 auto;padding:22px}
h1{font-size:21px;margin:0 0 4px}
.sub{color:#777;font-size:13px;margin-bottom:18px}
.cards{display:flex;gap:14px;margin-bottom:18px}
.card{flex:1;background:#fff;border-radius:8px;padding:16px 18px;
      box-shadow:0 1px 3px rgba(0,0,0,.08)}
.card .k{font-size:13px;color:#888;margin-bottom:8px}
.card .v{font-size:27px;font-weight:600}
.pos{color:#c0392b}.neg{color:#27ae60}
.panel{background:#fff;border-radius:8px;padding:18px 20px;margin-bottom:18px;
       box-shadow:0 1px 3px rgba(0,0,0,.08)}
.panel h2{font-size:15px;margin:0 0 14px;padding-bottom:8px;
          border-bottom:1px solid #eee}
table{border-collapse:collapse;width:100%;font-size:13px}
th,td{padding:7px 10px;text-align:right;border-bottom:1px solid #f0f0f0}
th{background:#fafafa;color:#666;font-weight:500;text-align:right}
th:first-child,td:first-child{text-align:left}
img{width:100%;display:block;margin:6px 0 14px}
.grid2{display:grid;grid-template-columns:1fr 1fr;gap:26px}
.note{font-size:12px;color:#888;line-height:1.7}
.warn{background:#fff8e6;border-left:3px solid #f0ad4e;padding:10px 14px;
      font-size:12.5px;color:#7a5a12;border-radius:4px;margin-top:10px}
"""


def _pct(x, sign=True) -> str:
    if x is None or (isinstance(x, float) and not np.isfinite(x)):
        return "-"
    return f"{x:+.2%}" if sign else f"{x:.2%}"


def _num(x, nd=3) -> str:
    if x is None or (isinstance(x, float) and not np.isfinite(x)):
        return "-"
    return f"{x:+.{nd}f}"


def _cls(x) -> str:
    if x is None or (isinstance(x, float) and not np.isfinite(x)):
        return ""
    return "pos" if x >= 0 else "neg"


def _table(df: pd.DataFrame, pct_cols=(), num_cols=(), idx_name="") -> str:
    if df is None or df.empty:
        return "<p class='note'>无数据</p>"
    head = "".join(f"<th>{c}</th>" for c in [idx_name] + list(df.columns))
    rows = []
    for i, r in df.iterrows():
        tds = [f"<td>{i}</td>"]
        for c in df.columns:
            v = r[c]
            if c in pct_cols:
                tds.append(f"<td class='{_cls(v)}'>{_pct(v)}</td>")
            elif c in num_cols:
                tds.append(f"<td class='{_cls(v)}'>{_num(v)}</td>")
            else:
                tds.append(f"<td>{v}</td>")
        rows.append("<tr>" + "".join(tds) + "</tr>")
    return f"<table><thead><tr>{head}</tr></thead><tbody>{''.join(rows)}</tbody></table>"


def build_jq_report(equity: pd.Series, benchmark: pd.Series, title: str,
                    subtitle: str = "", trades: pd.DataFrame = None,
                    turnover: float = None, split_date=None,
                    out_path: str = None, extra_notes: str = "") -> str:
    """生成聚宽风格的 HTML 回测报告，返回 HTML 文本（并可选写盘）"""
    bench = pd.Series(benchmark).astype(float).reindex(equity.index).ffill() \
        if benchmark is not None and len(benchmark) else None
    rp = to_returns(equity)
    rb = to_returns(bench) if bench is not None else None
    summ = performance_summary(equity, bench)
    ts = trade_stats(trades)
    dsr = downside_risk(rp)

    # ---- 顶部三卡 ----
    cards = [
        ("策略收益", summ["total_return"], "pos"),
        ("基准收益", summ.get("bench_total_return"), ""),
        ("超额收益", summ.get("excess_return"), _cls(summ.get("excess_return"))),
    ]
    card_html = ""
    for k, v, c in cards:
        card_html += (f"<div class='card'><div class='k'>{k}</div>"
                      f"<div class='v {c}'>{_pct(v)}</div></div>")

    # ---- 指标表（对齐聚宽口径）----
    rows = [
        ("策略年化收益", _pct(summ["annual_return"]), "基准年化收益",
         _pct(summ.get("bench_annual_return"))),
        ("策略波动率", _pct(summ["annual_volatility"], False), "基准波动率",
         _pct(summ.get("bench_annual_volatility"), False)),
        ("夏普比率", _num(summ["sharpe_ratio"]), "索提诺比率",
         _num(summ["sortino_ratio"])),
        ("最大回撤", _pct(summ["max_drawdown"]), "最大回撤区间",
         f"{summ.get('max_drawdown_peak','')} ~ {summ.get('max_drawdown_trough','')}"),
        ("信息比率", _num(summ.get("information_ratio")), "跟踪误差",
         _pct(summ.get("tracking_error"), False)),
        ("Alpha", _pct(summ.get("alpha_annual")), "Beta", _num(summ.get("beta"))),
        ("Alpha t 值", _num(summ.get("alpha_tstat"), 2), "Alpha p 值",
         f"{summ.get('alpha_pvalue', float('nan')):.3f}"),
        ("下行风险", _pct(dsr, False), "收益/风险",
         _num(return_risk_ratio(summ["annual_return"], summ["annual_volatility"]))),
        ("卡玛比率", _num(summ["calmar_ratio"]), "R²", _num(summ.get("r_squared"))),
        ("胜率", f"{ts['胜率']:.1%}", "盈亏比", _num(ts["盈亏比"], 2)),
        ("盈利次数", f"{ts['盈利次数']}", "亏损次数", f"{ts['亏损次数']}"),
        ("交易笔数", f"{ts['交易笔数']}", "平均换手",
         f"{turnover:.1%}" if turnover is not None else "-"),
    ]
    metric_html = "<table><tbody>"
    for a, b, c, d in rows:
        metric_html += (f"<tr><td>{a}</td><td>{b}</td>"
                        f"<td style='color:#666'>{c}</td><td>{d}</td></tr>")
    metric_html += "</tbody></table>"

    # ---- 分年度 / 分月 ----
    ann = annual_returns_table(rp, rb)
    mon = monthly_returns_table(rp)

    charts = _charts(equity, bench if bench is not None else equity, split_date)
    warn = (f"<div class='warn'>{extra_notes}</div>" if extra_notes else "")

    html = f"""<!DOCTYPE html><html lang="zh-CN"><head><meta charset="utf-8">
<title>{title} - 回测报告</title><style>{CSS}</style></head><body><div class="wrap">
<h1>{title}</h1><div class="sub">{subtitle}</div>
<div class="cards">{card_html}</div>
<div class="panel"><h2>关键指标</h2>{metric_html}</div>
<div class="panel"><h2>收益与回撤</h2>{charts}</div>
<div class="grid2">
  <div class="panel"><h2>分年度收益</h2>
    {_table(ann, pct_cols=("组合", "基准", "超额"), idx_name="年份")}</div>
  <div class="panel"><h2>分月收益（策略）</h2>
    {_table(mon, pct_cols=tuple(mon.columns) if not mon.empty else (), idx_name="年份")}</div>
</div>
{warn}
<div class="panel"><h2>说明</h2><div class="note">
策略收益与基准收益按复利计算；夏普用无风险利率 0（聚宽默认 0）；
Alpha/Beta 为日频 CAPM 回归（Alpha 年化 = 日截距 × 252）；
波动率、下行风险、跟踪误差均按 252 交易日年化；
盈亏比 = 平均盈利 / |平均亏损|（只统计已平仓交易）。
</div></div>
</div></body></html>"""

    if out_path:
        import os
        os.makedirs(os.path.dirname(os.path.abspath(out_path)), exist_ok=True)
        with open(out_path, "w", encoding="utf-8") as f:
            f.write(html)
    return html
