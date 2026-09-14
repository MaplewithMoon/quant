import pandas as pd
import numpy as np
from dataclasses import dataclass, field
from typing import Dict


METRICS = {
    "total_return":         "总收益率：期末净值/期初净值 - 1",
    "annual_return":        "年化收益率：将总收益换算为年化",
    "annual_volatility":    "年化波动率：日收益率标准差 × √252",
    "sharpe_ratio":         "夏普比率：(年化收益-无风险利率)/年化波动",
    "calmar_ratio":         "卡尔玛比率：年化收益/最大回撤",
    "max_drawdown":         "最大回撤：从峰值到谷底的最大跌幅",
    "avg_drawdown":         "平均回撤：所有回撤日的平均深度",
    "dd_std":               "回撤标准差：回撤的离散程度",
    "dd_percentile_95":     "95%分位回撤：95%时间回撤不超过该值",
    "max_drawdown_duration": "最长回撤天数：连续不回本的最长时间",
    "recovery_time":        "恢复时间：从谷底爬回峰值最长天数",
    "win_rate":             "胜率：盈利交易占总交易比例",
    "total_trades":         "总交易次数：完整买卖回合数",
    "profit_factor":        "盈亏比：总盈利/总亏损",
}


def help_metrics() -> str:
    """打印所有可用绩效指标的计算说明"""
    lines = ["可用绩效指标：", "=" * 70]
    for name, desc in METRICS.items():
        lines.append(f"  {name:<24} {desc}")
    lines.append("=" * 70)
    lines.append("用法: metrics = Metrics.compute(equity_curve, trades)")
    return "\n".join(lines)


@dataclass
class Metrics:
    total_return: float = 0.0
    annual_return: float = 0.0
    annual_volatility: float = 0.0
    sharpe_ratio: float = 0.0
    calmar_ratio: float = 0.0
    max_drawdown: float = 0.0
    avg_drawdown: float = 0.0
    dd_std: float = 0.0
    dd_percentile_95: float = 0.0
    max_drawdown_duration: int = 0
    recovery_time: int = 0
    win_rate: float = 0.0
    total_trades: int = 0
    profit_factor: float = 0.0

    def to_dict(self) -> Dict:
        import math
        out = {}
        for k, v in self.__dict__.items():
            if isinstance(v, float):
                if math.isinf(v):
                    out[k] = "inf" if v > 0 else "-inf"
                elif math.isnan(v):
                    out[k] = "nan"
                else:
                    out[k] = round(v, 4)
            else:
                out[k] = v
        return out

    @classmethod
    def compute(cls, equity_curve: pd.Series,
                trades: pd.DataFrame) -> "Metrics":
        if equity_curve.empty:
            return cls()

        total_return = equity_curve.iloc[-1] / equity_curve.iloc[0] - 1
        daily_returns = equity_curve.pct_change().dropna()

        n_days = len(daily_returns)
        ann_ret = (1 + total_return) ** (252 / n_days) - 1 if n_days > 0 else 0.0
        ann_vol = daily_returns.std() * np.sqrt(252)

        sharpe = (ann_ret - 0.03) / ann_vol if ann_vol > 0 else 0.0

        roll_max = equity_curve.expanding().max()
        drawdown = (equity_curve - roll_max) / roll_max
        max_dd = drawdown.min()

        calmar = ann_ret / abs(max_dd) if max_dd != 0 else 0.0

        dd_vals = drawdown[drawdown < 0]
        avg_dd = dd_vals.mean() if len(dd_vals) > 0 else 0.0
        dd_stdev = dd_vals.std() if len(dd_vals) > 0 else 0.0
        dd_p95 = dd_vals.quantile(0.05) if len(dd_vals) > 0 else 0.0

        dd_duration = _max_drawdown_duration(drawdown)
        rec_time = _recovery_time(equity_curve)

        # 胜率 / 盈亏比只在"已平仓"交易上计算。
        # 修正：旧实现把买入行也算进分母，而买入行的 pnl 恒为 0，
        # 于是胜率被稀释近一半（实测报 19.05%，真实 40%）。
        if not trades.empty and "action" in trades.columns:
            closed = trades[trades["action"] == "sell"]
        else:
            closed = trades.iloc[0:0]

        total_trades = len(closed)

        if total_trades and "pnl" in closed.columns:
            pnl_series = closed["pnl"]
            wins = pnl_series[pnl_series > 0]
            losses = pnl_series[pnl_series < 0]
            win_rate = len(wins) / len(pnl_series)
            profit_factor = (wins.sum() / abs(losses.sum())
                             if len(losses) > 0 and losses.sum() != 0 else float("inf"))
        else:
            win_rate = 0.0
            profit_factor = 0.0

        return cls(
            total_return=total_return,
            annual_return=ann_ret,
            annual_volatility=ann_vol,
            sharpe_ratio=sharpe,
            calmar_ratio=calmar,
            max_drawdown=max_dd,
            avg_drawdown=avg_dd,
            dd_std=dd_stdev,
            dd_percentile_95=dd_p95,
            max_drawdown_duration=dd_duration,
            recovery_time=rec_time,
            win_rate=win_rate,
            total_trades=total_trades,
            profit_factor=profit_factor,
        )


def _max_drawdown_duration(drawdown: pd.Series) -> int:
    is_dd = drawdown < 0
    if not is_dd.any():
        return 0
    lengths = []
    cur = 0
    for v in is_dd:
        if v:
            cur += 1
        elif cur > 0:
            lengths.append(cur)
            cur = 0
    if cur > 0:
        lengths.append(cur)
    return max(lengths) if lengths else 0


def _recovery_time(equity_curve: pd.Series) -> int:
    """从回撤谷底恢复到前高所用的最长天数"""
    peak = equity_curve.expanding().max()
    in_drawdown = equity_curve < peak

    if not in_drawdown.any():
        return 0

    recoveries = []
    trough_idx = None
    for i in range(len(equity_curve)):
        if not in_drawdown.iloc[i]:
            if trough_idx is not None:
                recoveries.append(i - trough_idx)
                trough_idx = None
        else:
            if trough_idx is None:
                trough_idx = i

    return max(recoveries) if recoveries else 0
