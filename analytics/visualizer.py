import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import pandas as pd
import numpy as np
from typing import Optional


class Visualizer:
    def __init__(self, figsize=(14, 8)):
        self.figsize = figsize

    def plot_equity_curve(self, equity: pd.Series, trades: Optional[pd.DataFrame] = None,
                          title: str = "Equity Curve"):
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=self.figsize, gridspec_kw={"height_ratios": [3, 1]})

        ax1.plot(equity.index, equity.values, label="Portfolio", linewidth=1.5, color="#1f77b4")
        if trades is not None and not trades.empty:
            buys = trades[trades["action"] == "buy"]
            sells = trades[trades["action"] == "sell"]
            ax1.scatter(buys.index, [equity.loc[t] if t in equity.index else equity.iloc[-1] for t in buys.index],
                        marker="^", color="g", s=40, label="Buy", alpha=0.7)
            ax1.scatter(sells.index, [equity.loc[t] if t in equity.index else equity.iloc[-1] for t in sells.index],
                        marker="v", color="r", s=40, label="Sell", alpha=0.7)
        ax1.set_title(title, fontsize=13)
        ax1.set_ylabel("Equity")
        ax1.legend()
        ax1.grid(True, alpha=0.3)

        roll_max = equity.expanding().max()
        drawdown = (equity - roll_max) / roll_max * 100
        ax2.fill_between(drawdown.index, 0, drawdown.values, color="tomato", alpha=0.5)
        ax2.set_ylabel("Drawdown %")
        ax2.set_xlabel("Date")
        ax2.grid(True, alpha=0.3)

        plt.tight_layout()
        plt.show()

    def plot_returns_distribution(self, returns: pd.Series, bins: int = 50):
        fig, ax = plt.subplots(figsize=(self.figsize[0], 4))
        ax.hist(returns.dropna(), bins=bins, edgecolor="white", color="steelblue", alpha=0.7)
        ax.axvline(returns.mean(), color="red", linestyle="--", label=f"Mean: {returns.mean():.4f}")
        ax.axvline(0, color="gray", linestyle="-", linewidth=0.8)
        ax.set_title("Returns Distribution")
        ax.set_xlabel("Return")
        ax.set_ylabel("Frequency")
        ax.legend()
        plt.tight_layout()
        plt.show()

    def plot_metrics_table(self, metrics: dict):
        fig, ax = plt.subplots(figsize=(self.figsize[0], 3))
        ax.axis("off")
        rows = [[k, f"{v:.2%}" if isinstance(v, float) and abs(v) < 1 else f"{v:.4f}"]
                for k, v in metrics.items()]
        table = ax.table(cellText=rows, colLabels=["Metric", "Value"],
                         loc="center", cellLoc="left")
        table.auto_set_font_size(False)
        table.set_fontsize(10)
        table.scale(1, 1.5)
        ax.set_title("Performance Summary", fontsize=13)
        plt.tight_layout()
        plt.show()

    def plot_cumulative_returns(self, strategy_ret: pd.Series, benchmark_ret: Optional[pd.Series] = None):
        fig, ax = plt.subplots(figsize=self.figsize)
        ax.plot((1 + strategy_ret).cumprod(), label="Strategy", linewidth=1.5)
        if benchmark_ret is not None:
            ax.plot((1 + benchmark_ret).cumprod(), label="Benchmark", linewidth=1.5, alpha=0.7)
        ax.set_title("Cumulative Returns")
        ax.set_ylabel("Cumulative Return")
        ax.legend()
        ax.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.show()
