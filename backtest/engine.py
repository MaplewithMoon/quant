import pandas as pd
import numpy as np
from typing import Optional
from dataclasses import dataclass, field
from datetime import datetime

from data.dataset import DataSet
from strategy.base import Strategy, Action
from execution.broker import SimulatedBroker
from execution.order import Order, OrderSide, OrderType
from portfolio.portfolio import Portfolio
from risk.manager import RiskManager
from utils.logger import setup_logger


@dataclass
class TradeRecord:
    timestamp: datetime
    symbol: str
    side: str
    size: float
    price: float
    pnl: float
    balance: float
    commission: float = 0.0
    reason: str = ""


class BacktestEngine:
    def __init__(
        self,
        initial_capital: float = 1_000_000,
        commission: float = 0.0001,
        slippage: float = 0.001,
        min_commission: float = 5.0,
    ):
        """回测引擎
        commission:     佣金费率（默认万分之一 = 0.0001）
        slippage:       滑点（默认0.1%）
        min_commission: 单笔最低佣金（默认5元）
        """
        self.initial_capital = initial_capital
        self.broker = SimulatedBroker(slippage=slippage, commission=commission,
                                      min_commission=min_commission)
        self.portfolio = Portfolio(initial_capital)
        self.logger = setup_logger("backtest")
        self.trades: list[TradeRecord] = []
        self.risk_manager = RiskManager()
        self._equity: list[tuple] = []      # 逐 bar 权益快照 [(日期, 总资产)]
        self.fees_paid: float = 0.0         # 累计交易费用

    # ---------- 权益曲线（唯一账本） ----------
    @property
    def equity_curve(self) -> pd.Series:
        """逐 bar 账户权益，由引擎自己产生

        这是**唯一**的净值来源。历史上 scripts/backtest.py 另有一个
        build_equity_curve() 用成交记录重推现金与持仓，两本账实测差 0.96%
        （账户 1,040,192.63 vs 曲线 1,030,567.10），而对外指标全部取自后者。
        现在指标直接消费这条曲线，账实必然一致。
        """
        if not self._equity:
            return pd.Series(dtype=float)
        idx = pd.Index([d for d, _ in self._equity], name="trade_date")
        return pd.Series([v for _, v in self._equity], index=idx, dtype=float)

    @property
    def ledger_gap(self) -> float:
        """权益曲线末值与账户总资产的差额（健康值 = 0）"""
        if not self._equity:
            return 0.0
        return abs(self._equity[-1][1] - self.portfolio.total_value)

    # ---------- 主循环 ----------
    def run(self, dataset: DataSet, strategy: Strategy) -> pd.DataFrame:
        data = dataset.data.copy()
        position = 0.0
        self.trades = []
        self._equity = []
        self.fees_paid = 0.0

        for i in range(len(data)):
            bar = data.iloc[i]
            close = float(bar["close"])
            signal = strategy.on_bar(data, i)

            signal = self.risk_manager.filter(signal, self.portfolio)

            if signal.action == Action.BUY and position <= 0:
                # 下单量在扣掉滑点与佣金后算出，避免成交后被"静默缩量"
                size = self.broker.max_affordable_size(self.portfolio.cash, close)
                if size > 0:
                    order = Order(
                        symbol=dataset.symbol,
                        side=OrderSide.BUY,
                        size=size,
                        order_type=OrderType.MARKET,
                        price=close,
                    )
                    self.broker.place_order(order)
                    filled = self.broker.execute(order, close)
                    filled_qty = self.portfolio.buy(
                        dataset.symbol, filled.filled_qty, filled.avg_fill_price,
                        commission=filled.commission,
                    )
                    position = filled_qty
                    self.fees_paid += filled.commission
                    self._record_trade(bar, dataset.symbol, "buy", filled, 0.0)

            elif signal.action == Action.SELL and position > 0:
                order = Order(
                    symbol=dataset.symbol,
                    side=OrderSide.SELL,
                    size=position,
                    order_type=OrderType.MARKET,
                    price=close,
                )
                self.broker.place_order(order)
                filled = self.broker.execute(order, close)
                pnl = self.portfolio.sell(
                    dataset.symbol, filled.filled_qty, filled.avg_fill_price,
                    commission=filled.commission,
                )
                position = 0.0
                self.fees_paid += filled.commission
                self._record_trade(bar, dataset.symbol, "sell", filled, pnl)

            self.portfolio.update_price(dataset.symbol, close)
            self._equity.append((bar.name, self.portfolio.total_value))

        report = self._build_report(dataset)

        gap = self.ledger_gap
        if gap > 1e-6:
            self.logger.error(
                f"账目不变量被破坏: 权益曲线末值 {self._equity[-1][1]:,.4f} "
                f"!= 账户总资产 {self.portfolio.total_value:,.4f} (差 {gap:,.4f})"
            )
        return report

    def _record_trade(self, bar, symbol: str, side: str,
                      order: Order, pnl: float):
        self.trades.append(TradeRecord(
            timestamp=bar.name,
            symbol=symbol,
            side=side,
            size=order.filled_qty,
            price=order.avg_fill_price,          # 干净成交价，不含费用
            pnl=pnl,
            balance=self.portfolio.total_value,
            commission=order.commission,
        ))

    def _build_report(self, dataset: DataSet) -> pd.DataFrame:
        snapshots = []
        for trade in self.trades:
            snapshots.append({
                "timestamp": trade.timestamp,
                "action": trade.side,
                "price": trade.price,
                "size": trade.size,
                "pnl": trade.pnl,
                "commission": trade.commission,
                "balance": trade.balance,
            })

        df = pd.DataFrame(snapshots)
        if not df.empty:
            df.set_index("timestamp", inplace=True)
        return df
