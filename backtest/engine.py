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

    def run(self, dataset: DataSet, strategy: Strategy) -> pd.DataFrame:
        data = dataset.data.copy()
        position = 0.0

        for i in range(len(data)):
            bar = data.iloc[i]
            signal = strategy.on_bar(data, i)

            signal = self.risk_manager.filter(signal, self.portfolio)

            if signal.action == Action.BUY and position <= 0:
                size = self.portfolio.cash * signal.size // bar["close"]
                if size > 0:
                    order = Order(
                        symbol=dataset.symbol,
                        side=OrderSide.BUY,
                        size=size,
                        order_type=OrderType.MARKET,
                        price=float(bar["close"]),
                    )
                    self.broker.place_order(order)
                    filled = self.broker.execute(order, float(bar["close"]))
                    actual_size = self.portfolio.buy(
                        dataset.symbol, filled.filled_qty,
                        filled.filled_amount / filled.filled_qty
                    )
                    position = actual_size
                    self._record_trade(bar, dataset.symbol, "buy", filled, 0.0)

            elif signal.action == Action.SELL and position > 0:
                order = Order(
                    symbol=dataset.symbol,
                    side=OrderSide.SELL,
                    size=position,
                    order_type=OrderType.MARKET,
                    price=float(bar["close"]),
                )
                self.broker.place_order(order)
                filled = self.broker.execute(order, float(bar["close"]))
                pnl = self.portfolio.sell(
                    dataset.symbol, filled.filled_qty,
                    filled.filled_amount / filled.filled_qty
                )
                position = 0.0
                self._record_trade(bar, dataset.symbol, "sell", filled, pnl)

            self.portfolio.update_price(dataset.symbol, float(bar["close"]))

        return self._build_report(dataset)

    def _record_trade(self, bar, symbol: str, side: str,
                      order: Order, pnl: float):
        self.trades.append(TradeRecord(
            timestamp=bar.name,
            symbol=symbol,
            side=side,
            size=order.filled_qty,
            price=order.filled_amount / order.filled_qty,
            pnl=pnl,
            balance=self.portfolio.total_value,
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
                "balance": trade.balance,
            })

        df = pd.DataFrame(snapshots)
        if not df.empty:
            df.set_index("timestamp", inplace=True)
        return df
