from dataclasses import dataclass, field
from typing import Dict


@dataclass
class Position:
    symbol: str
    size: float = 0.0
    avg_cost: float = 0.0
    current_price: float = 0.0

    @property
    def market_value(self) -> float:
        return self.size * self.current_price

    @property
    def unrealized_pnl(self) -> float:
        return self.size * (self.current_price - self.avg_cost)

    @property
    def unrealized_pnl_pct(self) -> float:
        if self.avg_cost == 0:
            return 0.0
        return (self.current_price / self.avg_cost - 1) * self.size


class Portfolio:
    def __init__(self, initial_cash: float = 1_000_000):
        self.initial_cash = initial_cash
        self.cash = initial_cash
        self.positions: Dict[str, Position] = {}
        self._realized_pnl: float = 0.0
        self._peak = initial_cash

    @property
    def total_value(self) -> float:
        return self.cash + sum(p.market_value for p in self.positions.values())

    @property
    def total_unrealized_pnl(self) -> float:
        return sum(p.unrealized_pnl for p in self.positions.values())

    @property
    def drawdown_pct(self) -> float:
        return (self.total_value / self._peak - 1) if self._peak > 0 else 0.0

    @property
    def drawdown(self) -> float:
        return self._peak - self.total_value

    def _update_peak(self):
        if self.total_value > self._peak:
            self._peak = self.total_value

    def buy(self, symbol: str, size: float, price: float) -> float:
        cost = size * price
        if cost > self.cash:
            size = self.cash // price
            cost = size * price
        self.cash -= cost

        if symbol in self.positions:
            pos = self.positions[symbol]
            total_size = pos.size + size
            total_cost = pos.size * pos.avg_cost + cost
            pos.size = total_size
            pos.avg_cost = total_cost / total_size if total_size > 0 else 0
        else:
            self.positions[symbol] = Position(
                symbol=symbol, size=size, avg_cost=price, current_price=price
            )
        self._update_peak()
        return size

    def sell(self, symbol: str, size: float, price: float) -> float:
        pos = self.positions.get(symbol)
        if not pos or pos.size < size:
            raise ValueError(f"insufficient {symbol} to sell")

        revenue = size * price
        self.cash += revenue
        pnl = size * (price - pos.avg_cost)
        self._realized_pnl += pnl

        pos.size -= size
        if pos.size <= 1e-8:
            del self.positions[symbol]
        self._update_peak()
        return pnl

    def update_price(self, symbol: str, price: float):
        if symbol in self.positions:
            self.positions[symbol].current_price = price
        self._update_peak()
