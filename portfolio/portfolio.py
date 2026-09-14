from dataclasses import dataclass, field
from typing import Dict


@dataclass
class Position:
    symbol: str
    size: float = 0.0
    available: float = 0.0      # 可卖数量（T+1：当日买入的当日不可卖）
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
        """未实现收益率（比率，不含杠杆）

        修正：旧实现写成 (price/avg_cost - 1) * size，把股数乘了进去，
        结果不是"率"而是被放大了 size 倍的数。
        """
        if self.avg_cost == 0:
            return 0.0
        return self.current_price / self.avg_cost - 1


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
    def realized_pnl(self) -> float:
        """已实现盈亏（已扣除买卖两次的全部费用）"""
        return self._realized_pnl

    @property
    def drawdown_pct(self) -> float:
        return (self.total_value / self._peak - 1) if self._peak > 0 else 0.0

    @property
    def drawdown(self) -> float:
        return self._peak - self.total_value

    def _update_peak(self):
        if self.total_value > self._peak:
            self._peak = self.total_value

    # ---------- 查询 ----------
    def position_size(self, symbol: str) -> float:
        p = self.positions.get(symbol)
        return p.size if p else 0.0

    def sellable_size(self, symbol: str) -> float:
        """当前可卖数量（T+1 约束：当日买入的部分不可卖）"""
        p = self.positions.get(symbol)
        return p.available if p else 0.0

    # ---------- T+1 ----------
    def new_day(self):
        """进入新交易日：昨日及更早买入的持仓解锁为可卖

        A股 T+1 制度：当日买入的股票当日不能卖出。引擎在每个 bar 开始时
        调用本方法；当日买入的持仓不会立刻解锁，因此同一根 bar 内
        "买入后立即卖出"会被 sellable_size() 挡下。
        """
        for p in self.positions.values():
            p.available = p.size

    # ---------- 交易 ----------
    def buy(self, symbol: str, size: float, price: float, fees: float = 0.0) -> float:
        """买入。返回成交股数。

        费用单列（不再混进成交价），并且**现金不足时直接报错，不做静默缩量**。

        为什么取消静默缩量：旧实现会偷偷把 size 改小，但引擎的成交记录仍按
        原始委托量写盘，于是记录（125,371 股）与真实持仓（125,300 股）不一致，
        最终使"净值曲线"和"账户总资产"两本账差出 0.96%。
        下单量应由调用方用 SimulatedBroker.max_affordable_size() 先算准。
        """
        if size <= 0:
            return 0.0
        cost = size * price + fees
        if cost > self.cash + 1e-9:
            raise ValueError(
                f"现金不足，拒绝买入: 需 {cost:,.2f}（成交额 {size * price:,.2f} + "
                f"费用 {fees:,.2f}），可用 {self.cash:,.2f}"
            )
        self.cash -= cost

        # 每股成本含买入费用，这样卖出时算出的 PnL 才是真实的净盈亏
        unit_cost = price + fees / size
        if symbol in self.positions:
            pos = self.positions[symbol]
            total_size = pos.size + size
            total_cost = pos.size * pos.avg_cost + size * unit_cost
            pos.size = total_size
            pos.avg_cost = total_cost / total_size if total_size > 0 else 0.0
            # 当日买入不加进 available（T+1）
        else:
            self.positions[symbol] = Position(
                symbol=symbol, size=size, available=0.0,
                avg_cost=unit_cost, current_price=price,
            )
        self._update_peak()
        return size

    def sell(self, symbol: str, size: float, price: float, fees: float = 0.0) -> float:
        """卖出（受 T+1 限制）。返回该笔的净盈亏。"""
        pos = self.positions.get(symbol)
        if not pos:
            raise ValueError(f"没有 {symbol} 的持仓，无法卖出")
        if size > pos.available + 1e-8:
            raise ValueError(
                f"{symbol} 可卖不足（T+1）: 想卖 {size:,.0f}，可卖 {pos.available:,.0f}，"
                f"总持仓 {pos.size:,.0f}"
            )

        proceeds = size * price - fees
        self.cash += proceeds

        pnl = size * (price - pos.avg_cost) - fees
        self._realized_pnl += pnl

        pos.size -= size
        pos.available -= size
        if pos.size <= 1e-8:
            del self.positions[symbol]
        self._update_peak()
        return pnl

    def update_price(self, symbol: str, price: float):
        if symbol in self.positions:
            self.positions[symbol].current_price = price
        self._update_peak()
