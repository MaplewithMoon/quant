from abc import ABC, abstractmethod
from typing import List
from .order import Order, OrderSide, OrderStatus


class Broker(ABC):
    @abstractmethod
    def place_order(self, order: Order) -> str:
        ...

    @abstractmethod
    def cancel_order(self, order_id: str) -> bool:
        ...

    @abstractmethod
    def get_order(self, order_id: str) -> Order:
        ...


class SimulatedBroker(Broker):
    def __init__(self, slippage: float = 0.001, commission: float = 0.0001,
                 min_commission: float = 5.0):
        """模拟券商
        slippage:       滑点（默认0.1%）
        commission:     佣金费率（默认万分之一 = 0.0001）
        min_commission: 单笔最低佣金（默认5元，与国内券商一致）
        """
        self.slippage = slippage
        self.commission = commission
        self.min_commission = min_commission
        self._orders: dict[str, Order] = {}

    def place_order(self, order: Order) -> str:
        self._orders[order.order_id] = order
        return order.order_id

    def cancel_order(self, order_id: str) -> bool:
        order = self._orders.get(order_id)
        if order and order.is_active:
            order.status = OrderStatus.CANCELED
            return True
        return False

    def get_order(self, order_id: str) -> Order:
        return self._orders[order_id]

    # ---------- 费用与撮合 ----------
    def commission_of(self, turnover: float) -> float:
        """单笔费用 = max(成交额 × 费率, 最低佣金)"""
        if turnover <= 0:
            return 0.0
        return max(turnover * self.commission, self.min_commission)

    def exec_price(self, market_price: float, side: OrderSide) -> float:
        """滑点后的成交价：买入加价、卖出折价"""
        return round(market_price * (1 + self.slippage * side.value), 2)

    def execute(self, order: Order, current_price: float) -> Order:
        """立即全部成交

        修正要点：费用**不再累加进 filled_amount**。
        旧实现 `order.filled_amount += cost` 让成交额里混进了佣金，引擎随后
        用 filled_amount/filled_qty 当成交价：
          - 买入侧 → 价格偏高，等于多扣了一次佣金（方向恰好对）
          - 卖出侧 → 价格偏高，等于把佣金当成了收入加回现金（方向错）
        结果一个完整往返的手续费净额为 0，回测系统性偏乐观。
        """
        exec_price = self.exec_price(current_price, order.side)
        turnover = order.size * exec_price
        fee = self.commission_of(turnover)

        order.fill(order.size, exec_price)   # 只记成交额
        order.commission += fee              # 费用单独记
        return order

    # ---------- 下单前的资金可行性 ----------
    def estimate_buy_cost(self, size: float, market_price: float) -> float:
        """预估买入总支出（含滑点与佣金）"""
        if size <= 0:
            return 0.0
        price = self.exec_price(market_price, OrderSide.BUY)
        turnover = size * price
        return turnover + self.commission_of(turnover)

    def max_affordable_size(self, cash: float, market_price: float) -> float:
        """在预留佣金与滑点后，现金能买的最大股数

        引擎用它来定下单量，从而不必让 Portfolio 在成交后"静默缩量"——
        静默缩量正是历史上成交记录（125,371 股）与真实持仓（125,300 股）
        脱钩、进而导致两本账差 0.96% 的根因。
        """
        if market_price <= 0 or cash <= 0:
            return 0.0
        price = self.exec_price(market_price, OrderSide.BUY)
        size = cash // price
        while size > 0 and self.estimate_buy_cost(size, market_price) > cash:
            size -= 1
        return float(max(size, 0))

    @property
    def orders(self) -> List[Order]:
        return list(self._orders.values())
