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

    def execute(self, order: Order, current_price: float) -> Order:
        exec_price = current_price * (1 + self.slippage * order.side.value)
        exec_price = round(exec_price, 2)
        # 佣金 = max(成交额 × 费率, 单笔最低佣金)
        turnover = order.size * exec_price
        cost = max(turnover * self.commission, self.min_commission)
        order.fill(order.size, exec_price)
        order.filled_amount += cost
        return order

    @property
    def orders(self) -> List[Order]:
        return list(self._orders.values())
