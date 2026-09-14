from dataclasses import dataclass, field
from enum import Enum
from typing import Optional
from datetime import datetime
import uuid


class OrderSide(Enum):
    BUY = 1
    SELL = -1


class OrderType(Enum):
    MARKET = "market"
    LIMIT = "limit"


class OrderStatus(Enum):
    PENDING = "pending"
    FILLED = "filled"
    PARTIAL = "partial"
    CANCELED = "canceled"
    REJECTED = "rejected"


@dataclass
class Order:
    symbol: str
    side: OrderSide
    size: float
    order_type: OrderType = OrderType.MARKET
    price: Optional[float] = None
    status: OrderStatus = OrderStatus.PENDING
    order_id: str = field(default_factory=lambda: uuid.uuid4().hex[:12])
    created_at: datetime = field(default_factory=datetime.now)
    filled_qty: float = 0.0
    filled_amount: float = 0.0      # 成交额 = Σ(数量 × 成交价)，**不含任何费用**
    commission: float = 0.0         # 佣金（双边）
    stamp_duty: float = 0.0         # 印花税（仅卖出）
    transfer_fee: float = 0.0       # 过户费（双边）

    @property
    def is_active(self) -> bool:
        return self.status in (OrderStatus.PENDING, OrderStatus.PARTIAL)

    @property
    def avg_fill_price(self) -> float:
        """实际成交均价（不含费用）

        注意：以前引擎用 filled_amount/filled_qty 当成交价，而佣金被加进了
        filled_amount，导致这个"价格"里混入了费用。现在成交额与费用分列，
        这里返回的就是干净的成交价。
        """
        return self.filled_amount / self.filled_qty if self.filled_qty else 0.0

    @property
    def total_fee(self) -> float:
        """合计交易费用 = 佣金 + 印花税 + 过户费"""
        return self.commission + self.stamp_duty + self.transfer_fee

    @property
    def buy_cost(self) -> float:
        """买入总支出 = 成交额 + 费用"""
        return self.filled_amount + self.total_fee

    @property
    def sell_proceeds(self) -> float:
        """卖出净收入 = 成交额 - 费用"""
        return self.filled_amount - self.total_fee

    def fill(self, qty: float, price: float):
        self.filled_qty += qty
        self.filled_amount += qty * price
        if abs(self.filled_qty - self.size) < 1e-8:
            self.status = OrderStatus.FILLED
        else:
            self.status = OrderStatus.PARTIAL
