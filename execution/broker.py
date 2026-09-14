from abc import ABC, abstractmethod
from typing import List
from .order import Order, OrderSide, OrderStatus
from .market_rules import round_lot


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
                 min_commission: float = 5.0, stamp_duty: float = 0.0005,
                 transfer_fee: float = 0.00001, lot_size: int = 100):
        """模拟券商（含 A 股实际费用与一手取整）

        slippage:       滑点（默认 0.1%）
        commission:     佣金费率（默认万分之一）
        min_commission: 单笔最低佣金（默认 5 元，与国内券商一致）
        stamp_duty:     印花税，**仅卖出单边**（默认 0.05%，2023-08 减半后税率）
        transfer_fee:   过户费，双边（默认 0.001%）
        lot_size:       一手股数，A股为 100

        注意：印花税按成交额单边征收，是佣金的好几倍，忽略它会明显低估
        交易成本——尤其是高换手的短周期策略。
        """
        self.slippage = slippage
        self.commission = commission
        self.min_commission = min_commission
        self.stamp_duty_rate = stamp_duty
        self.transfer_fee_rate = transfer_fee
        self.lot_size = lot_size
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

    # ---------- 费用 ----------
    def commission_of(self, turnover: float) -> float:
        """佣金 = max(成交额 × 费率, 最低佣金)"""
        if turnover <= 0:
            return 0.0
        return max(turnover * self.commission, self.min_commission)

    def fee_breakdown(self, turnover: float, side: OrderSide) -> dict:
        """按买卖方向拆解费用：印花税只在卖出单边征收"""
        if turnover <= 0:
            return {"commission": 0.0, "stamp_duty": 0.0, "transfer_fee": 0.0}
        return {
            "commission": self.commission_of(turnover),
            "stamp_duty": turnover * self.stamp_duty_rate if side == OrderSide.SELL else 0.0,
            "transfer_fee": turnover * self.transfer_fee_rate,
        }

    def fees_of(self, turnover: float, side: OrderSide) -> float:
        f = self.fee_breakdown(turnover, side)
        return f["commission"] + f["stamp_duty"] + f["transfer_fee"]

    # ---------- 撮合 ----------
    def exec_price(self, market_price: float, side: OrderSide) -> float:
        """滑点后的成交价：买入加价、卖出折价"""
        return round(market_price * (1 + self.slippage * side.value), 2)

    def execute(self, order: Order, current_price: float) -> Order:
        """立即全部成交

        费用**不再累加进 filled_amount**。旧实现 `order.filled_amount += cost`
        让成交额里混进了佣金，引擎随后用 filled_amount/filled_qty 当成交价：
          - 买入侧 → 价格偏高，等于多扣了一次佣金（方向恰好对）
          - 卖出侧 → 价格偏高，等于把佣金当成了收入加回现金（方向错）
        结果一个完整往返的手续费净额为 0，回测系统性偏乐观。
        """
        exec_price = self.exec_price(current_price, order.side)
        turnover = order.size * exec_price
        fees = self.fee_breakdown(turnover, order.side)

        order.fill(order.size, exec_price)     # 只记成交额
        order.commission += fees["commission"]
        order.stamp_duty += fees["stamp_duty"]
        order.transfer_fee += fees["transfer_fee"]
        return order

    # ---------- 下单前的资金可行性 ----------
    def estimate_buy_cost(self, size: float, market_price: float) -> float:
        """预估买入总支出（含滑点、佣金、过户费）"""
        if size <= 0:
            return 0.0
        price = self.exec_price(market_price, OrderSide.BUY)
        turnover = size * price
        return turnover + self.fees_of(turnover, OrderSide.BUY)

    def max_affordable_size(self, cash: float, market_price: float,
                            lot_size: int = None) -> float:
        """在预留费用与滑点、并按一手取整后，现金能买的最大股数

        引擎用它来定下单量，从而不必让 Portfolio 在成交后"静默缩量"——
        静默缩量正是历史上成交记录（125,371 股）与真实持仓（125,300 股）
        脱钩、进而导致两本账差 0.96% 的根因。

        A股必须整手买入，所以结果一定是 lot_size 的整数倍。
        """
        lot = self.lot_size if lot_size is None else lot_size
        if market_price <= 0 or cash <= 0:
            return 0.0
        price = self.exec_price(market_price, OrderSide.BUY)
        size = round_lot(cash // price, lot)
        step = lot if lot and lot > 1 else 1
        while size > 0 and self.estimate_buy_cost(size, market_price) > cash:
            size -= step
        return float(max(size, 0))

    @property
    def orders(self) -> List[Order]:
        return list(self._orders.values())
