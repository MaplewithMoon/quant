from abc import ABC, abstractmethod
from typing import List
from .order import Order, OrderSide, OrderStatus
from .market_rules import round_lot
from .impact import SlippageModel, FixedSlippage, build_model


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
                 transfer_fee: float = 0.00001, lot_size: int = 100,
                 slippage_model: SlippageModel = None,
                 max_participation: float = 0.10):
        """模拟券商（A股费用 + 一手取整 + 冲击成本 + 参与率上限）

        slippage:          固定滑点率（当 slippage_model 为 None 时生效，默认 0.1%）
        commission:        佣金费率（默认万分之一）
        min_commission:    单笔最低佣金（默认 5 元，与国内券商一致）
        stamp_duty:        印花税，**仅卖出单边**（默认 0.05%）
        transfer_fee:      过户费，双边（默认 0.001%）
        lot_size:          一手股数，A股为 100
        slippage_model:    成交价模型（见 execution/impact.py）。传 None 则用
                           固定滑点模型；传 SqrtImpact(k) 启用平方根冲击成本
        max_participation: 单笔最多吃掉当日成交量的比例（默认 10%，0 = 不限）
                           —— 防止回测里一笔几千万的单子"无摩擦"地全部成交

        注意：印花税按成交额单边征收，是佣金的好几倍，忽略它会明显低估交易成本。
        """
        self.slippage = slippage
        self.slippage_model = slippage_model or FixedSlippage(slippage)
        self.commission = commission
        self.min_commission = min_commission
        self.stamp_duty_rate = stamp_duty
        self.transfer_fee_rate = transfer_fee
        self.lot_size = lot_size
        self.max_participation = max_participation
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

    # ---------- 成交价 ----------
    def exec_price(self, market_price: float, side: OrderSide,
                   size: float = 0.0, volume: float = None) -> float:
        """滑点/冲击后的成交价：买入加价、卖出折价

        平方根模型下成交价与下单量有关，所以 size / volume 需要传进来。
        """
        return self.slippage_model.adjust(market_price, side, size, volume)

    def impact_pct(self, size: float, volume: float) -> float:
        return self.slippage_model.impact_pct(size, volume)

    # ---------- 流动性约束 ----------
    def max_tradable_size(self, volume: float, is_buy: bool = True) -> float:
        """受参与率上限约束的可成交股数（0 或 volume 缺失 = 不限）"""
        if not self.max_participation or volume is None or volume <= 0:
            return float("inf")
        cap = float(volume) * self.max_participation
        if is_buy:
            return round_lot(cap, self.lot_size)
        return cap

    # ---------- 撮合 ----------
    def execute(self, order: Order, current_price: float,
                volume: float = None) -> Order:
        """立即全部成交

        费用**不再累加进 filled_amount**。旧实现 `order.filled_amount += cost`
        让成交额里混进了佣金，引擎随后用 filled_amount/filled_qty 当成交价：
          - 买入侧 → 价格偏高，等于多扣了一次佣金（方向恰好对）
          - 卖出侧 → 价格偏高，等于把佣金当成了收入加回现金（方向错）
        结果一个完整往返的手续费净额为 0，回测系统性偏乐观。
        """
        exec_price = self.exec_price(current_price, order.side, order.size, volume)
        turnover = order.size * exec_price
        fees = self.fee_breakdown(turnover, order.side)

        order.fill(order.size, exec_price)     # 只记成交额
        order.commission += fees["commission"]
        order.stamp_duty += fees["stamp_duty"]
        order.transfer_fee += fees["transfer_fee"]
        return order

    # ---------- 下单前的可行性 ----------
    def estimate_buy_cost(self, size: float, market_price: float,
                          volume: float = None) -> float:
        """预估买入总支出（含冲击成本、佣金、过户费）"""
        if size <= 0:
            return 0.0
        price = self.exec_price(market_price, OrderSide.BUY, size, volume)
        turnover = size * price
        return turnover + self.fees_of(turnover, OrderSide.BUY)

    def max_affordable_size(self, cash: float, market_price: float,
                            volume: float = None, lot_size: int = None) -> float:
        """在预留费用与冲击成本、并按一手取整后，现金能买的最大股数

        引擎用它来定下单量，从而不必让 Portfolio 在成交后"静默缩量"——
        静默缩量正是历史上成交记录（125,371 股）与真实持仓（125,300 股）
        脱钩、进而导致两本账差 0.96% 的根因。

        平方根冲击模型下"下单量越大成交价越高"，因此这里用迭代收敛：
        先按不含冲击的量估算，再逐步下调直到总支出不超过现金。
        """
        lot = self.lot_size if lot_size is None else lot_size
        if market_price <= 0 or cash <= 0:
            return 0.0
        price = self.exec_price(market_price, OrderSide.BUY, 0.0, volume)
        size = round_lot(cash // price, lot)
        step = lot if lot and lot > 1 else 1
        for _ in range(10000):
            if size <= 0:
                break
            if self.estimate_buy_cost(size, market_price, volume) <= cash:
                break
            size -= step
        return float(max(size, 0))

    @property
    def orders(self) -> List[Order]:
        return list(self._orders.values())
