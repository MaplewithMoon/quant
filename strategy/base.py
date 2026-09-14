from abc import ABC, abstractmethod
from dataclasses import dataclass
from enum import Enum
from typing import Optional
import pandas as pd


class Action(Enum):
    """标准交易动作（策略层唯一允许输出的动作）"""
    BUY = 1      # 买入/增加持仓
    SELL = -1    # 卖出/减少持仓
    HOLD = 0     # 维持不动


@dataclass
class Signal:
    """策略输出标准信号（策略层不得直接操作订单/账户）

    【约定】
        - action:  只允许 Action.BUY / SELL / HOLD
        - size:    目标持仓比例（0~1），指"占组合可投资资产的比例"，
                   具体下单股数/资金由组合层(portfolio)根据资金与价格决定
        - 策略层职责止于"表达意图"，不得调用 broker/portfolio 下订单

    【为什么这样设计】
        1. 组合层统一决定资金分配 → 多策略叠加、风控拦截都方便
        2. 策略换仓逻辑与成交执行解耦 → 回测/实盘可复用同一套信号
        3. 避免策略各自乱下订单导致仓位失控
    """
    action: Action
    size: float = 1.0          # 目标持仓比例 0~1
    price: Optional[float] = None   # 建议价（可空，由组合层决定）
    reason: str = ""


class Strategy(ABC):
    """策略基类

    【子类必须遵守】
        - 只能返回 Signal，不得创建 Order、不得调用 broker/portfolio
        - Signal.size 表示目标持仓比例，由组合层换算成股数
        - 如需跨K线记忆状态（如金叉后持仓），用实例属性记录
    """
    def __init__(self, name: str = ""):
        self.name = name or self.__class__.__name__

    @abstractmethod
    def on_bar(self, data: pd.DataFrame, idx: int) -> Signal:
        """在每根K线调用，返回标准信号"""
        ...

    def compute(self, data: pd.DataFrame) -> pd.Series:
        """批量计算信号序列（仅供分析/可视化，不用于实际回测下单）"""
        signals = []
        for i in range(len(data)):
            sig = self.on_bar(data, i)
            signals.append(sig.action.value * sig.size)
        return pd.Series(signals, index=data.index)
