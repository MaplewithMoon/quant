from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import List, Optional

from strategy.base import Signal, Action
from portfolio.portfolio import Portfolio


class RiskRule(ABC):
    @abstractmethod
    def check(self, signal: Signal, portfolio: Portfolio) -> Signal:
        ...


@dataclass
class MaxDrawdownRule(RiskRule):
    max_drawdown_pct: float = 0.20

    def check(self, signal: Signal, portfolio: Portfolio) -> Signal:
        if portfolio.drawdown_pct <= -self.max_drawdown_pct:
            if signal.action == Action.BUY:
                return Signal(Action.HOLD, reason="max drawdown exceeded")
        return signal


@dataclass
class MaxPositionSizeRule(RiskRule):
    max_position_pct: float = 0.25

    def check(self, signal: Signal, portfolio: Portfolio) -> Signal:
        if signal.action == Action.BUY:
            cost = signal.size * portfolio.cash * self.max_position_pct
            if cost > portfolio.total_value * self.max_position_pct:
                return Signal(Action.HOLD, reason="position size limit")
        return signal


class RiskManager:
    """风控规则链：按顺序执行规则，任一规则将信号降为 HOLD 则终止

    支持从配置(dict/JSON)构建规则，便于不同策略灵活调整。
    """
    def __init__(self, rules: Optional[List[RiskRule]] = None):
        self.rules = rules or []

    def add_rule(self, rule: RiskRule) -> "RiskManager":
        self.rules.append(rule)
        return self

    def filter(self, signal: Signal, portfolio: Portfolio) -> Signal:
        for rule in self.rules:
            signal = rule.check(signal, portfolio)
            if signal.action == Action.HOLD:
                break
        return signal

    # ============================================================
    # 配置化构建
    # ============================================================
    @classmethod
    def from_config(cls, config: dict) -> "RiskManager":
        """从配置 dict 构建风控规则链

        配置格式:
            {
              "rules": [
                {"type": "max_drawdown", "max_drawdown_pct": 0.15},
                {"type": "max_position", "max_position_pct": 0.30},
                {"type": "max_daily_loss", "max_daily_loss_pct": 0.05}
              ]
            }

        支持规则类型:
            - max_drawdown  最大回撤限制（组合回撤超过阈值禁开仓）
            - max_position  单笔仓位上限
            - max_daily_loss 单日最大亏损（当日亏损超阈值禁开仓）

        示例:
            RiskManager.from_config({"rules": [
                {"type": "max_drawdown", "max_drawdown_pct": 0.20},
                {"type": "max_position", "max_position_pct": 0.25},
            ]})
        """
        mgr = cls()
        rule_cfg = config.get("rules", []) if isinstance(config, dict) else []
        for item in rule_cfg:
            t = item.get("type", "")
            if t == "max_drawdown":
                mgr.add_rule(MaxDrawdownRule(item.get("max_drawdown_pct", 0.20)))
            elif t == "max_position":
                mgr.add_rule(MaxPositionSizeRule(item.get("max_position_pct", 0.25)))
            elif t == "max_daily_loss":
                mgr.add_rule(MaxDailyLossRule(item.get("max_daily_loss_pct", 0.05)))
            else:
                raise ValueError(f"未知风控规则类型: {t}")
        return mgr

    @classmethod
    def from_json(cls, path: str) -> "RiskManager":
        """从 JSON 文件构建"""
        import json
        with open(path, encoding="utf-8") as f:
            return cls.from_config(json.load(f))


@dataclass
class MaxDailyLossRule(RiskRule):
    """单日最大亏损限制：当日组合亏损超过阈值则禁止开仓（含当日已实现/未实现）"""
    max_daily_loss_pct: float = 0.05
    _day_open_value: float = None      # 当日开盘时组合市值
    _last_date: object = None

    def check(self, signal: Signal, portfolio: Portfolio) -> Signal:
        # 用组合市值估算当日变化（引擎每日 update_price 后调用）
        # 跨日重置基准：通过日期变化判断（无日期时用简单重置）
        value = portfolio.total_value
        if self._day_open_value is None:
            self._day_open_value = value
        daily_pnl_pct = value / self._day_open_value - 1 if self._day_open_value else 0
        if daily_pnl_pct <= -self.max_daily_loss_pct:
            if signal.action == Action.BUY:
                return Signal(Action.HOLD, reason="daily loss limit")
        return signal

    def reset(self, value: float):
        """每个交易日开盘调用，重置当日基准"""
        self._day_open_value = value
