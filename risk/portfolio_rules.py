# -*- coding: utf-8 -*-
"""组合层风控：作用在**目标权重宽表**上

【为什么要有这个文件 —— 与 risk/manager.py 的区别】
-----------------------------------------------
`risk/manager.py` 的规则签名是：

    def check(self, signal: Signal, portfolio: Portfolio) -> Signal

`Signal` 是**单标的**概念（`Action.BUY/SELL/HOLD` + `size`）。这套东西能用在
`backtest/engine.py`（一次一只股票）上，但**接不进组合引擎**：组合引擎收到的是
一张 index=交易日、columns=代码 的**权重宽表**，没有"某一只股票的买卖信号"。

这就是 D2 的真实含义：不是"忘了把风控接上"，而是**风控层是按单标的语义写的**，
而所有真实策略（多因子、板块轮动、聚宽移植）都走组合路径。所以需要一套
组合层规则，而不是给 `RiskManager` 补一行调用。

本项目里同一结构病已经出现了三次 —— 冲击成本模型、未来函数自检、
风控链，**全都只装在单标的路径上**。

【规则约定】
    每条规则实现 `adjust(target, ctx) -> (target, reason or None)`：
      - `target`: pd.Series，index=代码，值=目标权重（和为 1 或更小）
      - `ctx`:    RiskContext（含组合状态、日期、上一期权重）
      - 返回调整后的 target；未触发时原样返回且 reason=None

【方向约定】
    风控只会**降低风险**：削减权重、压低总仓位、限制换手。
    绝不允许规则放大仓位（那会把风控变成加杠杆）。`PortfolioRiskManager`
    会在最后统一 clip 到 [0, max_gross]，防止规则写错方向。
"""
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import pandas as pd


@dataclass
class RiskContext:
    """规则做判断所需的组合状态"""
    date: object = None
    total_value: float = 0.0
    peak_value: float = 0.0
    drawdown_pct: float = 0.0          # 相对峰值的回撤（负数）
    cash: float = 0.0
    # 上一期实际权重（用于换手类规则）；无则 None
    prev_weights: Optional[pd.Series] = None

    @classmethod
    def from_portfolio(cls, portfolio, date=None, prev_weights=None):
        return cls(date=date,
                   total_value=portfolio.total_value,
                   peak_value=getattr(portfolio, "_peak", portfolio.total_value),
                   drawdown_pct=portfolio.drawdown_pct,
                   cash=portfolio.cash,
                   prev_weights=prev_weights)


class PortfolioRiskRule:
    """组合层风控规则基类"""
    name = "rule"

    def adjust(self, target: pd.Series, ctx: RiskContext):
        raise NotImplementedError


@dataclass
class MaxDrawdownDeRisk(PortfolioRiskRule):
    """回撤超阈值 -> 按比例压低**总仓位**（而非清仓）

    为什么要"按比例"而不是"清仓"：清仓会在阈值附近来回抖动（回撤刚好在
    阈值上下时，一次调仓清仓、下一次又满仓），产生巨额无谓交易。用线性
    斜坡把仓位从 1 平滑压到 `min_exposure`，能显著减少这种抖动。

        drawdown <= -max_drawdown_pct                -> min_exposure
        -max_drawdown_pct < drawdown <= -start_pct   -> 线性过渡
        drawdown > -start_pct                        -> 不调整

    参数:
        max_drawdown_pct: 达到该回撤时仓位压到最低（正数，如 0.20 表示 -20%）
        start_pct:        从该回撤开始减仓（正数，如 0.10）
        min_exposure:     最低总仓位（0.0 = 允许清仓）
    """
    max_drawdown_pct: float = 0.20
    start_pct: float = 0.10
    min_exposure: float = 0.0
    name = "max_drawdown_derisk"

    def adjust(self, target: pd.Series, ctx: RiskContext):
        dd = -float(ctx.drawdown_pct)              # 转成正数表示回撤幅度
        if dd <= self.start_pct:
            return target, None
        if dd >= self.max_drawdown_pct:
            scale = self.min_exposure
        else:
            span = max(1e-12, self.max_drawdown_pct - self.start_pct)
            ratio = (dd - self.start_pct) / span      # 0 -> 1
            scale = 1.0 - ratio * (1.0 - self.min_exposure)
        return target * scale, (f"回撤 {dd:.1%} 超 {self.start_pct:.0%}，"
                                f"总仓位降至 {scale:.0%}")


@dataclass
class MaxWeightRule(PortfolioRiskRule):
    """单标的上限：任何一只的权重不得超过 max_weight

    超出的部分**削掉不外扩**（不做再分配）—— 再分配会把权重推给别的股票，
    可能又顶到上限，还可能把仓位集中到第二大的那只上，越控越集中。
    削掉后总仓位可能略小于 1，等价于留一点现金，方向是**保守**的。
    """
    max_weight: float = 0.10
    name = "max_weight"

    def adjust(self, target: pd.Series, ctx: RiskContext):
        if target is None or target.empty:
            return target, None
        over = target[target > self.max_weight + 1e-12]
        if over.empty:
            return target, None
        out = target.clip(upper=self.max_weight)
        return out, (f"{len(over)} 只超过单票上限 {self.max_weight:.0%}"
                     f"（最大 {over.max():.1%}）")


@dataclass
class MaxTurnoverRule(PortfolioRiskRule):
    """换手上限：单次调仓的目标权重相对上期变动过大时，按比例向现状靠拢

        new = prev + (target - prev) * (max_turnover / turnover)

    只在**换手超过上限**时生效，是纯粹的降风险操作。
    ⚠️ `prev_weights` 缺失时不触发 —— 宁可不动，也不要凭空假设一个持仓。
    """
    max_turnover: float = 0.5
    name = "max_turnover"

    def adjust(self, target: pd.Series, ctx: RiskContext):
        prev = ctx.prev_weights
        if prev is None or target is None or target.empty:
            return target, None
        idx = target.index.union(prev.index)
        t = target.reindex(idx).fillna(0.0)
        p = prev.reindex(idx).fillna(0.0)
        turnover = float((t - p).abs().sum())
        if turnover <= self.max_turnover + 1e-12 or turnover <= 0:
            return target, None
        k = self.max_turnover / turnover
        blended = p + (t - p) * k
        return blended.reindex(target.index), (
            f"换手 {turnover:.0%} 超上限 {self.max_turnover:.0%}，"
            f"向现有持仓靠拢至 {self.max_turnover:.0%}")


class PortfolioRiskManager:
    """组合层风控规则链

    用法:
        from risk import PortfolioRiskManager, MaxDrawdownDeRisk, MaxWeightRule
        rm = PortfolioRiskManager([MaxDrawdownDeRisk(0.20), MaxWeightRule(0.10)])
        eng.run(panel, weights, risk_manager=rm)

    ⚠️ 与 `risk.RiskManager` 不是同一个东西：那个是单标的信号过滤，见模块文档。
    """

    def __init__(self, rules: Optional[List[PortfolioRiskRule]] = None,
                 max_gross: float = 1.0):
        self.rules = list(rules or [])
        self.max_gross = float(max_gross)
        self._prev_weights: Optional[pd.Series] = None

    def add_rule(self, rule: PortfolioRiskRule) -> "PortfolioRiskManager":
        self.rules.append(rule)
        return self

    # ---------- 主入口 ----------
    def adjust(self, target: pd.Series, portfolio=None, date=None,
               holdings=None) -> Tuple[pd.Series, List[Dict]]:
        """按规则链调整目标权重，返回 (新权重, 触发记录)

        触发记录是**给报告用的**：没有它，回测结果里看不出"这次减仓是策略
        决定的还是风控砍的"，事后归因会一头雾水。
        """
        if target is None or target.empty or not self.rules:
            return target, []
        ctx = (RiskContext.from_portfolio(portfolio, date=date,
                                          prev_weights=self._prev_weights)
               if portfolio is not None else RiskContext(date=date,
                                                         prev_weights=self._prev_weights))
        events, cur = [], target
        for rule in self.rules:
            new, reason = rule.adjust(cur, ctx)
            if reason:
                events.append({"date": date, "rule": rule.name, "reason": reason})
            cur = new
        # 兜底：任何规则都不得把仓位放大（防"风控变杠杆"）
        cur = cur.clip(lower=0.0)
        gross = float(cur.sum())
        if gross > self.max_gross + 1e-12:
            cur = cur * (self.max_gross / gross)
            events.append({"date": date, "rule": "max_gross",
                           "reason": f"总仓位 {gross:.1%} 超上限 "
                                     f"{self.max_gross:.0%}，已等比压缩"})
        self._prev_weights = cur.copy()
        return cur, events

    def reset(self):
        """跑新的一段回测前清掉内部状态（prev_weights）"""
        self._prev_weights = None


def build_risk_manager(config: Optional[dict]) -> Optional[PortfolioRiskManager]:
    """从 dict 构建（便于脚本传参 / 落盘配置）

        {"max_drawdown_pct": 0.20, "start_pct": 0.10, "min_exposure": 0.0,
         "max_weight": 0.10, "max_turnover": 0.5, "max_gross": 1.0}

    全为 None / 0 时返回 None（= 不启用风控，保持原行为）。
    """
    if not config:
        return None
    rules = []
    if config.get("max_drawdown_pct"):
        rules.append(MaxDrawdownDeRisk(
            max_drawdown_pct=float(config["max_drawdown_pct"]),
            start_pct=float(config.get("start_pct") or
                            float(config["max_drawdown_pct"]) / 2),
            min_exposure=float(config.get("min_exposure", 0.0))))
    if config.get("max_weight"):
        rules.append(MaxWeightRule(float(config["max_weight"])))
    if config.get("max_turnover"):
        rules.append(MaxTurnoverRule(float(config["max_turnover"])))
    if not rules:
        return None
    return PortfolioRiskManager(rules, max_gross=float(config.get("max_gross", 1.0)))


__all__ = ["PortfolioRiskRule", "PortfolioRiskManager", "RiskContext",
           "MaxDrawdownDeRisk", "MaxWeightRule", "MaxTurnoverRule",
           "build_risk_manager"]
