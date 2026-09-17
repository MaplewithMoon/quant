"""风控层

**这里有两套风控，别用错：**

| | `RiskManager`（manager.py） | `PortfolioRiskManager`（portfolio_rules.py） |
|---|---|---|
| 作用对象 | `Signal`（单标的买卖信号） | 目标**权重宽表**（index=交易日, columns=代码） |
| 用在哪 | `backtest/engine.py`（一次一只股票） | `backtest/multi_engine.py`（组合） |
| 规则 | 最大回撤禁买 / 单笔仓位 / 单日亏损 | 回撤降仓 / 单票上限 / 换手上限 |

所有真实策略（多因子、板块轮动、聚宽移植）都走**组合**路径，所以实际生效的
是 `PortfolioRiskManager`。`RiskManager` 保留给单标的验证脚本。
"""
from .manager import RiskManager, RiskRule, MaxDrawdownRule, MaxPositionSizeRule
from .portfolio_rules import (PortfolioRiskManager, PortfolioRiskRule, RiskContext,
                              MaxDrawdownDeRisk, MaxWeightRule, MaxTurnoverRule,
                              build_risk_manager)

__all__ = ["RiskManager", "RiskRule", "MaxDrawdownRule", "MaxPositionSizeRule",
           "PortfolioRiskManager", "PortfolioRiskRule", "RiskContext",
           "MaxDrawdownDeRisk", "MaxWeightRule", "MaxTurnoverRule",
           "build_risk_manager"]
