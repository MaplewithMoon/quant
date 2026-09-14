# -*- coding: utf-8 -*-
"""滑点与市场冲击成本模型

为什么需要
----------
回测里最容易低估的就是「你的单子自己会推动价格」。固定比例滑点假设无论下
多大的单子成本都一样，真实市场并不如此：**下单量占当日成交量的比例越高，
冲击成本越大**，经验上是平方根关系。

本项目原本在 `scripts/semiconductor_rotation.py` 里手写了一套平方根模型，
但框架的 `SimulatedBroker` 只有固定滑点 —— 于是同一份数据在两处跑出不同结果。
这里把它上收进执行层，成为统一、可切换、可测试的实现。

模型
----
    FixedSlippage(rate)   成交价 = 市价 × (1 ± rate)          与下单量无关（默认，旧行为）
    SqrtImpact(k, cap)    成交价 = 市价 × (1 ± k·√(Q/V))      与下单量占比成平方根
                          Q = 下单股数, V = 当日成交量(股)

⚠️ 关于 k 的取值（重要）
------------------------
本模型是**参与率形式**（只跟 Q/V 有关），k 是无量纲系数，**必须针对自己的
标的与资金规模做校准，不能照抄**。实测同一策略（000001/MA/2024，100 万本金，
下单量约占当日成交量的 0.1%）：

| k | 总收益率 | 平均冲击 | 相对无滑点 |
|---:|---:|---:|---:|
| 0（无滑点） | +3.95% | 0 | — |
| 固定滑点 0.1% | +1.89% | 0.10% | -2.06pp |
| 0.02 | +2.72% | 0.06% | -1.24pp |
| 0.1（项目原实现） | -2.52% | 0.31% | -6.48pp |
| 0.5 | -23.49% | 1.45% | -27.45pp |

k 从 0.1 变到 0.5 收益差 21pp —— **这个参数足以决定策略生死**，
所以默认保持 `fixed`（不启用），必须显式 `--impact-model sqrt` 才生效。

理论/业界更常用的形式带上了波动率：

    impact ≈ c · σ_daily · √(Q/V)      （σ 为该股日波动率, c 约 0.5~1）

上例参与率 0.1%、σ=2%、c=1 时冲击仅约 **6.3bp**，对应 k ≈ 0.02 ——
也就是说 k=0.1 对一只低波动大盘股偏激进约 5 倍。若你的标的波动率高、
流动性差，k 应当更大；请用 `--impact-model sqrt` 跑一遍，看输出的
「平均冲击 / 单笔最大冲击」是否落在你对真实交易成本的认知范围内，再反调 k。

校准辅助：`.deps/calibrate_impact.py` 会打印 k 从 0 到 0.5 的收益-冲击曲线，
并给出波动率形式对应的量级供对照。
"""
from abc import ABC, abstractmethod
from math import sqrt


class SlippageModel(ABC):
    """成交价模型基类"""

    name = "abstract"

    @abstractmethod
    def impact_pct(self, size: float, volume: float) -> float:
        """返回冲击/滑点比例（正数，如 0.001 = 0.1%）"""

    def adjust(self, market_price: float, side, size: float = 0.0,
               volume: float = None) -> float:
        """把市价调整为成交价：买入加价、卖出折价

        side 为 OrderSide（BUY.value=+1 / SELL.value=-1），这里只用其 value，
        避免与 execution.order 形成循环依赖。
        """
        p = self.impact_pct(size, volume)
        return round(market_price * (1 + p * side.value), 2)


class NoSlippage(SlippageModel):
    """零滑点：仅用于隔离变量做对照实验"""
    name = "none"

    def impact_pct(self, size, volume):
        return 0.0


class FixedSlippage(SlippageModel):
    """固定比例滑点（默认模型，与历史行为一致）"""
    name = "fixed"

    def __init__(self, rate: float = 0.001):
        self.rate = rate

    def impact_pct(self, size, volume):
        return self.rate


class SqrtImpact(SlippageModel):
    """平方根市场冲击模型

    参数:
        k:          冲击系数（默认 0.1）
        max_impact: 冲击上限（默认 5%，防止极端情况算出离谱价格）
        min_volume: 当日成交量低于该值时视为流动性极差，直接给到 max_impact

    说明:
        volume 缺失或为 0 时返回 0 冲击（不阻塞回测）。若希望"没有成交量
        数据就不许成交"，应在 MarketRules 里加约束，而不是让价格模型
        悄悄给出错误价格。
    """
    name = "sqrt"

    def __init__(self, k: float = 0.1, max_impact: float = 0.05):
        self.k = k
        self.max_impact = max_impact

    def impact_pct(self, size, volume):
        if volume is None or volume <= 0 or size is None or size <= 0:
            return 0.0
        participation = size / float(volume)
        return min(self.k * sqrt(participation), self.max_impact)


def build_model(kind: str, rate: float = 0.001, k: float = 0.1,
                max_impact: float = 0.05) -> SlippageModel:
    """按名字构造模型，便于 CLI 传参"""
    kind = (kind or "fixed").lower()
    if kind in ("none", "no", "0"):
        return NoSlippage()
    if kind == "fixed":
        return FixedSlippage(rate)
    if kind == "sqrt":
        return SqrtImpact(k=k, max_impact=max_impact)
    raise ValueError(f"未知滑点模型 {kind!r}，可选: none / fixed / sqrt")
