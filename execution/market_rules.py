# -*- coding: utf-8 -*-
"""A股交易制度约束（撮合前检查）

回测若把制度约束当成不存在，结果会系统性偏乐观：

| 约束 | 忽略后的偏差 |
|---|---|
| **T+1** | 当日买入当日即可卖 → 高估短周期/日内策略 |
| **涨跌停** | 涨停还能买入、跌停还能卖出 → 高估动量与突破策略 |
| **停牌** | 停牌日照样成交 → 高估流动性差的股票 |
| **一手 100 股** | 可买零股 → 小资金回测被高估 |
| **印花税/过户费** | 只算佣金 → 低估交易成本（印花税是佣金的好几倍） |

这里集中实现这些规则，由 `BacktestEngine` 在撮合前调用。

所需数据都已存在于数据库：
  - 涨跌停价 → `db/cleaned/limit_price/`（由配方 `limit_price` 派生）
  - 停牌     → `db/frozen/suspend/`
通过 `database.loader.load_trading_status()` 对齐到日线后即可生效；
数据缺失时规则自动跳过（不会误拦），因此对 mock 数据完全向后兼容。
"""
from dataclasses import dataclass, field

import pandas as pd

EPS = 1e-6
# 涨跌停价比较容差：价格四舍五入到分，留半个分钱的余量
LIMIT_TOL = 5e-3


def _num(bar, col):
    """从 bar 里安全取浮点数：列不存在或为 NaN 时返回 None

    ⚠️ **NaN 是有语义的**：`limit_up` / `limit_down` 为 NaN 表示该日
    **不设涨跌幅限制**（新股上市初期 —— 科创板/创业板注册制/主板 2023 起的
    前 5 个交易日、主板 2013-2023 首日不适用 44% 的重新上市等）。
    返回 None 会让调用方**跳过**涨跌停检查，正是"无限制"应有的行为。
    见 `database/limit_rules.py` 的 `LISTING_REGIMES`。
    """
    try:
        v = bar.get(col)
    except Exception:
        return None
    if v is None:
        return None
    try:
        f = float(v)
    except (TypeError, ValueError):
        return None
    return None if pd.isna(f) else f


def _flagged(bar, col) -> bool:
    """从 bar 里安全取布尔标记（0/1、True/False、NaN 都兼容）"""
    try:
        v = bar.get(col)
    except Exception:
        return False
    if v is None:
        return False
    try:
        f = float(v)
    except (TypeError, ValueError):
        return bool(v)
    if pd.isna(f):
        return False
    return f > 0


def round_lot(size: float, lot_size: int = 100) -> float:
    """按一手股数向下取整"""
    if lot_size and lot_size > 1:
        return float(int(size // lot_size) * lot_size)
    return float(int(size))


@dataclass
class MarketRules:
    """A股交易制度规则

    enabled:                总开关（False 则全部放行，用于对照组）
    lot_size:               一手股数，A股为 100
    t_plus:                 交割制度，A股为 T+1
    block_suspended:        停牌不可交易
    block_limit_up_buy:     涨停（封板）不可买入
    block_limit_down_sell:  跌停（封板）不可卖出

    ⚠️ `limit_up` / `limit_down` 为 **NaN 表示"不设涨跌幅限制"**（新股上市
    初期），此时检查被**跳过**（放行）—— 这是有意为之，不是缺数据。
    """
    enabled: bool = True
    lot_size: int = 100
    t_plus: int = 1
    block_suspended: bool = True
    block_limit_up_buy: bool = True
    block_limit_down_sell: bool = True

    # ---------- 买入检查 ----------
    def check_buy(self, bar, ref_price: float):
        """返回 (是否允许, 原因)

        "买不到"的判定用**封板**口径，而不是简单地比价格：
          - 参考价已在涨停价上（如开盘即涨停），或
          - 全天最低价都没跌破涨停价（一字板 / 封死）
        这样即使涨跌停价数据有几分钱误差也不易误判；
        若股票盘中曾跌破涨停价，说明确实买得到，就不该拦。
        """
        if not self.enabled:
            return True, ""
        if self.block_suspended and _flagged(bar, "suspended"):
            return False, "停牌不可买"
        if self.block_limit_up_buy:
            lu = _num(bar, "limit_up")
            if lu is not None:
                low = _num(bar, "low")
                if ref_price >= lu - LIMIT_TOL:
                    return False, f"参考价已在涨停({ref_price:.2f}≥{lu:.2f})，买不到"
                if low is not None and low >= lu - LIMIT_TOL:
                    return False, f"涨停封板不可买(全天最低 {low:.2f}≥涨停 {lu:.2f})"
        return True, ""

    # ---------- 卖出检查 ----------
    def check_sell(self, bar, ref_price: float):
        """返回 (是否允许, 原因)；判定口径与 check_buy 对称"""
        if not self.enabled:
            return True, ""
        if self.block_suspended and _flagged(bar, "suspended"):
            return False, "停牌不可卖"
        if self.block_limit_down_sell:
            ld = _num(bar, "limit_down")
            if ld is not None:
                high = _num(bar, "high")
                if ref_price <= ld + LIMIT_TOL:
                    return False, f"参考价已在跌停({ref_price:.2f}≤{ld:.2f})，卖不掉"
                if high is not None and high <= ld + LIMIT_TOL:
                    return False, f"跌停封板不可卖(全天最高 {high:.2f}≤跌停 {ld:.2f})"
        return True, ""


@dataclass
class RejectedTrade:
    """被制度约束拦下的委托（用于回测后汇总，把"没成交"的原因讲清楚）"""
    timestamp: object
    action: str
    reason: str


@dataclass
class RejectionLog:
    items: list = field(default_factory=list)

    def add(self, timestamp, action, reason):
        self.items.append(RejectedTrade(timestamp, action, reason))

    def summary(self) -> dict:
        """按原因汇总，便于一眼看出回测被哪条规则影响最大"""
        out = {}
        for r in self.items:
            out[r.reason] = out.get(r.reason, 0) + 1
        return out
