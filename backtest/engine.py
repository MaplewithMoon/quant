import pandas as pd
import numpy as np
from typing import Optional
from dataclasses import dataclass, field
from datetime import datetime

from data.dataset import DataSet
from strategy.base import Strategy, Action
from execution.broker import SimulatedBroker
from execution.order import Order, OrderSide, OrderType
from execution.market_rules import MarketRules, RejectionLog
from portfolio.portfolio import Portfolio
from risk.manager import RiskManager
from utils.logger import setup_logger

# 成交时点
FILL_NEXT_OPEN = "next_open"      # 第 i 根 bar 决策 → 第 i+1 根 bar 开盘成交（推荐）
FILL_SAME_CLOSE = "same_close"    # 第 i 根 bar 决策 → 第 i 根 bar 收盘成交（旧行为，含未来函数）


@dataclass
class TradeRecord:
    timestamp: datetime
    symbol: str
    side: str
    size: float
    price: float
    pnl: float
    balance: float
    commission: float = 0.0
    stamp_duty: float = 0.0
    transfer_fee: float = 0.0
    reason: str = ""

    @property
    def total_fee(self) -> float:
        return self.commission + self.stamp_duty + self.transfer_fee


class BacktestEngine:
    def __init__(
        self,
        initial_capital: float = 1_000_000,
        commission: float = 0.0001,
        slippage: float = 0.001,
        min_commission: float = 5.0,
        stamp_duty: float = 0.0005,
        transfer_fee: float = 0.00001,
        fill_timing: str = FILL_NEXT_OPEN,
        market_rules: Optional[MarketRules] = None,
        truncate_data: bool = True,
    ):
        """回测引擎

        commission:     佣金费率（默认万分之一）
        slippage:       滑点（默认 0.1%）
        min_commission: 单笔最低佣金（默认 5 元）
        stamp_duty:     印花税，仅卖出单边（默认 0.05%）
        transfer_fee:   过户费，双边（默认 0.001%）
        fill_timing:    成交时点，"next_open"（默认，无未来函数）
                        或 "same_close"（旧行为：用第 i 根 bar 收盘价决策并成交）
        market_rules:   A股交易制度规则（T+1 / 涨跌停 / 停牌 / 一手取整），
                        传 MarketRules(enabled=False) 可关闭做对照
        truncate_data:  喂给策略的数据只保留到当前 bar。
                        True 可从物理上封堵"策略偷看未来"的通道。
        """
        self.initial_capital = initial_capital
        self.broker = SimulatedBroker(
            slippage=slippage, commission=commission, min_commission=min_commission,
            stamp_duty=stamp_duty, transfer_fee=transfer_fee,
            lot_size=(market_rules or MarketRules()).lot_size,
        )
        self.portfolio = Portfolio(initial_capital)
        self.logger = setup_logger("backtest")
        self.trades: list[TradeRecord] = []
        self.risk_manager = RiskManager()
        self.fill_timing = fill_timing
        self.market_rules = market_rules if market_rules is not None else MarketRules()
        self.truncate_data = truncate_data

        self._equity: list[tuple] = []      # 逐 bar 权益快照 [(日期, 总资产)]
        self.fees_paid: float = 0.0         # 累计交易费用
        self.rejections = RejectionLog()    # 被制度约束拦下的委托
        self.pending_cancelled: int = 0     # 最后一根 bar 未能成交的挂单

    # ---------- 权益曲线（唯一账本） ----------
    @property
    def equity_curve(self) -> pd.Series:
        """逐 bar 账户权益，由引擎自己产生

        这是**唯一**的净值来源。历史上 scripts/backtest.py 另有一个
        build_equity_curve() 用成交记录重推现金与持仓，两本账实测差 0.96%
        （账户 1,040,192.63 vs 曲线 1,030,567.10），而对外指标全部取自后者。
        现在指标直接消费这条曲线，账实必然一致。
        """
        if not self._equity:
            return pd.Series(dtype=float)
        idx = pd.Index([d for d, _ in self._equity], name="trade_date")
        return pd.Series([v for _, v in self._equity], index=idx, dtype=float)

    @property
    def ledger_gap(self) -> float:
        """权益曲线末值与账户总资产的差额（健康值 = 0）"""
        if not self._equity:
            return 0.0
        return abs(self._equity[-1][1] - self.portfolio.total_value)

    # ---------- 主循环 ----------
    def run(self, dataset: DataSet, strategy: Strategy) -> pd.DataFrame:
        data = dataset.data.copy()
        symbol = dataset.symbol

        self.trades = []
        self._equity = []
        self.fees_paid = 0.0
        self.rejections = RejectionLog()
        self.pending_cancelled = 0

        pending = None      # 上一根 bar 产生、待本根 bar 执行的委托

        for i in range(len(data)):
            bar = data.iloc[i]
            close = float(bar["close"])

            # T+1：新交易日开始，解锁昨日及更早买入的持仓
            self.portfolio.new_day()

            # 1) 先执行上一根 bar 挂下的委托（next_open：用本根开盘价成交）
            if pending is not None:
                ref = float(bar["open"]) if self.fill_timing == FILL_NEXT_OPEN else close
                self._execute(pending, bar, ref, symbol)
                pending = None

            # 2) 本根 bar 产生信号
            #    截断数据：策略只能看到 0..i，无法访问未来行
            view = data.iloc[:i + 1] if self.truncate_data else data
            signal = strategy.on_bar(view, i)
            signal = self.risk_manager.filter(signal, self.portfolio)

            # 3) 成交
            if self.fill_timing == FILL_SAME_CLOSE:
                self._execute(signal, bar, close, symbol)
            elif signal.action in (Action.BUY, Action.SELL):
                pending = signal      # 留到下一根 bar 开盘执行

            self.portfolio.update_price(symbol, close)
            self._equity.append((bar.name, self.portfolio.total_value))

        if pending is not None:
            self.pending_cancelled += 1
            self.logger.debug("最后一根 bar 的信号无下一根 bar 可成交，已作废")

        report = self._build_report(dataset)

        gap = self.ledger_gap
        if gap > 1e-6:
            self.logger.error(
                f"账目不变量被破坏: 权益曲线末值 {self._equity[-1][1]:,.4f} "
                f"!= 账户总资产 {self.portfolio.total_value:,.4f} (差 {gap:,.4f})"
            )
        return report

    # ---------- 撮合 ----------
    def _execute(self, signal, bar, ref_price: float, symbol: str):
        """在参考价上尝试撮合一个信号；被制度约束拦下时记入 rejections"""
        held = self.portfolio.position_size(symbol)

        if signal.action == Action.BUY:
            if held > 0:
                return None                      # 已持仓（本引擎为单标的多头）
            ok, why = self.market_rules.check_buy(bar, ref_price)
            if not ok:
                self.rejections.add(bar.name, "buy", why)
                return None
            size = self.broker.max_affordable_size(self.portfolio.cash, ref_price)
            if size <= 0:
                return None
            order = Order(symbol=symbol, side=OrderSide.BUY, size=size,
                          order_type=OrderType.MARKET, price=ref_price)
            self.broker.place_order(order)
            filled = self.broker.execute(order, ref_price)
            self.portfolio.buy(symbol, filled.filled_qty, filled.avg_fill_price,
                               fees=filled.total_fee)
            self.fees_paid += filled.total_fee
            self._record_trade(bar, symbol, "buy", filled, 0.0)
            return filled

        if signal.action == Action.SELL:
            if held <= 0:
                return None
            # T+1：只在规则开启时生效；关闭规则时视为全部可卖（用于对照组）
            if self.market_rules.enabled and self.market_rules.t_plus > 0:
                sellable = self.portfolio.sellable_size(symbol)
            else:
                sellable = held
            if sellable <= 0:
                self.rejections.add(bar.name, "sell", "T+1 当日买入不可卖")
                return None
            ok, why = self.market_rules.check_sell(bar, ref_price)
            if not ok:
                self.rejections.add(bar.name, "sell", why)
                return None
            order = Order(symbol=symbol, side=OrderSide.SELL, size=sellable,
                          order_type=OrderType.MARKET, price=ref_price)
            self.broker.place_order(order)
            filled = self.broker.execute(order, ref_price)
            pnl = self.portfolio.sell(symbol, filled.filled_qty, filled.avg_fill_price,
                                      fees=filled.total_fee)
            self.fees_paid += filled.total_fee
            self._record_trade(bar, symbol, "sell", filled, pnl)
            return filled

        return None

    def _record_trade(self, bar, symbol: str, side: str,
                      order: Order, pnl: float):
        self.trades.append(TradeRecord(
            timestamp=bar.name,
            symbol=symbol,
            side=side,
            size=order.filled_qty,
            price=order.avg_fill_price,          # 干净成交价，不含费用
            pnl=pnl,
            balance=self.portfolio.total_value,
            commission=order.commission,
            stamp_duty=order.stamp_duty,
            transfer_fee=order.transfer_fee,
        ))

    def _build_report(self, dataset: DataSet) -> pd.DataFrame:
        snapshots = []
        for trade in self.trades:
            snapshots.append({
                "timestamp": trade.timestamp,
                "action": trade.side,
                "price": trade.price,
                "size": trade.size,
                "pnl": trade.pnl,
                "commission": trade.commission,
                "stamp_duty": trade.stamp_duty,
                "transfer_fee": trade.transfer_fee,
                "fee": trade.total_fee,
                "balance": trade.balance,
            })

        df = pd.DataFrame(snapshots)
        if not df.empty:
            df.set_index("timestamp", inplace=True)
        return df
