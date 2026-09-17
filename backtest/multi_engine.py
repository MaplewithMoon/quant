# -*- coding: utf-8 -*-
"""多标的组合回测引擎

与单标的引擎（backtest/engine.py）的关系
----------------------------------------
`BacktestEngine` 是"一次一只股票、策略直接给买卖信号"，适合验证单票择时。
组合策略需要的是完全不同的东西：**在 N 只股票之间分配资金、按目标权重调仓**。
本模块负责后者。

设计要点
--------
1. **输入是目标权重面板**，不是信号。权重由 `portfolio.construction` 从
   因子打分 + 股票池算出，引擎只负责"把当前持仓调到目标权重"。
2. **每个 bar 只做一次调仓**：先卖后买（卖出释放现金），买入按可用现金等比缩减。
3. **完整复用执行层**：`Portfolio`（T+1 可卖数量 + 费用单列）、
   `SimulatedBroker`（滑点/冲击/佣金/印花税/过户费/整手）、
   `MarketRules`（停牌 / 涨跌停封板）。
4. **唯一账本**：逐 bar 快照 `portfolio.total_value`，`ledger_gap` 必须为 0。
5. **成交时点**：默认 `next_open`（T 日收盘算权重 → T+1 开盘成交），
   与单标的引擎口径一致，避免"用当日收盘价决策并按同一价格成交"的未来函数。

用法
----
    from backtest.multi_engine import PortfolioBacktestEngine

    eng = PortfolioBacktestEngine(initial_capital=1_000_000)
    res = eng.run(price_panel, target_weights)   # 都是宽表 / dict of 宽表
    print(res.metrics.total_return, res.equity.iloc[-1])

    # 带市场级择时（总仓位控制）：
    from factors.market import market_signals, exposure_from_signal
    sig = market_signals("2020-01-01", "2024-12-31")
    expo = exposure_from_signal(sig["IC_basis"])
    res = eng.run(price_panel, target_weights, exposure=expo)
"""
from dataclasses import dataclass, field
from typing import Dict, Optional

import numpy as np
import pandas as pd

from backtest.metrics import Metrics
from execution.broker import SimulatedBroker
from execution.market_rules import MarketRules, round_lot
from execution.order import OrderSide
from portfolio.portfolio import Portfolio
from utils.logger import setup_logger

FILL_NEXT_OPEN = "next_open"
FILL_SAME_CLOSE = "same_close"


@dataclass
class MultiBacktestResult:
    equity: pd.Series                  # 逐日总资产
    trades: pd.DataFrame               # 成交明细
    holdings: pd.DataFrame             # 逐日持仓权重
    metrics: Metrics
    rejections: Dict[str, int] = field(default_factory=dict)
    ledger_gap: float = 0.0
    lookahead_report: str = ""         # 成交时点前视自检报告
    lookahead_violations: int = 0      # >0 说明成交早于决策，必须查
    same_day_fills: int = 0            # fill=same_close 时的同日成交笔数
    risk_events: list = field(default_factory=list)   # 风控触发的记录

    def summary(self) -> str:
        m = self.metrics
        L = ["=" * 74, "组合回测结果", "=" * 74]
        L.append(f"  总收益率  : {m.total_return:>9.2%}")
        L.append(f"  年化收益率: {m.annual_return:>9.2%}")
        L.append(f"  年化波动  : {m.annual_volatility:>9.2%}")
        L.append(f"  夏普比率  : {m.sharpe_ratio:>9.2f}")
        L.append(f"  最大回撤  : {m.max_drawdown:>9.2%}")
        L.append(f"  交易笔数  : {m.total_trades:>9d}")
        L.append(f"  胜率      : {m.win_rate:>9.2%}")
        L.append(f"  账目差额  : {self.ledger_gap:>9.2e}  (应=0)")
        if self.rejections:
            L.append("")
            L.append("  —— 被制度约束拦下的委托 ——")
            for k, v in sorted(self.rejections.items(), key=lambda kv: -kv[1]):
                L.append(f"    {v:>6} 次  {k}")
        if self.risk_events:
            L.append("")
            L.append("  —— 风控触发 ——")
            for e in self.risk_events[:10]:
                L.append(f"    {str(e.get('date'))[:10]}  {e.get('reason')}")
            if len(self.risk_events) > 10:
                L.append(f"    ... 另有 {len(self.risk_events) - 10} 次")
        if self.lookahead_report:
            L.append("")
            L.append("  —— 前视自检 ——")
            L.append(self.lookahead_report)
        L.append("=" * 74)
        return "\n".join(L)


class PortfolioBacktestEngine:
    def __init__(self, initial_capital: float = 1_000_000,
                 commission: float = 0.0001, min_commission: float = 5.0,
                 stamp_duty: float = 0.0005, transfer_fee: float = 0.00001,
                 lot_size: int = 100, slippage: float = 0.001,
                 slippage_model=None, max_participation: float = 0.10,
                 fill_timing: str = FILL_NEXT_OPEN,
                 market_rules: Optional[MarketRules] = None):
        self.initial_capital = initial_capital
        self.broker = SimulatedBroker(
            slippage=slippage, commission=commission,
            min_commission=min_commission, stamp_duty=stamp_duty,
            transfer_fee=transfer_fee, lot_size=lot_size,
            slippage_model=slippage_model, max_participation=max_participation)
        self.lot_size = lot_size
        self.fill_timing = fill_timing
        self.market_rules = market_rules if market_rules is not None else MarketRules()
        self.logger = setup_logger("portfolio_backtest")

    # ---------- 主循环 ----------
    def run(self, panel: dict, target_weights: pd.DataFrame,
            exposure: Optional[pd.Series] = None,
            risk_manager=None,
            audit_lookahead: bool = True,
            verbose: bool = False) -> MultiBacktestResult:
        """执行组合回测

        参数:
            panel:  dict，至少包含 'close'、'open'；可选 'volume'、
                    'limit_up'、'limit_down'、'suspended'（都是宽表）
            target_weights: 宽表，调仓日为权重、非调仓日为 NaN
                             （portfolio.construction.build_target_weights 的产出）
            exposure: 目标总仓位（index=交易日, 0~1），可选。给定时**只在调仓日**
                    缩放目标权重 —— 即择时只通过"这次调仓多买还是少买"生效，
                    调仓日之间不因为信号变化而临时加减仓（那会引入大量无谓交易，
                    也会把"信号噪声"当成调仓理由）。非调仓日与缺失日按 1.0 处理。
            risk_manager: 组合层风控（`risk.PortfolioRiskManager`），在**每次调仓前**
                    调整目标权重。注意这**不是** `risk.RiskManager`：那套规则签名是
                    `check(signal: Signal, ...)`，是**单标的**语义，套不到权重宽表上。
            audit_lookahead: 回测后做成交时点前视自检（默认开）。
                    ⚠️ 策略**实际用了哪些数据**引擎无从得知 —— 那部分要由调用方用
                    `backtest.lookahead.verify_point_in_time` 或
                    `audit_decision_inputs` 逐决策校验。引擎只能保证
                    "成交不早于决策"，但这一条恰恰是组合路径最容易出错的地方
                    （决策日和成交日共用一个日期索引）。
        """
        if exposure is not None:
            from factors.market import apply_exposure
            target_weights = apply_exposure(target_weights, exposure)
        close = panel["close"]
        opn = panel.get("open", close)

        dates = close.index
        codes = list(close.columns)
        tw = target_weights.reindex(index=dates, columns=codes)

        pf = Portfolio(self.initial_capital)
        equity, trades, holds = [], [], []
        rejects: Dict[str, int] = {}
        pending = None      # (调仓日, 目标权重) —— next_open 模式下待执行
        rebalance_dates = []          # 决策日（供前视自检）
        risk_events = []              # 风控触发记录

        for i, d in enumerate(dates):
            pf.new_day()                     # T+1 解锁

            # 1) 先执行上一交易日挂下的调仓（next_open：用今日开盘价）
            if pending is not None:
                tdate, tw_row = pending
                ref = self._ref_prices(opn, d)
                self._rebalance(pf, tw_row, ref, d, panel, trades, rejects)
                pending = None

            # 2) 今日若是调仓日，产生目标权重
            row = tw.loc[d]
            is_reb = row.notna().any()
            if is_reb:
                rebalance_dates.append(d)
                # 组合层风控：在挂单**之前**调整目标权重
                if risk_manager is not None:
                    row, ev = risk_manager.adjust(
                        row, portfolio=pf, date=d, holdings=pf.positions)
                    risk_events.extend(ev)
                if self.fill_timing == FILL_SAME_CLOSE:
                    ref = self._ref_prices(close, d)
                    self._rebalance(pf, row, ref, d, panel, trades, rejects)
                else:
                    pending = (d, row)

            # 3) 收盘估值
            last = close.loc[d]
            for c in codes:
                px = last.get(c)
                if px is not None and np.isfinite(px):
                    pf.update_price(c, float(px))
            total = pf.total_value
            equity.append((d, total))
            holds.append(self._weights_of(pf, last, total))

        eq = pd.Series([v for _, v in equity],
                       index=pd.Index([d for d, _ in equity], name="trade_date"))
        tdf = self._trades_frame(trades)
        hdf = pd.DataFrame(holds, index=eq.index)
        gap = abs(eq.iloc[-1] - pf.total_value) if len(eq) else 0.0
        if gap > 1e-6:
            self.logger.error(f"账目不变量被破坏: 权益 {eq.iloc[-1]:,.4f} "
                              f"!= 账户 {pf.total_value:,.4f} (差 {gap:,.4f})")

        # 成交时点前视自检（引擎能自己保证的那部分 PIT）
        la_report, la_bad, same_day = "", 0, 0
        if audit_lookahead:
            from backtest.lookahead import check_fill_timing
            r = check_fill_timing(rebalance_dates, tdf, fill_timing=self.fill_timing)
            la_report = r["report"]
            la_bad = len(r["violations"])
            same_day = r["same_day_fills"]
            if la_bad:
                self.logger.error(
                    f"前视自检发现 {la_bad} 笔成交早于其决策日 —— "
                    f"回测结果不可信，请检查 fill_timing 与调仓日的对齐")

        return MultiBacktestResult(equity=eq, trades=tdf, holdings=hdf,
                                   metrics=Metrics.compute(eq, tdf),
                                   rejections=rejects, ledger_gap=gap,
                                   lookahead_report=la_report,
                                   lookahead_violations=la_bad,
                                   same_day_fills=same_day,
                                   risk_events=risk_events)

    # ---------- 内部 ----------
    @staticmethod
    def _ref_prices(px: pd.DataFrame, d) -> Dict[str, float]:
        row = px.loc[d]
        out = {}
        for c, v in row.items():
            if v is not None and np.isfinite(v) and v > 0:
                out[c] = float(v)
        return out

    def _bar(self, panel: dict, d, code: str) -> dict:
        """构造 MarketRules 需要的 bar（缺失字段返回 None）"""
        bar = {}
        for k in ("low", "high", "limit_up", "limit_down", "suspended"):
            df = panel.get(k)
            if df is None:
                continue
            try:
                v = df.at[d, code]
            except KeyError:
                continue
            bar[k] = None if (v is None or (isinstance(v, float) and np.isnan(v))) else v
        return bar

    def _volume(self, panel: dict, d, code: str):
        v = panel.get("volume")
        if v is None:
            return None
        try:
            x = v.at[d, code]
        except KeyError:
            return None
        return None if (x is None or not np.isfinite(x)) else float(x)

    def _weights_of(self, pf: Portfolio, last: pd.Series, total: float) -> Dict[str, float]:
        out = {}
        if total <= 0:
            return out
        for c, pos in pf.positions.items():
            px = last.get(c)
            if px is not None and np.isfinite(px):
                out[c] = pos.size * float(px) / total
        return out

    def _rebalance(self, pf: Portfolio, target_w: pd.Series,
                   ref: Dict[str, float], d, panel: dict,
                   trades: list, rejects: Dict[str, int]):
        """把当前持仓调整到目标权重：先卖后买"""
        total = pf.total_value
        if total <= 0:
            return
        symbols = sorted(set(target_w.dropna().index) | set(pf.positions))
        tw = target_w.reindex(symbols).fillna(0.0)

        # ---- 目标股数（无参考价的股票保持不动）----
        desired: Dict[str, Optional[float]] = {}
        for c in symbols:
            px = ref.get(c)
            if px is None:
                desired[c] = None                     # 停牌/无价 -> 不动
                continue
            w = float(tw.get(c, 0.0) or 0.0)
            desired[c] = round_lot(max(total * w, 0.0) / px, self.lot_size)

        # ---- 卖出（先卖释放现金）----
        for c in symbols:
            tgt = desired[c]
            if tgt is None:
                continue
            cur = pf.position_size(c)
            if tgt >= cur:
                continue
            qty = cur - tgt
            if self.market_rules.enabled and self.market_rules.t_plus > 0:
                qty = min(qty, pf.sellable_size(c))
            if qty <= 0:
                self._reject(rejects, "T+1 当日买入不可卖")
                continue
            px = ref[c]
            ok, why = self.market_rules.check_sell(self._bar(panel, d, c), px)
            if not ok:
                self._reject(rejects, why)
                continue
            vol = self._volume(panel, d, c)
            qty = min(qty, self.broker.max_tradable_size(vol, is_buy=False))
            if qty <= 0:
                continue
            filled = self.broker.exec_price(px, OrderSide.SELL, qty, vol)
            turnover = qty * filled
            fees = self.broker.fees_of(turnover, OrderSide.SELL)
            try:
                pnl = pf.sell(c, qty, filled, fees=fees)
            except ValueError:
                continue
            trades.append(self._rec(d, c, "sell", qty, filled, fees, pnl, total))

        # ---- 买入（按可用现金等比缩减）----
        buys = []
        for c in symbols:
            tgt = desired[c]
            if tgt is None:
                continue
            cur = pf.position_size(c)
            if tgt <= cur:
                continue
            px = ref[c]
            ok, why = self.market_rules.check_buy(self._bar(panel, d, c), px)
            if not ok:
                self._reject(rejects, why)
                continue
            vol = self._volume(panel, d, c)
            qty = tgt - cur
            qty = min(qty, self.broker.max_tradable_size(vol, is_buy=True))
            if qty <= 0:
                continue
            buys.append((c, qty, px, vol))

        if buys:
            cash = pf.cash
            # 先按目标量估计总花费
            est = [self.broker.estimate_buy_cost(qty, px, vol)
                   for c, qty, px, vol in buys]
            need = sum(est)
            scale = min(1.0, cash / need) if need > 0 else 0.0
            for (c, qty, px, vol), e in zip(buys, est):
                if scale < 1.0:
                    qty = round_lot(qty * scale, self.lot_size)
                if qty <= 0:
                    continue
                ep = self.broker.exec_price(px, OrderSide.BUY, qty, vol)
                turnover = qty * ep
                fees = self.broker.fees_of(turnover, OrderSide.BUY)
                if turnover + fees > pf.cash + 1e-9:
                    qty = round_lot(max(pf.cash - fees, 0) / ep, self.lot_size)
                    if qty <= 0:
                        continue
                    turnover = qty * ep
                    fees = self.broker.fees_of(turnover, OrderSide.BUY)
                try:
                    pf.buy(c, qty, ep, fees=fees)
                except ValueError:
                    continue
                trades.append(self._rec(d, c, "buy", qty, ep, fees, 0.0, total))

    @staticmethod
    def _reject(rejects: Dict[str, int], why: str):
        key = why.split("(")[0].strip()
        rejects[key] = rejects.get(key, 0) + 1

    @staticmethod
    def _rec(d, code, side, qty, price, fees, pnl, total) -> dict:
        return {"timestamp": d, "code": code, "action": side, "size": float(qty),
                "price": float(price), "fee": float(fees), "pnl": float(pnl),
                "balance": float(total)}

    @staticmethod
    def _trades_frame(rows) -> pd.DataFrame:
        if not rows:
            return pd.DataFrame(columns=["timestamp", "code", "action", "size",
                                         "price", "fee", "pnl", "balance"])
        df = pd.DataFrame(rows)
        df["stamp_duty"] = np.nan
        return df.set_index("timestamp")
