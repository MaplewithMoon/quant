"""现代量化策略集"""
import numpy as np
from .base import Strategy, Signal, Action


class DualThrust(Strategy):
    """Dual Thrust — 价格通道突破策略

    源自《Futures Truth》杂志排名前列的策略。
    原理：以昨日价格区间为基准，向上/向下突破特定阈值入场。
    在 A 股期货和股票中广泛使用。
    """
    def __init__(self, k1: float = 0.5, k2: float = 0.5, lookback: int = 1):
        super().__init__(f"DualThrust_{k1}_{k2}")
        self.k1 = k1
        self.k2 = k2
        self.lookback = lookback
        self._in_position = False

    def on_bar(self, data, idx):
        if idx < self.lookback + 1:
            return Signal(Action.HOLD)

        close = data["close"].iloc[idx]
        prev = data.iloc[idx - self.lookback]

        hh = max(prev["high"], prev["close"])
        ll = min(prev["low"], prev["close"])
        hc = prev["close"] - min(prev["low"], prev["close"])
        lc = max(prev["high"], prev["close"]) - prev["close"]
        r = max(hh - ll, hc, lc)
        if r == 0:
            return Signal(Action.HOLD)

        buy_line = prev["open"] + self.k1 * r
        sell_line = prev["open"] - self.k2 * r

        if not self._in_position:
            if close > buy_line:
                self._in_position = True
                return Signal(Action.BUY, reason="dual_thrust_buy")
        else:
            if close < sell_line:
                self._in_position = False
                return Signal(Action.SELL, reason="dual_thrust_sell")

        return Signal(Action.HOLD)


class GridStrategy(Strategy):
    """网格交易策略

    在价格区间内等距挂买单和卖单，高抛低吸。
    适合震荡行情，不依赖方向判断，自动低买高卖。
    """
    def __init__(self, lower_pct: float = -0.10, upper_pct: float = 0.10,
                 grids: int = 10, initial_capital_ratio: float = 0.5):
        super().__init__(f"Grid_{grids}g")
        self.lower_pct = lower_pct
        self.upper_pct = upper_pct
        self.grids = grids
        self.initial_capital_ratio = initial_capital_ratio
        self._base_price = 0.0
        self._grid_levels = []
        self._position_per_grid = 0.0
        self._initialized = False

    def on_bar(self, data, idx):
        if idx < 20:
            return Signal(Action.HOLD)

        if not self._initialized:
            self._base_price = data["close"].iloc[:20].mean()
            step = (self.upper_pct - self.lower_pct) / self.grids
            self._grid_levels = [
                self._base_price * (1 + self.lower_pct + i * step)
                for i in range(self.grids + 1)
            ]
            self._position_per_grid = self.initial_capital_ratio / self.grids
            self._initialized = True
            return Signal(Action.HOLD)

        close = data["close"].iloc[idx]
        prev_close = data["close"].iloc[idx - 1]

        for i, level in enumerate(self._grid_levels):
            if prev_close <= level < close:
                if i < len(self._grid_levels) // 2:
                    return Signal(Action.BUY, size=self._position_per_grid,
                                  reason=f"grid_buy_{i}")
            elif prev_close >= level > close:
                if i > len(self._grid_levels) // 2:
                    return Signal(Action.SELL, size=self._position_per_grid,
                                  reason=f"grid_sell_{i}")

        return Signal(Action.HOLD)


class RSIDivergence(Strategy):
    """RSI背离策略

    价格创新低但 RSI 没创新低 → 底背离 → 买入
    价格创新高但 RSI 没创新高 → 顶背离 → 卖出
    比单纯的 RSI 超买超卖更可靠。
    """
    def __init__(self, period: int = 14, oversold: float = 30,
                 overbought: float = 70, lookback: int = 10):
        super().__init__(f"RSIDiv_{period}")
        self.period = period
        self.oversold = oversold
        self.overbought = overbought
        self.lookback = lookback

    def on_bar(self, data, idx):
        if idx < self.period + self.lookback + 1:
            return Signal(Action.HOLD)

        if "rsi" not in data.columns:
            return Signal(Action.HOLD)

        close = data["close"]
        rsi = data["rsi"]
        curr_close = close.iloc[idx]
        curr_rsi = rsi.iloc[idx]
        window = data.iloc[idx - self.lookback:idx + 1]
        prev_window = data.iloc[idx - self.lookback - 1:idx]

        close_low = close.iloc[idx - self.lookback:idx + 1].min()
        close_high = close.iloc[idx - self.lookback:idx + 1].max()

        if curr_rsi < self.oversold:
            if curr_close == close_low:
                prev_close_low = prev_window["close"].min()
                prev_rsi_low = prev_window["rsi"].min()
                if curr_close < prev_close_low and curr_rsi > prev_rsi_low:
                    return Signal(Action.BUY, reason="rsi_bullish_div")

        if curr_rsi > self.overbought:
            if curr_close == close_high:
                prev_close_high = prev_window["close"].max()
                prev_rsi_high = prev_window["rsi"].max()
                if curr_close > prev_close_high and curr_rsi < prev_rsi_high:
                    return Signal(Action.SELL, reason="rsi_bearish_div")

        return Signal(Action.HOLD)


class PullbackStrategy(Strategy):
    """突破回踩策略

    价格突破前期高点后回踩均线不破 → 买入
    追高容易被套，等回踩确认支撑再入场更安全。
    """
    def __init__(self, lookback: int = 20, ma_period: int = 10,
                 pullback_pct: float = 0.02):
        super().__init__(f"Pullback_{lookback}")
        self.lookback = lookback
        self.ma_period = ma_period
        self.pullback_pct = pullback_pct
        self._breakout_high = 0.0

    def on_bar(self, data, idx):
        if idx < max(self.lookback, self.ma_period) + 1:
            return Signal(Action.HOLD)

        close = data["close"]
        high = data["high"]
        curr_close = close.iloc[idx]
        recent_high = high.iloc[idx - self.lookback:idx].max()
        ma = close.iloc[idx - self.ma_period:idx].mean()

        if self._breakout_high == 0.0:
            if curr_close > recent_high:
                self._breakout_high = curr_close
                return Signal(Action.HOLD, reason="breakout_detected")
            return Signal(Action.HOLD)

        pullback_low = self._breakout_high * (1 - self.pullback_pct)
        if curr_close <= pullback_low and curr_close >= ma * 0.98:
            self._breakout_high = 0.0
            return Signal(Action.BUY, reason="pullback_buy")
        if curr_close < ma * 0.95:
            self._breakout_high = 0.0

        return Signal(Action.HOLD)


class AdaptiveAMA(Strategy):
    """Kaufman自适应均线策略

    根据市场波动率自动调整均线速度：
    - 趋势强时 → 均线变快（紧跟趋势）
    - 震荡时 → 均线变慢（过滤噪音）
    比固定周期均线更聪明。
    """
    def __init__(self, fast: int = 2, slow: int = 30, lookback: int = 10):
        super().__init__(f"AMA_{fast}_{slow}")
        self.fast = fast
        self.slow = slow
        self.lookback = lookback
        self._prev_ama = 0.0
        self._prev_signal = Action.HOLD

    def on_bar(self, data, idx):
        if idx < self.lookback + self.slow:
            return Signal(Action.HOLD)

        close = data["close"]
        if idx == self.lookback + self.slow:
            self._prev_ama = close.iloc[:idx].mean()

        ama = self._calc_ama(close, idx)
        self._prev_ama = ama

        curr_close = close.iloc[idx]
        if curr_close > ama and self._prev_signal != Action.BUY:
            self._prev_signal = Action.BUY
            return Signal(Action.BUY, reason="ama_bullish")
        elif curr_close < ama and self._prev_signal != Action.SELL:
            self._prev_signal = Action.SELL
            return Signal(Action.SELL, reason="ama_bearish")

        return Signal(Action.HOLD)

    def _calc_ama(self, close, idx):
        n = self.lookback
        price_change = abs(close.iloc[idx] - close.iloc[idx - n])
        volatility = sum(abs(close.iloc[i] - close.iloc[i - 1])
                         for i in range(idx - n + 1, idx + 1))
        if volatility == 0:
            return self._prev_ama

        er = price_change / volatility
        fast_sc = 2 / (self.fast + 1)
        slow_sc = 2 / (self.slow + 1)
        ssc = er * (fast_sc - slow_sc) + slow_sc
        sc = (ssc / 2) ** 2

        return self._prev_ama + sc * (close.iloc[idx] - self._prev_ama)
