"""经典策略集合"""
import numpy as np
from .base import Strategy, Signal, Action


class TurtleStrategy(Strategy):
    """海龟交易法则 — 唐奇安通道突破

    入场：价格突破N日高点做多，突破N日低点做空
    止损：ATR * 2
    加仓：每涨0.5ATR加仓一次，最多加3次
    出场：跌破M日低点平多，涨破M日高点平空
    """
    def __init__(self, entry_window: int = 20, exit_window: int = 10,
                 atr_period: int = 14, add_unit_step: float = 0.5,
                 stop_loss_atr: float = 2.0, max_add_units: int = 3):
        super().__init__(f"Turtle_{entry_window}_{exit_window}")
        self.entry_window = entry_window
        self.exit_window = exit_window
        self.atr_period = atr_period
        self.add_unit_step = add_unit_step
        self.stop_loss_atr = stop_loss_atr
        self.max_add_units = max_add_units
        self._position = 0           # 当前持仓方向：1多, -1空, 0空仓
        self._entry_price = 0.0
        self._units = 0

    def on_bar(self, data, idx):
        if idx < max(self.entry_window, self.exit_window, self.atr_period) + 1:
            return Signal(Action.HOLD)

        high = data["high"]
        low = data["low"]
        close = data["close"]
        atr = data["atr"].iloc[idx] if "atr" in data.columns else \
            self._calc_atr(data, idx)

        entry_high = high.iloc[idx - self.entry_window:idx].max()
        entry_low = low.iloc[idx - self.entry_window:idx].min()
        exit_low = low.iloc[idx - self.exit_window:idx].min()
        exit_high = high.iloc[idx - self.exit_window:idx].max()

        current_price = close.iloc[idx]

        if self._position == 0:
            if current_price > entry_high:
                self._position = 1
                self._entry_price = current_price
                self._units = 1
                return Signal(Action.BUY, reason="turtle_long_entry")
            elif current_price < entry_low:
                self._position = -1
                self._entry_price = current_price
                self._units = 1
                return Signal(Action.SELL, reason="turtle_short_entry")

        elif self._position == 1:
            stop_loss = self._entry_price - self.stop_loss_atr * atr
            if current_price < stop_loss:
                self._position = 0
                self._units = 0
                return Signal(Action.SELL, reason="turtle_long_stop")

            if current_price < exit_low:
                self._position = 0
                self._units = 0
                return Signal(Action.SELL, reason="turtle_long_exit")

            add_price = self._entry_price + self.add_unit_step * atr * self._units
            if current_price > add_price and self._units < self.max_add_units + 1:
                self._units += 1
                return Signal(Action.BUY, reason=f"turtle_add_unit_{self._units}")

        elif self._position == -1:
            stop_loss = self._entry_price + self.stop_loss_atr * atr
            if current_price > stop_loss:
                self._position = 0
                self._units = 0
                return Signal(Action.BUY, reason="turtle_short_stop")

            if current_price > exit_high:
                self._position = 0
                self._units = 0
                return Signal(Action.BUY, reason="turtle_short_exit")

            add_price = self._entry_price - self.add_unit_step * atr * self._units
            if current_price < add_price and self._units < self.max_add_units + 1:
                self._units += 1
                return Signal(Action.SELL, reason=f"turtle_add_unit_{self._units}")

        return Signal(Action.HOLD)

    def _calc_atr(self, data, idx):
        tr = np.max([
            data["high"].iloc[idx] - data["low"].iloc[idx],
            abs(data["high"].iloc[idx] - data["close"].iloc[idx - 1]),
            abs(data["low"].iloc[idx] - data["close"].iloc[idx - 1]),
        ])
        return data["atr"].iloc[idx - 1] * 13 / 14 + tr / 14 if idx > self.atr_period else tr


class BollingerBandStrategy(Strategy):
    """布林带均值回归策略

    价格触及下轨 → 买入
    价格触及上轨 → 卖出
    回到中轨 → 平仓
    """
    def __init__(self, window: int = 20, num_std: float = 2.0):
        super().__init__(f"BBand_{window}_{num_std}")
        self.window = window
        self.num_std = num_std
        self._in_position = False

    def on_bar(self, data, idx):
        if idx < self.window:
            return Signal(Action.HOLD)

        close = data["close"].iloc[idx]
        bb_upper = data["bb_upper"].iloc[idx]
        bb_lower = data["bb_lower"].iloc[idx]
        bb_mid = data["bb_mid"].iloc[idx]

        if not self._in_position:
            if close <= bb_lower:
                self._in_position = True
                return Signal(Action.BUY, reason="bb_oversold")
        else:
            if close >= bb_mid:
                self._in_position = False
                return Signal(Action.SELL, reason="bb_exit")
            elif close >= bb_upper:
                self._in_position = False
                return Signal(Action.SELL, reason="bb_overbought")

        return Signal(Action.HOLD)


class MACDStrategy(Strategy):
    """MACD柱状图背离策略

    经典MACD金叉死叉，加入零轴过滤
    """
    def __init__(self, signal_period: int = 9):
        super().__init__(f"MACD_{signal_period}")
        self.signal_period = signal_period
        self._prev_signal = Action.HOLD

    def on_bar(self, data, idx):
        if "macd" not in data.columns or "macd_signal" not in data.columns:
            return Signal(Action.HOLD)
        if idx < 2:
            return Signal(Action.HOLD)

        macd = data["macd"].iloc[idx]
        signal = data["macd_signal"].iloc[idx]
        prev_macd = data["macd"].iloc[idx - 1]
        prev_signal = data["macd_signal"].iloc[idx - 1]

        if prev_macd < prev_signal and macd > signal and macd < 0:
            # 零轴下方金叉 — 更可靠的买入信号
            if self._prev_signal != Action.BUY:
                self._prev_signal = Action.BUY
                return Signal(Action.BUY, reason="macd_bullish_cross")
        elif prev_macd > prev_signal and macd < signal and macd > 0:
            # 零轴上方死叉 — 卖出信号
            if self._prev_signal != Action.SELL:
                self._prev_signal = Action.SELL
                return Signal(Action.SELL, reason="macd_bearish_cross")

        return Signal(Action.HOLD)


class DualMovingAverageCross(Strategy):
    """双均线交叉（增强版 — 含成交量过滤）

    金叉 + 成交量放大 → 买入
    死叉 → 卖出
    """
    def __init__(self, fast: int = 10, slow: int = 30, vol_factor: float = 1.2):
        super().__init__(f"DMA_{fast}_{slow}")
        self.fast = fast
        self.slow = slow
        self.vol_factor = vol_factor
        self._prev_signal = Action.HOLD

    def on_bar(self, data, idx):
        if idx < self.slow + 1:
            return Signal(Action.HOLD)

        close = data["close"]
        volume = data["volume"]
        ma_fast = close.iloc[idx - self.fast:idx].mean()
        ma_slow = close.iloc[idx - self.slow:idx].mean()
        vol_ma = volume.iloc[idx - 20:idx].mean()

        if ma_fast > ma_slow and self._prev_signal != Action.BUY:
            vol_surge = volume.iloc[idx] > vol_ma * self.vol_factor if self.vol_factor > 0 else True
            if vol_surge:
                self._prev_signal = Action.BUY
                return Signal(Action.BUY, reason="golden_cross_vol_surge")
        elif ma_fast < ma_slow and self._prev_signal != Action.SELL:
            self._prev_signal = Action.SELL
            return Signal(Action.SELL, reason="death_cross")

        return Signal(Action.HOLD)
