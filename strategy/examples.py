import pandas as pd
from .base import Strategy, Signal, Action


class MovingAverageCross(Strategy):
    def __init__(self, fast: int = 5, slow: int = 20):
        super().__init__(f"MA_{fast}_{slow}")
        self.fast = fast
        self.slow = slow
        self._prev_signal = Action.HOLD

    def on_bar(self, data: pd.DataFrame, idx: int) -> Signal:
        if idx < self.slow:
            return Signal(Action.HOLD)

        ma_fast = data["close"].iloc[idx - self.fast:idx].mean()
        ma_slow = data["close"].iloc[idx - self.slow:idx].mean()

        if ma_fast > ma_slow and self._prev_signal != Action.BUY:
            self._prev_signal = Action.BUY
            return Signal(Action.BUY, reason="golden cross")
        elif ma_fast < ma_slow and self._prev_signal != Action.SELL:
            self._prev_signal = Action.SELL
            return Signal(Action.SELL, reason="death cross")
        return Signal(Action.HOLD)


class MeanReversion(Strategy):
    def __init__(self, window: int = 20, entry_std: float = 2.0):
        super().__init__(f"MeanRev_{window}_{entry_std}")
        self.window = window
        self.entry_std = entry_std

    def on_bar(self, data: pd.DataFrame, idx: int) -> Signal:
        if idx < self.window:
            return Signal(Action.HOLD)

        close = data["close"].iloc[idx]
        mean = data["close"].iloc[idx - self.window:idx].mean()
        std = data["close"].iloc[idx - self.window:idx].std()
        upper = mean + self.entry_std * std
        lower = mean - self.entry_std * std

        if close > upper:
            return Signal(Action.SELL, reason="overbought")
        if close < lower:
            return Signal(Action.BUY, reason="oversold")
        return Signal(Action.HOLD)
