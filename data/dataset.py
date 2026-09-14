import pandas as pd
import numpy as np
from dataclasses import dataclass, field
from typing import Dict, Optional
from .source import DataSource


@dataclass
class DataSet:
    symbol: str
    data: pd.DataFrame = field(repr=False)

    def __post_init__(self):
        self._ensure_ohlcv()

    def _ensure_ohlcv(self):
        required = {"open", "high", "low", "close", "volume"}
        missing = required - set(self.data.columns)
        if missing:
            raise ValueError(f"missing columns: {missing}")

    @property
    def close(self) -> pd.Series:
        return self.data["close"]

    @property
    def returns(self) -> pd.Series:
        return self.data["close"].pct_change().fillna(0)

    @property
    def log_returns(self) -> pd.Series:
        return np.log(self.data["close"] / self.data["close"].shift(1)).fillna(0)

    @property
    def high_low_pct(self) -> pd.Series:
        return (self.data["high"] - self.data["low"]) / self.data["low"]

    def resample(self, freq: str) -> "DataSet":
        ohlc = {
            "open": "first",
            "high": "max",
            "low": "min",
            "close": "last",
            "volume": "sum",
        }
        resampled = self.data.resample(freq).agg(
            {k: v for k, v in ohlc.items() if k in self.data.columns}
        ).dropna()
        return DataSet(symbol=self.symbol, data=resampled)

    def add_feature(self, name: str, func) -> "DataSet":
        self.data[name] = func(self.data)
        return self

    def align_valuation(self, valuation: pd.DataFrame, ffill: bool = True,
                        overwrite: bool = False) -> "DataSet":
        """把估值/因子数据按交易日对齐到日线时间轴（左连接 + 前向填充）

        Args:
            valuation: 含 'trade_date' 列（或已是 DatetimeIndex）的估值表，
                       例如 database.loader.load_valuation() 的返回值
            ffill:     对齐后是否用前值填充缺口。估值按日发布但可能缺失，
                       前向填充表示"沿用最近一次已知的估值"
            overwrite: True 则覆盖已有同名列；默认 False，保护 OHLCV 不被冲掉

        Returns:
            self（便于链式调用）

        说明:
            估值日期可能比日线少（新股/停牌/接口缺失），因此用左连接以日线
            时间轴为准，不用 inner join，避免丢失行情行。
        """
        if valuation is None or len(valuation) == 0:
            return self
        if not isinstance(self.data.index, pd.DatetimeIndex):
            raise TypeError("align_valuation 要求 DataSet.data 的索引是交易日(DatetimeIndex)")

        v = valuation.copy()
        if "trade_date" in v.columns:
            v["trade_date"] = pd.to_datetime(v["trade_date"])
            v = v.set_index("trade_date")
        v.index = pd.to_datetime(v.index)
        v = v[~v.index.duplicated(keep="last")].sort_index()

        # 默认只补充新列，避免估值表里的同名字段覆盖行情列
        cols = [c for c in v.columns
                if c != "code" and (overwrite or c not in self.data.columns)]
        if not cols:
            return self

        out = self.data.join(v[cols], how="left")
        if ffill:
            out[cols] = out[cols].ffill()
        self.data = out
        return self

    def attach_market_status(self, status: pd.DataFrame) -> "DataSet":
        """把交易日状态（涨跌停价 / 停牌标记）按交易日对齐到日线

        Args:
            status: 含 trade_date（或 DatetimeIndex）与 limit_up/limit_down/
                    suspended 等列的表，例如 database.loader.load_trading_status()

        Returns:
            self（便于链式调用）

        注意: **不做前向填充**。涨跌停价必须逐日精确，用前值填充会把
              不该拦的交易拦掉（或反之），因此这里用普通左连接。
        """
        if status is None or len(status) == 0:
            return self
        if not isinstance(self.data.index, pd.DatetimeIndex):
            raise TypeError("attach_market_status 要求 DataSet.data 的索引是交易日")

        s = status.copy()
        if "trade_date" in s.columns:
            s["trade_date"] = pd.to_datetime(s["trade_date"])
            s = s.set_index("trade_date")
        s.index = pd.to_datetime(s.index)
        s = s[~s.index.duplicated(keep="last")].sort_index()

        cols = [c for c in s.columns if c not in self.data.columns]
        if not cols:
            return self
        self.data = self.data.join(s[cols], how="left")
        return self

    @classmethod
    def load(cls, symbol: str, start: str, end: str, freq: str = "1d",
             source: Optional[DataSource] = None) -> "DataSet":
        from .source import RemoteSource
        source = source or RemoteSource()
        df = source.fetch(symbol, start, end, freq)
        return cls(symbol=symbol, data=df)

    @classmethod
    def from_db(cls, symbol: str, start: str = None, end: str = None,
                adjust: str = "qfq") -> "DataSet":
        """从量化数据库加载（原始价 + 复权因子现场算前复权）
        adjust: 'qfq' 前复权(默认) / '' 原始价
        """
        from database.loader import load_daily
        df = load_daily(symbol, start, end, adjust=adjust)
        if df.empty:
            raise ValueError(f"{symbol} 数据库无数据")
        return cls(symbol=symbol, data=df.set_index("trade_date"))

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, key):
        return self.data[key]
