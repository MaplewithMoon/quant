from abc import ABC, abstractmethod
import pandas as pd
import numpy as np
from typing import Optional


class DataSource(ABC):
    @abstractmethod
    def fetch(
        self, symbol: str, start: str, end: str, freq: str = "1d"
    ) -> pd.DataFrame:
        ...


class LocalSource(DataSource):
    def __init__(self, base_path: str = "data/storage"):
        self.base_path = base_path

    def fetch(
        self, symbol: str, start: str, end: str, freq: str = "1d"
    ) -> pd.DataFrame:
        file_path = f"{self.base_path}/{symbol}_{freq}.csv"
        df = pd.read_csv(file_path, parse_dates=["datetime"], index_col="datetime")
        return df.loc[start:end]


class RemoteSource(DataSource):
    def __init__(self, backend: str = "akshare"):
        self.backend = backend

    def fetch(
        self, symbol: str, start: str, end: str, freq: str = "1d"
    ) -> pd.DataFrame:
        df = self._download(symbol, start, end, freq)
        df.columns = [c.lower() for c in df.columns]
        return df

    def _download(self, symbol: str, start: str, end: str, freq: str) -> pd.DataFrame:
        if self.backend == "akshare":
            import akshare as ak
            import time
            for attempt in range(3):
                try:
                    raw = ak.stock_zh_a_hist(
                        symbol=symbol,
                        start_date=start.replace("-", ""),
                        end_date=end.replace("-", ""),
                        adjust="qfq",
                    )
                    break
                except Exception as e:
                    if attempt < 2:
                        time.sleep(2 ** attempt)
                    else:
                        raise ConnectionError(
                            f"failed to fetch {symbol} after 3 retries: {e}\n"
                            "tip: try using MockSource for offline testing"
                        )
            raw.rename(columns={
                "日期": "datetime", "开盘": "open", "最高": "high",
                "最低": "low", "收盘": "close", "成交量": "volume",
                "成交额": "amount", "振幅": "amplitude",
                "涨跌幅": "pct_chg", "涨跌额": "change",
                "换手率": "turnover",
            }, inplace=True)
            raw["datetime"] = pd.to_datetime(raw["datetime"])
            raw.sort_values("datetime", inplace=True)
            raw.set_index("datetime", inplace=True)
            return raw
        if self.backend == "baostock":
            import baostock as bs
            import datetime
            bs.login()
            try:
                code = f"sz.{symbol}" if symbol.startswith(("0", "3")) else f"sh.{symbol}"
                rs = bs.query_history_k_data_plus(
                    code,
                    "date,open,high,low,close,volume,amount",
                    start_date=start,
                    end_date=end,
                    frequency="d",
                    adjustflag="2",
                )
                data = rs.get_data()
            finally:
                bs.logout()
            if data is None or data.empty:
                raise ValueError(f"baostock returned no data for {symbol}")
            raw = data.rename(columns={"date": "datetime"})
            raw["datetime"] = pd.to_datetime(raw["datetime"])
            raw.set_index("datetime", inplace=True)
            for col in ["open", "high", "low", "close", "volume", "amount"]:
                raw[col] = raw[col].astype(float)
            raw.sort_index(inplace=True)
            return raw
        raise NotImplementedError(f"backend {self.backend} not supported")


class MockSource(DataSource):
    """生成随机模拟数据，用于离线开发和调试"""

    def __init__(self, seed: int = 42):
        self.seed = seed

    def fetch(self, symbol: str, start: str, end: str, freq: str = "1d") -> pd.DataFrame:
        rng = np.random.default_rng(self.seed)
        dates = pd.date_range(start, end, freq="B")
        n = len(dates)
        price = 100 * np.exp(np.cumsum(rng.normal(0, 0.01, n)))
        df = pd.DataFrame({
            "open": price * (1 + rng.uniform(-0.005, 0.005, n)),
            "high": price * (1 + rng.uniform(0, 0.015, n)),
            "low": price * (1 - rng.uniform(0, 0.015, n)),
            "close": price,
            "volume": rng.integers(1e5, 1e7, n),
            "amount": price * rng.integers(1e5, 1e7, n),
        }, index=dates)
        df.index.name = "datetime"
        return df

