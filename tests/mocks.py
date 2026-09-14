# -*- coding: utf-8 -*-
"""单元测试：Mock 数据源生成固定行情，供各测试复用"""
import sys, io, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import pandas as pd
import numpy as np


def make_daily(code="TEST", n=100, start="2024-01-01", seed=42):
    """生成确定性日线（同一seed结果一致，便于断言）"""
    rng = np.random.default_rng(seed)
    dates = pd.bdate_range(start, periods=n)
    close = 10 + np.cumsum(rng.normal(0, 0.1, n))  # 随机游走
    open_ = close + rng.normal(0, 0.02, n)
    high = np.maximum(open_, close) + rng.uniform(0, 0.1, n)
    low = np.minimum(open_, close) - rng.uniform(0, 0.1, n)
    df = pd.DataFrame({
        "code": code,
        "trade_date": dates,
        "open": open_,
        "high": high,
        "low": low,
        "close": close,
        "volume": rng.integers(1e5, 1e6, n).astype(float),
        "amount": close * rng.integers(1e5, 1e6, n),
    })
    return df


def make_dataset(code="TEST", n=100, start="2024-01-01", seed=42):
    """返回 DataSet（trade_date 为索引）"""
    from data.dataset import DataSet
    df = make_daily(code, n, start, seed)
    return DataSet(symbol=code, data=df.set_index("trade_date"))


def make_valuation(code="TEST", n=100, start="2024-01-01"):
    """生成估值数据（pe_ttm 确定性）"""
    dates = pd.bdate_range(start, periods=n)
    return pd.DataFrame({
        "code": code,
        "trade_date": dates,
        "pe_ttm": 10 + np.arange(n) * 0.01,     # 单调递增，便于断言
        "total_mv": 1e6 + np.arange(n) * 1000,
    })


def make_rising_daily(code="TEST", n=100, start="2024-01-01", daily_ret=0.002):
    """确定性上涨行情（每日+0.2%，便于测试"应该盈利"）"""
    dates = pd.bdate_range(start, periods=n)
    close = 10 * (1 + daily_ret) ** np.arange(n)
    open_ = close * (1 - daily_ret / 2)
    high = close * 1.01
    low = close * 0.99
    df = pd.DataFrame({
        "code": code,
        "trade_date": dates,
        "open": open_,
        "high": high,
        "low": low,
        "close": close,
        "volume": np.full(n, 1e6, dtype=float),
        "amount": close * 1e6,
    })
    return df


def make_factor(code="TEST"):
    """生成复权因子（前复权，最新=1）"""
    dates = pd.to_datetime(["2024-01-01", "2024-06-01"])
    return pd.DataFrame({"trade_date": dates, "factor": [0.9, 1.0]})
