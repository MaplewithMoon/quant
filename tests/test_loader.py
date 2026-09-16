# -*- coding: utf-8 -*-
"""测试：数据加载与复权"""
import sys
import io
import os
# 不要在 import 阶段替换 sys.stdout：pytest 会在导入时接管 stdout，
# 替换后其捕获机制会抛 "ValueError: I/O operation on closed file"。
# 脚本直跑时的编码修正放到文件末尾的 __main__ 分支里。
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import pandas as pd
import numpy as np
from mocks import make_dataset, make_valuation, make_factor
from database.loader import to_qfq


def test_dataset_ohlcv():
    """DataSet 应有完整 OHLCV"""
    ds = make_dataset(n=50)
    assert len(ds) == 50
    for col in ["open", "high", "low", "close", "volume", "amount"]:
        assert col in ds.data.columns, f"缺 {col}"
    print("[OK] DataSet 含完整 OHLCV")


def test_high_low_consistency():
    """OHLC 关系应自洽（high>=low, close 在区间内）"""
    ds = make_dataset(n=100)
    d = ds.data
    assert (d["high"] >= d["low"]).all()
    assert (d["high"] >= d["close"]).all()
    assert (d["low"] <= d["close"]).all()
    print("[OK] OHLC 关系自洽")


def test_qfq_computation():
    """前复权 = 原始价 × 因子"""
    df = pd.DataFrame({
        "trade_date": pd.to_datetime(["2024-01-01", "2024-06-01", "2024-12-01"]),
        "open": [99, 119, 149], "high": [101, 121, 151],
        "low": [98, 118, 148], "close": [100, 120, 150],
    })
    factor = pd.DataFrame({"trade_date": pd.to_datetime(["2024-01-01", "2024-06-01"]),
                           "factor": [0.9, 1.0]})
    qfq = to_qfq(df, factor)
    # 2024-06-01 之后因子=1，前复权=原始价；之前因子=0.9
    assert abs(qfq["close"].iloc[1] - 120) < 1e-3      # 因子1.0段
    assert abs(qfq["close"].iloc[0] - 90) < 1e-3       # 因子0.9段
    print("[OK] 前复权计算正确")


def test_align_valuation():
    """估值对齐到日线轴（前向填充）"""
    ds = make_dataset(n=10)
    v = make_valuation(n=5)   # 估值只有前5天
    ds.align_valuation(v)
    # 前5天有 pe_ttm，后5天前向填充
    assert ds.data["pe_ttm"].iloc[4] > 0
    assert not ds.data["pe_ttm"].iloc[8:].isna().all()  # 有前值填充
    print("[OK] 估值时间轴对齐（前向填充）")


if __name__ == "__main__":
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")  # 修正 Windows 控制台中文编码
    test_dataset_ohlcv()
    test_high_low_consistency()
    test_qfq_computation()
    test_align_valuation()
    print("\n全部数据加载测试通过")
