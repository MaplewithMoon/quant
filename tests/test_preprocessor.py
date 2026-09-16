# -*- coding: utf-8 -*-
"""测试：指标计算"""
import sys
import io
import os
# 不在 import 阶段替换 sys.stdout（会破坏 pytest 的输出捕获，见 test_loader.py 注释）
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np
import pandas as pd
from mocks import make_daily
from data.preprocessor import sma, ema, rsi, atr, bollinger, macd


def _df():
    return make_daily(n=60)


def test_sma_ema():
    df = _df()
    s5 = sma(df, 5)
    # 第0天窗口不足 → NaN，第4天开始有值
    assert pd.isna(s5.iloc[3])
    assert not pd.isna(s5.iloc[4])
    # 手工校验第4天
    expected = df["close"].iloc[:5].mean()
    assert abs(s5.iloc[4] - expected) < 1e-6
    print("[OK] SMA 正确")


def test_bollinger_band_order():
    df = _df()
    bb = bollinger(df, 20, 2.0)
    # 上轨 >= 中轨 >= 下轨
    valid = bb.dropna()
    assert (valid["bb_upper"] >= valid["bb_mid"]).all()
    assert (valid["bb_mid"] >= valid["bb_lower"]).all()
    print("[OK] 布林带轨序正确")


def test_rsi_range():
    df = _df()
    r = rsi(df, 14).dropna()
    assert ((r >= 0) & (r <= 100)).all()
    print("[OK] RSI 在 0~100 区间")


def test_atr_positive():
    df = _df()
    a = atr(df, 14).dropna()
    assert (a > 0).all()   # 真实波幅恒正
    print("[OK] ATR 恒为正")


def test_macd_structure():
    df = _df()
    m = macd(df)
    assert set(["macd", "macd_signal", "macd_hist"]).issubset(m.columns)
    print("[OK] MACD 三列结构正确")


if __name__ == "__main__":
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")  # 修正 Windows 控制台中文编码
    test_sma_ema()
    test_bollinger_band_order()
    test_rsi_range()
    test_atr_positive()
    test_macd_structure()
    print("\n全部指标测试通过")
