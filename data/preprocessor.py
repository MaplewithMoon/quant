import pandas as pd
import numpy as np
from typing import Callable, List
from .dataset import DataSet


# ============ 技术指标计算函数（输入 DataFrame，输出 Series 或 DataFrame） ============

def sma(df: pd.DataFrame, period: int = 20, col: str = "close") -> pd.Series:
    """简单移动平均线 SMA"""
    return df[col].rolling(period).mean()


def ema(df: pd.DataFrame, period: int = 12, col: str = "close") -> pd.Series:
    """指数移动平均线 EMA"""
    return df[col].ewm(span=period, adjust=False).mean()


def macd(df: pd.DataFrame, fast: int = 12, slow: int = 26,
         signal: int = 9) -> pd.DataFrame:
    """MACD 指标：返回 macd 线、信号线、柱状图"""
    ema_fast = df["close"].ewm(span=fast, adjust=False).mean()
    ema_slow = df["close"].ewm(span=slow, adjust=False).mean()
    macd_line = ema_fast - ema_slow
    macd_signal = macd_line.ewm(span=signal, adjust=False).mean()
    hist = macd_line - macd_signal
    return pd.DataFrame({"macd": macd_line, "macd_signal": macd_signal, "macd_hist": hist})


def rsi(df: pd.DataFrame, period: int = 14, col: str = "close") -> pd.Series:
    """相对强弱指标 RSI (0-100)"""
    delta = df[col].diff()
    gain = delta.clip(lower=0).rolling(period).mean()
    loss = (-delta.clip(upper=0)).rolling(period).mean()
    rs = gain / loss.replace(0, np.nan)
    return 100 - (100 / (1 + rs))


def bollinger(df: pd.DataFrame, period: int = 20, num_std: float = 2.0) -> pd.DataFrame:
    """布林带：返回中轨、上轨、下轨"""
    mid = df["close"].rolling(period).mean()
    std = df["close"].rolling(period).std()
    return pd.DataFrame({
        "bb_mid": mid,
        "bb_upper": mid + num_std * std,
        "bb_lower": mid - num_std * std,
    })


def atr(df: pd.DataFrame, period: int = 14) -> pd.Series:
    """平均真实波幅 ATR"""
    high_low = df["high"] - df["low"]
    high_close = (df["high"] - df["close"].shift()).abs()
    low_close = (df["low"] - df["close"].shift()).abs()
    tr = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
    return tr.rolling(period).mean()


def obv(df: pd.DataFrame, col: str = "close", vol: str = "volume") -> pd.Series:
    """能量潮指标 OBV（累计成交量方向指标）"""
    direction = np.sign(df[col].diff().fillna(0))
    return (direction * df[vol]).cumsum()


def kdj(df: pd.DataFrame, n: int = 9, m1: int = 3, m2: int = 3) -> pd.DataFrame:
    """随机指标 KDJ"""
    low_n = df["low"].rolling(n).min()
    high_n = df["high"].rolling(n).max()
    rsv = (df["close"] - low_n) / (high_n - low_n).replace(0, np.nan) * 100
    k = rsv.ewm(com=m1 - 1, adjust=False).mean()
    d = k.ewm(com=m2 - 1, adjust=False).mean()
    j = 3 * k - 2 * d
    return pd.DataFrame({"kdj_k": k, "kdj_d": d, "kdj_j": j})


def wr(df: pd.DataFrame, period: int = 14) -> pd.Series:
    """威廉指标 W%R (0-100)"""
    high_n = df["high"].rolling(period).max()
    low_n = df["low"].rolling(period).min()
    return (high_n - df["close"]) / (high_n - low_n).replace(0, np.nan) * 100


def cci(df: pd.DataFrame, period: int = 14) -> pd.Series:
    """顺势指标 CCI"""
    tp = (df["high"] + df["low"] + df["close"]) / 3
    ma = tp.rolling(period).mean()
    md = (tp - ma).abs().rolling(period).mean()
    return (tp - ma) / (0.015 * md.replace(0, np.nan))


def adx(df: pd.DataFrame, period: int = 14) -> pd.DataFrame:
    """平均趋向指标 ADX（趋势强度）"""
    up_move = df["high"].diff()
    down_move = -df["low"].diff()
    plus_dm = np.where((up_move > down_move) & (up_move > 0), up_move, 0.0)
    minus_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0.0)
    tr = pd.concat([
        df["high"] - df["low"],
        (df["high"] - df["close"].shift()).abs(),
        (df["low"] - df["close"].shift()).abs(),
    ], axis=1).max(axis=1)
    atr_val = tr.ewm(alpha=1 / period, adjust=False).mean()
    plus_di = 100 * pd.Series(plus_dm, index=df.index).ewm(alpha=1 / period, adjust=False).mean() / atr_val
    minus_di = 100 * pd.Series(minus_dm, index=df.index).ewm(alpha=1 / period, adjust=False).mean() / atr_val
    dx = (plus_di - minus_di).abs() / (plus_di + minus_di).replace(0, np.nan) * 100
    adx_val = dx.ewm(alpha=1 / period, adjust=False).mean()
    return pd.DataFrame({"plus_di": plus_di, "minus_di": minus_di, "adx": adx_val})


# ============ 指标注册表 ============
INDICATORS = {
    "sma":      {"func": sma,      "desc": "简单移动平均线", "params": "period=20, col='close'"},
    "ema":      {"func": ema,      "desc": "指数移动平均线", "params": "period=12, col='close'"},
    "macd":     {"func": macd,     "desc": "MACD 金叉死叉", "params": "fast=12, slow=26, signal=9"},
    "rsi":      {"func": rsi,      "desc": "相对强弱指标", "params": "period=14, col='close'"},
    "bollinger": {"func": bollinger, "desc": "布林带", "params": "period=20, num_std=2.0"},
    "atr":      {"func": atr,      "desc": "平均真实波幅", "params": "period=14"},
    "obv":      {"func": obv,      "desc": "能量潮", "params": "col='close', vol='volume'"},
    "kdj":      {"func": kdj,      "desc": "随机指标", "params": "n=9, m1=3, m2=3"},
    "wr":       {"func": wr,       "desc": "威廉指标", "params": "period=14"},
    "cci":      {"func": cci,      "desc": "顺势指标", "params": "period=14"},
    "adx":      {"func": adx,      "desc": "平均趋向指标", "params": "period=14"},
}


def help_indicators() -> str:
    """打印所有可用技术指标的计算说明"""
    lines = ["可用技术指标：", "=" * 70]
    for name, info in INDICATORS.items():
        lines.append(f"  {name:<10} {info['desc']:<12} 参数: {info['params']}")
    lines.append("=" * 70)
    lines.append("用法示例: from data.preprocessor import macd; result = macd(df)")
    return "\n".join(lines)


def add_indicators(ds: DataSet, names: List[str] = None) -> DataSet:
    """按名称批量添加技术指标到 DataSet"""
    df = ds.data
    selected = names if names else list(INDICATORS.keys())
    for name in selected:
        if name not in INDICATORS:
            continue
        result = INDICATORS[name]["func"](df)
        if isinstance(result, pd.DataFrame):
            for col in result.columns:
                df[col] = result[col]
        else:
            df[name] = result
    return ds


# ============ 预处理器 ============
class Preprocessor:
    def __init__(self):
        self._steps: List[Callable[[DataSet], DataSet]] = []

    def add(self, step: Callable[[DataSet], DataSet]) -> "Preprocessor":
        self._steps.append(step)
        return self

    def run(self, ds: DataSet) -> DataSet:
        for step in self._steps:
            ds = step(ds)
        return ds


def fillna(method: str = "ffill") -> Callable[[DataSet], DataSet]:
    def _fn(ds: DataSet) -> DataSet:
        if method == "ffill":
            ds.data = ds.data.ffill().dropna()
        elif method == "bfill":
            ds.data = ds.data.bfill().dropna()
        else:
            ds.data = ds.data.fillna(0)
        return ds
    return _fn


def drop_outliers(zscore_thresh: float = 3.0) -> Callable[[DataSet], DataSet]:
    def _fn(ds: DataSet) -> DataSet:
        z = np.abs((ds.data - ds.data.mean()) / ds.data.std())
        ds.data = ds.data[(z < zscore_thresh).all(axis=1)]
        return ds
    return _fn


def add_technical_indicators(ds: DataSet) -> DataSet:
    """兼容旧接口：一次性添加常用技术指标（内部复用注册表指标函数）"""
    df = ds.data
    df["sma_5"] = sma(df, 5)
    df["sma_20"] = sma(df, 20)
    df["ema_12"] = ema(df, 12)
    df["ema_26"] = ema(df, 26)
    df["macd"] = df["ema_12"] - df["ema_26"]
    df["macd_signal"] = df["macd"].ewm(span=9, adjust=False).mean()
    df["rsi"] = rsi(df, 14)
    bb = bollinger(df)
    df["bb_mid"] = bb["bb_mid"]
    df["bb_upper"] = bb["bb_upper"]
    df["bb_lower"] = bb["bb_lower"]
    df["atr"] = atr(df, 14)
    return ds
