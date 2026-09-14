import math
import sys
import io


TIMEFRAME_MAP = {
    "1min": 60, "5min": 300, "15min": 900, "30min": 1800,
    "1h": 3600, "4h": 14400, "1d": 86400, "1w": 604800,
}


def help_all(kind: str = "") -> str:
    """统一指标帮助入口
    kind: '' 全部 | 'tech' 技术指标 | 'perf' 绩效指标 | 'util' 工具函数
    """
    from data.preprocessor import help_indicators
    from backtest.metrics import help_metrics

    parts = []
    if kind in ("", "tech"):
        parts.append(help_indicators())
        parts.append("")
    if kind in ("", "perf"):
        parts.append(help_metrics())
        parts.append("")
    if kind in ("", "util"):
        parts.append("工具函数：")
        parts.append("-" * 70)
        parts.append("  timeframe_to_seconds(tf)       时间频率转秒: '1d'→86400")
        parts.append("  format_currency(value)         货币格式化: 123456789→1.23亿")
        parts.append("  annualized_return(tr, n)       年化收益率")
        parts.append("  annualized_volatility(std)     年化波动率")
        parts.append("  disp_width(s) / pad(s,w)       终端宽度计算与填充")
        parts.append("=" * 70)
    return "\n".join(parts)


def _disp_width(s) -> int:
    """字符串在终端中的显示宽度（中文占2，英文/数字占1）"""
    n = 0
    for c in str(s):
        if '\u4e00' <= c <= '\u9fff':
            n += 2
        else:
            n += 1
    return n


def _pad(s, width: int, align: str = "<") -> str:
    """按显示宽度填充字符串到指定宽度"""
    s = str(s)
    pad = width - _disp_width(s)
    if pad <= 0:
        return s
    return (" " * pad) + s if align == ">" else s + (" " * pad)


def timeframe_to_seconds(tf: str) -> int:
    return TIMEFRAME_MAP.get(tf, 86400)


def format_currency(value: float, decimals: int = 2) -> str:
    prefix = "-" if value < 0 else ""
    abs_v = abs(value)
    if abs_v >= 1e8:
        return f"{prefix}{abs_v / 1e8:.{decimals}f}亿"
    if abs_v >= 1e4:
        return f"{prefix}{abs_v / 1e4:.{decimals}f}万"
    return f"{prefix}{abs_v:.{decimals}f}"


def annualized_return(total_return: float, periods: int,
                      periods_per_year: int = 252) -> float:
    return (1 + total_return) ** (periods_per_year / periods) - 1


def annualized_volatility(daily_std: float,
                          periods_per_year: int = 252) -> float:
    return daily_std * math.sqrt(periods_per_year)
