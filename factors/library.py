# -*- coding: utf-8 -*-
"""内置因子库

全部基于 `factors.panel.load_panel` 产出的面板，只用**当日及以前**的数据。

类别:
    动量/反转    mom_20 / mom_60 / mom_120 / rev_5 / rev_20
    波动/风险    vol_20 / vol_60 / amp_20 / max_ret_20 / skew_20
    流动性       turn_20 / illiq_20
    估值         ep / bp / sp
    规模         size
    量价         vol_ratio_20 / price_volume_corr_20
"""
import numpy as np
import pandas as pd

from .base import register
from .panel import adjusted_close


def _close(panel):
    return adjusted_close(panel)


def _returns(panel):
    return _close(panel).pct_change(fill_method=None)


# ============================================================
# 动量 / 反转
# ============================================================
@register("mom_20", "20 日动量（复权）", direction=1, category="动量反转")
def mom_20(panel):
    c = _close(panel)
    return c / c.shift(20) - 1


@register("mom_60", "60 日动量（复权）", direction=1, category="动量反转")
def mom_60(panel):
    c = _close(panel)
    return c / c.shift(60) - 1


@register("mom_120", "120 日动量（复权）", direction=1, category="动量反转")
def mom_120(panel):
    c = _close(panel)
    return c / c.shift(120) - 1


@register("rev_5", "5 日反转（取负的短期收益）", direction=1, category="动量反转")
def rev_5(panel):
    c = _close(panel)
    return -(c / c.shift(5) - 1)


@register("rev_20", "20 日反转（取负的中期收益）", direction=1, category="动量反转")
def rev_20(panel):
    c = _close(panel)
    return -(c / c.shift(20) - 1)


# ============================================================
# 波动 / 风险
# ============================================================
@register("vol_20", "20 日收益波动率（低波更优）", direction=-1, category="波动风险")
def vol_20(panel):
    return _returns(panel).rolling(20).std()


@register("vol_60", "60 日收益波动率（低波更优）", direction=-1, category="波动风险")
def vol_60(panel):
    return _returns(panel).rolling(60).std()


@register("amp_20", "20 日平均振幅 (high-low)/pre_close", direction=-1, category="波动风险")
def amp_20(panel):
    c = _close(panel)
    h = panel.get("high_adj", panel.get("high"))
    l = panel.get("low_adj", panel.get("low"))
    amp = (h - l) / c.shift(1)
    return amp.rolling(20).mean()


@register("max_ret_20", "20 日内最大单日收益（彩票效应，偏高者后续偏弱）",
          direction=-1, category="波动风险")
def max_ret_20(panel):
    return _returns(panel).rolling(20).max()


@register("skew_20", "20 日收益偏度（左偏/右偏）", direction=-1, category="波动风险")
def skew_20(panel):
    return _returns(panel).rolling(20).skew()


# ============================================================
# 流动性
# ============================================================
@register("turn_20", "20 日平均换手率（高换手后续偏弱）", direction=-1, category="流动性")
def turn_20(panel):
    if "turnover_rate" in panel:
        return panel["turnover_rate"].rolling(20).mean()
    # 回退：成交额 / 流通市值（tushare 流通市值单位为万元）
    if "amount" in panel and "circ_mv" in panel:
        return (panel["amount"] / (panel["circ_mv"] * 1e4)).rolling(20).mean()
    raise KeyError("缺少 turnover_rate 与 circ_mv，无法计算换手率")


@register("illiq_20", "Amihud 非流动性 = 平均(|收益|/成交额)（越不流动越需要补偿）",
          direction=1, category="流动性")
def illiq_20(panel):
    r = _returns(panel).abs()
    amt = panel["amount"].replace(0, np.nan)
    return (r / amt * 1e8).rolling(20).mean()


# ============================================================
# 估值
# ============================================================
@register("ep", "盈利收益率 1/PE_TTM（低估值更优）", direction=1, category="估值")
def ep(panel):
    if "pe_ttm" not in panel:
        raise KeyError("缺少 pe_ttm")
    pe = panel["pe_ttm"].replace(0, np.nan)
    return 1.0 / pe.where(pe > 0)


@register("bp", "账面市值比 1/PB（低估值更优）", direction=1, category="估值")
def bp(panel):
    if "pb" not in panel:
        raise KeyError("缺少 pb")
    pb = panel["pb"].replace(0, np.nan)
    return 1.0 / pb.where(pb > 0)


@register("sp", "市销率倒数 1/PS_TTM", direction=1, category="估值")
def sp(panel):
    if "ps_ttm" in panel:
        ps = panel["ps_ttm"].replace(0, np.nan)
        return 1.0 / ps.where(ps > 0)
    raise KeyError("缺少 ps_ttm")


@register("size", "规模 = ln(总市值)（小市值效应）", direction=-1, category="规模")
def size(panel):
    if "total_mv" not in panel:
        raise KeyError("缺少 total_mv")
    return np.log(panel["total_mv"].where(panel["total_mv"] > 0))


# ============================================================
# 量价
# ============================================================
@register("vol_ratio_20", "量比 = 当日成交量 / 20 日均量", direction=1, category="量价")
def vol_ratio_20(panel):
    v = panel["volume"]
    return v / v.rolling(20).mean()


@register("price_volume_corr_20", "20 日价量相关性（量价背离）",
          direction=-1, category="量价")
def price_volume_corr_20(panel):
    r = _returns(panel)
    v = panel["volume"].pct_change(fill_method=None)
    return r.rolling(20).corr(v)
