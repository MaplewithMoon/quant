# -*- coding: utf-8 -*-
"""聚宽（JoinQuant）兼容层

把 `from jqdata import *` 换成 `from joinquant import *`，
聚宽策略代码即可在本项目的引擎上运行。

    from joinquant import *
    from joinquant.data import JQData
    from joinquant.api import run_strategy

设计要点见 `joinquant/data.py` 与 `joinquant/api.py` 的模块文档。
"""
import datetime
from datetime import date, time, timedelta

from .api import (  # noqa: F401
    Context, FixedSlippage, GlobalNamespace, JQEngine, Order, OrderCost,
    OrderStatus, PriceRelatedSlippage, Query, SecurityInfo, balance, cash_flow,
    get_all_securities, get_current_data, get_fundamentals, get_industries,
    get_index_stocks, get_industry_stocks, get_price, get_security_info,
    get_trade_days, history, income, log, order, order_target,
    order_target_percent, order_target_value, order_value, query, run_daily,
    run_strategy, run_weekly, set_benchmark, set_option, set_order_cost,
    set_slippage, set_universe, valuation,
)
from .data import JQData  # noqa: F401

__all__ = [
    # 上下文与全局
    "Context", "GlobalNamespace", "log", "g",
    # 调度
    "run_daily", "run_weekly",
    # 设置
    "set_benchmark", "set_option", "set_slippage", "set_order_cost",
    "set_universe", "FixedSlippage", "PriceRelatedSlippage", "OrderCost",
    "OrderStatus", "Order",
    # 数据
    "get_price", "history", "get_current_data", "get_index_stocks",
    "get_security_info", "get_trade_days", "get_all_securities",
    "get_fundamentals", "query", "valuation", "income", "balance", "cash_flow",
    "get_industries", "get_industry_stocks", "SecurityInfo",
    # 交易
    "order", "order_target", "order_value", "order_target_value",
    "order_target_percent",
    # 运行
    "run_strategy", "JQEngine", "JQData",
    # 时间（策略里直接用 datetime / timedelta）
    "datetime", "date", "time", "timedelta",
]


def g():
    """占位：聚宽的 g 是 initialize 里赋值的全局对象，
    这里由引擎在 run 时注入到策略模块的命名空间。"""
    raise RuntimeError("g 由引擎注入，不要在 joinquant 命名空间直接取")
