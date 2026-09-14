# -*- coding: utf-8 -*-
"""因子库：面板构建 + 因子计算 + 有效性评估（IC / 分层 / 换手 / 衰减）

快速上手
--------
    from factors import load_panel, get_factor, evaluate, list_factors

    print(list_factors())                       # 看有哪些因子
    panel = load_panel("2020-01-01", "2024-12-31", limit=800)
    rep = evaluate(get_factor("mom_20"), panel)
    print(rep.summary())

自定义因子
----------
    from factors.base import register

    @register("my_factor", "我的因子", direction=1, category="自定义")
    def my_factor(panel):
        c = panel["close_adj"]
        return c / c.shift(30) - 1      # 只用当日及以前的数据！

注册后即可用 CLI 评估：python scripts/factor_eval.py --factor my_factor ...
"""
from .base import Factor, FACTORS, register, add_factor, get_factor, list_factors
from .panel import (load_panel, forward_returns, adjusted_close,
                    universe_mask)
from .evaluation import (ic_series, ic_stats, quantile_returns,
                         long_short_curve, quantile_turnover, factor_decay,
                         monotonicity, evaluate, FactorReport)

# 导入内置因子库以完成注册（必须放在 base 之后）
from . import library  # noqa: F401,E402

__all__ = [
    "Factor", "FACTORS", "register", "add_factor", "get_factor", "list_factors",
    "load_panel", "forward_returns", "adjusted_close", "universe_mask",
    "ic_series", "ic_stats", "quantile_returns", "long_short_curve",
    "quantile_turnover", "factor_decay", "monotonicity", "evaluate",
    "FactorReport",
]
