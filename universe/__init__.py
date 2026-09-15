# -*- coding: utf-8 -*-
"""股票池：历史指数成分 + 多维过滤，消除幸存者偏差

快速上手
--------
    from factors import load_panel
    from universe import UniverseSpec, build_universe, universe_size

    panel = load_panel("2020-01-01", "2024-12-31")
    mask = build_universe(panel, UniverseSpec(index_code="000300.SH",
                                             min_listed_days=60,
                                             min_amount=5e7))
    print(universe_size(mask).describe())      # 每日可选股票数

关键点
------
`index_member_panel()` 在每个交易日只使用**当时已经公布的**指数成分快照，
而不是今天的成分 —— 后者会让回测收益系统性虚高（幸存者偏差）。
"""
from .pool import (UniverseSpec, build_universe, universe_size,
                   index_member_panel, index_weight_panel, load_index_members,
                   st_panel, suspended_panel, listed_days_panel)

__all__ = [
    "UniverseSpec", "build_universe", "universe_size",
    "index_member_panel", "index_weight_panel", "load_index_members",
    "st_panel", "suspended_panel", "listed_days_panel",
]
