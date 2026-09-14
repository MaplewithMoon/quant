# -*- coding: utf-8 -*-
"""策略参数优化与样本外验证

模块:
    search.py       参数空间定义、网格搜索、随机搜索
    walkforward.py  Walk-Forward 前进式样本外验证（防过拟合）

⚠️ 使用警告
-----------
`grid_search` / `random_search` 是在**同一段数据**上挑参数，
挑出来的"最优参数"天然被数据窥探（data snooping）污染：
试的参数组合越多，最好看的那个越可能只是噪音。

**正确做法是 `walk_forward`**：用前一段数据选参、紧接的一段做样本外验证，
滚动前进，最后只看拼接出来的样本外表现。

本包在搜索结果的 summary 里会明确提示试了多少组参数，
避免把"万里挑一的好看结果"当成真实能力。
"""
from .search import (ParamSpace, expand_space, sample_space,
                     grid_search, random_search, OBJECTIVES)
from .walkforward import (make_windows, walk_forward, Window,
                          WalkForwardResult, slice_ds)

__all__ = [
    "ParamSpace", "expand_space", "sample_space",
    "grid_search", "random_search", "OBJECTIVES",
    "make_windows", "walk_forward", "Window", "WalkForwardResult", "slice_ds",
]
