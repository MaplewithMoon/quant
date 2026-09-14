# -*- coding: utf-8 -*-
"""因子定义与注册表

一个"因子"就是：把面板数据映射成"日期 × 股票"的因子值矩阵。
注册后就能被 IC 分析、分层回测、参数化调用统一处理。

约定
----
- `func(panel)` 只允许使用**当日及以前**的数据（rolling/expanding），
  绝不能用 `shift(-n)` 之类的前视操作。evaluation 会用前瞻收益做验证，
  但因子本身的因果性要靠写因子时守住。
- `direction`: +1 表示因子值越大越看多；-1 表示越小越看多。
  分层回测与多空组合会据此调整方向，保证"多头组合"始终是看多那一端。
"""
from dataclasses import dataclass
from typing import Callable, Dict, List

import pandas as pd


@dataclass
class Factor:
    name: str
    func: Callable[[dict], pd.DataFrame]
    desc: str = ""
    direction: int = 1
    category: str = ""

    def compute(self, panel: dict) -> pd.DataFrame:
        df = self.func(panel)
        if not isinstance(df, pd.DataFrame):
            raise TypeError(f"因子 {self.name} 必须返回 DataFrame，实际 {type(df)}")
        return df


FACTORS: Dict[str, Factor] = {}


def register(name: str, desc: str = "", direction: int = 1, category: str = ""):
    """装饰器：把函数注册成因子"""
    def deco(fn):
        FACTORS[name] = Factor(name=name, func=fn, desc=desc,
                               direction=direction, category=category)
        return fn
    return deco


def add_factor(name: str, fn, desc: str = "", direction: int = 1,
               category: str = "") -> Factor:
    """以编程方式注册因子（不常用装饰器时）"""
    FACTORS[name] = Factor(name=name, func=fn, desc=desc,
                           direction=direction, category=category)
    return FACTORS[name]


def get_factor(name: str) -> Factor:
    if name not in FACTORS:
        raise KeyError(f"未注册的因子 {name!r}，可用: {sorted(FACTORS)}")
    return FACTORS[name]


def list_factors() -> pd.DataFrame:
    """列出所有已注册因子"""
    rows = [{"因子": f.name, "类别": f.category, "方向": "越大越好" if f.direction > 0 else "越小越好",
             "说明": f.desc} for f in FACTORS.values()]
    return pd.DataFrame(rows).sort_values(["类别", "因子"]).reset_index(drop=True)
