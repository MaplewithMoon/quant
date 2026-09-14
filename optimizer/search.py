# -*- coding: utf-8 -*-
"""参数空间定义与搜索

参数空间写法
------------
    {"fast": [5, 10, 20]}      离散取值
    {"fast": (2, 20)}          整数区间（随机搜索时均匀采样；网格搜索时展开成 2..20）
    {"vol_factor": (0.8, 2.0)} 浮点区间（网格搜索会展开成 10 档）

搜索方式
--------
    grid_search    笛卡尔积全枚举 —— 适合参数少（≤3 个）且范围窄
    random_search  随机采样 N 组 —— 适合参数多、范围大；也是 Bergstra 建议的
                   做法（随机搜索在高维下比网格更有效）

⚠️ 两者都是在同一段数据上挑参数，结果必然含数据窥探偏差。
   要评估真实能力请用 optimizer.walkforward.walk_forward。
"""
import itertools
import math
import time
from dataclasses import dataclass, field
from typing import Callable, Dict, List

import pandas as pd

from backtest.engine import BacktestEngine
from backtest.metrics import Metrics
from execution.market_rules import MarketRules

# 可作为优化目标的绩效指标
OBJECTIVES = {
    "sharpe": "sharpe_ratio",
    "total_return": "total_return",
    "annual_return": "annual_return",
    "calmar": "calmar_ratio",
    "profit_factor": "profit_factor",
    "win_rate": "win_rate",
    # 越低越好的目标会在内部取负
}
MINIMIZE = {"max_drawdown"}


@dataclass
class ParamSpace:
    """参数空间

    grid:    离散值列表，或 (lo, hi) 区间
    n_grid:  (lo, hi) 区间在网格搜索时展开的档数（默认展开成全部整数）
    is_int:  随机采样时是否取整
    """
    spec: Dict[str, object] = field(default_factory=dict)

    def names(self) -> List[str]:
        return list(self.spec.keys())

    def grid_values(self, name: str, n_grid: int = 10) -> List:
        v = self.spec[name]
        if isinstance(v, (list, tuple)) and len(v) == 2 and all(
                isinstance(x, (int, float)) for x in v) and not isinstance(v, list):
            lo, hi = v
            if isinstance(lo, int) and isinstance(hi, int) and (hi - lo) <= 60:
                return list(range(lo, hi + 1))
            step = (hi - lo) / max(n_grid - 1, 1)
            return [round(lo + i * step, 6) for i in range(n_grid)]
        if isinstance(v, (list, tuple)):
            return list(v)
        return [v]

    def sample(self, rng) -> Dict:
        """随机采样一组参数"""
        out = {}
        for name, v in self.spec.items():
            if isinstance(v, (list, tuple)) and len(v) == 2 and all(
                    isinstance(x, (int, float)) for x in v) and not isinstance(v, list):
                lo, hi = v
                if isinstance(lo, int) and isinstance(hi, int):
                    out[name] = int(rng.integers(lo, hi + 1))
                else:
                    out[name] = float(rng.uniform(lo, hi))
            elif isinstance(v, (list, tuple)):
                out[name] = v[int(rng.integers(len(v)))]
            else:
                out[name] = v
        return out

    def size(self, n_grid: int = 10) -> int:
        n = 1
        for name in self.spec:
            n *= len(self.grid_values(name, n_grid))
        return n


def expand_space(space, n_grid: int = 10) -> List[Dict]:
    """把参数空间展开成笛卡尔积列表"""
    if isinstance(space, ParamSpace):
        space = space.spec
    names = list(space.keys())
    values = []
    ps = ParamSpace(space)
    for name in names:
        values.append(ps.grid_values(name, n_grid))
    return [dict(zip(names, combo)) for combo in itertools.product(*values)]


def sample_space(space, n_iter: int, seed: int = 42) -> List[Dict]:
    """随机采样 n_iter 组参数（去重）"""
    import numpy as np
    if isinstance(space, ParamSpace):
        space = space.spec
    ps = ParamSpace(space)
    rng = np.random.default_rng(seed)
    seen = set()
    out = []
    for _ in range(n_iter * 5):
        if len(out) >= n_iter:
            break
        p = ps.sample(rng)
        key = tuple(sorted(p.items()))
        if key in seen:
            continue
        seen.add(key)
        out.append(p)
    return out


def evaluate(ds, factory: Callable[[Dict], object], params: Dict,
             objective: str = "sharpe", bt_kwargs: Dict = None) -> Dict:
    """跑一次回测并返回绩效

    factory: 参数字典 -> 策略实例
    bt_kwargs: 传给 BacktestEngine 的参数（费用/成交时点/制度约束等）
    """
    bt_kwargs = dict(bt_kwargs or {})
    bt_kwargs.setdefault("market_rules", MarketRules())
    engine = BacktestEngine(**bt_kwargs)
    try:
        trades = engine.run(ds, factory(params))
        m = Metrics.compute(engine.equity_curve, trades)
    except Exception as e:                      # 参数组合非法（如 fast>=slow）
        return {**params, "_error": str(e)[:80], "_score": float("-inf"),
                "total_return": float("nan"), "sharpe_ratio": float("nan"),
                "max_drawdown": float("nan"), "total_trades": 0,
                "equity": None, "engine": None}
    score = _score_of(m, objective)
    return {**params, "_score": score, "_error": "",
            "total_return": m.total_return, "annual_return": m.annual_return,
            "sharpe_ratio": m.sharpe_ratio, "calmar_ratio": m.calmar_ratio,
            "max_drawdown": m.max_drawdown, "win_rate": m.win_rate,
            "total_trades": m.total_trades,
            "equity": engine.equity_curve, "engine": engine}


def _score_of(m: Metrics, objective: str) -> float:
    if objective == "max_drawdown":
        return -abs(getattr(m, "max_drawdown"))
    attr = OBJECTIVES.get(objective, objective)
    v = getattr(m, attr, float("nan"))
    try:
        v = float(v)
    except (TypeError, ValueError):
        return float("-inf")
    if math.isnan(v) or math.isinf(v):
        return float("-inf")
    return v


def _run(param_list, ds, factory, objective, bt_kwargs, verbose):
    rows = []
    t0 = time.time()
    for i, p in enumerate(param_list):
        rows.append(evaluate(ds, factory, p, objective, bt_kwargs))
        if verbose and (i + 1) % max(1, len(param_list) // 10) == 0:
            print(f"  {i+1}/{len(param_list)}  用时 {time.time()-t0:.0f}s", flush=True)
    return rows


def _to_frame(rows) -> pd.DataFrame:
    return pd.DataFrame([{k: v for k, v in r.items()
                          if k not in ("equity", "engine")} for r in rows])


def grid_search(ds, factory, space, objective: str = "sharpe",
                bt_kwargs: Dict = None, n_grid: int = 10,
                verbose: bool = True) -> pd.DataFrame:
    """网格搜索：枚举参数空间的所有组合

    返回按目标值降序排列的 DataFrame（含 _score 列）。
    """
    combos = expand_space(space, n_grid=n_grid)
    if verbose:
        print(f"网格搜索: {len(combos)} 组参数, 目标={objective}")
    rows = _run(combos, ds, factory, objective, bt_kwargs, verbose)
    df = _to_frame(rows).sort_values("_score", ascending=False).reset_index(drop=True)
    df.attrs["n_trials"] = len(combos)
    return df


def random_search(ds, factory, space, n_iter: int = 100, objective: str = "sharpe",
                  seed: int = 42, bt_kwargs: Dict = None,
                  verbose: bool = True) -> pd.DataFrame:
    """随机搜索：采样 n_iter 组参数"""
    combos = sample_space(space, n_iter, seed=seed)
    if verbose:
        print(f"随机搜索: {len(combos)} 组参数, 目标={objective}")
    rows = _run(combos, ds, factory, objective, bt_kwargs, verbose)
    df = _to_frame(rows).sort_values("_score", ascending=False).reset_index(drop=True)
    df.attrs["n_trials"] = len(combos)
    return df


def overfit_warning(df: pd.DataFrame) -> str:
    """给出数据窥探提示：试了多少组、最优比中位数好多少"""
    n = df.attrs.get("n_trials", len(df))
    if df.empty:
        return ""
    best = df["_score"].iloc[0]
    med = df["_score"].median()
    return (f"⚠️ 本次共试了 {n} 组参数，最优 {best:.3f} vs 中位数 {med:.3f}。\n"
            f"   在 {n} 组里挑最好的，这个数字含有数据窥探偏差，不能当作真实能力。\n"
            f"   请用 optimizer.walk_forward（scripts/optimize.py --walk-forward）做样本外验证。")
