# -*- coding: utf-8 -*-
"""多标的（组合）参数搜索 + 训练/测试集切分

`optimizer/search.py` 是给**单标的** `BacktestEngine` 用的（`engine.run(ds, strategy)`）。
组合策略走的是 `PortfolioBacktestEngine`（输入是目标权重面板），签名完全不同，
所以在这里补一套同风格的组合版搜索。

【为什么必须切训练/测试集】
    在同一个区间上既选参数又报业绩，等价于"用未来信息挑参数"，Sharpe 会被
    系统性高估。本模块强制把区间切成 train / test：
        - 网格搜索只在 train 上做
        - test 上只用**最优那一组**参数跑一次，且允许只跑一次
        - 报出两者的落差（`overfit_gap`）作为过拟合的量化证据
"""
from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional

import numpy as np
import pandas as pd

from backtest.multi_engine import PortfolioBacktestEngine
from backtest.metrics import Metrics
from analytics.performance import information_ratio, to_returns

OBJECTIVES = {
    "sharpe": "sharpe_ratio",
    "calmar": "calmar_ratio",
    "annual_return": "annual_return",
    "total_return": "total_return",
    "max_drawdown": "max_drawdown",
}


# ============================================================
# 区间切分
# ============================================================
@dataclass
class Split:
    """训练 / 测试 / 全样本 三段区间"""
    start: str
    split: str
    end: str

    @property
    def train(self):
        return (self.start, self.split)

    @property
    def test(self):
        return (self.split, self.end)

    @property
    def full(self):
        return (self.start, self.end)

    def describe(self) -> str:
        return (f"训练集 {self.start} ~ {self.split}    "
                f"测试集 {self.split} ~ {self.end}")


def make_split(start: str, split: str, end: str) -> Split:
    """校验区间顺序，避免把测试集放在训练集前面"""
    s, m, e = pd.Timestamp(start), pd.Timestamp(split), pd.Timestamp(end)
    if not (s < m < e):
        raise ValueError(f"区间顺序必须 start < split < end，实得 {start} / {split} / {end}")
    return Split(s.strftime("%Y-%m-%d"), m.strftime("%Y-%m-%d"), e.strftime("%Y-%m-%d"))


def slice_panel(panel: dict, start: str, end: str) -> dict:
    """按日期切面板（所有宽表一起切，保证维度一致）"""
    s, e = pd.Timestamp(start), pd.Timestamp(end)
    out = {}
    for k, v in panel.items():
        if isinstance(v, pd.DataFrame):
            out[k] = v.loc[(v.index >= s) & (v.index <= e)]
        else:
            out[k] = v
    return out


def slice_mask(mask: pd.DataFrame, start: str, end: str) -> pd.DataFrame:
    s, e = pd.Timestamp(start), pd.Timestamp(end)
    return mask.loc[(mask.index >= s) & (mask.index <= e)]


def slice_benchmark(bench: pd.Series, start: str, end: str) -> pd.Series:
    if bench is None or bench.empty:
        return bench
    s, e = pd.Timestamp(start), pd.Timestamp(end)
    return bench.loc[(bench.index >= s) & (bench.index <= e)]


# ============================================================
# 单次评估
# ============================================================
def score_of(m: Metrics, objective: str, returns: pd.Series = None,
             bench_returns: pd.Series = None) -> float:
    """把绩效压成一个可排序的标量（越大越好）"""
    if objective == "information_ratio":
        if returns is None or bench_returns is None:
            return float("-inf")
        return information_ratio(returns, bench_returns)
    if objective == "max_drawdown":
        return -abs(m.max_drawdown)
    attr = OBJECTIVES.get(objective, objective)
    v = getattr(m, attr, float("nan"))
    try:
        v = float(v)
    except (TypeError, ValueError):
        return float("-inf")
    if not np.isfinite(v):
        return float("-inf")
    return v


def evaluate_portfolio(panel: dict, mask: pd.DataFrame, factory: Callable,
                       params: Dict, objective: str = "sharpe",
                       benchmark: pd.Series = None,
                       engine_kwargs: Dict = None,
                       verbose: bool = False) -> Dict:
    """跑一次组合回测

    factory: (panel, mask, params) -> 目标权重面板（宽表，非调仓日 NaN）

    返回 dict，含 `_score` 与完整 `metrics`；失败时 `_score=-inf` 且 `_error` 有值，
    **不会**把异常抛出去打断整个网格搜索。
    """
    engine_kwargs = dict(engine_kwargs or {})
    try:
        weights = factory(panel, mask, params)
        engine = PortfolioBacktestEngine(**engine_kwargs)
        res = engine.run(panel, weights, verbose=verbose)
        m = res.metrics
        r = to_returns(res.equity)
        br = to_returns(benchmark) if benchmark is not None and not benchmark.empty else None
        sc = score_of(m, objective, r, br)
        return {**params, "_score": sc, "_error": "",
                "total_return": m.total_return, "annual_return": m.annual_return,
                "annual_volatility": m.annual_volatility,
                "sharpe_ratio": m.sharpe_ratio, "calmar_ratio": m.calmar_ratio,
                "max_drawdown": m.max_drawdown, "win_rate": m.win_rate,
                "total_trades": m.total_trades, "ledger_gap": res.ledger_gap,
                "metrics": m, "equity": res.equity, "result": res, "weights": weights}
    except Exception as e:
        return {**params, "_score": float("-inf"), "_error": f"{type(e).__name__}: {e}"[:120],
                "total_return": float("nan"), "annual_return": float("nan"),
                "annual_volatility": float("nan"), "sharpe_ratio": float("nan"),
                "calmar_ratio": float("nan"), "max_drawdown": float("nan"),
                "win_rate": float("nan"), "total_trades": 0, "ledger_gap": float("nan"),
                "metrics": None, "equity": None, "result": None, "weights": None}


def _to_frame(rows: List[Dict]) -> pd.DataFrame:
    return pd.DataFrame([{k: v for k, v in r.items()
                          if k not in ("metrics", "equity", "result", "weights")}
                         for r in rows])


def grid_search_portfolio(panel: dict, mask: pd.DataFrame, factory: Callable,
                          space, objective: str = "sharpe",
                          benchmark: pd.Series = None,
                          engine_kwargs: Dict = None, n_grid: int = 10,
                          verbose: bool = True, label: str = "") -> pd.DataFrame:
    """网格搜索（组合版）

    返回按 `_score` 降序的 DataFrame。**只在训练集上调用**。
    """
    from optimizer.search import expand_space

    combos = expand_space(space, n_grid=n_grid)
    if verbose:
        tag = f"[{label}] " if label else ""
        print(f"{tag}网格搜索: {len(combos)} 组参数, 目标={objective}")
    rows = []
    for i, p in enumerate(combos):
        rows.append(evaluate_portfolio(panel, mask, factory, p, objective,
                                       benchmark, engine_kwargs))
        if verbose and (i + 1) % max(1, len(combos) // 10) == 0:
            print(f"  {i+1}/{len(combos)}", flush=True)
    df = _to_frame(rows)
    df = df.sort_values("_score", ascending=False).reset_index(drop=True)
    df.attrs["n_trials"] = len(combos)
    df.attrs["objective"] = objective
    return df


# ============================================================
# 训练/测试对照
# ============================================================
@dataclass
class OverfitReport:
    best_params: Dict = field(default_factory=dict)
    train_score: float = float("nan")
    test_score: float = float("nan")
    gap: float = float("nan")               # train - test，越大越过拟合
    train_metrics: Dict = field(default_factory=dict)
    test_metrics: Dict = field(default_factory=dict)
    search_table: pd.DataFrame = field(default_factory=pd.DataFrame)
    n_trials: int = 0

    def describe(self) -> str:
        L = ["=" * 74, "样本内外对照（过拟合体检）", "=" * 74]
        L.append(f"  搜索次数      : {self.n_trials}")
        L.append(f"  最优参数      : {self.best_params}")
        L.append(f"  样本内得分    : {self.train_score:+.4f}")
        L.append(f"  样本外得分    : {self.test_score:+.4f}")
        L.append(f"  过拟合落差    : {self.gap:+.4f}   "
                 f"({'落差较大，警惕过拟合' if self.gap > 0.5 else '落差可接受'})")
        L.append("=" * 74)
        return "\n".join(L)


def train_test_search(panel_train: dict, mask_train: pd.DataFrame,
                      panel_test: dict, mask_test: pd.DataFrame,
                      factory: Callable, space, objective: str = "sharpe",
                      benchmark_train: pd.Series = None,
                      benchmark_test: pd.Series = None,
                      engine_kwargs: Dict = None, n_grid: int = 10,
                      verbose: bool = True) -> OverfitReport:
    """在训练集上网格搜索，把最优参数拿到测试集上跑一次"""
    search = grid_search_portfolio(panel_train, mask_train, factory, space,
                                   objective, benchmark_train, engine_kwargs,
                                   n_grid, verbose, label="训练集")
    if search.empty or not np.isfinite(search.iloc[0]["_score"]):
        return OverfitReport(search_table=search, n_trials=len(search))

    param_names = [c for c in search.columns
                   if not c.startswith("_") and c not in
                   ("total_return", "annual_return", "annual_volatility",
                    "sharpe_ratio", "calmar_ratio", "max_drawdown", "win_rate",
                    "total_trades", "ledger_gap")]
    best = {k: search.iloc[0][k] for k in param_names}

    tr = evaluate_portfolio(panel_train, mask_train, factory, best, objective,
                            benchmark_train, engine_kwargs)
    te = evaluate_portfolio(panel_test, mask_test, factory, best, objective,
                            benchmark_test, engine_kwargs)
    return OverfitReport(
        best_params=best,
        train_score=float(tr["_score"]),
        test_score=float(te["_score"]),
        gap=float(tr["_score"] - te["_score"]),
        train_metrics=tr,
        test_metrics=te,
        search_table=search,
        n_trials=len(search),
    )


def params_of(row: pd.Series, drop_prefix: str = "_",
              exclude=("total_return", "annual_return", "annual_volatility",
                       "sharpe_ratio", "calmar_ratio", "max_drawdown", "win_rate",
                       "total_trades", "ledger_gap", "_error")) -> Dict:
    """从搜索结果的一行里取出参数（剔除绩效列）"""
    return {k: row[k] for k in row.index
            if not str(k).startswith(drop_prefix) and k not in exclude}
