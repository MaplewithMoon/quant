# -*- coding: utf-8 -*-
"""Walk-Forward 前进式样本外验证（防过拟合）

为什么必须做
------------
在同一段历史上试 100 组参数、挑夏普最高的那组，得到的"夏普 3.4"是**数据窥探**
的产物——它在样本外大概率失效。项目里 `scripts/backtest.py --compare`
把 11 个策略按夏普排序，就是这个问题的温和版本。

Walk-Forward 的做法
-------------------
    |---- 训练(选参) ----|---- 测试(验证) ----|
                  |---- 训练 ----|---- 测试 ----|
                                |---- 训练 ----|---- 测试 ----|
    ← 滚动前进，测试段首尾相接，互不重叠 →

每个窗口只用**它之前**的数据选参，然后在**紧接着的、没参与选参的**数据上跑，
最后把各测试段拼接成一条**样本外净值曲线**——只有这条曲线的表现才是诚实的。

同时输出「样本内 vs 样本外」的落差：落差越大，说明参数越是在拟合噪音。
"""
import time
from dataclasses import dataclass, field
from typing import Callable, Dict, List

import pandas as pd

from backtest.engine import BacktestEngine
from backtest.metrics import Metrics
from execution.market_rules import MarketRules

from .search import (expand_space, sample_space, _score_of, OBJECTIVES)


@dataclass
class Window:
    """一个 walk-forward 窗口"""
    idx: int
    train_start: pd.Timestamp
    train_end: pd.Timestamp
    test_start: pd.Timestamp
    test_end: pd.Timestamp

    def __str__(self):
        return (f"W{self.idx}: 训练 {self.train_start.date()}~{self.train_end.date()} "
                f"→ 测试 {self.test_start.date()}~{self.test_end.date()}")


def make_windows(index: pd.DatetimeIndex, train_days: int = 252,
                 test_days: int = 63, step_days: int = None,
                 anchored: bool = False) -> List[Window]:
    """按交易日切分 walk-forward 窗口

    参数:
        index:      交易日索引
        train_days: 训练段长度（交易日）
        test_days:  测试段长度（交易日）
        step_days:  每次前进多少（默认 = test_days，即测试段首尾相接）
        anchored:   True=训练段从最早开始逐窗扩张（anchored）；
                    False=训练段固定长度滚动（rolling，默认）
    """
    if test_days <= 0 or train_days <= 0:
        raise ValueError("train_days / test_days 必须为正")
    step = step_days or test_days
    n = len(index)
    out = []
    i = 0
    while True:
        train_lo = 0 if anchored else i
        train_hi = i + train_days - 1
        test_lo = train_hi + 1
        test_hi = test_lo + test_days - 1
        if test_hi >= n:
            break
        out.append(Window(
            idx=len(out),
            train_start=index[train_lo], train_end=index[train_hi],
            test_start=index[test_lo], test_end=index[test_hi],
        ))
        i += step
    return out


def slice_ds(ds, start, end):
    """按日期切出子 DataSet（保持原始顺序与指标列）"""
    from data.dataset import DataSet
    sub = ds.data.loc[(ds.data.index >= start) & (ds.data.index <= end)]
    return DataSet(symbol=ds.symbol, data=sub.copy())


@dataclass
class WalkForwardResult:
    windows: pd.DataFrame                       # 每窗口：最佳参数 + IS/OOS 表现
    oos_equity: pd.Series                       # 拼接的样本外净值（起点 1.0）
    oos_trades: pd.DataFrame
    oos_metrics: Metrics
    n_trials_per_window: int = 0
    is_sharpe_mean: float = 0.0
    oos_sharpe_mean: float = 0.0
    param_stability: Dict = field(default_factory=dict)

    def overfit_gap(self) -> float:
        """样本内与样本外的目标值落差（越大越可能是过拟合）"""
        return self.is_sharpe_mean - self.oos_sharpe_mean

    def summary(self) -> str:
        L = ["=" * 74, "Walk-Forward 样本外验证结果", "=" * 74]
        if self.windows.empty:
            L.append("  没有可用窗口（数据长度不足）")
            return "\n".join(L)
        w = self.windows
        L.append(f"  窗口数            : {len(w)}")
        if "train_days" in w.columns:
            L.append(f"  每窗训练/测试     : {int(w['train_days'].iloc[0])} / "
                     f"{int(w['test_days'].iloc[0])} 交易日")
        traded = int((w["OOS_交易"] > 0).sum()) if "OOS_交易" in w.columns else 0
        L.append(f"  测试段有交易的窗口: {traded}/{len(w)}"
                 f"{'   ← 多数窗口没触发信号，样本外结论参考价值有限' if traded < len(w) * 0.5 else ''}")
        L.append(f"  每窗尝试参数组数  : {self.n_trials_per_window}")
        L.append("")
        L.append(f"  样本内(IS) 平均夏普: {self.is_sharpe_mean:>8.3f}   ← 选参时看到的表现")
        L.append(f"  样本外(OOS) 平均夏普:{self.oos_sharpe_mean:>8.3f}   ← 真实拿到的表现")
        gap = self.overfit_gap()
        verdict = ("落差很小，参数较稳健" if gap < 0.3 else
                   "落差明显，参数很可能在拟合噪音" if gap < 1.0 else
                   "落差极大，几乎可以断定是过拟合")
        L.append(f"  过拟合落差        : {gap:>8.3f}   {verdict}")
        L.append("")
        m = self.oos_metrics
        L.append("  —— 拼接后的样本外业绩（唯一诚实的结果）——")
        L.append(f"    总收益率  : {m.total_return:>8.2%}")
        L.append(f"    年化收益率: {m.annual_return:>8.2%}")
        L.append(f"    夏普比率  : {m.sharpe_ratio:>8.2f}")
        L.append(f"    最大回撤  : {m.max_drawdown:>8.2%}")
        L.append(f"    交易次数  : {m.total_trades:>8d}")
        L.append(f"    胜率      : {m.win_rate:>8.2%}")
        if self.param_stability:
            L.append("")
            L.append("  —— 各窗口选出的参数（越集中越稳定）——")
            for name, counts in self.param_stability.items():
                top = sorted(counts.items(), key=lambda kv: -kv[1])[:6]
                L.append(f"    {name:<12}: " + ", ".join(f"{k}×{v}" for k, v in top))
        L.append("=" * 74)
        return "\n".join(L)


def walk_forward(ds, factory: Callable[[Dict], object], space,
                 train_days: int = 252, test_days: int = 63, step_days: int = None,
                 mode: str = "random", n_iter: int = 30, objective: str = "sharpe",
                 seed: int = 42, bt_kwargs: Dict = None,
                 anchored: bool = False, verbose: bool = True) -> WalkForwardResult:
    """执行 walk-forward 验证

    参数:
        ds:        预处理后的完整 DataSet（含指标）
        factory:   参数字典 -> 策略实例
        space:     参数空间（见 optimizer.search.ParamSpace）
        train_days/test_days/step_days/anchored: 窗口切分方式
        mode:      "random"（随机采样，默认）或 "grid"（全枚举）
        n_iter:    random 模式下每窗尝试的组数
        objective: 选参目标（sharpe / total_return / calmar ...）
        bt_kwargs: 传给 BacktestEngine 的参数（费用、成交时点、制度约束等）

    返回:
        WalkForwardResult
    """
    bt_kwargs = dict(bt_kwargs or {})
    bt_kwargs.setdefault("market_rules", MarketRules())

    index = ds.data.index
    windows = make_windows(index, train_days, test_days, step_days, anchored)
    if not windows:
        return WalkForwardResult(pd.DataFrame(), pd.Series(dtype=float),
                                 pd.DataFrame(), Metrics(), n_iter)

    if verbose:
        print(f"Walk-Forward: {len(windows)} 个窗口, "
              f"训练 {train_days} 日 / 测试 {test_days} 日, "
              f"每窗 {n_iter if mode == 'random' else '全'} 组参数")

    combos = (sample_space(space, n_iter, seed=seed) if mode == "random"
              else expand_space(space))
    n_trials = len(combos)

    rows = []
    pieces = []
    all_trades = []
    level = 1.0
    is_scores = []
    oos_scores = []
    stability = {}

    t0 = time.time()
    for w in windows:
        train_ds = slice_ds(ds, w.train_start, w.train_end)
        test_ds = slice_ds(ds, w.test_start, w.test_end)
        if len(train_ds) < 20 or len(test_ds) < 2:
            continue

        # ---- 只用训练段选参 ----
        best, best_score = None, float("-inf")
        for p in combos:
            eng = BacktestEngine(**bt_kwargs)
            try:
                tr = eng.run(train_ds, factory(p))
                sc = _score_of(Metrics.compute(eng.equity_curve, tr), objective)
            except Exception:
                continue
            if sc > best_score:
                best, best_score = dict(p), sc
        if best is None:
            continue

        # ---- 在紧随其后的测试段验证（参数不再调整）----
        eng = BacktestEngine(**bt_kwargs)
        try:
            test_trades = eng.run(test_ds, factory(best))
        except Exception as e:
            if verbose:
                print(f"  {w} 测试段失败: {e}")
            continue
        test_eq = eng.equity_curve
        if len(test_eq) < 2 or test_eq.iloc[0] <= 0:
            continue
        oos_m = Metrics.compute(test_eq, test_trades)

        for k, v in best.items():
            stability.setdefault(k, {})
            stability[k][v] = stability[k].get(v, 0) + 1

        norm = test_eq / test_eq.iloc[0]
        pieces.append(norm * level)
        level = float((norm * level).iloc[-1])
        if not test_trades.empty:
            all_trades.append(test_trades)

        is_scores.append(best_score)
        oos_scores.append(_score_of(oos_m, objective))
        rows.append({
            "窗口": w.idx,
            "train_days": len(train_ds),
            "test_days": len(test_ds),
            "训练区间": f"{w.train_start.date()}~{w.train_end.date()}",
            "测试区间": f"{w.test_start.date()}~{w.test_end.date()}",
            "最佳参数": ", ".join(f"{k}={v}" for k, v in best.items()),
            "IS_目标值": best_score,
            "OOS_目标值": _score_of(oos_m, objective),
            "OOS_收益": oos_m.total_return,
            "OOS_夏普": oos_m.sharpe_ratio,
            "OOS_回撤": oos_m.max_drawdown,
            "OOS_交易": oos_m.total_trades,
        })
        if verbose:
            print(f"  W{w.idx} 训练→测试 {w.test_start.date()}~{w.test_end.date()}  "
                  f"IS={best_score:.3f} OOS={_score_of(oos_m, objective):.3f}  "
                  f"{best}")

    if not pieces:
        return WalkForwardResult(pd.DataFrame(), pd.Series(dtype=float),
                                 pd.DataFrame(), Metrics(), n_trials)

    oos_equity = pd.concat(pieces)
    oos_equity = oos_equity[~oos_equity.index.duplicated(keep="last")].sort_index()
    oos_trades = pd.concat(all_trades) if all_trades else pd.DataFrame()
    oos_metrics = Metrics.compute(oos_equity, oos_trades)

    if verbose:
        print(f"完成，用时 {time.time()-t0:.0f}s")

    return WalkForwardResult(
        windows=pd.DataFrame(rows),
        oos_equity=oos_equity,
        oos_trades=oos_trades,
        oos_metrics=oos_metrics,
        n_trials_per_window=n_trials,
        is_sharpe_mean=float(sum(is_scores) / len(is_scores)) if is_scores else 0.0,
        oos_sharpe_mean=float(sum(oos_scores) / len(oos_scores)) if oos_scores else 0.0,
        param_stability=stability,
    )
