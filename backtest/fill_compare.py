# -*- coding: utf-8 -*-
"""成交时点对照：next_open vs same_close（T1·②）

【为什么对照要**默认跑**，而不是"可以传开关跑两版"】
`sector_rotation_backtest.py` 里早就写了"可用 --fill open/close 做敏感性"，
但**没人会真的去跑两版**，于是"结论对成交时点多敏感"这个问题一直没有答案。
所以对照必须是**报告的默认动作**，不是可选实验。

【口径（写死）】
    next_open   T日收盘决策 → T+1开盘成交     **可实现，主结果**
    same_close  当日收盘决策且按当日收盘成交    **上界，实盘不可复现**

两版的**差值**就是"成交时点近似"带来的不确定度。这个数字本身是结论的一部分：
差值很小 → 结论稳健，F1（与聚宽的差异）可以往别处找；
差值很大 → 结论高度依赖一个不可实现的假设，必须先解决它再去比聚宽。
"""
from typing import Callable, List, Optional

import pandas as pd

from execution.setup import FILL_NEXT_OPEN, FILL_SAME_CLOSE

_METRICS = (("年化", "annual_return", "{:+.2%}"),
            ("夏普", "sharpe_ratio", "{:+.3f}"),
            ("最大回撤", "max_drawdown", "{:+.2%}"),
            ("年化波动", "annual_volatility", "{:+.2%}"))


def compare_fill_timing(engine_factory: Callable, panel: dict,
                        target_weights: pd.DataFrame,
                        base_kwargs: dict = None,
                        run_kwargs: dict = None) -> dict:
    """跑两版（只差 fill_timing），返回 {main, alt, text, diff}

    参数:
        engine_factory: `lambda **kw: PortfolioBacktestEngine(**kw)`，
                        便于不引入 backtest 依赖（避免循环 import）
        base_kwargs:    引擎参数（会覆盖 fill_timing）
        run_kwargs:     run() 的参数（risk_manager / context 等）
    """
    base_kwargs = dict(base_kwargs or {})
    run_kwargs = dict(run_kwargs or {})
    out = {"main": None, "alt": None, "text": "", "diff": {}}

    results = {}
    for mode in (FILL_NEXT_OPEN, FILL_SAME_CLOSE):
        kw = dict(base_kwargs)
        kw["fill_timing"] = mode
        try:
            eng = engine_factory(**kw)
            results[mode] = eng.run(panel, target_weights, **run_kwargs)
        except Exception as e:
            results[mode] = None
            out["text"] = f"（成交时点对照失败：{mode} -> {type(e).__name__}: {e}）"
            return out

    main, alt = results.get(FILL_NEXT_OPEN), results.get(FILL_SAME_CLOSE)
    out["main"], out["alt"] = main, alt
    if main is None or alt is None:
        out["text"] = "（成交时点对照：有一版没跑出来）"
        return out

    L = ["", "-" * 78, "成交时点对照（**same_close 是上界，不可复现**）", "-" * 78]
    L.append("  为什么默认跑两版：不跑就永远不知道结论对成交时点有多敏感。")
    L.append("  两版**只差** fill_timing，其余参数完全相同。")
    L.append("")
    hdr = f"  {'口径':<14}" + "".join(f"{n:>12}" for n, _, _ in _METRICS)
    L.append(hdr)
    for mode, res, tag in ((FILL_NEXT_OPEN, main, "主结果(可实现)"),
                           (FILL_SAME_CLOSE, alt, "上界(不可复现)")):
        row = f"  {tag:<14}"
        for _, attr, fmt in _METRICS:
            v = getattr(res.metrics, attr, float("nan"))
            try:
                row += f"{fmt.format(v):>12}"
            except (ValueError, TypeError):
                row += f"{'-':>12}"
        L.append(row)

    row = f"  {'差值(上界-主)':<14}"
    for _, attr, _ in _METRICS:
        a = getattr(main.metrics, attr, float("nan"))
        b = getattr(alt.metrics, attr, float("nan"))
        d = (b - a) if pd.notna(a) and pd.notna(b) else float("nan")
        out["diff"][attr] = d
        row += f"{d:>+12.4f}" if pd.notna(d) else f"{'-':>12}"
    L.append(row)

    # 一句话判读：差值大到什么程度才值得担心
    da = out["diff"].get("annual_return", float("nan"))
    if pd.notna(da):
        if abs(da) < 0.01:
            verdict = ("差值 <1%/年 —— 成交时点近似**不构成主要不确定性**，"
                       "F1 的差异应往别处找")
        elif abs(da) < 0.03:
            verdict = "差值 1%~3%/年 —— 需要注意，但结论方向大概率不变"
        else:
            verdict = ("差值 >3%/年 —— **结论高度依赖一个不可实现的假设**"
                       "（same_close），必须先解决成交时点再谈其它对比")
        L.append("")
        L.append(f"  判读：{verdict}")
    L.append("")
    L.append("  对外引用请用 **next_open**（主结果）；same_close 只用于说明"
             "「若假设更乐观会怎样」。")
    out["text"] = "\n".join(L)
    return out


def compare_fill_on_report(run_fn: Callable, base_kwargs: dict,
                           panel: dict, weights: pd.DataFrame,
                           run_kwargs: dict = None) -> List[str]:
    """给报告用的一层薄封装：返回可直接塞进 extra_sections 的行列表"""
    r = compare_fill_timing(run_fn, panel, weights,
                            base_kwargs=base_kwargs, run_kwargs=run_kwargs)
    return r["text"].split("\n") if r["text"] else []


__all__ = ["compare_fill_timing", "compare_fill_on_report"]
