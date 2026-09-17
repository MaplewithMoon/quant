# -*- coding: utf-8 -*-
"""执行层的**唯一配置入口**（T1·② 的根本修法）

【为什么要有这个模块】
原来的问题**不是"两个脚本缺开关"**，而是**成交时点与冲击模型的实现散落在
每个脚本里各写一遍**：

    脚本                          成交时点开关  冲击模型
    backtest.py                       有          有
    portfolio_backtest.py             有          有
    jq_backtest.py                    有          有
    optimize.py                       有          **无**
    multifactor_backtest.py          **无**        有
    sector_rotation_backtest.py      **无**        有
    semiconductor_rotation.py        **无**       **无**

每加一个脚本就抄一遍，抄漏一处就永久缺失。所以修法是**收敛成单一入口**：
脚本只调 `add_execution_args()` + `build_execution()`，
新增脚本自动获得全部执行参数，不再有"忘了接"这回事。

【口径（写死，不允许调用方改）】
    next_open   T日收盘决策 → T+1开盘成交
                **可实现口径，对外引用请用这个（主结果）**
    same_close  当日收盘决策且按当日收盘成交
                **上界：收盘价要收盘后才知道，实盘不可复现**，只能作对照

这个声明在 `analytics/result_report.py::FILL_CAVEAT` 里也会打印 ——
报告里不写死它，读者就会拿上界当结论。
"""
import argparse

# ---- 成交时点：**唯一常量定义处**（backtest 层从这里 import，避免两处漂移）----
FILL_NEXT_OPEN = "next_open"
FILL_SAME_CLOSE = "same_close"
FILL_CHOICES = (FILL_NEXT_OPEN, FILL_SAME_CLOSE)

# ---- 默认交易成本 ----
# ⚠️ 滑点默认 **0.001**（1bp）。历史上多个脚本默认 0.0，那会让结果偏乐观，
#    而且掩盖容量问题（--capital 调大也没反应）。别改回 0。
DEFAULT_SLIPPAGE = 0.001
DEFAULT_COMMISSION = 0.0001
DEFAULT_MIN_COMMISSION = 5.0
DEFAULT_STAMP_DUTY = 0.0005      # 卖出印花税
DEFAULT_TRANSFER_FEE = 0.00001   # 过户费
DEFAULT_LOT_SIZE = 100
DEFAULT_MAX_PARTICIPATION = 0.10
IMPACT_CHOICES = ("none", "fixed", "sqrt")


def add_execution_args(parser: argparse.ArgumentParser,
                       fill_default: str = FILL_NEXT_OPEN,
                       include_compare: bool = True,
                       skip=()) -> argparse.ArgumentParser:
    """给任意回测脚本挂上统一的执行参数

    新增脚本只要调它，就自动拥有成交时点 / 冲击模型 / 全部费用参数 ——
    这是"抄漏一处就永久缺失"的解法。

    skip: 已经自己定义过、不要重复定义的参数名（如 `("slippage",)`）。
          用显式参数而不是去翻 `parser._actions` 私有属性做去重 ——
          后者依赖 argparse 内部实现，脆弱且难查。
    """
    skip = set(skip)
    g = parser.add_argument_group("执行层（execution/setup.py 统一提供）")
    specs = [
        (("--fill-timing",), dict(default=fill_default, choices=FILL_CHOICES,
                                  dest="fill_timing",
                                  help="成交时点。next_open=可实现口径（主结果）；"
                                       "same_close=上界，不可复现，仅作对照")),
        (("--slippage",), dict(type=float, default=DEFAULT_SLIPPAGE,
                               help=f"固定滑点（默认 {DEFAULT_SLIPPAGE:.4f}）；"
                                    f"0 会让结果偏乐观并掩盖容量问题")),
        (("--impact-model",), dict(default="fixed", choices=IMPACT_CHOICES,
                                   help="市场冲击模型；sqrt 需配合加大 "
                                        "--capital 做容量分析")),
        (("--impact-k",), dict(type=float, default=0.1, help="平方根冲击系数")),
        (("--max-participation",), dict(type=float,
                                        default=DEFAULT_MAX_PARTICIPATION,
                                        help="单日成交量参与率上限（0=不限）")),
        (("--commission",), dict(type=float, default=DEFAULT_COMMISSION)),
        (("--min-commission",), dict(type=float,
                                     default=DEFAULT_MIN_COMMISSION)),
        (("--stamp-duty",), dict(type=float, default=DEFAULT_STAMP_DUTY)),
        (("--lot-size",), dict(type=int, default=DEFAULT_LOT_SIZE)),
    ]
    if include_compare:
        specs.append((("--no-fill-compare",),
                      dict(action="store_true",
                           help="跳过 next_open vs same_close 对照（默认**会跑**；"
                                "对照是报告的默认动作，不是可选实验）")))
    for flags, kw in specs:
        if kw.get("dest", flags[0].lstrip("-").replace("-", "_")) in skip:
            continue
        g.add_argument(*flags, **kw)
    return parser


def build_execution(args, *, with_impact: bool = True) -> dict:
    """把参数转成引擎 kwargs（`PortfolioBacktestEngine(**build_execution(args))`）

    只产出**执行层**关心的键；资金/调仓等由各脚本自己补。
    """
    out = {
        "slippage": float(getattr(args, "slippage", DEFAULT_SLIPPAGE)),
        "fill_timing": str(getattr(args, "fill_timing", FILL_NEXT_OPEN)),
    }
    for k, d in (("commission", DEFAULT_COMMISSION),
                 ("min_commission", DEFAULT_MIN_COMMISSION),
                 ("stamp_duty", DEFAULT_STAMP_DUTY)):
        v = getattr(args, k, None)
        if v is not None:
            out[k] = float(v)
    v = getattr(args, "lot_size", None)
    if v is not None:
        out["lot_size"] = int(v)
    v = getattr(args, "max_participation", None)
    if v is not None:
        out["max_participation"] = float(v)

    if with_impact:
        model = str(getattr(args, "impact_model", "none") or "none")
        if model != "none":
            from .impact import build_model
            out["slippage_model"] = build_model(
                model, rate=out["slippage"],
                k=float(getattr(args, "impact_k", 0.1)))
    return out


def describe_execution(exec_kwargs: dict) -> str:
    """一行执行口径描述，供报告头部/启动日志用"""
    ft = exec_kwargs.get("fill_timing", FILL_NEXT_OPEN)
    slip = exec_kwargs.get("slippage", DEFAULT_SLIPPAGE)
    model = exec_kwargs.get("slippage_model")
    mname = type(model).__name__ if model is not None else "固定滑点"
    return (f"成交时点 {ft}   滑点 {slip:.2%}   模型 {mname}   "
            f"参与率上限 {exec_kwargs.get('max_participation', 0):.0%}")


__all__ = ["FILL_NEXT_OPEN", "FILL_SAME_CLOSE", "FILL_CHOICES",
           "DEFAULT_SLIPPAGE", "DEFAULT_COMMISSION", "DEFAULT_MIN_COMMISSION",
           "DEFAULT_STAMP_DUTY", "DEFAULT_TRANSFER_FEE", "DEFAULT_LOT_SIZE",
           "DEFAULT_MAX_PARTICIPATION", "IMPACT_CHOICES",
           "add_execution_args", "build_execution", "describe_execution"]
