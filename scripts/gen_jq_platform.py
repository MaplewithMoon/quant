# -*- coding: utf-8 -*-
"""生成**聚宽平台原生**策略文件（供在 joinquant.com 上直接跑回测做对照）

【为什么要单独生成，而不是直接用手写的那个】
`strategies/jq/*.py` 是**我们本地**的移植版：主体与原版逐行一致，但文件头
换成了 `from joinquant import *`（走本项目的兼容层）。平台上的文件不能有这一行
—— 聚宽把 API 注入全局命名空间。

【对照实验的设计】
同一份策略主体：

    本地：joinquant 兼容层 + 我们的执行模型（T+1、涨跌停、整手、停牌…）
    平台：聚宽真实撮合（分钟级，但我们只用日线数据），聚宽自己的费用/滑点

两边的差 = **兼容层与撮合的保真度**，正是 F1~F7 那批待解释的近似。
所以文件头里必须写死本次本地跑的参数（区间/本金/费用/滑点），
否则两边参数不一致，差值就没有意义。

⚠️ 生成的文件**不参与本项目的测试与 CI** —— 它是给平台用的，本地跑不了
（本地没有 `jqdata` 模块）。

用法：
    python scripts/gen_jq_platform.py
输出：
    jq_platform/<name>.py
"""
import sys
from pathlib import Path

sys.path.insert(0, ".")
if __name__ == "__main__":
    from utils.console import force_utf8_stdout
    force_utf8_stdout()

SRC_DIR = Path("strategies/jq")
OUT_DIR = Path("jq_platform")

# 与 jq_backtest.py 的默认参数保持一致（改这里要同步改对照组说明）
LOCAL_PARAMS = {
    "区间": "2024-01-01 ~ 2025-12-31",
    "本金": "100,000",
    "成交价": "auto（上午开盘 / 下午收盘，日线近似）",
    "制度约束": "开（T+1、涨跌停封板、整手、停牌停交）",
    "冲击模型": "none（仅固定滑点）",
    "附注": "策略内 set_slippage/set_order_cost 会覆盖命令行的默认值，"
            "以策略里写的为准",
}

# 平台版要还原的 import —— 对照 port_joinquant.py 的反向操作
HEADER_TMPL = '''# -*- coding: utf-8 -*-
"""国九小市值策略（**聚宽平台原生版**，仅用日线数据）

本文件由 `scripts/gen_jq_platform.py` 从 `{src}` 自动生成。
它**不改策略主体**，只做两件事：
  1. 还原平台 import：`from joinquant import *` -> `from jqdata import *`
     （聚宽把 API 注入全局命名空间，本地兼容层才需要那行 import）
  2. 加上下面的对照说明

────────────────────────────────────────────────────────────────
【对照实验说明】请用完全相同的参数跑，否则差值不可解释

{params}

策略内部已经调用 `set_option('avoid_future_data', True)`、
`set_slippage(FixedSlippage(...))`、`set_order_cost(OrderCost(...))`，
**以策略里写的为准**；平台默认的滑点/费率不要另外改。

【这份文件为什么存在】
本项目把同一份策略主体在**本地**跑了一遍（`scripts/jq_backtest.py`），
用的是自建的聚宽兼容层 + 自建执行模型（T+1 / 涨跌停封板 / 整手 / 停牌）。
平台上跑出来的是聚宽**真实撮合**。两边的差就是**兼容层与撮合的保真度**，
对应本项目已知的 7 条移植近似（F1~F7），其中最主要的是：

  F1  成交时点：聚宽是分钟撮合，我们只能用日线近似
      -> **差异的首要来源**
  F3  `history(1,'1m')` / `get_current_data().last_price` 这类"当前价"，
      我们只能拿当日开盘价近似 -> 涨跌停判断会有出入
  F6  指数成分：我们只有月末权重快照，平台是精确的历史成分
  F7  涨跌停/ST 口径：我们按自己的规则算，平台用自己的

所以**有差异是预期的**，重点不是"能不能完全对上"，而是**差多少、差在哪**。
建议对比时看：年化 / 最大回撤 / 夏普 / 换手 / 换手次数 / 逐月收益相关性。

⚠️ 2026-09 追加：第一轮对照发现本地结果与平台**差了整整一个量级**
（本地 +2.36% vs 平台 +185.52%），根因**不是上面这些近似**，而是本地兼容层
一个静默的数据读取 bug：`joinquant/data.py::_load_financials` 把同目录下的
balance/cashflow/profit 三张报表一起读了，去重后系统性留下资产负债表行 ->
`get_fundamentals` 的收入/利润字段恒为 NaN -> 选股结果恒为空 -> 策略全程
空仓买货币 ETF。已修复并加了回归测试。**近似只会带来几个百分点的偏差，
几十上百个百分点的差一定是 bug**，这个判据请记住。
────────────────────────────────────────────────────────────────
"""
from jqdata import *  # noqa: F401,F403
'''


def _params_block():
    w = max(len(k) for k in LOCAL_PARAMS)
    return "\n".join(f"  {k.ljust(w)} : {v}" for k, v in LOCAL_PARAMS.items())


def convert(src: Path, out: Path) -> int:
    lines = src.read_text(encoding="utf-8").split("\n")
    # 定位策略主体起点：找到原版的版权注释行或 `import numpy`
    start = next((i for i, ln in enumerate(lines)
                  if "克隆自聚宽文章" in ln), None)
    if start is None:
        start = next(i for i, ln in enumerate(lines) if ln.startswith("import numpy"))
    body = "\n".join(lines[start:])

    # 正文里若有 `from joinquant` 残留，还原（主体通常没有）
    body = body.replace("from joinquant import *", "from jqdata import *")
    body = body.replace("from six import BytesIO", "from io import BytesIO")

    header = HEADER_TMPL.format(src=src.as_posix(), params=_params_block())
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(header + "\n" + body + "\n", encoding="utf-8")
    return body.count("\n") + 1


def main():
    if not SRC_DIR.exists():
        print(f"✗ 找不到 {SRC_DIR}")
        return 1
    srcs = sorted(SRC_DIR.glob("guojiu_smallcap*.py"))
    if not srcs:
        print(f"✗ {SRC_DIR} 下没有 guojiu_smallcap*.py")
        return 1
    print(f"生成聚宽平台原生版 -> {OUT_DIR}/")
    for s in srcs:
        name = s.stem + "_jq_platform.py"
        n = convert(s, OUT_DIR / name)
        print(f"  ✓ {OUT_DIR / name}  （策略主体 {n} 行）")
    print()
    print("用法：把 jq_platform/ 下的文件内容整份复制到聚宽「策略」编辑器，")
    print("      参数只改回测区间与初始资金，其余保持文件里写的。")
    return 0


if __name__ == "__main__":
    sys.exit(main())
