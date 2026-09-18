# -*- coding: utf-8 -*-
"""交易记录 / 收益分析导出（对齐聚宽平台导出的那两份文本）

【为什么需要这一层】
    聚宽回测结束后可以导出两份文本：「交易记录」（逐笔成交）与「收益分析」
    （指标表）。我们原先只有权益 CSV + HTML 报告 —— **出问题时无法逐笔核对**
    "这笔到底成交没有、成交在哪天、成本多少"。做平台对照实验时，这一层不是
    可选项：净值曲线的差异要能落到具体某一笔成交上。

【和聚宽导出的两点差别（照实说明）】
    1. 聚宽的「交易记录」里**含未成交的委托**（成交数量 0、成交价为空）——
       涨停没买进、停牌卖不出都会留一行。我们的组合引擎把这类事件记在
       `reject_log` 里（`MultiBacktestResult.rejection_detail`），本模块把
       它作为**独立的「委托失败明细」**导出，而不是混进成交里 ——
       混进去会让"成交笔数/成交额"这类统计口径变味。
    2. 聚宽记的是**委托时间**（10:00 / 14:00 / 14:50），我们的引擎只有
       成交日。默认按成交时点口径填（`next_open` -> 09:30，`same_close` -> 15:00），
       可用 `fill_time` 覆盖。

【附带的对账（`audit_trades`）】
    只靠成交表也能查出的硬错误：现金变负、持仓变负、卖超、手数不是整手。
    这三项一旦不通过，**任何逐笔归因都不成立**，必须先修引擎。
"""
from typing import Dict, List, Optional

import re

import numpy as np
import pandas as pd

from .performance import (align, drawdown_info, drawdown_series,
                          performance_summary, to_returns)

# 聚宽导出文本的列（顺序照抄，方便并排 diff）
RECORD_COLS = ["日期", "委托时间", "标的", "交易类型", "下单类型",
               "成交数量", "成交价", "成交额", "平仓盈亏", "手续费"]

_ACTION_ZH = {"buy": "买", "sell": "卖"}
_SIDE_TO_ACTION = {v: k for k, v in _ACTION_ZH.items()}
# 聚宽导出的数字带单位与千分位："900股" / "90,103.50" / "-1,100股"
_NUM_JUNK = re.compile(r"[^0-9eE.\-+]")


def _to_num(s) -> pd.Series:
    """宽容地把一列转成数字：去掉千分位逗号与"股"这类单位"""
    if pd.api.types.is_numeric_dtype(s):
        return pd.to_numeric(s, errors="coerce")
    cleaned = (pd.Series(s).astype(str).str.replace(_NUM_JUNK, "", regex=True)
               .replace({"": None, "-": None, ".": None, "nan": None, "None": None}))
    return pd.to_numeric(cleaned, errors="coerce")


# ============================================================
# 1) 归一化：把两种引擎的成交表变成同一套字段
# ============================================================
def normalize_trades(trades, fill_time: str = "09:30") -> pd.DataFrame:
    """统一成交表 -> (date, code, action, shares, price, fee, pnl, amount)

    * `multi_engine`：列 action/size/price/fee/pnl，索引是 timestamp
    * `joinquant`   ：列 action/size/price/**fees**/pnl
    * 中文列（"交易类型"/"成交数量"…）也接受，便于把聚宽导出的文本回灌做对照
    """
    cols = ["date", "time", "code", "name", "action", "order_type",
            "shares", "price", "amount", "pnl", "fee"]
    if trades is None or len(trades) == 0:
        return pd.DataFrame(columns=cols)
    df = pd.DataFrame(trades).copy()
    if df.empty:
        return pd.DataFrame(columns=cols)

    # ⚠️ 列名归一必须在解析日期**之前**：聚宽导出的列名是中文（"日期"/"委托时间"），
    # 先按英文列名找日期会走 else 分支塞一个全 NaT 的 "date"，rename 再加一列同名列
    # -> `df["date"]` 变成 DataFrame，最后在构造结果表时炸 "Buffer has wrong
    # number of dimensions"。这个顺序不能调。
    ren = {"日期": "date", "委托时间": "time", "成交数量": "shares_raw",
           "成交价": "price", "成交额": "amount_raw",
           "平仓盈亏": "pnl", "手续费": "fee", "交易类型": "action_raw",
           "标的": "security", "下单类型": "order_type"}
    df = df.rename(columns={k: v for k, v in ren.items() if k in df.columns})

    if "timestamp" in df.columns:
        df["date"] = pd.to_datetime(df["timestamp"], errors="coerce")
    elif "date" in df.columns:
        df["date"] = pd.to_datetime(df["date"], errors="coerce")
    elif not isinstance(df.index, pd.RangeIndex):
        df["date"] = pd.to_datetime(df.index, errors="coerce")
    else:
        df["date"] = pd.NaT

    if "action" not in df.columns and "action_raw" in df.columns:
        df["action"] = df["action_raw"].map(_SIDE_TO_ACTION).fillna(df["action_raw"])
    if "action" not in df.columns and "side" in df.columns:
        # 聚宽导出的文本解析后常见的中文列（买/卖）
        df["action"] = df["side"].map(_SIDE_TO_ACTION).fillna(df["side"])
    if "shares_raw" in df.columns or "shares" not in df.columns:
        src = df["shares_raw"] if "shares_raw" in df.columns else df.get("size", 0.0)
        df["shares"] = _to_num(src)
    if "size" not in df.columns:
        df["size"] = df["shares"]
    if "fees" in df.columns and "fee" not in df.columns:
        df["fee"] = df["fees"]
    if "fee" not in df.columns:
        df["fee"] = 0.0
    if "pnl" not in df.columns:
        df["pnl"] = 0.0
    if "code" not in df.columns:
        df["code"] = ""
    if "security" in df.columns:
        # 聚宽导出的"银华日利(511880.XSHG)" -> 只留代码，名字另存
        sec = df["security"].astype(str)
        if "name" not in df.columns:
            df["name"] = sec.str.replace(r"\(.*\)", "", regex=True).str.strip()
        code = sec.str.extract(r"\(([0-9A-Za-z.]+)\)", expand=False)
        df["code"] = code.fillna(df["code"])
    if "name" not in df.columns:
        df["name"] = ""
    if "order_type" not in df.columns:
        df["order_type"] = "市价单"

    if "amount_raw" in df.columns:
        amount_src = _to_num(df["amount_raw"])
    elif "amount" in df.columns:
        amount_src = _to_num(df["amount"])
    else:
        amount_src = pd.Series(np.nan, index=df.index)

    out = pd.DataFrame({
        "date": df["date"],
        "time": df.get("time", pd.Series([fill_time] * len(df), index=df.index)),
        "code": df["code"].astype(str),
        "name": df["name"].fillna("").astype(str),
        "action": df["action"].astype(str),
        "order_type": df["order_type"].fillna("市价单").astype(str),
        "shares": _to_num(df["shares"]).fillna(0.0).astype(float),
        "price": _to_num(df["price"]),
        "amount": amount_src,
        "pnl": _to_num(df["pnl"]).fillna(0.0),
        "fee": _to_num(df["fee"]).fillna(0.0),
    })
    # 金额：买正卖负（聚宽口径），缺列时按 股数×价格 补
    sign = np.where(out["action"] == "sell", -1.0, 1.0)
    # 股数统一成**带符号**（聚宽导出就是 -1100股）。组合引擎的 size 是正数 +
    # action 区分方向，不统一的话下游（FIFO 配对、现金对账）会把卖出当买入。
    # 用 abs() 是为了对"本来就是负数的聚宽导出"幂等。
    out["shares"] = out["shares"].abs() * sign
    calc = out["shares"].abs() * out["price"]
    out["amount"] = out["amount"].fillna(calc * sign)
    out = out.dropna(subset=["date"]).sort_values("date", kind="stable")
    return out.reset_index(drop=True)


# ============================================================
# 2) 名称查询（可选依赖；拿不到就用代码）
# ============================================================
def name_map() -> Dict[str, str]:
    try:
        from database.loader import load_industry_map
        d = load_industry_map()
    except Exception:
        return {}
    if d is None or getattr(d, "empty", True) or "name" not in d.columns:
        return {}
    code = (d["code"] if "code" in d.columns
            else d["ts_code"].astype(str).str.split(".").str[0])
    return dict(zip(code.astype(str).str.zfill(6), d["name"].astype(str)))


def _label(code: str, name: str = "") -> str:
    name = name or ""
    return f"{name}({code})" if name else code


# ============================================================
# 3) 成交回合（FIFO 配对）——聚宽导出里没有，是我们多给的一层
# ============================================================
def round_trips(trades: pd.DataFrame) -> pd.DataFrame:
    """按 FIFO 把买卖配成回合，算净盈亏 / 持仓天数

    ⚠️ 只按**成交价**算，不含费用；净盈亏单列。加仓/减仓会被拆成多个回合，
    这是有意的：聚宽的「平仓盈亏」也是逐笔按均价算的，但要复核"这笔到底赚没赚"
    还是 FIFO 更直观。
    """
    rows: List[dict] = []
    for code, g in trades.groupby("code", sort=False):
        open_lots: List[list] = []          # [shares, price, date, fee_per_share]
        for _, r in g.sort_values("date", kind="stable").iterrows():
            sh = float(r["shares"])
            px = float(r["price"]) if np.isfinite(r["price"]) else np.nan
            if sh > 0:
                open_lots.append([sh, px, r["date"], float(r["fee"]) / sh if sh else 0.0])
                continue
            left = -sh
            while left > 1e-9 and open_lots:
                lot = open_lots[0]
                take = min(left, lot[0])
                gross = (px - lot[1]) * take
                fee = float(r["fee"]) * (take / left if left else 0) + lot[3] * take
                rows.append({
                    "code": code, "name": r.get("name", ""),
                    "开仓日": lot[2].date(), "平仓日": r["date"].date(),
                    "持仓天数": int((r["date"] - lot[2]).days),
                    "股数": take, "买入价": lot[1], "卖出价": px,
                    "毛利": gross, "费用": fee, "净利": gross - fee,
                    "收益率": (gross - fee) / (lot[1] * take) if lot[1] * take else np.nan,
                })
                lot[0] -= take
                left -= take
                if lot[0] <= 1e-9:
                    open_lots.pop(0)
            # 卖超（left 仍有剩余）说明成交表不自洽，交由 audit_trades 报
    df = pd.DataFrame(rows)
    if df.empty:
        return pd.DataFrame(columns=["code", "name", "开仓日", "平仓日", "持仓天数",
                                     "股数", "买入价", "卖出价", "毛利", "费用",
                                     "净利", "收益率"])
    return df


# ============================================================
# 4) 对账 / 自检
# ============================================================
def audit_trades(trades: pd.DataFrame, initial_cash: float = None,
                 lot_size: int = 100) -> Dict:
    """只用成交表就能查出的硬错误

    不通过就说明成交记录本身不可信，别拿它做归因。
    """
    out = {"n": int(len(trades)), "ok": True, "issues": [], "final_positions": {},
           "cash": None, "fees": float(trades["fee"].sum()) if len(trades) else 0.0}
    if trades is None or len(trades) == 0:
        out["ok"] = False
        out["issues"].append("成交表为空")
        return out

    cash = float(initial_cash) if initial_cash is not None else None
    pos: Dict[str, float] = {}
    neg_cash, neg_pos, odd_lot, bad_price = [], [], [], []
    for _, r in trades.iterrows():
        code, sh, px = r["code"], float(r["shares"]), r["price"]
        if not np.isfinite(px) or px <= 0:
            bad_price.append((r["date"], code))
            continue
        if abs(sh) % lot_size != 0:
            odd_lot.append((r["date"], code, sh))
        pos[code] = pos.get(code, 0.0) + sh
        if cash is not None:
            cash += (-sh) * float(px) - (+1 if sh > 0 else -1) * float(r["fee"])
            if cash < -1e-6:
                neg_cash.append((r["date"], code, cash))
        if pos[code] < -1e-6:
            neg_pos.append((r["date"], code, pos[code]))
    out["cash"] = cash
    out["final_positions"] = {k: v for k, v in pos.items() if abs(v) > 1e-9}
    for tag, bad, msg in (("现金为负", neg_cash, "买入超过可用现金，或漏记了卖出"),
                          ("持仓为负", neg_pos, "卖出超过持仓，成交表不自洽"),
                          ("非整手", odd_lot, "A 股必须是 100 股整数倍"),
                          ("成交价非法", bad_price, "价格为 0/NaN")):
        if bad:
            out["ok"] = False
            out["issues"].append(f"{tag} {len(bad)} 处：{bad[:3]}")
    return out


def reconcile_equity(equity: pd.Series, trades: pd.DataFrame) -> Dict:
    """成交表与权益曲线的一致性检查（能查的几条）

    * 成交日必须都在权益曲线的交易日里（否则净值曲线漏了调仓日）
    * 期末权益 − 期初权益 与 Σ(平仓盈亏) − Σ(费用) + Δ持仓浮盈 的关系无法在
      没有逐日行情时闭合，因此这里**不做**假对账，只报可验证项与残差口径。
    """
    e = pd.Series(equity).astype(float).dropna()
    out = {"n_days": int(len(e)), "trades_off_calendar": [], "trades_on_holiday": []}
    if len(e) == 0 or trades is None or len(trades) == 0:
        return out
    idx = set(pd.DatetimeIndex(e.index).normalize())
    for d in pd.DatetimeIndex(trades["date"]).normalize().unique():
        if pd.Timestamp(d) not in idx:
            out["trades_off_calendar"].append(str(pd.Timestamp(d).date()))
    out["total_return_from_equity"] = float(e.iloc[-1] / e.iloc[0] - 1)
    out["sum_realized_pnl"] = float(trades["pnl"].sum())
    out["sum_fees"] = float(trades["fee"].sum())
    return out


# ============================================================
# 5) 文本导出（聚宽格式）
# ============================================================
def _fmt_money(x, nd=2) -> str:
    if x is None or (isinstance(x, float) and not np.isfinite(x)):
        return ""
    return f"{x:,.{nd}f}"


def trade_records_text(trades: pd.DataFrame, names: Dict[str, str] = None) -> str:
    """聚宽「交易记录」同构文本（制表符分隔，含表头）"""
    names = names or {}
    lines = ["\t".join(RECORD_COLS)]
    for _, r in trades.iterrows():
        sh = float(r["shares"])
        lines.append("\t".join([
            pd.Timestamp(r["date"]).strftime("%Y-%m-%d"),
            str(r.get("time", "") or ""),
            _label(r["code"], (r.get("name") or names.get(r["code"], ""))),
            _ACTION_ZH.get(r["action"], r["action"]),
            str(r.get("order_type", "市价单")),
            f"{int(sh)}股",
            _fmt_money(r["price"], 3),
            _fmt_money(r["amount"]),
            _fmt_money(r["pnl"]),
            _fmt_money(r["fee"]),
        ]))
    return "\n".join(lines) + "\n"


def perf_analysis_text(equity: pd.Series, benchmark: pd.Series = None,
                       trades: pd.DataFrame = None, label: str = "") -> str:
    """聚宽「收益分析」同构文本（每行：指标名 / 值）"""
    from .jq_report import trade_stats
    e = pd.Series(equity).astype(float).dropna()
    r = to_returns(e)
    summ = performance_summary(e, benchmark)
    ts = trade_stats(trades)

    def pct(x, sign=False):
        return "-" if x is None or not np.isfinite(x) else (
            f"{x:+.2%}" if sign else f"{x:.2%}")

    def num(x, nd=3):
        return "-" if x is None or not np.isfinite(x) else f"{x:.{nd}f}"

    pairs: List[tuple] = []
    if label:
        pairs.append(("策略名称", label))
    pairs += [
        ("策略收益", pct(summ["total_return"], True)),
        ("策略年化收益", pct(summ["annual_return"], True)),
        ("策略波动率", num(summ["annual_volatility"])),
        ("夏普比率", num(summ["sharpe_ratio"])),
        ("索提诺比率", num(summ["sortino_ratio"])),
        ("最大回撤", pct(summ["max_drawdown"])),
        ("最大回撤区间", f"{summ.get('max_drawdown_peak','')},{summ.get('max_drawdown_trough','')}"),
    ]
    if benchmark is not None and len(pd.Series(benchmark).dropna()):
        b = pd.Series(benchmark).astype(float)
        br = to_returns(b)
        ex_nav = ((1 + r).cumprod() / (1 + br.reindex(r.index).fillna(0)).cumprod())
        ex_dd = drawdown_series(ex_nav)
        # 超额收益最大回撤：超额净值自己的回撤
        ex_dd_min = float(ex_dd.min()) if len(ex_dd) else np.nan
        excess_daily = (r - br.reindex(r.index).fillna(0.0))
        pairs += [
            ("基准收益", pct(summ.get("bench_total_return"), True)),
            ("超额收益", pct(summ.get("excess_return"), True)),
            ("阿尔法", num(summ.get("alpha_annual"))),
            ("贝塔", num(summ.get("beta"))),
            ("信息比率", num(summ.get("information_ratio"))),
            ("跟踪误差", num(summ.get("tracking_error"))),
            ("日均超额收益", pct(float(excess_daily.mean()), True)),
            ("超额收益最大回撤", pct(ex_dd_min)),
            ("超额收益夏普比率",
             num(float(excess_daily.mean() / excess_daily.std() * np.sqrt(252))
                 if excess_daily.std() > 0 else np.nan)),
            ("基准波动率", num(summ.get("bench_annual_volatility"))),
        ]
    pairs += [
        ("日胜率", f"{float((r > 0).mean()):.3f}" if len(r) else "-"),
        ("盈利次数", f"{ts['盈利次数']}"),
        ("亏损次数", f"{ts['亏损次数']}"),
        ("胜率", f"{ts['胜率']:.3f}"),
        ("盈亏比", num(ts["盈亏比"])),
        ("交易笔数", f"{ts['交易笔数']}"),
        ("交易日数", f"{summ['n_days']}"),
        ("卡玛比率", num(summ["calmar_ratio"])),
        ("最大回撤持续(天)", f"{summ.get('max_drawdown_duration', 0)}"),
        ("最大回撤收复日", str(summ.get("max_drawdown_recover", ""))),
    ]
    lines: List[str] = []
    for k, v in pairs:
        lines.append(str(k))
        lines.append(str(v))
    return "\n".join(lines) + "\n"


def rejection_text(rejects) -> str:
    """委托失败明细（聚宽把这些混在成交记录里，我们单独出一份）"""
    lines = ["日期\t时间\t标的\t方向\t数量\t价格\t原因"]
    if rejects is None or len(rejects) == 0:
        return "\n".join(lines) + "\n"
    df = pd.DataFrame(rejects)
    if df.empty:
        return "\n".join(lines) + "\n"
    for col in ("date", "code", "side", "qty", "px", "reason"):
        if col not in df.columns:
            df[col] = ""
    for _, r in df.iterrows():
        d = r["date"]
        d = pd.Timestamp(d).strftime("%Y-%m-%d") if d is not None and str(d) else ""
        lines.append("\t".join([d, "", str(r["code"]), str(r["side"]),
                                "" if r["qty"] in ("", None) else f"{r['qty']}",
                                _fmt_money(r["px"]) if isinstance(r["px"], (int, float)) else "",
                                str(r["reason"])]))
    return "\n".join(lines) + "\n"


# ============================================================
# 6) 一键导出
# ============================================================
def export_trade_log(trades, equity: pd.Series, benchmark: pd.Series = None,
                     outdir: str = "results", tag: str = "strategy",
                     rejects=None, initial_cash: float = None,
                     fill_time: str = "09:30", label: str = "") -> Dict:
    """写出 交易记录 / 委托失败明细 / 成交回合 / 收益分析 + 机器可读 CSV

    返回 {"files": {...}, "audit": {...}, "reconcile": {...}}
    """
    import os
    os.makedirs(outdir, exist_ok=True)
    names = name_map()
    df = normalize_trades(trades, fill_time=fill_time)
    if len(df):
        df["name"] = [n or names.get(c, "") for n, c in zip(df["name"], df["code"])]
    rt = round_trips(df)
    audit = audit_trades(df, initial_cash=initial_cash)
    rec = reconcile_equity(equity, df)

    files = {
        "交易记录": os.path.join(outdir, f"交易记录{tag}.txt"),
        "委托失败明细": os.path.join(outdir, f"委托失败明细{tag}.txt"),
        "收益分析": os.path.join(outdir, f"收益分析{tag}.txt"),
        "成交明细csv": os.path.join(outdir, f"成交明细{tag}.csv"),
        "成交回合csv": os.path.join(outdir, f"成交回合{tag}.csv"),
    }
    with open(files["交易记录"], "w", encoding="utf-8") as fh:
        fh.write(trade_records_text(df, names))
    with open(files["委托失败明细"], "w", encoding="utf-8") as fh:
        fh.write(rejection_text(rejects))
    with open(files["收益分析"], "w", encoding="utf-8") as fh:
        fh.write(perf_analysis_text(equity, benchmark, df, label=label))
    df.to_csv(files["成交明细csv"], index=False, encoding="utf-8-sig")
    rt.to_csv(files["成交回合csv"], index=False, encoding="utf-8-sig")
    return {"files": files, "audit": audit, "reconcile": rec,
            "round_trips": rt, "trades": df}


def audit_line(audit: Dict) -> str:
    """一行式对账结论，直接打进回测报告"""
    if not audit.get("n"):
        return "  成交记录对账: 无成交"
    state = "✓ 通过" if audit["ok"] else "✗ 不通过"
    pos = audit.get("final_positions") or {}
    return (f"  成交记录对账: {state}  成交 {audit['n']} 笔  "
            f"费用合计 {audit['fees']:,.2f}  期末持仓 {len(pos)} 只"
            + ("" if audit["ok"] else "\n      " + "\n      ".join(audit["issues"])))
