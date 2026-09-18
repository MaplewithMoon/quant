# -*- coding: utf-8 -*-
"""聚宽 API 的本地实现 + 事件循环

策略代码原样保留（`from jqdata import *` 换成 `from joinquant import *` 即可），
本模块提供它需要的全部 API。**撮合与账本复用项目自己的
`Portfolio` / `SimulatedBroker` / `MarketRules`**，不另起一套，
这样 JQ 移植策略和本项目其它策略走的是同一套执行层。

【事件循环】逐交易日、按调度时间顺序执行 run_daily / run_weekly 注册的函数，
与聚宽一致。`context.current_dt` / `context.previous_date` 在每个任务前更新。

【成交价近似】
    聚宽是分钟级撮合（10:00 的单在 10:00 成交）。我们只有日线，所以：
        --fill auto（默认）：上午的任务（<=11:30）用**当日开盘价**，
                             下午的任务（>11:30）用**当日收盘价**
        --fill open / close ：全部用开盘/收盘
    这是移植里最主要的近似，报告里会单列。
"""
import datetime
import re
from dataclasses import dataclass
from datetime import date as _date, timedelta
from typing import Callable, Dict, List, Optional

import numpy as np
import pandas as pd

from execution.broker import SimulatedBroker
from execution.market_rules import MarketRules
from execution.order import OrderSide
from portfolio.portfolio import Portfolio

from .data import JQData, LegacyFrame

# ============================================================
# 全局状态（聚宽策略是无参自由函数，只能靠模块级单例）
# ============================================================
_ENGINE: "JQEngine" = None


class GlobalNamespace:
    """聚宽的 g 对象：任意属性"""
    def __repr__(self):
        return f"g({self.__dict__})"


# ============================================================
# 配置对象（与聚宽同名同参）
# ============================================================
@dataclass
class FixedSlippage:
    """固定滑点。聚宽口径是**双边**，单边 = x/2"""
    value: float = 0.0

    @property
    def per_side(self) -> float:
        return self.value / 2.0


@dataclass
class PriceRelatedSlippage:
    """按价格比例的滑点。聚宽口径同样是双边，单边 = x/2"""
    value: float = 0.0

    @property
    def per_side(self) -> float:
        return self.value / 2.0


@dataclass
class OrderCost:
    open_tax: float = 0.0
    close_tax: float = 0.0
    open_commission: float = 0.0
    close_commission: float = 0.0
    close_today_commission: float = 0.0
    min_commission: float = 5.0


class OrderStatus:
    """聚宽 OrderStatus 的最小子集"""
    held = "held"
    filled = "filled"
    rejected = "rejected"


@dataclass
class Order:
    security: str
    amount: float = 0.0
    filled: float = 0.0
    price: float = 0.0
    status: str = OrderStatus.rejected
    fees: float = 0.0
    reason: str = ""


class _Log:
    """聚宽 log 的最小实现"""
    LEVELS = {"debug": 10, "info": 20, "warning": 30, "error": 40}

    def __init__(self):
        self.level = 20
        self.lines: List[str] = []
        self.verbose = False

    def set_level(self, kind, level):
        pass

    def _emit(self, lvl, msg):
        if self.LEVELS[lvl] >= self.level:
            self.lines.append(f"[{lvl.upper()}] {msg}")
            if self.verbose:
                print(f"  {lvl.upper()}: {msg}")

    def debug(self, m): self._emit("debug", m)
    def info(self, m): self._emit("info", m)
    def warning(self, m): self._emit("warning", m)
    def error(self, m): self._emit("error", m)


log = _Log()


def print_(*a, **k):          # 策略里直接用了 print()
    if log.verbose:
        print(*a, **k)


# ============================================================
# query DSL（只实现这两个策略用到的部分）
# ============================================================
class Cond:
    def __init__(self, col, op, val):
        self.col, self.op, self.val = col, op, val


class Col:
    def __init__(self, table, name):
        self.table, self.name = table, name

    def _c(self, op, v):
        return Cond(self, op, v)

    def __gt__(self, v): return self._c(">", v)
    def __lt__(self, v): return self._c("<", v)
    def __ge__(self, v): return self._c(">=", v)
    def __le__(self, v): return self._c("<=", v)
    def __eq__(self, v): return self._c("==", v)
    def in_(self, v): return self._c("in", v)
    def between(self, a, b): return self._c("between", (a, b))
    def asc(self): return (self, "asc")
    def desc(self): return (self, "desc")

    def __repr__(self):
        return f"{self.table}.{self.name}"


class _Table:
    def __init__(self, name, cols):
        self._name = name
        for c in cols:
            setattr(self, c, Col(name, c))

    def __repr__(self):
        return f"<table {self._name}>"


valuation = _Table("valuation", ["code", "market_cap", "circulating_market_cap",
                                 "pe_ratio", "pb_ratio", "turnover_ratio"])
income = _Table("income", ["code", "np_parent_company_owners", "net_profit",
                           "operating_revenue", "total_operating_revenue"])
balance = _Table("balance", ["code", "total_assets", "total_liability"])
cash_flow = _Table("cash_flow", ["code", "net_operate_cash_flow"])


class Query:
    def __init__(self, cols):
        self.cols = list(cols)
        self.conds: List[Cond] = []
        self.order = None
        self.limit_n = None

    def filter(self, *conds):
        self.conds.extend([c for c in conds if isinstance(c, Cond)])
        return self

    def order_by(self, *keys):
        self.order = keys[0] if keys else None
        return self

    def limit(self, n):
        self.limit_n = int(n)
        return self


def query(*cols):
    return Query(cols)


# ============================================================
# 上下文对象
# ============================================================
class PositionView:
    """聚宽 position 对象"""
    def __init__(self, code, pos, price):
        self.security = code
        self.total_amount = float(pos.size)
        self.closeable_amount = float(getattr(pos, "available", pos.size))
        self.avg_cost = float(pos.avg_cost)
        self.price = float(price)
        self.value = self.total_amount * self.price

    def __repr__(self):
        return f"<Position {self.security} {self.total_amount:.0f}股>"


class _Positions(dict):
    """聚宽的 `context.portfolio.positions` 是"访问即返回对象"的字典：

    对**没有持仓**的股票取 `positions['xxx']` 不会 KeyError，而是返回一个
    total_amount=0 的 Position 对象。策略里就是这么用的：
        if context.portfolio.positions[stock].total_amount == 0:
    用普通 dict 会直接 KeyError（第一轮 506 笔"拒绝委托"全是这个原因）。
    """
    _engine = None

    def __missing__(self, code):
        eng = self._engine
        d = eng.current_date if eng is not None else None
        px = (eng.price_of(code, d) if eng is not None else None) or 0.0
        return _ZeroPosition(code, px)


class _ZeroPosition:
    def __init__(self, code, price=0.0):
        self.security = code
        self.total_amount = 0.0
        self.closeable_amount = 0.0
        self.avg_cost = 0.0
        self.price = float(price or 0.0)
        self.value = 0.0


class PortfolioView:
    def __init__(self, engine):
        self._engine = engine

    @property
    def positions(self) -> Dict[str, PositionView]:
        pf = self._engine.pf
        d = self._engine.current_date
        out = _Positions()
        out._engine = self._engine
        for c, p in pf.positions.items():
            if p.size <= 0:
                continue
            px = self._engine.price_of(c, d)
            out[c] = PositionView(c, p, px if px else p.current_price)
        return out

    @property
    def cash(self) -> float:
        return float(self._engine.pf.cash)

    @property
    def available_cash(self) -> float:
        return float(self._engine.pf.cash)

    @property
    def total_value(self) -> float:
        return float(self._engine.pf.total_value)


class Context:
    def __init__(self, engine):
        self._engine = engine
        self.g = engine.g

    @property
    def portfolio(self) -> PortfolioView:
        return PortfolioView(self._engine)

    @property
    def current_dt(self) -> datetime.datetime:
        return self._engine.current_dt

    @property
    def previous_date(self) -> _date:
        return self._engine.previous_date

    def __repr__(self):
        return f"<Context {self._engine.current_dt}>"


# ============================================================
# 调度
# ============================================================
@dataclass
class Task:
    func: Callable
    time: str
    weekly: Optional[int] = None       # None=每日；1~5=每周第 N 个交易日
    reference_security: str = ""

    @property
    def minutes(self) -> int:
        h, m = self.time.split(":")[:2]
        return int(h) * 60 + int(m)


def _norm_time(t) -> str:
    if t in ("open", None, ""):
        return "09:30"
    if t == "close":
        return "15:00"
    if t == "before_open":
        return "09:00"
    if t == "after_close":
        return "15:30"
    return str(t)


def run_daily(func, time="open", reference_security=None):
    _ENGINE.tasks.append(Task(func, _norm_time(time), None, reference_security or ""))


def run_weekly(func, weekday, time="open", reference_security=None):
    _ENGINE.tasks.append(Task(func, _norm_time(time), int(weekday),
                              reference_security or ""))


# ============================================================
# 设置
# ============================================================
def set_benchmark(code):
    _ENGINE.benchmark = code


def set_option(key, value):
    _ENGINE.options[key] = value


def set_slippage(obj):
    _ENGINE.slippage_obj = obj
    # 标记"策略显式设过滑点"：CLI 的默认滑点/冲击模型不应覆盖策略的显式选择
    _ENGINE.slippage_set_by_strategy = True


def set_order_cost(obj, type="stock"):
    _ENGINE.order_cost = obj


def set_universe(codes):
    pass


# ============================================================
# 数据 API
# ============================================================
def _d():
    return _ENGINE.data


def get_price(security, end_date=None, frequency="daily", fields=None,
              count=1, panel=False, fill_paused=False, skip_paused=False,
              fq=None, start_date=None):
    """行情查询

    支持单只（返回 DataFrame）与多只（panel=False 返回长表，含 code 列）。
    `frequency='1m'` 用当日开盘价近似（我们只有日线）。
    """
    eng = _ENGINE
    if end_date is None:
        end_date = eng.current_dt
    ed = pd.Timestamp(end_date)
    single = isinstance(security, str)
    codes = [security] if single else list(security)
    fields = [fields] if isinstance(fields, str) else (fields or ["close"])

    if frequency in ("1m", "minute") or frequency.startswith("1"):
        # 分钟线没有：用当日"盘中快照"当当前价（`--fill auto` 口径：上午看开盘、
        # 下午看收盘，见 JQEngine._fill_price）。
        #
        # ⚠️⚠️ `current_bar()` 的返回是 **index=股票代码, columns=open/high/low/close**
        # （`pd.DataFrame({字段: Series(按代码)})` 的自然结果）。原先这里按
        # "index=字段, columns=代码" 去取，于是 `'close' in bar.index` 恒 False
        # -> `src` 全部退回 "open"，而 `c in bar.columns` 也恒 False -> **整列 NaN**。
        # 症状：v1 的 `check_limit_up()` 里 `close < high_limit` 恒为 False
        # （NaN 比较），"涨停打开就卖、第二天回补"这条路径**完全死掉** ——
        # 本地 v1 只有 29 个个股成交日，聚宽有 89 天，差的就是这条路径。
        #
        # 另外 `high_limit` / `low_limit` 聚宽是真实涨跌停价，**不能拿开盘价顶替**：
        # 直接取 cleaned/limit_price 面板（`get_current_data()` 就是这么做的）。
        bar = _d().current_bar(pd.Timestamp(eng.current_date))
        if bar.empty:
            return LegacyFrame(columns=fields)
        dd = pd.Timestamp(eng.current_date)
        lu, ld = _d().limit_prices(dd)
        pre = getattr(_d(), "_pre_close", None)
        pre_row = pre.loc[dd] if pre is not None and dd in pre.index else None
        recs = []
        for c in codes:
            rec = {"code": c}
            for f in fields:
                if c in bar.index and f in bar.columns:
                    v = bar.at[c, f]
                elif f == "high_limit":
                    v = lu.get(c) if lu is not None and c in lu.index else np.nan
                elif f == "low_limit":
                    v = ld.get(c) if ld is not None and c in ld.index else np.nan
                elif f == "pre_close":
                    v = pre_row.get(c) if pre_row is not None else np.nan
                elif f in ("last_price", "price"):
                    v = bar.at[c, "close"] if c in bar.index else np.nan
                else:
                    # 既不是盘中快照字段也没有真实来源 -> 退回开盘价（**近似**，见文档）
                    v = bar.at[c, "open"] if c in bar.index else np.nan
                rec[f] = float(v) if pd.notna(v) else np.nan
            recs.append(rec)
        long = pd.DataFrame(recs)
        if single:
            out = long.iloc[[0]][fields].copy()
            out.index = pd.DatetimeIndex([eng.current_dt])
            return LegacyFrame(out)
        out = long[fields + ["code"]].copy()
        out.index = pd.DatetimeIndex([eng.current_dt] * len(out))
        return LegacyFrame(out)

    data = {}
    nonstock = [c for c in codes if _d().is_non_stock(c)]
    for f in fields:
        if f == "close" and nonstock:
            # 指数 / ETF：只有 close 可用（重建指数或合成的货币 ETF）
            frames = {}
            stock_codes = [c for c in codes if c not in nonstock]
            if stock_codes:
                frames.update({c: _d().history(f, [c], ed, count if start_date is None
                                               else 10 ** 6)[c]
                               for c in stock_codes})
            for c in nonstock:
                frames[c] = _d().any_close(c, ed, count if start_date is None else 10 ** 6)
            df = pd.DataFrame(frames)
            if start_date is not None:
                df = df[df.index >= pd.Timestamp(start_date)]
        else:
            df = _d().history(f, codes, ed, count if start_date is None else 10 ** 6)
            if start_date is not None:
                df = df[df.index >= pd.Timestamp(start_date)]
        data[f] = df
    if not data:
        return LegacyFrame()
    # 统一补列：请求的代码若在某个字段里不存在（如 ETF 没有 high_limit），
    # 补成 NaN 列而不是抛 KeyError —— 聚宽返回的是带 NaN 的完整宽表
    for f in fields:
        data[f] = data[f].reindex(columns=codes)
    if single:
        c = codes[0]
        df0 = data[fields[0]]
        if c not in df0.columns:
            return LegacyFrame(columns=fields)
        out = pd.DataFrame({f: data[f][c] if c in data[f].columns else np.nan
                            for f in fields})
        return LegacyFrame(out)
    # 多只：panel=False 的长表（index=日期, 列=fields + code）
    frames = []
    for c in codes:
        if c not in data[fields[0]].columns:
            continue
        part = pd.DataFrame({f: data[f][c] for f in fields})
        part["code"] = c
        frames.append(part)
    return LegacyFrame(pd.concat(frames)) if frames else LegacyFrame()


def history(count, unit="1d", field="close", security_list=None, df=True,
            skip_paused=False, fq="pre"):
    """历史行情。`unit='1d'` 取到**上一交易日**为止（当日日线未完成）；
    `unit='1m'` 用当日开盘价近似当前价。"""
    eng = _ENGINE
    codes = list(security_list) if security_list else list(_d().codes)
    if unit in ("1m", "minute"):
        row = _d().current_bar(pd.Timestamp(eng.current_date))
        if row.empty:
            return LegacyFrame(columns=codes)
        col = row["open"] if field == "close" else row.get(field, row["open"])
        return LegacyFrame(pd.DataFrame([col.reindex(codes)], index=[eng.current_dt]))
    return _d().history(field, codes, eng.previous_date, int(count))


class _CurrentData:
    """get_current_data() 的返回：`data[code].xxx`"""
    def __init__(self, engine):
        self._eng = engine
        d = pd.Timestamp(engine.current_date)
        data = engine.data
        self._open = data._open.loc[d] if d in data._open.index else pd.Series(dtype=float)
        self._paused = data.paused(d)
        lu, ld = data.limit_prices(d)
        self._lu, self._ld = lu, ld
        self._missing = 0

    def __getitem__(self, code):
        eng = self._eng
        nam = eng.data.name_of(code)
        if nam == "" and code not in eng.data._close.columns:
            self._missing += 1
        lu = float(self._lu.get(code)) if self._lu is not None and code in self._lu.index \
            and pd.notna(self._lu.get(code)) else np.nan
        ld = float(self._ld.get(code)) if self._ld is not None and code in self._ld.index \
            and pd.notna(self._ld.get(code)) else np.nan
        op = float(self._open.get(code)) if code in self._open.index and \
            pd.notna(self._open.get(code)) else np.nan
        paused = bool(self._paused.get(code, True)) if code in self._paused.index else True
        return _Bar(name=nam or code, paused=paused, is_st=eng.data.is_st(code),
                    high_limit=lu, low_limit=ld, day_open=op, last_price=op)


@dataclass
class _Bar:
    name: str = ""
    paused: bool = True
    is_st: bool = False
    high_limit: float = np.nan
    low_limit: float = np.nan
    day_open: float = np.nan
    last_price: float = np.nan


def get_current_data():
    return _CurrentData(_ENGINE)


def get_index_stocks(index_symbol, date=None):
    return _d().index_stocks(index_symbol, date or _ENGINE.current_date)


class SecurityInfo:
    def __init__(self, code, data):
        self.code = code
        self.display_name = data.name_of(code)
        ld = data.list_date_of(code)
        self.start_date = ld.date() if pd.notna(ld) else _date(1990, 1, 1)
        self.end_date = None
        self.type = "stock"


def get_security_info(code):
    return SecurityInfo(code, _d())


def get_trade_days(start_date=None, end_date=None, count=None):
    """交易日列表

    ⚠️ 返回 **datetime.date 列表**，不是 DatetimeIndex。
    策略里写的是 `g.trading_day = get_trade_days(...)[0]` 然后
    `today == g.trading_day`（today 是 `context.current_dt.date()`，即 datetime.date）。
    pandas 的 `Timestamp == datetime.date` **恒为 False**，所以返回 Timestamp 会让
    策略的"到调仓日了吗"永远不成立 —— v2 因此整段回测一次都没调仓、全程持 ETF。
    """
    d = _d().trade_days(start=start_date, end=end_date, count=count)
    return [x.date() if hasattr(x, "date") else x for x in pd.DatetimeIndex(d)]


def get_all_securities(types=None, date=None):
    d = _d()
    codes = [c for c in d.codes
             if pd.isna(d.list_date_of(c)) or d.list_date_of(c) <= pd.Timestamp(date or d.end)]
    return pd.DataFrame({"display_name": [d.name_of(c) for c in codes]},
                        index=pd.Index(codes, name="code"))


def get_industries(name="sw_l1", date=None):
    d = _d()
    ind = sorted(set(d._industry.reindex(d.codes).dropna()))
    return pd.DataFrame({"name": ind}, index=pd.Index(ind, name="code"))


def get_industry_stocks(industry_code, date=None):
    d = _d()
    return [c for c in d.codes if d._industry.get(c) == industry_code]


# ============================================================
# 财务查询
# ============================================================
def get_fundamentals(q, date=None):
    """执行 query DSL。对齐口径与聚宽一致：取**查询日已公告**的最新一期财报"""
    eng = _ENGINE
    d = pd.Timestamp(date or eng.previous_date)
    universe = set(eng.data.codes)
    for c in q.conds:
        if c.col.name == "code" and c.op == "in":
            universe &= set(c.val)
    codes = sorted(universe)
    out = pd.DataFrame({"code": codes})

    # ⚠️ 需要取数的列 = SELECT 的列 **并上** WHERE 里用到的列。
    # 聚宽允许对没写进 query() 的列做过滤（如 query(valuation.code) 里过滤 income.*），
    # 只按 q.cols 取数会让那些过滤条件被静默跳过 —— 实测会把营收不达标的股票放进来。
    want = {c.name for c in q.cols if isinstance(c, Col)}
    want |= {c.col.name for c in q.conds}
    if "market_cap" in want:
        mc = eng.data.market_cap_yi(d).reindex(codes)
        out["market_cap"] = mc.values
    if want & {"np_parent_company_owners", "net_profit", "operating_revenue"}:
        fin = eng.data.financials_asof(d).reindex(codes)
        if "np_parent_company_owners" in want:
            out["np_parent_company_owners"] = fin.get(
                "n_income_attr_p", pd.Series(index=codes, dtype=float)).values
        if "net_profit" in want:
            out["net_profit"] = fin.get(
                "n_income", pd.Series(index=codes, dtype=float)).values
        if "operating_revenue" in want:
            out["operating_revenue"] = fin.get(
                "revenue", pd.Series(index=codes, dtype=float)).values

    for c in q.conds:
        name, op, val = c.col.name, c.op, c.val
        if name not in out.columns:
            continue
        s = pd.to_numeric(out[name], errors="coerce")
        with np.errstate(invalid="ignore"):
            if op == ">":
                mask = s > val
            elif op == "<":
                mask = s < val
            elif op == ">=":
                mask = s >= val
            elif op == "<=":
                mask = s <= val
            elif op == "==":
                mask = s == val
            elif op == "between":
                mask = (s >= val[0]) & (s <= val[1])
            else:
                continue
        out = out[mask.fillna(False)]
    if q.order is not None:
        col, how = q.order
        if col.name in out.columns:
            out = out.sort_values(col.name, ascending=(how == "asc"))
    if q.limit_n:
        out = out.head(q.limit_n)
    # 去重：query(valuation.code, ...) 里已经含 code，再拼一次会出现重复列
    keep = list(dict.fromkeys(
        ["code"] + [c.name for c in q.cols
                    if isinstance(c, Col) and c.name in out.columns]))
    return out[keep].reset_index(drop=True)


# ============================================================
# 交易 API
# ============================================================
def order_target_value(security, value):
    return _ENGINE.order_target_value(security, value)


def order_value(security, value):
    eng = _ENGINE
    cur = eng.pf.position_size(security) * (eng.price_of(security) or 0.0)
    return eng.order_target_value(security, cur + value)


def order(security, amount):
    eng = _ENGINE
    cur = eng.pf.position_size(security)
    return eng.order_target_shares(security, cur + amount)


def order_target(security, amount):
    return _ENGINE.order_target_shares(security, amount)


def order_target_percent(security, pct):
    eng = _ENGINE
    return eng.order_target_value(security, eng.pf.total_value * pct)


# ============================================================
# 引擎
# ============================================================
class JQEngine:
    def __init__(self, data: JQData, initial_cash: float = 1_000_000,
                 fill: str = "auto", rules: bool = True,
                 verbose: bool = False, volume_limit: bool = False,
                 slippage: float = 0.001, impact_model: str = "none",
                 impact_k: float = 0.1):
        global _ENGINE
        self.data = data
        self.initial_cash = float(initial_cash)
        self.fill = fill
        self.tasks: List[Task] = []
        self.g = GlobalNamespace()
        self.options: Dict = {}
        self.benchmark = "000300.XSHG"
        # ⚠️ 原先这里是 `FixedSlippage(0.0)` —— 两个 Clone 策略的回测因此
        # **零滑点、零冲击**，与聚宽对不上不只差在成交时点。现在默认 1bp，
        # 并且可以换成平方根冲击模型；策略若显式调用 `set_slippage()`，
        # 以策略为准（聚宽语义）。
        self._default_slippage = float(slippage)
        self._impact_model = impact_model
        self._impact_k = float(impact_k)
        self.slippage_set_by_strategy = False
        self.slippage_obj = FixedSlippage(self._default_slippage)
        self.order_cost = OrderCost(open_commission=2.5e-4, close_commission=2.5e-4,
                                    close_tax=1e-3, min_commission=5.0)
        self.pf = Portfolio(self.initial_cash)
        self.rules = MarketRules() if rules else MarketRules(enabled=False)
        self.log = log
        log.verbose = verbose
        self.verbose = verbose
        self.volume_limit = volume_limit
        self.rejections: Dict[str, int] = {}
        # 逐笔拒单明细。聚宽的「交易记录」导出里**含未成交的委托**
        # （成交数量 0、成交价为空），只统计次数的话对照实验里就没法回答
        # "这一笔是没买进还是根本没下单"。字段与 multi_engine 的 reject_log 对齐。
        self.reject_log: List[dict] = []
        self.trades: List[dict] = []
        self.current_dt = datetime.datetime.combine(data.dates[0].date(),
                                                    datetime.time(9, 30))
        self.previous_date = data.dates[0].date()
        self.current_date = data.dates[0]
        self._rebuild_broker()
        _ENGINE = self

    # ---------- 费率/滑点 ----------
    def _rebuild_broker(self):
        from execution.impact import build_model
        common = dict(commission=float(self.order_cost.open_commission),
                      min_commission=float(self.order_cost.min_commission),
                      stamp_duty=float(self.order_cost.close_tax),
                      transfer_fee=0.0, lot_size=100)
        if self._impact_model != "none" and not self.slippage_set_by_strategy:
            # 冲击模型（需成交量）：冲击 ∝ 下单量/成交量，会随资金规模放大
            self.broker = SimulatedBroker(
                slippage_model=build_model(self._impact_model,
                                           rate=self._default_slippage,
                                           k=self._impact_k), **common)
        else:
            slip = float(getattr(self.slippage_obj, "per_side", 0.0))
            self.broker = SimulatedBroker(slippage=slip, **common)
        # 聚宽 close_commission 与 open_commission 可以不同，这里分别记账
        self._close_comm = float(self.order_cost.close_commission)

    def fees_of(self, turnover: float, side: OrderSide) -> float:
        comm = turnover * (self.order_cost.open_commission if side == OrderSide.BUY
                           else self._close_comm)
        comm = max(comm, self.order_cost.min_commission) if turnover > 0 else 0.0
        tax = turnover * self.order_cost.close_tax if side == OrderSide.SELL else 0.0
        return comm + tax

    # ---------- 价格 ----------
    def price_of(self, code, d) -> Optional[float]:
        # 指数 / ETF 不在股票面板里，走重建指数或合成的货币 ETF 序列
        if self.data.is_non_stock(code):
            s = self.data.any_close(code, pd.Timestamp(d), 1)
            return float(s.iloc[-1]) if len(s) and np.isfinite(s.iloc[-1]) else None
        try:
            v = self.data._close.at[pd.Timestamp(d), code]
        except (KeyError, TypeError):
            return None
        return float(v) if v is not None and np.isfinite(v) and v > 0 else None

    def _fill_price(self, code, d, minutes: int) -> Optional[float]:
        """成交价：auto 模式下上午用开盘、下午用收盘"""
        d = pd.Timestamp(d)
        if self.data.is_non_stock(code):
            # 指数/ETF 只有一条合成/重建序列，没有开收盘之分
            return self.price_of(code, d)
        if self.fill == "open":
            use_open = True
        elif self.fill == "close":
            use_open = False
        else:
            use_open = minutes <= 11 * 60 + 30
        src = self.data._open if use_open else self.data._close
        try:
            v = src.at[d, code]
        except (KeyError, TypeError):
            return None
        return float(v) if v is not None and np.isfinite(v) and v > 0 else None

    def _bar(self, code, d) -> dict:
        if self.data.is_non_stock(code):
            # 指数/ETF 没有涨跌停与停牌的概念
            return {"low": None, "high": None, "limit_up": None,
                    "limit_down": None, "suspended": False}
        bar = {}
        for k, df in (("low", self.data._low), ("high", self.data._high)):
            try:
                v = df.at[pd.Timestamp(d), code]
                bar[k] = None if v is None or not np.isfinite(v) else float(v)
            except (KeyError, TypeError):
                bar[k] = None
        lu, ld = self.data.limit_prices(pd.Timestamp(d))
        bar["limit_up"] = float(lu.get(code)) if lu is not None and code in lu.index \
            and pd.notna(lu.get(code)) else None
        bar["limit_down"] = float(ld.get(code)) if ld is not None and code in ld.index \
            and pd.notna(ld.get(code)) else None
        try:
            bar["suspended"] = bool(self.data.paused(pd.Timestamp(d)).get(code, True))
        except Exception:
            bar["suspended"] = False
        return bar

    def _volume(self, code, d):
        vol = self.data.panel.get("volume")
        if vol is None:
            return None
        try:
            v = vol.at[pd.Timestamp(d), code]
            return float(v) if v is not None and np.isfinite(v) and v > 0 else None
        except (KeyError, TypeError):
            return None

    def _reject(self, why, code=None, side=None, qty=None, px=None):
        self.rejections[why] = self.rejections.get(why, 0) + 1
        self.reject_log.append({
            "date": self.current_date, "code": code or "",
            "side": side or "", "qty": qty if qty is not None else "",
            "px": float(px) if isinstance(px, (int, float)) else "",
            "reason": why,
        })

    # ---------- 下单 ----------
    def order_target_value(self, code, value) -> Optional[Order]:
        eng = self
        d = pd.Timestamp(eng.current_date)
        px = eng._fill_price(code, d, eng._minutes)
        if px is None or not np.isfinite(px) or px <= 0:
            eng._reject("无有效成交价（停牌/未上市）", code, "buy", value, None)
            return None
        target_size = max(float(value), 0.0) / px
        # 按一手取整（聚宽同样只接受 100 股整数倍）
        target_size = float(int(target_size // 100) * 100)
        return eng._trade_to(code, target_size, px, d)

    def order_target_shares(self, code, amount) -> Optional[Order]:
        d = pd.Timestamp(self.current_date)
        px = self._fill_price(code, d, self._minutes)
        if px is None or px <= 0:
            self._reject("无有效成交价（停牌/未上市）", code, "sell", amount, None)
            return None
        tgt = float(int(max(amount, 0.0) // 100) * 100)
        return self._trade_to(code, tgt, px, d)

    @property
    def _minutes(self) -> int:
        return self.current_dt.hour * 60 + self.current_dt.minute

    def _trade_to(self, code, target_size, px, d) -> Order:
        cur = self.pf.position_size(code)
        diff = target_size - cur
        if abs(diff) < 1:
            return Order(code, amount=0.0, filled=0.0, price=px,
                         status=OrderStatus.held)
        bar = self._bar(code, d)
        order = Order(code, amount=abs(diff), price=px)
        if diff > 0:                      # 买入
            ok, why = self.rules.check_buy(bar, px)
            if not ok:
                self._reject(why, code, "buy", diff, px)
                order.reason = why
                return order
            cash = self.pf.cash
            size = self.broker.max_affordable_size(cash, px, self._volume(code, d))
            size = min(diff, size)
            if size < 100:
                self._reject("现金不足一手", code, "buy", diff, px)
                order.reason = "现金不足"
                return order
            fees = self.fees_of(size * px, OrderSide.BUY)
            try:
                self.pf.buy(code, size, px, fees=fees)
            except ValueError as e:
                self._reject(str(e)[:40], code, "buy", size, px)
                order.reason = str(e)[:60]
                return order
            order.filled, order.fees, order.status = size, fees, OrderStatus.held
            self.trades.append({"timestamp": pd.Timestamp(self.current_date),
                                "code": code, "action": "buy", "size": float(size),
                                "price": float(px), "fees": float(fees),
                                "pnl": 0.0})
        else:                             # 卖出
            ok, why = self.rules.check_sell(bar, px)
            if not ok:
                self._reject(why, code, "sell", abs(diff), px)
                order.reason = why
                return order
            size = min(abs(diff), self.pf.sellable_size(code))
            if size < 1:
                self._reject("T+1 当日买入不可卖", code, "sell", abs(diff), px)
                order.reason = "T+1"
                return order
            fees = self.fees_of(size * px, OrderSide.SELL)
            try:
                pnl = self.pf.sell(code, size, px, fees=fees)
            except ValueError as e:
                self._reject(str(e)[:40], code, "sell", size, px)
                order.reason = str(e)[:60]
                return order
            order.filled, order.fees, order.status = size, fees, OrderStatus.held
            self.trades.append({"timestamp": pd.Timestamp(self.current_date),
                                "code": code, "action": "sell", "size": float(size),
                                "price": float(px), "fees": float(fees),
                                "pnl": float(pnl)})
        return order

    # ---------- 事件循环 ----------
    def run(self, initialize, dates=None, progress=None) -> dict:
        dates = pd.DatetimeIndex(dates if dates is not None else self.data.dates)
        # ⚠️ initialize 里读到的 current_dt / previous_date 必须是**回测起始日**，
        # 不是面板首日。面板为了预热会多带一年多的历史，若拿面板首日，
        # 策略在 initialize 里算出来的日子（如 v2 的 g.trading_day）会落在预热期，
        # 之后 `today == g.trading_day` 永远不成立 -> 整个回测一次都不调仓。
        self.current_date = dates[0]
        self.current_dt = datetime.datetime.combine(dates[0].date(),
                                                    datetime.time(9, 30))
        self.previous_date = dates[0].date()
        self.context = Context(self)
        initialize(self.context)
        self._rebuild_broker()
        equity, holdings = [], []
        prev = None
        for i, d in enumerate(dates):
            self.current_date = d
            self.previous_date = prev.date() if prev is not None else d.date()
            self.pf.new_day()
            for task in sorted(self.tasks, key=lambda t: t.minutes):
                if task.weekly is not None and not self._is_nth_trading_day(d, task.weekly):
                    continue
                self.current_dt = datetime.datetime.combine(d.date(), self._time_of(task))
                try:
                    task.func(self.context)
                except Exception as e:          # 与聚宽一致：单任务异常不终止回测
                    # 拒绝原因里带上异常正文与位置，否则只看到 "策略异常:KeyError"
                    # 根本不知道是哪个 key、哪一行
                    import traceback
                    tb = traceback.extract_tb(e.__traceback__)
                    where = ""
                    for fr in reversed(tb):
                        if "strategies" in (fr.filename or ""):
                            where = f" @{fr.filename.split(chr(92))[-1]}:{fr.lineno}"
                            break
                    msg = f"{type(e).__name__}: {e}{where}"
                    log.error(f"{task.func.__name__} 异常: {msg}\n"
                              f"{traceback.format_exc()}")
                    self._reject(f"策略异常 {task.func.__name__}: {msg[:90]}",
                                 code="", side="", qty="", px=None)
            # 收盘估值
            for c in list(self.pf.positions.keys()):
                px = self.price_of(c, d)
                if px:
                    self.pf.update_price(c, px)
            equity.append((d, self.pf.total_value))
            holdings.append(self._weights(d))
            prev = d
            if progress and (i + 1) % max(1, len(dates) // 10) == 0:
                progress(i + 1, len(dates))
        eq = pd.Series([v for _, v in equity],
                       index=pd.Index([x for x, _ in equity], name="trade_date"))
        return {"equity": eq,
                "holdings": pd.DataFrame(holdings, index=eq.index),
                "trades": pd.DataFrame(self.trades),
                "rejections": dict(self.rejections),
                "reject_log": list(self.reject_log),
                "log": list(log.lines)}

    @staticmethod
    def _time_of(task: Task) -> datetime.time:
        h, m = task.time.split(":")[:2]
        return datetime.time(int(h), int(m))

    def _is_nth_trading_day(self, d, n: int) -> bool:
        """聚宽 run_weekly(func, weekday)：weekday = 每周的第几个**交易日**"""
        idx = self.data.dates
        pos = idx.get_loc(d)
        week = idx[pos].isocalendar()[:2]
        same_week = [x for x in idx if x.isocalendar()[:2] == week]
        return len(same_week) >= n and same_week[n - 1] == d

    def _weights(self, d) -> Dict[str, float]:
        tv = self.pf.total_value
        if tv <= 0:
            return {}
        out = {}
        for c, p in self.pf.positions.items():
            if p.size <= 0:
                continue
            px = self.price_of(c, d) or p.current_price
            out[c] = p.size * px / tv
        return out


# ============================================================
# 运行入口
# ============================================================
def run_strategy(module, panel, start, end, initial_cash=1_000_000, fill="auto",
                 etf_yield=0.02, rules=True, verbose=False,
                 volume_limit=False, slippage=0.001,
                 impact_model="none", impact_k=0.1) -> dict:
    """跑一个聚宽策略模块

    module: 含 `initialize` 的模块对象（策略文件 import 进来即可）

    slippage / impact_model: 默认 1bp 固定滑点。**默认值原先是 0.0**，见
        JQEngine 的说明。要评估容量用 impact_model="sqrt"。
        策略内显式 `set_slippage()` 时以策略为准。
    """
    data = JQData(panel, start, end, etf_yield=etf_yield, verbose=verbose)
    eng = JQEngine(data, initial_cash, fill=fill, rules=rules, verbose=verbose,
                   volume_limit=volume_limit, slippage=slippage,
                   impact_model=impact_model, impact_k=impact_k)
    # 聚宽的 g 是平台注入到策略模块命名空间的全局对象。
    # 策略里写的是 `g.trading_signal = True`，所以必须把 g 绑到模块全局。
    module.g = eng.g
    if not hasattr(module, "datetime"):
        module.datetime = datetime
    dates = data.dates[(data.dates >= pd.Timestamp(start)) & (data.dates <= pd.Timestamp(end))]
    res = eng.run(module.initialize, dates=dates)
    res["engine"] = eng
    return res
