# -*- coding: utf-8 -*-
"""聚宽数据 API 的本地实现（只覆盖这两个策略用到的部分）

【为什么需要这一层】
    用户引入的两份策略是聚宽平台的代码（`from jqdata import *`）。
    要在本项目里复现并且能与聚宽回测结果对照，最可靠的做法不是"重写一遍逻辑"，
    而是**原样保留策略代码**，只把 `jqdata` 的 API 换成本层的实现。
    这样策略逻辑零改动，差异只可能来自数据与撮合，便于归因。

【本层做出的近似（都会在报告里列出）】
    1. **399101.XSHE（中小综指）与 000985.XSHG（中证全指）成分股库里没有**
       -> 用规则重建：399101 ≈ 002/003 开头的股票；000985 ≈ 全部 A 股
    2. **399101 指数点位库里没有** -> 用重建成分股的**等权**日收益合成指数
    3. **511880.XSHG（银华日利 ETF）日线库里没有**（etf 表只有月度快照）
       -> 合成为"现金等价物"：按固定年化收益增长（默认 2%），可用 --etf-yield 调整
    4. **微盘股指数 tiny_index.csv 没有** -> 策略 2 里本来就有 399101 的备用分支，
       改为使用该分支（`g.index = '399101.XSHE'`），这是一个**行为差异**，必须说明
    5. **分钟线没有** -> `history(1,'1m','close')` 与 `get_current_data().last_price`
       统一近似为**当日开盘价**；`high_limit/low_limit` 用 cleaned/limit_price 真实值

【没有做出的近似（保证与聚宽口径一致）】
    - 价格用**不复权**真实价（`use_real_price=True` 的语义）
    - 财务数据按 **ann_date 做 point-in-time 对齐**，不用 end_date
    - 市值单位换算成"亿元"（tushare 的 total_mv 是万元）
"""
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

# ---------------------------------------------------------------
# 指数代理规则：库里没有的指数，用代码前缀/全市场规则重建
# ---------------------------------------------------------------
REAL_INDEXES = ("000300.SH", "000905.SH", "000852.SH", "932000.CSI",
                "399101.SZ", "000985.CSI", "H00985.CSI")

# 聚宽代码 -> 本库 ts_code。库里现在有真实的 399101/000985，优先用真实数据；
# 拿不到时才退回下面的规则重建。
INDEX_CODE_MAP = {
    "399101.XSHE": "399101.SZ",     # 中小综指
    "399101.SZ": "399101.SZ",
    "000985.XSHG": "000985.CSI",    # 中证全指
    "000985.CSI": "000985.CSI",
    "H00985.CSI": "H00985.CSI",
    "000300.XSHG": "000300.SH",
    "000905.XSHG": "000905.SH",
    "000852.XSHG": "000852.SH",
    "932000.CSI": "932000.CSI",
}

INDEX_PROXY_PREFIX = {
    "399101.XSHE": ("002", "003"),      # 中小企业板综指 ≈ 002/003 开头
    "399101.SZ": ("002", "003"),
}
FULL_MARKET_INDEXES = ("000985.XSHG", "000985.CSI", "800007.choice")

# 银华日利等货币 ETF：库里没有日线，按现金等价物合成
ETF_YIELD_DEFAULT = 0.02


# ===========================================================
# pandas 1.x 兼容：整数索引的位置回退
# ===========================================================
class LegacySeries(pd.Series):
    """还原 pandas 1.x 的行为：索引里找不到的整数键回退成**位置**索引

    聚宽平台跑的是 pandas 1.x，策略代码里大量出现
        last_prices[stock][-1]        # 取最新一根
    在 pandas 1.x 里 `s[-1]` 会退化成位置索引；pandas 2 起告警、**3 起直接抛
    KeyError**（因为 -1 被当成标签）。实测 v1 策略 84 次 weekly_adjustment
    全部死在这一行。

    保持策略代码零改动的前提下，只能在这里把老语义补回来。
    """
    def __getitem__(self, key):
        try:
            return super().__getitem__(key)
        except (KeyError, IndexError):
            if isinstance(key, (int, np.integer)) and not isinstance(key, bool):
                return self.iloc[int(key)]
            raise


def _norm_date_key(key):
    """把 datetime.date 归一成 Timestamp（含元组/列表里的）

    pandas 3 起 `df.loc[datetime.date(2024,1,2)]` 在 DatetimeIndex 上**匹配不到**
    （标签必须精确是 Timestamp）；聚宽的 pandas 1.x 是可以的。
    v2 的 time_choosing 里就是 `df.loc[context.previous_date, 'close']`，
    在 pandas 3 下每天抛 KeyError。
    """
    import datetime as _dt
    if isinstance(key, _dt.date) and not isinstance(key, _dt.datetime):
        return pd.Timestamp(key)
    if isinstance(key, tuple):
        return tuple(_norm_date_key(k) for k in key)
    if isinstance(key, list):
        return [_norm_date_key(k) for k in key]
    return key


class _LegacyLoc:
    """`.loc` 的兼容包装：先转成普通 DataFrame 再取，避免递归调用自身属性"""

    def __init__(self, obj):
        self._obj = obj

    def __getitem__(self, key):
        return pd.DataFrame(self._obj).loc[_norm_date_key(key)]

    def __setitem__(self, key, value):
        self._obj.loc[_norm_date_key(key)] = value


class LegacyFrame(pd.DataFrame):
    """history()/get_price() 的返回类型

    - 列切片给出 LegacySeries（整数键位置回退，还原 pandas 1.x 语义）
    - `.loc` 接受 datetime.date（pandas 3 起不再自动匹配 DatetimeIndex）
    """
    @property
    def _constructor(self):
        return LegacyFrame

    @property
    def _constructor_sliced(self):
        return LegacySeries

    @property
    def loc(self):
        return _LegacyLoc(self)


def to_ts_code(code: str) -> str:
    """聚宽代码 -> 本库 ts_code（511880.XSHG -> 511880.SH）"""
    c = str(code).upper()
    if c.endswith(".XSHG"):
        return c[:-5] + ".SH"
    if c.endswith(".XSHE"):
        return c[:-5] + ".SZ"
    return c


class JQData:
    """聚宽数据接口的本地实现（预加载到内存，查询走内存索引）"""

    def __init__(self, panel: dict, start, end, etf_yield: float = ETF_YIELD_DEFAULT,
                 verbose: bool = False):
        self.panel = panel
        self.start = pd.Timestamp(start)
        self.end = pd.Timestamp(end)
        self.etf_yield = float(etf_yield)
        self.verbose = verbose

        close = panel["close"]
        self.dates: pd.DatetimeIndex = close.index
        self.codes: List[str] = list(close.columns)
        self._close = close
        self._open = panel.get("open", close)
        self._high = panel.get("high", close)
        self._low = panel.get("low", close)
        self._pre_close = panel.get("pre_close")
        self._limit_up = panel.get("limit_up")
        self._limit_down = panel.get("limit_down")
        self._suspended = panel.get("suspended")
        # tushare total_mv 单位是万元；聚宽的 valuation.market_cap 是亿元
        mv = panel.get("total_mv")
        self._mktcap_yi = (mv / 1e4) if mv is not None else None

        self._load_security_info()
        self._load_financials()
        self._fund_cache: Dict[str, Optional[pd.Series]] = {}
        self._build_index_cache()

    # ===========================================================
    # 股票基础信息
    # ===========================================================
    def _load_security_info(self):
        from database.loader import load_industry_map
        info = load_industry_map()
        self._name = pd.Series(dtype=object)
        self._list_date = pd.Series(dtype="datetime64[ns]")
        self._industry = pd.Series(dtype=object)
        if info is None or info.empty:
            return
        d = info.copy()
        if "code" not in d.columns:
            d["code"] = d["ts_code"].astype(str).str.split(".").str[0]
        d["code"] = d["code"].astype(str).str.zfill(6)
        d = d.drop_duplicates("code").set_index("code")
        if "name" in d.columns:
            self._name = d["name"].astype(str)
        if "list_date" in d.columns:
            self._list_date = pd.to_datetime(d["list_date"], errors="coerce")
        if "industry" in d.columns:
            self._industry = d["industry"].astype(str)

    def name_of(self, code: str) -> str:
        return str(self._name.get(code, "") or "")

    def list_date_of(self, code: str):
        v = self._list_date.get(code)
        return pd.Timestamp(v) if pd.notna(v) else pd.NaT

    def is_st(self, code: str) -> bool:
        """按股票名称判断 ST（聚宽 get_current_data().is_st 的口径）"""
        n = self.name_of(code).upper().replace(" ", "")
        return ("ST" in n) or ("*" in n)

    def is_delisting(self, code: str) -> bool:
        return "退" in self.name_of(code)

    # ===========================================================
    # 财务数据（PIT）
    # ===========================================================
    def _load_financials(self):
        """读利润表并做 point-in-time 对齐所需的最小字段

        只保留 (code, ann_date, end_date, revenue, n_income, n_income_attr_p)。
        同一 (code, end_date) 多次公告只认**最早**那次（as-reported），
        与 factors/fundamental.py 的口径完全一致。
        """
        from database.config import FROZEN_ROOT, connect_duckdb
        con = connect_duckdb()
        glob = f"{(FROZEN_ROOT / 'financial').as_posix()}/year=*/profit_*.parquet"
        try:
            df = con.execute(
                f"SELECT code, ann_date, end_date, report_type, revenue, "
                f"n_income, n_income_attr_p "
                f"FROM read_parquet('{glob}', union_by_name=True) "
                f"WHERE report_type = '1'").fetchdf()
        finally:
            con.close()
        if df.empty:
            self._fin = pd.DataFrame()
            self._fin_dates = np.array([], dtype="datetime64[ns]")
            return
        df["code"] = df["code"].astype(str).str.zfill(6)
        df["ann_date"] = pd.to_datetime(df["ann_date"], format="%Y%m%d", errors="coerce")
        df["end_date"] = pd.to_datetime(df["end_date"], format="%Y%m%d", errors="coerce")
        df = df.dropna(subset=["ann_date", "end_date"])
        df = (df.sort_values(["code", "end_date", "ann_date"])
                .drop_duplicates(["code", "end_date"], keep="first"))
        df = df.sort_values("ann_date").reset_index(drop=True)
        self._fin = df
        self._fin_dates = df["ann_date"].values

    def financials_asof(self, date) -> pd.DataFrame:
        """截至 date **已公告**的最新一期财务数据（每只股票一行）

        ⚠️ 对齐键是 ann_date。用 end_date 会让财报在公告前就生效 —— 典型前视。
        """
        if self._fin.empty:
            return pd.DataFrame(columns=["revenue", "n_income", "n_income_attr_p"])
        d = np.datetime64(pd.Timestamp(date))
        sub = self._fin[self._fin["ann_date"].values <= d]
        if sub.empty:
            return pd.DataFrame(columns=["revenue", "n_income", "n_income_attr_p"])
        # 同一股票可能有多期，取报告期最新的一期
        sub = (sub.sort_values(["code", "end_date"])
                  .drop_duplicates("code", keep="last"))
        return sub.set_index("code")[["revenue", "n_income", "n_income_attr_p"]]

    def market_cap_yi(self, date) -> pd.Series:
        """市值（亿元），与聚宽 valuation.market_cap 同口径"""
        if self._mktcap_yi is None or date not in self._mktcap_yi.index:
            return pd.Series(dtype=float)
        return self._mktcap_yi.loc[date].dropna()

    # ===========================================================
    # 指数（重建）
    # ===========================================================
    def _build_index_cache(self):
        self._index_cache: Dict[str, pd.DataFrame] = {}
        self._index_close_cache: Dict[str, pd.Series] = {}
        self._real_members: Dict[str, pd.DataFrame] = {}
        # 真实指数（行情库里有）；用聚宽代码和本库 ts_code 两种键都缓存一份
        from database.loader import load_index_daily
        for jq_code, ts_code in INDEX_CODE_MAP.items():
            if ts_code in self._index_cache:
                self._index_cache[jq_code] = self._index_cache[ts_code]
                continue
            df = load_index_daily(ts_code, str(self.start)[:10], str(self.end)[:10])
            if not df.empty:
                d = df.set_index("trade_date")
                self._index_cache[ts_code] = d
                self._index_cache[jq_code] = d
        # 真实成分（月度快照）
        from universe import load_index_members
        for ts_code in set(INDEX_CODE_MAP.values()):
            try:
                m = load_index_members(ts_code)
            except Exception:
                m = None
            if m is not None and not m.empty:
                self._real_members[ts_code] = m

    def has_real_members(self, code: str) -> bool:
        return INDEX_CODE_MAP.get(code, code) in self._real_members

    def real_index(self, code: str) -> Optional[pd.DataFrame]:
        return self._index_cache.get(code)

    def index_stocks(self, code: str, date=None) -> List[str]:
        """指数成分股

        优先级：**真实月度成分快照** > 规则重建。
        库里现在有 399101.SZ（中小综指）与 000985.CSI（中证全指）的真实成分，
        所以这两个指数不再用"002/003 前缀 / 全部 A 股"近似。
        """
        date = pd.Timestamp(date) if date is not None else self.dates[-1]
        ts_code = INDEX_CODE_MAP.get(code, code)
        # 1) 真实成分
        mem = self._real_members.get(ts_code)
        if mem is not None and not mem.empty:
            from universe import index_member_panel
            cols = sorted(mem["code"].unique())
            m = index_member_panel(ts_code, pd.DatetimeIndex([date]), cols)
            row = m.iloc[0]
            picked = [c for c in cols if bool(row.get(c, False))]
            if picked:
                return picked
        # 2) 真实指数行情里的成分（`index_weight` 之外的老数据）
        if code in REAL_INDEXES and ts_code in self._index_cache:
            from universe import load_index_members, index_member_panel
            mem2 = load_index_members(ts_code)
            if not mem2.empty:
                cols = sorted(mem2["code"].unique())
                m = index_member_panel(ts_code, pd.DatetimeIndex([date]), cols)
                row = m.iloc[0]
                return [c for c in cols if bool(row.get(c, False))]
        # 3) 规则重建（库里没有该指数时）
        if code in INDEX_PROXY_PREFIX:
            pre = INDEX_PROXY_PREFIX[code]
            return [c for c in self.codes
                    if str(c).startswith(pre)
                    and pd.notna(self.list_date_of(c))
                    and self.list_date_of(c) <= date]
        if code in FULL_MARKET_INDEXES:
            return [c for c in self.codes
                    if pd.notna(self.list_date_of(c))
                    and self.list_date_of(c) <= date]
        return []

    def index_close(self, code: str, end_date, count: int = 1) -> pd.Series:
        """指数收盘价序列

        真实指数直接用 frozen/index_daily；重建指数用成分股**等权**日收益合成。
        """
        end_date = pd.Timestamp(end_date)
        if code in self._index_cache:
            s = self._index_cache[code]["close"]
            return s[s.index <= end_date].tail(count)
        if code not in self._index_close_cache:
            codes = self.index_stocks(code, self.dates[-1])
            cols = [c for c in codes if c in self._close.columns]
            if not cols:
                return pd.Series(dtype=float)
            # 等权合成：每日成分股收益的均值，再复利累乘
            ret = self._close[cols].pct_change(fill_method=None)
            idx = (1 + ret.mean(axis=1).fillna(0.0)).cumprod()
            self._index_close_cache[code] = idx
        s = self._index_close_cache[code]
        return s[s.index <= end_date].tail(count)

    def any_close(self, code: str, end_date, count: int = 1) -> pd.Series:
        """任意证券的收盘序列：股票走行情面板，指数走重建/真实指数，其余按货币 ETF 合成

        策略里会出现三类代码：股票、指数（399101/000985）、ETF（511880）。
        只认股票面板会让指数和 ETF 直接 KeyError。
        """
        if code in self._close.columns:
            return self.history("close", [code], end_date, count)[code]
        if (code in self._index_cache or code in INDEX_PROXY_PREFIX
                or code in FULL_MARKET_INDEXES or code in REAL_INDEXES):
            return self.index_close(code, end_date, count)
        return self.etf_close(code, end_date, count)

    def is_non_stock(self, code: str) -> bool:
        """不在股票行情面板里的代码（指数 / ETF / 退市）"""
        return code not in self._close.columns

    # ===========================================================
    # ETF（合成现金等价物）
    # ===========================================================
    def etf_close(self, code: str, end_date, count: int = 1) -> pd.Series:
        """ETF 收盘序列

        **优先用真实日线**（frozen/fund_daily）。拿不到时退回"现金等价物"合成。

        ⚠️ 关于货币 ETF（511880 银华日利）：它的**真实价格恒在 100 附近**，
        收益靠"份额折算"发放而不是价格上涨 —— 2024-2025 真实价格累计只有
        −0.036%，而年化 2% 的合成值约 +4.0%。
        经济上合成口径更准（拿到了货币收益），但**聚宽回测同样不处理份额折算**，
        其 ETF 持仓也近似零收益 —— 要跟聚宽对齐就必须用真实价格。
        """
        end_date = pd.Timestamp(end_date)
        ts_code = INDEX_CODE_MAP.get(code, to_ts_code(code))
        s = self._fund_cache.get(ts_code)
        if s is None:
            s = self._load_fund_daily(ts_code)
            self._fund_cache[ts_code] = s
        if s is not None and len(s):
            return s[s.index <= end_date].tail(count)
        # 退回合成：净值以固定年化收益增长
        d = self.dates[self.dates <= end_date]
        if len(d) == 0:
            return pd.Series(dtype=float)
        base = 100.0
        growth = (1.0 + self.etf_yield) ** (np.arange(len(d)) / 252.0)
        return pd.Series(base * growth, index=d).tail(count)

    def _load_fund_daily(self, ts_code: str):
        """读 frozen/fund_daily 里某只基金的日线（没有则返回 None）"""
        from database.config import FROZEN_ROOT
        fs = sorted((FROZEN_ROOT / "fund_daily").rglob(f"{ts_code}.parquet"))
        if not fs:
            return None
        try:
            df = pd.concat([pd.read_parquet(f) for f in fs], ignore_index=True)
        except Exception:
            return None
        if df.empty or "close" not in df.columns:
            return None
        df["trade_date"] = pd.to_datetime(df["trade_date"])
        return df.sort_values("trade_date").drop_duplicates("trade_date") \
                 .set_index("trade_date")["close"].astype(float)

    def has_real_fund(self, code: str) -> bool:
        s = self._fund_cache.get(to_ts_code(code))
        return s is not None and len(s) > 0

    # ===========================================================
    # 交易日
    # ===========================================================
    def _load_calendar(self):
        """交易所交易日历（frozen/calendar）

        聚宽的 get_trade_days 走的是**交易所日历**，覆盖范围比我们加载的面板长。
        只用面板日期会让"下个月第一个交易日"在回测末期取不到 → 策略拿空列表
        去索引 [0] → IndexError（v2 实测 2 次）。
        """
        from database.config import FROZEN_ROOT
        fs = sorted((FROZEN_ROOT / "calendar").rglob("*.parquet"))
        if not fs:
            return None
        try:
            df = pd.concat([pd.read_parquet(f) for f in fs], ignore_index=True)
        except Exception:
            return None
        for col in ("cal_date", "trade_date", "date"):
            if col in df.columns:
                s = df[col]
                # trade_date 有的年份是 datetime、有的是 YYYYMMDD 字符串，两种都要认
                if pd.api.types.is_datetime64_any_dtype(s):
                    s = pd.to_datetime(s, errors="coerce")
                else:
                    s = pd.to_datetime(s.astype(str), format="%Y%m%d",
                                       errors="coerce")
                if "is_open" in df.columns:
                    s = s[pd.to_numeric(df["is_open"], errors="coerce") == 1]
                s = s.dropna().drop_duplicates().sort_values()
                return pd.DatetimeIndex(s)
        return None

    def trade_days(self, start=None, end=None, count=None) -> pd.DatetimeIndex:
        """交易日列表

        ⚠️ 只用**面板覆盖的交易日**，不用完整交易所日历。
        试过改用 frozen/calendar（覆盖到 2027，比面板长 486 天），
        结果 v2 每次都抛 `ValueError: truth value of an array is ambiguous` —— 
        日历里的交易日落在面板之外，策略据此算出的调仓日引擎跑不到，
        状态机就此卡死。**面板日期与策略实际能成交的日期一致**才是对的，
        即使代价是：回测末尾若策略要"找下个月的第一个交易日"会取空列表
        （v2 实测 2 次，属边界情形，已在移植报告里记录为已知限制）。
        """
        d = self.dates
        if start is not None:
            d = d[d >= pd.Timestamp(start)]
        if end is not None:
            d = d[d <= pd.Timestamp(end)]
        if count is not None:
            d = d[:int(count)]
        return pd.DatetimeIndex(d)

    # ===========================================================
    # 行情查询
    # ===========================================================
    def _frame(self, field: str) -> pd.DataFrame:
        return {"close": self._close, "open": self._open, "high": self._high,
                "low": self._low, "pre_close": self._pre_close}.get(field, self._close)

    def history(self, field: str, codes: List[str], end_date, count: int) -> pd.DataFrame:
        """返回 截止 end_date（含）的最后 count 根日线，columns=股票代码

        ⚠️ 缺失的代码要**补成 NaN 列**，不能静默丢掉：
        聚宽的 history 返回请求的全部证券，策略里会直接写 `last_prices[stock][-1]`，
        丢列会让它 KeyError。
        """
        f = self._frame(field)
        if f is None:
            return LegacyFrame()
        e = pd.Timestamp(end_date)
        sub = f.loc[f.index <= e]
        if sub.empty:
            return LegacyFrame(columns=list(codes))
        out = sub.reindex(columns=list(codes)).tail(int(count))
        return LegacyFrame(out)

    def current_bar(self, date) -> pd.DataFrame:
        """当日"盘中快照"：用开盘价近似（策略在 10:00 左右下单）"""
        if date not in self._open.index:
            return pd.DataFrame()
        d = {"open": self._open.loc[date], "high": self._high.loc[date],
             "low": self._low.loc[date], "close": self._close.loc[date]}
        return pd.DataFrame(d)

    def paused(self, date) -> pd.Series:
        if self._suspended is not None and date in self._suspended.index:
            return self._suspended.loc[date].fillna(True).astype(bool)
        # 没有停牌表时：当日无成交或价格缺失视为停牌
        if date not in self._close.index:
            return pd.Series(True, index=self.codes)
        return self._close.loc[date].isna()

    def limit_prices(self, date):
        if (self._limit_up is not None and self._limit_down is not None
                and date in self._limit_up.index):
            return self._limit_up.loc[date], self._limit_down.loc[date]
        return None, None
