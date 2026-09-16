# -*- coding: utf-8 -*-
"""A 股涨跌停价的**唯一**计算实现

【为什么要有这个模块】
    原先有三套实现，规则互相不一致，而且会写同一个 `cleaned/limit_price`：

    | 位置                     | ST 5% | 北交所 30%      | 用什么当昨收        |
    |--------------------------|-------|-----------------|---------------------|
    | database/downloader/meta.py | 有  | 无              | `close.shift(1)`  ✗ |
    | scripts/daily_update.py     | 无  | 只认 920        | `close.shift(1)`  ✗ |
    | scripts/rebuild_limit.py    | 有  | 920/43/83/87/88 | 官方 pre_close    ✓ |

    最要命的是**用 `close.shift(1)` 当昨收**：那是**未除权**的上一日收盘，
    每逢除权除息日算出的涨跌停价都是错的（10 送 10 的情况下会差一倍）。
    而增量更新 `daily_update.py` 会拿它去覆盖 `rebuild_limit.py` 生成的正确数据 ——
    这是**正在发生的污染**，不是理论风险。

    现在统一到这里：所有需要涨跌停价的地方都调本模块。

【规则】
    主板                        ±10%
    创业板 300/301              ±20%（**2020-08-24 注册制改革之前是 ±10%**）
    科创板 688/689              ±20%
    北交所 920/43/83/87/88      ±30%
    ST / *ST                    ±5%（优先于板块，取两者更严的）
    昨收                        一律用**官方 pre_close**（tushare 已做除权调整）
    舍入                        **四舍五入到分**（整数运算实现，见下方"舍入"一节）

【第三个坑：舍入】
    交易所是"四舍五入"，而 Python/numpy 的 `round()` 是**银行家舍入**。
    实测旧数据里涨停价 2.7%、跌停价 4.4% 的行因此差 1 分钱。
    本模块用整数分运算精确实现，不用 `round()`。
"""
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

BEIJING_PREFIXES = ("920", "43", "83", "87", "88")
STAR_PREFIXES = ("688", "689")          # 科创板
GEM_PREFIXES = ("300", "301")           # 创业板

# 创业板注册制改革：2020-08-24 起涨跌幅由 ±10% 放宽到 ±20%
GEM_20PCT_FROM = pd.Timestamp("2020-08-24")

# 主板 ST 的 ±5% 自 2026-07-06 起放宽到 ±10%
#
# 【这个日期是**从行情数据反推**出来的，不是抄某份公告】
#   2026-07-03 及之前：主板 ST 股每日 |pct_chg| 上限稳定在 5.11%~5.19%（141 只样本）
#   2026-07-06 起    ：上限跳到 10.08%~10.26%，并开始出现"最高价越过涨停价"的行
#   2015-2025 全年   ：超 5.5% 的行数为 0
# 断点非常干净，所以按制度变更处理。若日后拿到正式公告，核对/修正这里即可。
MAIN_ST_10PCT_FROM = pd.Timestamp("2026-07-06")

# ============================================================
# 上市初期的特殊涨跌幅规则
# ============================================================
# A 股对新股上市初期的涨跌幅有**独立于板块常规档位**的规定，而且随制度
# 改革反复变化。不建模会有两种方向的错：
#   - 该"不设限"却按常规档位算 -> 涨停价偏低 -> **多拦**买入（次新股首日的
#     合法大涨会被当成封板）
#   - 该 ±44% 却按 ±10% 算 -> 同样多拦
#
# 规则表按"板块 × 上市日所在制度区间"组织。规则类型：
#   NO_LIMIT  : 上市后前 n 个**市场交易日**不设涨跌幅 -> limit 置 NaN
#               （`MarketRules._num()` 对 NaN 返回 None -> 跳过涨跌停检查）
#   FIRST_44  : 上市首日有效申报价格 ≤ 发行价×1.44、≥×0.64
#               （2013-12-13《新股发行体制改革意见》起，沪深主板）
#               tushare 首日 `pre_close` 正是**发行价** —— 已用 002973/002971/
#               603551/603290 等实测确认（首日 high/pre_close−1 = 0.4399~0.4408）
#   None      : 无特殊规则，走板块常规档位
NO_LIMIT = "no_limit"
FIRST_44 = "first_44"

# 首日 ±44% 的整数因子：**不是对称的 ±44%**，跌的方向只有 −36%
K44_UP = 14400          # ×1.44
K44_DN = 6400           # ×0.64

LISTING_REGIMES = {
    # 主板：1990~2013-12-12 首日不设限；2013-12-13~2023-04-09 首日 ±44%；
    #       2023-04-10（全面注册制首批）起前 5 个交易日不设限
    "MAIN": (("1990-01-01", NO_LIMIT, 1),
             ("2013-12-13", FIRST_44, 0),
             ("2023-04-10", NO_LIMIT, 5)),
    # 创业板：2009-10-30 开板起首日不设限；2020-08-24 注册制改革起前 5 日不设限
    "GEM": (("2009-10-30", NO_LIMIT, 1),
            ("2020-08-24", NO_LIMIT, 5)),
    # 科创板：2019-07-22 开板起前 5 个交易日不设限
    "STAR": (("2019-07-22", NO_LIMIT, 5),),
    # 北交所：2020-07-27 精选层设立起首日不设限；2021-11-15 开市后同样首日不设限
    "BSE": (("2020-07-27", NO_LIMIT, 1),),
}

# 兼容旧名（外部可能引用）
NO_LIMIT_DAYS = 5


def board_of(code: str) -> str:
    """板块标识：STAR / GEM / BSE / MAIN"""
    c = str(code).zfill(6)
    if c.startswith(STAR_PREFIXES):
        return "STAR"
    if c.startswith(GEM_PREFIXES):
        return "GEM"
    if c.startswith(BEIJING_PREFIXES):
        return "BSE"
    return "MAIN"


def listing_rule(code: str, list_date) -> Tuple[Optional[str], int]:
    """该股适用的上市初期规则 -> (kind, n)；无特殊规则返回 (None, 0)"""
    if list_date is None or pd.isna(list_date):
        return (None, 0)
    d = pd.Timestamp(list_date)
    kind, n = None, 0
    for eff, k, cnt in LISTING_REGIMES.get(board_of(code), ()):
        if d >= pd.Timestamp(eff):
            kind, n = k, cnt
    return (kind, n)


_CAL_CACHE: Dict[str, object] = {}


def trading_calendar() -> pd.DatetimeIndex:
    """全市场交易日（**唯一实现在 database/calendar.py**）

    ⚠️ 返回的是**已裁剪到今天**的日历。`frozen/calendar` 本身含未来占位日
    （到 2027-12-31），把它当"已发生的交易日"用会出错 —— 详见
    `database/calendar.py` 的模块文档。
    """
    from database.calendar import trading_days
    cal = trading_days()
    _CAL_CACHE["cal"] = cal
    return cal


def load_list_dates() -> Dict[str, pd.Timestamp]:
    """{code: 上市日} —— 取 frozen/stocks 的 all.parquet

    ⚠️ 该目录下还有一份旧管线产物 `data.parquet`（code/name/list_status），
    两者 schema 不同，用 `year=*/*.parquet` 一把捞会因缺列抛 schema mismatch。
    """
    from database.config import FROZEN_ROOT, connect_duckdb
    f = FROZEN_ROOT / "stocks" / "year=2005" / "all.parquet"
    if not f.exists():
        return {}
    con = connect_duckdb()
    try:
        df = con.execute(f"SELECT code, list_date FROM "
                         f"read_parquet('{f.as_posix()}')").fetchdf()
    finally:
        con.close()
    out = {}
    for r in df.itertuples(index=False):
        d = pd.to_datetime(r.list_date, errors="coerce")
        if pd.notna(d):
            out[str(r.code).zfill(6)] = pd.Timestamp(d)
    return out


def listing_windows(list_dates: Dict[str, pd.Timestamp] = None,
                    cal: pd.DatetimeIndex = None) -> Dict[str, Tuple[str, pd.Timestamp, pd.Timestamp, int]]:
    """{code: (kind, start, until, n)} —— 上市初期特殊规则的**生效区间**

    区间含义：
        NO_LIMIT  -> [start, until] 内不设涨跌幅（limit 置空）
        FIRST_44  -> [start, until] 内适用 ×1.44/×0.64（首日，start == until）
    区间按**市场交易日**数：即"上市后的前 n 个交易日"，停牌不顺延。

    ⚠️ **必须同时判上下界**。只判 `trade_date <= until` 会把"数据起点早于
    list_date"的股票（重新上市股，如 `601399 国机重装` 的日线早于其重上市日）
    的**整段历史**都算进窗口 —— 实测会多出 6 万行错误的空涨跌停价。

    ⚠️ 已知局限：窗口起点取自 `frozen/stocks.list_date`。对于重新上市股，
    tushare 的 list_date 是重上市日、且这类股票不适用新股首日规则，会被误套
    ±44%。区分它们需要一张"重新上市"名单，见文档 C2b。
    """
    list_dates = list_dates if list_dates is not None else load_list_dates()
    cal = cal if cal is not None else trading_calendar()
    out = {}
    if len(cal) == 0:
        return out
    for code, ld in list_dates.items():
        kind, n = listing_rule(code, ld)
        if kind is None:
            continue
        i = int(cal.searchsorted(pd.Timestamp(ld), "left"))
        if i >= len(cal):
            continue
        j = min(i + max(n, 1) - 1, len(cal) - 1)
        out[code] = (kind, cal[i], cal[j], n)
    return out



def base_pct(code: str, date=None) -> float:
    """按板块给出基础涨跌幅限制

    date 用于处理创业板的制度切换；不传则一律按当前的 ±20%。
    """
    c = str(code).zfill(6)
    if c.startswith(STAR_PREFIXES):
        return 0.20
    if c.startswith(GEM_PREFIXES):
        if date is not None and pd.Timestamp(date) < GEM_20PCT_FROM:
            return 0.10
        return 0.20
    if c.startswith(BEIJING_PREFIXES):
        return 0.30
    return 0.10


def limit_pct(code: str, date=None, is_st: bool = False) -> float:
    """最终涨跌幅：**主板** ST 取 ±5%（2026-07-06 起放宽为 ±10%），其余按板块

    ⚠️ ST 的 ±5% **只适用于主板**。创业板 2020-08-24 注册制改革后，
    创业板与科创板的"风险警示股票"**不再单独设 5%**，仍按板块的 ±20%
    执行；北交所有自己的风险警示制度，同样不走 5%。
    早期实现给所有 ST 一律套 5%，会让这些股票的涨停价**偏低一半以上**
    —— 实测造成约 5,000 行"最高价越过涨停价"（被 `--check listing` 抓到）。
    """
    if is_st and board_of(code) == "MAIN":
        if date is not None and pd.Timestamp(date) >= MAIN_ST_10PCT_FROM:
            return 0.10
        return 0.05
    return base_pct(code, date)


# ============================================================
# 舍入：交易所是**四舍五入**，不是银行家舍入
# ============================================================
# 交易所口径：涨跌停价 = 前收盘价 ×(1±幅度)，按"四舍五入"取至价格最小变动单位。
# Python 的 round() / numpy 的 np.round() 是**银行家舍入**（round-half-to-even）：
#     round(11.045, 2) -> 11.04   （交易所口径应为 11.05）
# 而且浮点误差让结果"看运气"：10.05×1.1 的浮点值略大于 11.055，round() 恰好给了
# 正确的 11.06；换成 11.045 这类就会向下。所以绝不能依赖 round()。
# 实测（对旧 cleaned/limit_price 逐行比对）：涨停价 2.7%、跌停价 4.4% 的行
# 与交易所口径差 1 分钱 —— 1500 万行里约 50 万行。
#
# 做法：价格先换算成整数"分"，比例用整数分子/分母表示，全程整数运算。
#   limit_up = (pc_cents × K_up + 5000) // 10000  / 100
# 其中 K_up = round((1+p)×10000)，p ∈ {0.05, 0.10, 0.20, 0.30}；
# K_down = round((1-p)×10000)。整数除法与浮点无关，结果与交易所口径逐分一致。
PCT_DEN = 10000
TICK = 0.01          # A 股股票最小变动单位（1 分）


def _round_half_up(numer: np.ndarray, denom: int) -> np.ndarray:
    """整数四舍五入：round(numer / denom)，恰好 .5 时向上（远离零）"""
    return (numer + denom // 2) // denom


def _ticks(x, tick: float = TICK) -> np.ndarray:
    """价格 -> 整数 tick 单位（A 股价格正好落在 tick 网格上）"""
    return np.rint(np.asarray(x, dtype="float64") / tick).astype("int64")


def _pct_factors(pct: np.ndarray):
    """(K_up, K_down)：比例的整数分子，分母为 PCT_DEN"""
    k_up = np.rint((1.0 + pct) * PCT_DEN).astype("int64")
    k_dn = np.rint((1.0 - pct) * PCT_DEN).astype("int64")
    return k_up, k_dn


def compute(pre_close: float, code: str, date=None,
            is_st: bool = False, tick: float = TICK) -> Tuple[float, float]:
    """(limit_up, limit_down)，四舍五入到最小变动单位 —— 与交易所口径一致

    用整数运算实现"四舍五入"，避免 round() 的银行家舍入（见文件上方说明）。
    """
    if pre_close is None or not np.isfinite(pre_close) or pre_close <= 0:
        return np.nan, np.nan
    p = limit_pct(code, date, is_st)
    pc_t = int(round(pre_close / tick))
    k_up, k_dn = _pct_factors(np.array([p]))
    up = int(_round_half_up(np.array([pc_t], dtype="int64") * k_up, PCT_DEN)[0])
    dn = int(_round_half_up(np.array([pc_t], dtype="int64") * k_dn, PCT_DEN)[0])
    return round(up * tick, 10), round(dn * tick, 10)


# ============================================================
# ST 区间
# ============================================================
_ST_CACHE: Dict[str, Dict[str, List[Tuple]]] = {}

# frozen/st 的默认目录（供调用方判断"数据集是否存在"）
from .config import FROZEN_ROOT as _FROZEN_ROOT

ST_DIR_DEFAULT = _FROZEN_ROOT / "st"


def load_st_intervals(st_dir=None) -> Dict[str, List[Tuple]]:
    """读 frozen/st 的名称变更区间 -> {code: [(start, end), ...]}，只保留带 ST 的

    结果按目录缓存：`frozen/st` 有 4000+ 个 parquet 文件，而重建股票池时
    `st_panel` 会被反复调用，不缓存每次都要重读一遍。
    """
    from database.config import FROZEN_ROOT, connect_duckdb, year_globs
    d = st_dir or (FROZEN_ROOT / "st")
    key = str(d)
    if key in _ST_CACHE:
        return _ST_CACHE[key]
    out: Dict[str, List[Tuple]] = {}
    if not d.exists():
        _ST_CACHE[key] = out
        return out
    g = year_globs(d)
    if g == "[]":
        _ST_CACHE[key] = out
        return out
    # 用 DuckDB 一次扫完：4000+ 个小文件逐个 pd.read_parquet 要 20 秒以上，
    # DuckDB 一次扫描 ~1 秒。数据量很小，直接全取到内存。
    con = connect_duckdb()
    try:
        df = con.execute(f"""
            SELECT code, name, start_date, end_date
            FROM read_parquet({g})
            WHERE upper(coalesce(name, '')) LIKE '%ST%'
        """).fetchdf()
    finally:
        con.close()
    if df.empty:
        _ST_CACHE[key] = out
        return out
    df["_s"] = pd.to_datetime(df["start_date"], errors="coerce")
    df["_e"] = pd.to_datetime(df["end_date"], errors="coerce")
    df = df.dropna(subset=["_s"])
    for code, gg in df.groupby(df["code"].astype(str).str.zfill(6)):
        recs = [(s, e if pd.notna(e) else pd.Timestamp("2100-01-01"))
                for s, e in zip(gg["_s"], gg["_e"])]
        if recs:
            out[code] = recs
    _ST_CACHE[key] = out
    return out


def is_st_on(code: str, date, st_map: Dict[str, List[Tuple]]) -> bool:
    if not st_map:
        return False
    d = pd.Timestamp(date)
    for s, e in st_map.get(str(code).zfill(6), ()):
        if s <= d <= e:
            return True
    return False


def st_mask(dates, code: str,
            st_map: Dict[str, List[Tuple]]) -> np.ndarray:
    """给一串日期生成 ST 掩码（向量化）

    dates 可以是 Series 或 DatetimeIndex（两者都要支持，调用方各有习惯）。
    """
    d = np.asarray(getattr(dates, "values", dates), dtype="datetime64[ns]")
    m = np.zeros(len(d), dtype=bool)
    for s, e in st_map.get(str(code).zfill(6), ()):
        m |= (d >= np.datetime64(pd.Timestamp(s))) & (d <= np.datetime64(pd.Timestamp(e)))
    return m


# ============================================================
# DataFrame 入口
# ============================================================
def apply_limit_prices(df: pd.DataFrame, code: str,
                       st_map: Dict[str, List[Tuple]] = None,
                       pre_close_col: str = "pre_close",
                       tick: float = TICK,
                       listing_rule: Tuple = None) -> pd.DataFrame:
    """给带官方 `pre_close` 的日线表加上 `limit_up` / `limit_down`

    ⚠️ 必须传**官方 pre_close**（tushare 已按除权调整），
    绝不能用 `close.shift(1)` —— 那在除权日会算出错误的价格。

    listing_rule: 上市初期特殊规则 `(kind, until, n)`，来自
                  `listing_windows().get(code)`。不传 = 不做上市初期处理
                  （回测区间不覆盖次新股时结果一样，但校验会报不一致）。
    """
    out = df.copy()
    if pre_close_col not in out.columns:
        raise KeyError(f"缺少 {pre_close_col} 列：涨跌停价必须用官方昨收计算，"
                       f"不能用 close.shift(1)（除权日会错）")
    dates = pd.to_datetime(out["trade_date"])
    pc = pd.to_numeric(out[pre_close_col], errors="coerce").to_numpy(dtype="float64")
    pct = np.full(len(out), base_pct(code), dtype=float)
    # 创业板制度切换（按日）
    if str(code).zfill(6).startswith(GEM_PREFIXES):
        pct = np.where(dates < GEM_20PCT_FROM, 0.10, 0.20)
    # ST 区间覆盖（**只对主板**生效：创业板/科创板/北交所的 ST 仍走板块档位）
    # 2026-07-06 起主板 ST 也是 ±10%，与主板常规档位相同 -> 无需覆盖
    if st_map and board_of(code) == "MAIN":
        st = st_mask(dates, code, st_map) & (dates < MAIN_ST_10PCT_FROM).to_numpy()
        pct = np.where(st, 0.05, pct)

    # 整数 tick 运算 + 四舍五入（交易所口径，非银行家舍入）
    bad = ~(pc > 0)
    pc_t = _ticks(np.where(bad, 0.0, pc), tick)
    k_up, k_dn = _pct_factors(pct)
    up = _round_half_up(pc_t * k_up, PCT_DEN) * tick
    dn = _round_half_up(pc_t * k_dn, PCT_DEN) * tick

    # ---- 上市初期特殊规则（覆盖上面的常规档位）----
    if listing_rule:
        kind, start, until, _n = listing_rule
        # ⚠️ 必须**同时判上下界**：只判上界会把"数据起点早于 list_date"的股票
        # （重新上市股）的整段历史都算进窗口。
        in_win = ((dates >= pd.Timestamp(start))
                  & (dates <= pd.Timestamp(until))).to_numpy()
        if kind == NO_LIMIT:
            # 不设涨跌幅 -> NaN。引擎的 `_num()` 对 NaN 返回 None 会**跳过**
            # 涨跌停检查，正是"无限制"的语义。
            up = np.where(in_win, np.nan, up)
            dn = np.where(in_win, np.nan, dn)
            bad = bad | in_win
        elif kind == FIRST_44:
            # 首日**有效申报价格**不高于发行价×1.44、不低于×0.64 —— 注意这**不是**
            # 对称的 ±44%（跌的方向只有 −36%）。基准是发行价，而 tushare 首日的
            # `pre_close` 正是发行价（已用 002973 等实测确认）。
            up = np.where(in_win,
                          _round_half_up(pc_t * K44_UP, PCT_DEN) * tick, up)
            dn = np.where(in_win,
                          _round_half_up(pc_t * K44_DN, PCT_DEN) * tick, dn)
        else:
            raise ValueError(f"未知的上市初期规则 {kind!r}")

    out["limit_up"] = np.round(up, 10)
    out["limit_down"] = np.round(dn, 10)
    out.loc[bad, ["limit_up", "limit_down"]] = np.nan
    return out
