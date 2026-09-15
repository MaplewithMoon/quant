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

# 新股上市初期不设涨跌幅（注册制）：科创板/创业板/主板(2023-04-10 起) 前 5 个
# 交易日、北交所首日。
#
# ⚠️ **尚未实现**。这个常量只是把规则写在显眼处，避免误以为已经生效。
#    实测影响 29,289 行（占全库 0.183%），集中在次新股前 5 日。
#    实现路径已经验证过：`MarketRules` 的 `_num()` 对 NaN 返回 None 会**跳过**
#    涨跌停检查，所以把窗口内的 `limit_up/limit_down` 置为 NaN 即可表达"无限制"。
#    需要改的调用点：`apply_limit_prices` 增加 `list_date` 参数，并把
#    `rebuild_limit.py` / `daily_update.py` / `meta.py` / `validate_data.py`
#    四处都传进去（否则校验会误报不一致）。
#    分板块生效日：科创板 2019-07-22、创业板 2020-08-24、北交所 2021-11-15、
#    主板 2023-04-10。主板 2014-01-01 ~ 2023-04-09 的首日 ±44% 是另一条规则
#    （基准是发行价而非昨收），未在本模块处理。
NO_LIMIT_DAYS = 5


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
    """最终涨跌幅：ST 取 ±5%，否则按板块"""
    if is_st:
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
                       tick: float = TICK) -> pd.DataFrame:
    """给带官方 `pre_close` 的日线表加上 `limit_up` / `limit_down`

    ⚠️ 必须传**官方 pre_close**（tushare 已按除权调整），
    绝不能用 `close.shift(1)` —— 那在除权日会算出错误的价格。
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
    # ST 区间覆盖
    if st_map:
        pct = np.where(st_mask(dates, code, st_map), 0.05, pct)

    # 整数 tick 运算 + 四舍五入（交易所口径，非银行家舍入）
    bad = ~(pc > 0)
    pc_t = _ticks(np.where(bad, 0.0, pc), tick)
    k_up, k_dn = _pct_factors(pct)
    up = _round_half_up(pc_t * k_up, PCT_DEN) * tick
    dn = _round_half_up(pc_t * k_dn, PCT_DEN) * tick
    out["limit_up"] = np.round(up, 10)
    out["limit_down"] = np.round(dn, 10)
    out.loc[bad, ["limit_up", "limit_down"]] = np.nan
    return out
