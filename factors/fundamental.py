# -*- coding: utf-8 -*-
"""基本面因子：point-in-time 对齐 + 景气度 / 盈利质量 / 价值 / 拥挤度

【为什么 PIT（point-in-time）是本模块唯一重要的事】
财报有两个日期：
    end_date  报告期（如 2023-03-31）
    ann_date  公告日（如 2023-04-28）

如果按 `end_date` 把一季报对齐到 3 月 31 日，就等于在 3 月 31 日"已经知道"
4 月 28 日才公布的数据 —— 这是最典型、也最容易被忽略的前视偏差。
它会把因子 IC 抬得虚高：财报公布前后股价往往已经反应完毕，
用 end_date 对齐等于免费拿到这段行情。

本模块的做法：
    1. 读 `frozen/financial/{profit,balance,cashflow}`，解析 ann_date / end_date
    2. 同一 (code, end_date) 有多次公告时，**只保留最早公告的那条**（as-reported）
       —— 实测 71,186 组重复公告里 98.5% 数值完全相同（纯重复），
          只有 617 组营收不同、839 组净利不同，所以按最早公告去重是安全且正确的
    3. 生效日 effective = ann_date + lag_days（默认 1 天）
    4. 用"按 effective 排好序的宽表 + reindex(ffill)"把报告期数据对齐到每个交易日
       —— 即"截至 t 已公告的最新一期财报"
    5. 所有同比/环比都用**同一报告期**的口径，不做跨期混算

【因子清单与方向】
    景气度     rev_yoy        营收同比增速（累计口径）        越大越好
               np_yoy         归母净利同比增速                越大越好
    盈利修正   np_accel       净利增速的变化（增速的加速度）   越大越好
               sue            标准化预期外盈利（PEAD）        越大越好
    盈利质量   roe_ttm        TTM ROE                        越大越好
               gross_margin   毛利率                          越大越好
               gm_yoy_chg     毛利率同比变化                   越大越好
               ocf_to_np      TTM 经营现金流 / TTM 净利       越大越好
    价值       ep_ttm         1/PE_TTM                        越大越好
               bp             1/PB                            越大越好
    拥挤度     turnover_20    20 日均换手率                    见说明（反向）
               holder_chg     股东户数同比变化                 越小越好（户数减少=筹码集中）
"""
from dataclasses import dataclass, field
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

from database.config import FROZEN_ROOT, connect_duckdb

# 报告期数据在公告后多少天生效（留 1 天缓冲，避免"当天公告当天就能用"）
DEFAULT_LAG_DAYS = 1

# 因子名 -> (中文名, 方向)  方向 +1 表示越大越好，-1 表示越小越好
FACTOR_META: Dict[str, tuple] = {
    "rev_yoy":      ("营收同比增速", +1),
    "np_yoy":       ("归母净利同比增速", +1),
    "np_accel":     ("净利增速加速度", +1),
    "sue":          ("标准化预期外盈利", +1),
    "roe_ttm":      ("TTM ROE", +1),
    "gross_margin": ("毛利率", +1),
    "gm_yoy_chg":   ("毛利率同比变化", +1),
    "ocf_to_np":    ("现金流/净利", +1),
    "ep_ttm":       ("盈利收益率 1/PE", +1),
    "bp":           ("账面市值比 1/PB", +1),
    "turnover_20":  ("20日换手率", -1),
    "holder_chg":   ("股东户数同比变化", -1),
}

PROFIT_COLS = ["revenue", "oper_cost", "n_income_attr_p", "n_income"]
BALANCE_COLS = ["total_hldr_eqy_exc_min_int", "total_assets"]
CASHFLOW_COLS = ["n_cashflow_act"]


# ============================================================
# 读取与去重
# ============================================================
def _read_statements(kind: str, columns: List[str]) -> pd.DataFrame:
    """读一类报表（profit/balance/cashflow），解析日期，按最早公告去重

    返回长表: code, ann_date, end_date, <columns>
    """
    empty = pd.DataFrame(columns=["code", "ann_date", "end_date"] + columns)
    fin_dir = FROZEN_ROOT / "financial"
    # ⚠️ 两件事必须做对：
    #   1. glob 必须**限定到 {kind}_**（profit/balance/cashflow 三张表在同一个
    #      目录里，列完全不同；用 `year=*/*.parquet` 配 union_by_name 会把三张
    #      表的行混成一张）
    #   2. 没有 db/ 时要返回空表而不是让 DuckDB 抛
    #      `IOException: No files found`（这里的 try 只有 finally，异常会冒出去）
    if not any(fin_dir.glob(f"year=*/{kind}_*.parquet")):
        return empty
    con = connect_duckdb()
    glob = f"{fin_dir.as_posix()}/year=*/{kind}_*.parquet"
    cols = ", ".join(["code", "ann_date", "end_date", "report_type"] + columns)
    try:
        df = con.execute(
            f"SELECT {cols} FROM read_parquet('{glob}', union_by_name=True) "
            f"WHERE report_type = '1'").fetchdf()
    finally:
        con.close()
    if df.empty:
        return df
    df["code"] = df["code"].astype(str).str.zfill(6)
    # 用统一容错解析：tushare 少数记录的日期是 'YYYY-MM-DD HH:MM:SS'，
    # 写死 format="%Y%m%d" 会静默变 NaT 并在下一步被 dropna 丢掉。
    from database.dates import parse_tushare_date
    df["ann_date"] = parse_tushare_date(df["ann_date"])
    df["end_date"] = parse_tushare_date(df["end_date"])
    df = df.dropna(subset=["ann_date", "end_date"])
    # 同一 (code, end_date) 多次公告 -> 保留最早那次（as-reported，杜绝用后来的修正值）
    df = (df.sort_values(["code", "end_date", "ann_date"])
            .drop_duplicates(["code", "end_date"], keep="first"))
    return df[["code", "ann_date", "end_date"] + columns].reset_index(drop=True)


def _safe_div(a, b):
    """安全除法：分母为 0 / NaN 时给 NaN，而不是 inf

    支持 Series / DataFrame / 标量的任意组合。
    要点：**先判断是不是 pandas 对象**，是就走 pandas 的广播与对齐，
    绝不对 Series 调 np.isfinite（会抛 "truth value of a Series is ambiguous"）。
    """
    is_pd = (pd.Series, pd.DataFrame)
    if isinstance(a, is_pd) or isinstance(b, is_pd):
        if isinstance(b, is_pd):
            b = b.astype(float)
        out = a / b
        if isinstance(out, is_pd):
            out = out.replace([np.inf, -np.inf], np.nan)
        return out
    try:
        b = float(b)
    except (TypeError, ValueError):
        return np.nan
    if not np.isfinite(b) or b == 0:
        return np.nan
    try:
        return float(a) / b
    except (TypeError, ValueError):
        return np.nan


def _single_quarter(ytd: pd.Series, end_date: pd.Series,
                    code: pd.Series) -> pd.Series:
    """把年内累计口径（一季报/H1/三季报/年报）还原成**单季**值

    A 股利润表是年内累计的：二季报报的是 H1 累计，三季报报的是 9 个月累计。
    做 SUE、TTM 这类指标必须先还原成单季，否则"同比"会混入口径错误
    （比如拿 H1 累计去比 Q1 单季）。

    要求输入已按 (code, end_date) 排好序；输出与输入同索引。
    """
    s = pd.Series(np.asarray(ytd, dtype=float), index=ytd.index)
    ed = pd.to_datetime(pd.Series(np.asarray(end_date), index=ytd.index))
    cd = pd.Series(np.asarray(code), index=ytd.index)
    prev = s.groupby([cd, ed.dt.year]).shift(1)
    return (s - prev.fillna(0.0)).reindex(ytd.index)


def _dedupe_public(df: pd.DataFrame) -> pd.DataFrame:
    """按 (code, effective) 去重，同一生效日有多条时保留**报告期最新**的那条"""
    df = df.sort_values(["code", "effective", "end_date"])
    return df.drop_duplicates(["code", "effective"], keep="last")


@dataclass
class FundamentalData:
    """报告期层面的基本面因子表（PIT 对齐前的中间产物）

    reports: 长表，每行 = 一次公告，含 code / ann_date / end_date / 全部因子列
    """
    reports: pd.DataFrame = field(default_factory=pd.DataFrame)
    _wide_cache: Dict[str, pd.DataFrame] = field(default_factory=dict, repr=False)

    # ---------- 构建 ----------
    @classmethod
    def load(cls, verbose: bool = False) -> "FundamentalData":
        if verbose:
            print("  [基本面] 读取三大报表 ...")
        prof = _read_statements("profit", PROFIT_COLS)
        bal = _read_statements("balance", BALANCE_COLS)
        cf = _read_statements("cashflow", CASHFLOW_COLS)
        if prof.empty:
            return cls()
        rep = prof.merge(bal, on=["code", "ann_date", "end_date"], how="left") \
                  .merge(cf, on=["code", "ann_date", "end_date"], how="left")
        rep = rep.sort_values(["code", "end_date"]).reset_index(drop=True)

        # ---- 单季还原（用于 SUE / TTM）----
        for c in ("n_income_attr_p", "n_cashflow_act", "revenue"):
            rep[f"{c}_q"] = _single_quarter(rep[c], rep["end_date"], rep["code"])

        # ---- 同比：按**同一报告期**（去年的同一天）对齐 ----
        # ⚠️ 右表必须把 `end_date` 改名成 `ed_prev` 再做连接键。
        # 一开始写成"从 rep 里取 ed_prev 列"，等于拿 (E-1y) 去 join (E'-1y)，
        # 匹配条件退化成 E'==E —— 永远匹配到自己那一行，同比恒为 0。
        rep["ed_prev"] = rep["end_date"] - pd.DateOffset(years=1)
        prev = rep[["code", "end_date", "revenue", "n_income_attr_p", "oper_cost",
                    "n_income_attr_p_q"]].rename(columns={
            "end_date": "ed_prev", "revenue": "revenue_py",
            "n_income_attr_p": "n_income_attr_p_py", "oper_cost": "oper_cost_py",
            "n_income_attr_p_q": "n_income_attr_p_q_py"})
        rep = rep.merge(prev, on=["code", "ed_prev"], how="left")

        rep["rev_yoy"] = _safe_div(rep["revenue"] - rep["revenue_py"],
                                   rep["revenue_py"].abs())
        rep["np_yoy"] = _safe_div(rep["n_income_attr_p"] - rep["n_income_attr_p_py"],
                                  rep["n_income_attr_p_py"].abs())
        rep["gross_margin"] = _safe_div(rep["revenue"] - rep["oper_cost"], rep["revenue"])
        gm_py = _safe_div(rep["revenue_py"] - rep["oper_cost_py"], rep["revenue_py"])
        rep["gm_yoy_chg"] = rep["gross_margin"] - gm_py

        # ---- 净利增速的加速度：本期 YoY − 上一期 YoY ----
        rep["np_yoy_prev"] = rep.groupby("code")["np_yoy"].shift(1)
        rep["np_accel"] = rep["np_yoy"] - rep["np_yoy_prev"]

        # ---- SUE：单季净利的同比变化 / 过去 8 个季度该变化的标准差 ----
        rep["d_q"] = rep["n_income_attr_p_q"] - rep["n_income_attr_p_q_py"]
        g = rep.groupby("code")["d_q"]
        rep["sue"] = _safe_div(rep["d_q"], g.transform(
            lambda s: s.shift(1).rolling(8, min_periods=4).std(ddof=1)))

        # ---- TTM：最近 4 个单季之和 ----
        rep["np_ttm"] = rep.groupby("code")["n_income_attr_p_q"].transform(
            lambda s: s.rolling(4, min_periods=4).sum())
        rep["ocf_ttm"] = rep.groupby("code")["n_cashflow_act_q"].transform(
            lambda s: s.rolling(4, min_periods=4).sum())
        rep["roe_ttm"] = _safe_div(rep["np_ttm"], rep["total_hldr_eqy_exc_min_int"])
        rep["ocf_to_np"] = _safe_div(rep["ocf_ttm"], rep["np_ttm"].abs())

        # 只保留真正要用的列，减小后续对齐的开销
        # （日频因子 ep_ttm/bp/turnover_20/holder_chg 不在这里，由行情/股东户数单独提供）
        daily_only = ("ep_ttm", "bp", "turnover_20", "holder_chg")
        keep = ["code", "ann_date", "end_date"] + [
            c for c in FACTOR_META if c not in daily_only and c in rep.columns]
        return cls(reports=rep[keep].copy())

    # ---------- PIT 对齐 ----------
    def align(self, name: str, dates: pd.DatetimeIndex, codes,
              lag_days: int = DEFAULT_LAG_DAYS) -> pd.DataFrame:
        """把某个报告期因子对齐成日频宽表（index=交易日, columns=代码）

        ⚠️ 对齐键是 `effective = ann_date + lag_days`，不是 `end_date`。
        """
        dates = pd.DatetimeIndex(dates)
        if self.reports.empty or name not in self.reports.columns:
            return pd.DataFrame(index=dates, columns=list(codes), dtype=float)
        key = (name, int(lag_days))
        wide = self._wide_cache.get(key)
        if wide is None:
            df = self.reports[["code", "ann_date", "end_date", name]].dropna(subset=[name])
            if df.empty:
                return pd.DataFrame(index=dates, columns=list(codes), dtype=float)
            # ⚠️ as-reported 去重必须在这里也做一遍，不能只依赖 load()。
            # 否则只要有人用别的方式构造 reports（比如直接塞 DataFrame），
            # 同一报告期的事后修正值就会在公告前生效 —— 前视偏差。
            df = (df.sort_values(["code", "end_date", "ann_date"])
                    .drop_duplicates(["code", "end_date"], keep="first"))
            df = df.assign(effective=df["ann_date"] + pd.Timedelta(days=int(lag_days)))
            df = _dedupe_public(df)
            wide = df.pivot_table(index="effective", columns="code", values=name,
                                  aggfunc="last").sort_index()
            self._wide_cache[key] = wide
        # 用"并集 + ffill"把不定期公告摊到每个交易日
        idx = wide.index.union(dates)
        w = wide.reindex(idx).ffill().reindex(dates)
        return w.reindex(columns=list(codes))

    def align_many(self, names: List[str], dates: pd.DatetimeIndex, codes,
                   lag_days: int = DEFAULT_LAG_DAYS) -> Dict[str, pd.DataFrame]:
        return {n: self.align(n, dates, codes, lag_days) for n in names}


# ============================================================
# 股东户数（也是 PIT 数据）
# ============================================================
def load_holder_changes(dates: pd.DatetimeIndex, codes,
                        lag_days: int = DEFAULT_LAG_DAYS) -> pd.DataFrame:
    """股东户数同比变化（PIT 对齐）

    户数减少 = 筹码集中（通常偏多）；户数增加 = 筹码分散（偏空）。
    所以因子方向和别的相反：**越小越好**（FACTOR_META 里 direction = -1）。
    """
    con = connect_duckdb()
    glob = f"{(FROZEN_ROOT / 'holders').as_posix()}/year=*/*.parquet"
    try:
        df = con.execute(
            f"SELECT code, ann_date, end_date, holder_num "
            f"FROM read_parquet('{glob}', union_by_name=True)").fetchdf()
    except Exception:
        return pd.DataFrame(index=dates, columns=list(codes), dtype=float)
    finally:
        con.close()
    if df.empty:
        return pd.DataFrame(index=dates, columns=list(codes), dtype=float)

    df["code"] = df["code"].astype(str).str.zfill(6)
    # holders.ann_date 实测约 1.75% 的行是 'YYYY-MM-DD HH:MM:SS' 格式，
    # 写死 format="%Y%m%d" 会静默变 NaT 并被 dropna 丢掉（无声的数据损失）。
    from database.dates import parse_tushare_date
    df["ann_date"] = parse_tushare_date(df["ann_date"])
    df["end_date"] = parse_tushare_date(df["end_date"])
    df = df.dropna(subset=["ann_date", "end_date", "holder_num"])
    df = (df.sort_values(["code", "end_date", "ann_date"])
            .drop_duplicates(["code", "end_date"], keep="first"))
    df["ed_prev"] = df["end_date"] - pd.DateOffset(years=1)
    # ⚠️ 同 fundamental 的同比合并：右表必须把 `end_date` 改名成 `ed_prev` 当连接键。
    # 直接取当前行的 ed_prev 当右表键，会把 (E-1y) 和 (E'-1y) 对上，
    # 匹配条件退化成 E'==E —— 永远匹配自己，holder_chg 恒为 0。
    prev = df[["code", "end_date", "holder_num"]].rename(
        columns={"end_date": "ed_prev", "holder_num": "holder_num_py"})
    df = df.merge(prev, on=["code", "ed_prev"], how="left")
    df["holder_chg"] = _safe_div(df["holder_num"] - df["holder_num_py"],
                                 df["holder_num_py"].abs())
    df = df.dropna(subset=["holder_chg"])
    if df.empty:
        return pd.DataFrame(index=dates, columns=list(codes), dtype=float)
    df = df.assign(effective=df["ann_date"] + pd.Timedelta(days=int(lag_days)))
    df = df.sort_values(["code", "effective", "end_date"])
    df = df.drop_duplicates(["code", "effective"], keep="last")
    wide = df.pivot_table(index="effective", columns="code", values="holder_chg",
                          aggfunc="last").sort_index()
    idx = wide.index.union(pd.DatetimeIndex(dates))
    return wide.reindex(idx).ffill().reindex(pd.DatetimeIndex(dates)) \
               .reindex(columns=list(codes))


# ============================================================
# 日频因子（估值 / 换手）—— 本身就是行情数据，无需 PIT 对齐
# ============================================================
def valuation_factors(panel: dict, window: int = 20) -> Dict[str, pd.DataFrame]:
    """从价量面板里取/派生日频因子：ep_ttm、bp、turnover_20

    这三个用的是**日频行情**（市盈率、市净率、换手率），
    每个交易日都能拿到当日值，不存在公告滞后问题。
    """
    out: Dict[str, pd.DataFrame] = {}
    pe = panel.get("pe_ttm")
    pb = panel.get("pb")
    tr = panel.get("turnover_rate")
    if pe is not None:
        out["ep_ttm"] = _safe_div(1.0, pe)
    if pb is not None:
        out["bp"] = _safe_div(1.0, pb)
    if tr is not None:
        out["turnover_20"] = tr.astype(float).rolling(
            window, min_periods=_min_periods(window)).mean()
    return out


def _min_periods(window: int) -> int:
    """滚动窗口的最少样本数：既不能超过 window，也不能小到没有意义

    踩过的坑：直接写 `max(3, window // 3)`，当 window=2 时得到 3 > 2，
    pandas 直接抛 "min_periods 3 must be <= window 2"。
    """
    w = max(int(window), 1)
    return max(1, min(w, max(3, w // 3)))


# ============================================================
# 统一入口
# ============================================================
def load_all_factors(panel: dict, dates: pd.DatetimeIndex, codes,
                     lag_days: int = DEFAULT_LAG_DAYS,
                     verbose: bool = False) -> Dict[str, pd.DataFrame]:
    """一次产出全部基本面/拥挤度因子（日频宽表）

    panel: backtest.panel_data.load_price_panel 的产出（需要 with_valuation）
    """
    out: Dict[str, pd.DataFrame] = {}
    out.update(valuation_factors(panel))

    fd = FundamentalData.load(verbose=verbose)
    if not fd.reports.empty:
        names = [n for n in FACTOR_META if n in fd.reports.columns]
        out.update(fd.align_many(names, dates, codes, lag_days))
    if verbose:
        print(f"  [基本面] 报告期记录 {len(fd.reports):,} 行，"
              f"因子 {len(out)} 个")

    hc = load_holder_changes(dates, codes, lag_days)
    if not hc.empty and hc.notna().any().any():
        out["holder_chg"] = hc
    return out


def neutralized_score(factors: Dict[str, pd.DataFrame], names: List[str],
                      mask: pd.DataFrame, ind_map=None,
                      mv: pd.DataFrame = None, n_size: int = 5,
                      weights: Dict[str, float] = None) -> pd.DataFrame:
    """把多个因子合成一个横截面打分（越大越看多）

    流程（每一步都只在**当日横截面**内做，不看未来）：
        1. 按 FACTOR_META 的方向统一成"越大越好"
        2. 限定在股票池 mask 内
        3. 取横截面百分位秩（对 np_yoy 那种 P5=-245% 的极端值天然免疫）
        4. 在**行业内**和**市值组内**减去均值（秩中性化）
           —— 不做这一步，"营收增速有效"很可能只是"这个行业这两年好"
        5. 按 weights 加权平均（默认等权）

    `ind_map` 可以是 `pd.Series`（时不变，当前快照）或 `pd.DataFrame`
    （**PIT**，index=日期 columns=代码）。**强烈建议用 PIT 面板**：
    用当前快照做历史中性化是前视，详见 `database/industry.py`。

    ⚠️ 因子权重不在这里优化。样本内优化权重是过拟合的头号来源，
    而且上一轮的实验已经证明（docs/板块轮动改进报告.md）
    "样本内最优"在样本外近乎抽签。
    """
    names = [n for n in names if n in factors and factors[n] is not None]
    if not names:
        raise ValueError("没有可用的因子")
    w = {n: 1.0 for n in names}
    if weights:
        w.update({k: v for k, v in weights.items() if k in names})

    total = None
    wsum = 0.0
    for n in names:
        f = factors[n]
        if FACTOR_META.get(n, ("", 1))[1] < 0:
            f = -f
        r = f.where(mask)
        # 秩在整块面板上按行算，等价于逐日横截面秩
        r = r.rank(axis=1, pct=True)
        if ind_map is not None or mv is not None:
            r = _neutralize_panel(r, ind_map, mv, n_size)
        total = r * w[n] if total is None else total.add(r * w[n], fill_value=0.0)
        wsum += w[n]
    return total / wsum if wsum else total


def _group_demean(r: pd.DataFrame, g: pd.Series) -> pd.DataFrame:
    """按**不随时间变**的分组 g（index=股票代码）在每行内减去组均值

    向量化：转置后 `groupby(g).transform("sum")` 沿日期轴求组内和，
    再除以组内有效个数。NaN 必须先用 0 填充、并用 notna 单独计数，
    否则 NaN 会被当成 0 拉低组均值。

    ⚠️ 踩过的坑：最初写成"逐行 groupby + `r.loc[d, idx] = ...`"，
    11 个因子 × 2596 行 = 2.8 万次带标签列表的 setitem，
    实测 30 分钟跑不完一轮。向量化后是秒级。
    """
    g = g.reindex(r.columns)
    if not g.notna().any():
        return r
    rt = r.T
    filled = rt.fillna(0.0)
    cnt = rt.notna().astype(float)
    num = filled.groupby(g).transform("sum")
    den = cnt.groupby(g).transform("sum")
    mean = (num / den.replace(0.0, np.nan)).T.reindex(columns=r.columns)
    # 分组缺失的列（未分类）不减任何东西，绝不因为缺分组就变 NaN
    return r - mean.fillna(0.0)


def _demean_by_quintile(r: pd.DataFrame, mv: pd.DataFrame,
                        n_size: int = 5) -> pd.DataFrame:
    """按**逐日**市值分位去均值（分位随时间变化，不能按列分组）"""
    q = np.ceil(mv.astype(float).rank(axis=1, pct=True) * n_size)
    q = q.clip(lower=1, upper=n_size)
    filled = r.fillna(0.0)
    cnt = r.notna().astype(float)
    gm = pd.DataFrame(np.nan, index=r.index, columns=r.columns)
    for k in range(1, int(n_size) + 1):
        m = (q == k)
        if not m.any().any():
            continue
        num = filled.where(m).sum(axis=1)
        den = cnt.where(m).sum(axis=1)
        mk = num / den.replace(0.0, np.nan)
        gm = gm.mask(m, mk, axis=0)
    return r - gm.fillna(0.0)


def _group_demean_pit(r: pd.DataFrame, ind_panel: pd.DataFrame) -> pd.DataFrame:
    """按**逐日**行业分组去均值（PIT：行业归属随时间变）

    ⚠️ 为什么不能"逐日 groupby"：这个文件里已经踩过同款坑 ——
    逐行 groupby + 带标签 setitem，11 个因子 × 2596 行 = 2.8 万次操作，
    实测 30 分钟跑不完一轮。

    这里利用一个实测事实：**行业归属只在极少数日期发生变化**
    （全库仅 2,006 次变动，5,906 只股票里 4,260 只从未变过）。
    所以把日期按"行业配置快照"分块，**每块内部仍用原来的向量化实现**，
    块数 ≈ 变动次数，而不是日期数。

    只有"变过行业"的股票参与快照指纹 —— 恒定不变的那些对分组没有贡献，
    把它们排除能把指纹矩阵从 5,900 列降到约 1,600 列。
    """
    p = ind_panel.reindex(index=r.index, columns=r.columns)
    if p.empty:
        return r
    nun = p.nunique(dropna=False)
    varying = nun[nun > 1].index
    if len(varying) == 0:
        # 行业完全不变 -> 退化成时不变分组（快路径）
        g = p.iloc[0]
        g.index = r.columns
        return _group_demean(r, g)

    sub = p[varying].fillna(UNKNOWN_INDUSTRY_NAME).astype(str)
    # 逐列 factorize 后比较整数行：比 np.unique(axis=0) 处理 object 数组快得多
    arr = np.column_stack([pd.factorize(sub[c])[0] for c in sub.columns])
    _, inv = np.unique(arr, axis=0, return_inverse=True)
    out = pd.DataFrame(np.nan, index=r.index, columns=r.columns)
    for k in np.unique(inv):
        rows = r.index[inv == k]
        g = p.loc[rows[0]]
        g.index = r.columns
        out.loc[rows] = _group_demean(r.loc[rows], g)
    return out


# 与 analytics.attribution.UNKNOWN_INDUSTRY 保持一致（避免循环 import）
UNKNOWN_INDUSTRY_NAME = "未分类"


def _neutralize_panel(r: pd.DataFrame, ind_map,
                      mv: pd.DataFrame, n_size: int = 5) -> pd.DataFrame:
    """对整块秩面板做行业 + 市值中性（逐行去组均值）

    `ind_map` 支持两种形态：
      - `pd.Series`（index=代码）：**时不变**行业归属（当前快照）。
      - `pd.DataFrame`（index=日期, columns=代码）：**PIT** 行业面板，
        每个调仓日用它**当天**的行业归属。
        ⚠️ 用当前快照做历史中性化是前视（B7）：换过行业的股票会拿到
        它当时还不属于的那个行业的均值。全库有 1,646 只股票换过行业。
    """
    if ind_map is not None and len(ind_map):
        if isinstance(ind_map, pd.DataFrame):
            r = _group_demean_pit(r, ind_map)
        else:
            r = _group_demean(r, ind_map)
    if mv is not None and not mv.empty:
        r = _demean_by_quintile(r, mv.reindex(index=r.index, columns=r.columns),
                                n_size)
    return r


def combine_by_rule(factor_table: pd.DataFrame, max_p: float = 0.05,
                    min_spread: float = 0.0,
                    min_mono: float = 0.5) -> List[str]:
    """按**只看样本内**的预注册规则挑因子

    规则：样本内 p < max_p  且  样本内多空 > min_spread  且  样本内单调性 > min_mono

    ⚠️ 这三个条件必须全部来自样本内统计。一旦掺入样本外的数字，
    整个"样本外验证"就退化成事后选择，结论不可信。
    """
    need = {"内p", "内多空年化", "内单调性"}
    if not need.issubset(factor_table.columns):
        raise KeyError(f"因子表缺少列 {need - set(factor_table.columns)}")
    ok = ((factor_table["内p"] < max_p)
          & (factor_table["内多空年化"] > min_spread)
          & (factor_table["内单调性"] > min_mono))
    return list(factor_table.loc[ok, "因子"])


def factor_summary(factors: Dict[str, pd.DataFrame]) -> pd.DataFrame:
    """每个因子的覆盖率与量纲概览（用来快速发现数据问题）"""
    rows = []
    for name, df in factors.items():
        if df is None or df.empty:
            continue
        v = df.stack()
        rows.append({
            "因子": name,
            "中文名": FACTOR_META.get(name, ("", 0))[0],
            "方向": FACTOR_META.get(name, ("", 0))[1],
            "覆盖率": float(df.notna().mean().mean()),
            "均值": float(v.mean()), "标准差": float(v.std()),
            "P1": float(v.quantile(0.01)), "P99": float(v.quantile(0.99)),
        })
    return pd.DataFrame(rows)
