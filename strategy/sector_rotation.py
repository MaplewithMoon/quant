# -*- coding: utf-8 -*-
"""板块动量轮动策略（中低频，月度调仓）

【策略逻辑】
    1. 把全市场按行业分类切成 N 个板块（本库为 tushare 行业分类，110 个细分行业）
    2. 每个板块用**成分股等权**合成一条板块指数（日频）
    3. 板块动量 = 过去 `lookback` 日收益（跳过最近 `skip` 日，规避短期反转），
       可选地除以板块波动率做风险调整
    4. 每期选动量最强的 `top_sectors` 个板块
    5. 在选中板块内部按次级因子（流动性/低波动）选 `stocks_per_sector` 只股票
    6. 板块间等权（或按动量强弱加权），板块内等权

【为什么这样设计】
    - 库里 `frozen/index_daily` 只有 4 个宽基指数，**没有申万行业指数行情**
      （`frozen/industry/sw_l1.parquet` 只有 31 个行业代码，没有价格）。
      所以板块指数必须由成分股自己合成 —— 这也让"板块"的定义完全可追溯。
    - 策略输出的是**目标权重面板**（宽表），交给 `PortfolioBacktestEngine` 执行，
      A股制度约束（T+1/涨跌停/整手/费用）全部由执行层负责，策略层不碰。

【无未来函数】
    - 调仓日 t 的动量只用 ≤ t 的数据
    - 引擎默认 `next_open`：t 日收盘算权重 → t+1 开盘成交
    - 板块内选股用的次级因子同样是截至 t 的数据
"""
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from portfolio.construction import cap_weights, rebalance_dates

TRADING_DAYS = 252


# ============================================================
# 参数
# ============================================================
@dataclass
class SectorRotationSpec:
    """板块轮动参数（可被 optimizer 网格搜索的那些）"""
    lookback: int = 60            # 板块动量回看窗口（交易日）
    skip: int = 5                 # 跳过最近 N 日（规避短期反转）
    top_sectors: int = 5          # 持有板块数
    stocks_per_sector: int = 5    # 每个板块持股数（0=板块内全部等权）
    vol_adjust: bool = True       # 动量 / 板块波动率（风险调整动量）
    min_sectors: int = 30         # 当日有效板块数少于此值则空仓（数据不健康时保护）
    min_stocks_per_sector: int = 5   # 板块内有效股票数下限，低于此值该板块不可选
    secondary: str = "amount"     # 板块内选股的次级因子: amount / low_vol / none
    secondary_window: int = 20    # 次级因子回看窗口
    direction: str = "momentum"   # momentum=买最强板块 / reversal=买最弱板块
    sector_weighting: str = "equal"   # equal / momentum
    max_weight: float = 0.0       # 单票权重上限
    rebalance: str = "M"          # D / W / M / 整数

    # ---- 非寻优字段（执行/口径） ----
    exclude_sectors: Tuple[str, ...] = ()   # 需要剔除的板块（如"综合"这类壳公司聚集地）

    def describe(self) -> str:
        parts = [f"动量{self.lookback}日", f"跳过{self.skip}日",
                 f"前{self.top_sectors}板块", f"每板块{self.stocks_per_sector}只",
                 f"调仓{self.rebalance}"]
        if self.direction == "reversal":
            parts.append("方向=反转")
        if self.vol_adjust:
            parts.append("风险调整")
        if self.secondary != "none":
            parts.append(f"次级因子={self.secondary}")
        if self.sector_weighting != "equal":
            parts.append(f"板块加权={self.sector_weighting}")
        return " / ".join(parts)


# ============================================================
# 板块划分
# ============================================================
def load_sector_map(industry_map: pd.DataFrame = None) -> pd.Series:
    """股票代码(6位) -> 板块名

    来源 `frozen/industry/stock_industry.parquet`（tushare 行业分类，110 个细分行业）。

    ⚠️ 这是**当前**的行业归属，没有历史变更记录。用它做回测会有一点点前视
    （行业重新分类通常发生在年报后，且极少改变个股的市场属性），
    但比"用当前指数成分"那种幸存者偏差要轻微得多。要彻底消除需要
    逐期行业成分快照，本库暂未采集。
    """
    if industry_map is None:
        from database.loader import load_industry_map
        industry_map = load_industry_map()
    if industry_map is None or industry_map.empty:
        return pd.Series(dtype=object)
    df = industry_map.copy()
    if "code" not in df.columns:
        df["code"] = df["ts_code"].astype(str).str.split(".").str[0]
    df["code"] = df["code"].astype(str).str.zfill(6)
    df = df.dropna(subset=["industry"])
    return df.drop_duplicates("code").set_index("code")["industry"]


def _sector_columns(sector_map: pd.Series, codes) -> Dict[str, List[str]]:
    """板块名 -> 该板块在本面板中的列"""
    s = sector_map.reindex(list(codes)).dropna()
    return {str(name): list(grp.index) for name, grp in s.groupby(s)}


def _returns(panel: dict) -> pd.DataFrame:
    """优先用复权价算收益（分红送转不会造成假跳空）"""
    price = panel.get("close_adj")
    if price is None:
        price = panel["close"]
    return price.astype(float).pct_change(fill_method=None)


def sector_return_panel(panel: dict, sector_map: pd.Series,
                        min_stocks: int = 1) -> pd.DataFrame:
    """板块日收益（成分股等权平均），返回 宽表 index=日期, columns=板块"""
    ret = _returns(panel)
    if ret.empty:
        return pd.DataFrame()
    cols_by_sector = _sector_columns(sector_map, ret.columns)
    if not cols_by_sector:
        return pd.DataFrame()
    data = {}
    for name, cols in cols_by_sector.items():
        sub = ret[cols]
        cnt = sub.notna().sum(axis=1)
        # 板块内有效股票太少时该日置 NaN，避免 3 只票代表一个板块
        data[name] = sub.mean(axis=1).where(cnt >= min_stocks)
    return pd.DataFrame(data, index=ret.index).sort_index()


def sector_count_panel(panel: dict, sector_map: pd.Series) -> pd.DataFrame:
    """每个板块每日的有效成分股数（用于做可投资性过滤）"""
    ret = _returns(panel)
    cols_by_sector = _sector_columns(sector_map, ret.columns)
    return pd.DataFrame({name: ret[cols].notna().sum(axis=1)
                         for name, cols in cols_by_sector.items()},
                        index=ret.index).sort_index()


def sector_index(sector_ret: pd.DataFrame) -> pd.DataFrame:
    """板块等权指数（起点 1.0）

    注意：这是"每日再平衡到等权"的合成指数，与官方申万行业指数
    （自由流通市值加权）不同。它衡量的是板块的**平均个股表现**，
    对动量轮动而言口径一致、可比性更好。
    """
    if sector_ret.empty:
        return sector_ret
    return (1.0 + sector_ret.fillna(0.0)).cumprod()


@dataclass
class SectorData:
    """预计算的板块数据（板块日收益 / 每日有效成分股数）

    为什么要有这个类：`sector_return_panel` 每次都要对 ~5000 列做分组求均值，
    一次约 10~20 秒。网格搜索要跑几十组参数，而板块收益**只跟面板和分类有关、
    跟策略参数无关**，理应只算一次。把结果挂在 SectorData 上在参数间复用。

    用法：
        sd = SectorData.build(panel, sector_map, min_stocks=5)
        mom = sd.momentum(lookback=60, skip=5, vol_adjust=True)
    """
    sector_ret: pd.DataFrame
    count: pd.DataFrame = field(default_factory=pd.DataFrame)
    _index: pd.DataFrame = field(default=None, repr=False)

    @classmethod
    def build(cls, panel: dict, sector_map: pd.Series,
              min_stocks: int = 1) -> "SectorData":
        ret = _returns(panel)
        cols_by_sector = _sector_columns(sector_map, ret.columns)
        if not cols_by_sector:
            return cls(pd.DataFrame(), pd.DataFrame())
        sret, cnt = {}, {}
        for name, cols in cols_by_sector.items():
            sub = ret[cols]
            c = sub.notna().sum(axis=1)
            sret[name] = sub.mean(axis=1).where(c >= min_stocks)
            cnt[name] = c
        return cls(pd.DataFrame(sret, index=ret.index).sort_index(),
                   pd.DataFrame(cnt, index=ret.index).sort_index())

    @property
    def index(self) -> pd.DataFrame:
        if self._index is None:
            self._index = sector_index(self.sector_ret)
        return self._index

    def momentum(self, lookback: int = 60, skip: int = 5,
                 vol_adjust: bool = True, min_stocks: int = 5) -> pd.DataFrame:
        """板块动量打分（宽表 index=日期, columns=板块）"""
        sret = self.sector_ret
        if sret.empty:
            return pd.DataFrame()
        cnt = self.count
        keep = cnt.columns[(cnt >= min_stocks).any()] if not cnt.empty else sret.columns
        idx = self.index[keep]
        mom = idx.shift(skip) / idx.shift(skip + lookback) - 1.0
        if vol_adjust:
            vol = (sret[keep].rolling(lookback, min_periods=max(10, lookback // 3))
                   .std(ddof=1) * np.sqrt(TRADING_DAYS))
            mom = mom / vol.replace(0, np.nan)
        # 当日有效股票数不足的板块置 NaN
        if not cnt.empty:
            early = cnt[keep].rolling(lookback, min_periods=1).min() < min_stocks
            mom = mom.mask(early)
        return mom


# ============================================================
# 动量打分
# ============================================================
def sector_momentum(panel: dict, sector_map: pd.Series, lookback: int = 60,
                    skip: int = 5, vol_adjust: bool = True,
                    min_stocks: int = 5,
                    sector_data: "SectorData" = None) -> pd.DataFrame:
    """板块动量打分（宽表 index=日期, columns=板块）

    动量 = idx[t-skip] / idx[t-skip-lookback] - 1
    vol_adjust=True 时再除以板块年化波动率（信息比率形式）。

    ⚠️ 全部只用 ≤ t 的数据：`shift(skip)` 用到的最后一根 bar 是 t-skip ≤ t。

    传 `sector_data` 可复用预计算的板块收益（网格搜索时能省掉大部分耗时）。
    """
    sd = sector_data or SectorData.build(panel, sector_map, min_stocks=min_stocks)
    return sd.momentum(lookback=lookback, skip=skip,
                       vol_adjust=vol_adjust, min_stocks=min_stocks)


def select_sectors(momentum: pd.DataFrame, top_k: int,
                   min_sectors: int = 30) -> pd.DataFrame:
    """每个交易日选动量前 top_k 的板块（宽表 bool）

    当日有效板块数 < min_sectors 时全 False（数据缺失/市场异常时不开仓）。
    """
    if momentum.empty:
        return pd.DataFrame(dtype=bool)
    valid = momentum.notna()
    n_valid = valid.sum(axis=1)
    rank = momentum.where(valid).rank(axis=1, ascending=False, method="first")
    sel = (rank <= top_k) & valid.values & (n_valid >= min_sectors).values[:, None]
    return sel.fillna(False).astype(bool)


# ============================================================
# 目标权重
# ============================================================
def _secondary_score(panel: dict, spec: SectorRotationSpec,
                     codes) -> pd.DataFrame:
    """板块内选股用的次级因子（越大越优先）"""
    if spec.secondary == "none":
        return pd.DataFrame(1.0, index=panel["close"].index, columns=codes)
    w = max(int(spec.secondary_window), 1)
    if spec.secondary == "amount":
        amt = panel.get("amount")
        if amt is None:
            return pd.DataFrame(1.0, index=panel["close"].index, columns=codes)
        return amt.reindex(columns=codes).rolling(w, min_periods=max(3, w // 3)).mean()
    if spec.secondary == "low_vol":
        vol = _returns(panel).reindex(columns=codes).rolling(
            w, min_periods=max(3, w // 3)).std(ddof=1)
        return -vol                       # 波动越低分越高
    raise ValueError(f"未知次级因子 {spec.secondary!r}（amount/low_vol/none）")


@dataclass
class RotationPlan:
    """一次轮动决策的明细（用于报告与复盘）"""
    weights: pd.DataFrame                       # 目标权重面板（非调仓日 NaN）
    selected_sectors: pd.DataFrame              # 调仓日选中的板块（date × sector, bool）
    momentum: pd.DataFrame                      # 板块**打分**（date × sector）
                                                # ⚠️ direction="reversal" 时这里是取负后的
                                                # 选股打分，不是原始动量
    holding_detail: pd.DataFrame = field(default_factory=pd.DataFrame)  # 调仓日持仓明细
    n_holdings: pd.Series = field(default_factory=lambda: pd.Series(dtype=float))


def build_rotation_weights(panel: dict, mask: pd.DataFrame,
                           sector_map: pd.Series,
                           spec: SectorRotationSpec = None,
                           sector_data: SectorData = None) -> RotationPlan:
    """构造板块轮动的目标权重面板

    参数:
        panel:       `backtest.panel_data.load_price_panel` 的产出
        mask:        `universe.build_universe` 的产出（True=当日可选）
        sector_map:  股票代码 -> 板块名
        spec:        策略参数
        sector_data: 预计算的板块数据（网格搜索时务必传入，否则每组参数都要重算）

    返回:
        RotationPlan（weights 可直接喂给 PortfolioBacktestEngine）
    """
    spec = spec or SectorRotationSpec()
    close = panel["close"]
    dates, codes = close.index, list(close.columns)

    mom = sector_momentum(panel, sector_map, spec.lookback, spec.skip,
                          spec.vol_adjust, spec.min_stocks_per_sector,
                          sector_data=sector_data)
    if mom.empty:
        raise ValueError("板块动量计算为空：检查 sector_map 与 panel 的代码是否对得上")

    for bad in spec.exclude_sectors:
        if bad in mom.columns:
            mom[bad] = np.nan

    # 方向：momentum=买动量最强的板块；reversal=买跌得最惨的板块。
    # 后面选板块和板块间加权都用这个"打分"，所以在这里一次性翻转符号。
    # （A 股实证：行业层面短期反转远强于动量，见 docs/板块轮动策略报告.md）
    if spec.direction == "reversal":
        mom = -mom
    elif spec.direction != "momentum":
        raise ValueError(f"未知方向 {spec.direction!r}（momentum/reversal）")

    sel = select_sectors(mom, spec.top_sectors, spec.min_sectors)
    sec_score = _secondary_score(panel, spec, codes)
    cols_by_sector = _sector_columns(sector_map, codes)

    reb = rebalance_dates(dates, spec.rebalance)
    weights = pd.DataFrame(np.nan, index=dates, columns=codes)
    detail_rows = []

    for d in reb:
        chosen = [s for s in mom.columns if bool(sel.at[d, s])] if d in sel.index else []
        picks: List[str] = []
        for sec in chosen:
            members = cols_by_sector.get(sec, [])
            if not members:
                continue
            ok = mask.loc[d].reindex(members).fillna(False).astype(bool)
            cand = [c for c in members if bool(ok.get(c, False))]
            if len(cand) < spec.min_stocks_per_sector and spec.stocks_per_sector:
                continue                     # 可选标的太少，宁可不配这个板块
            if spec.stocks_per_sector and spec.stocks_per_sector < len(cand):
                sc = sec_score.loc[d].reindex(cand)
                sc = sc.dropna()
                if sc.empty:
                    continue
                cand = list(sc.sort_values(ascending=False).index[:spec.stocks_per_sector])
            picks.extend(cand)
            detail_rows.append({"date": d, "sector": sec,
                                "momentum": float(mom.at[d, sec]),
                                "n_candidates": len(cand)})

        picks = list(dict.fromkeys(picks))          # 去重保序
        if not picks:
            continue

        if spec.sector_weighting == "momentum" and chosen:
            # 板块按动量强弱分配权重。
            # ⚠️ 不能简单用 (m - m.min()) 归一化：动量横向差异一大，最弱的板块
            # 权重会被打到 1e-9，策略悄悄退化成"单板块押注"。
            # 这里先线性映射到 [0,1] 再抬底到 0.5，保证最弱的板块仍拿到
            # 最强板块一半的份额 —— 强弱有别但不过度集中。
            m = mom.loc[d, chosen].dropna()
            if len(m) >= 2 and m.max() > m.min():
                u = 0.5 + 0.5 * (m - m.min()) / (m.max() - m.min())
                sec_w = (u / u.sum()).to_dict()
            elif len(m) == 1:
                sec_w = {m.index[0]: 1.0}
            else:
                sec_w = {s: 1.0 / len(chosen) for s in chosen}
        else:
            sec_w = {s: 1.0 / max(len(chosen), 1) for s in chosen}

        w = {}
        for sec in chosen:
            members = [c for c in cols_by_sector.get(sec, []) if c in picks]
            if not members:
                continue
            for c in members:
                w[c] = sec_w.get(sec, 0.0) / len(members)
        if w:
            weights.loc[d, list(w.keys())] = list(w.values())

    # 归一化 + 单票上限
    row_sum = weights.sum(axis=1)
    weights = weights.div(row_sum.replace(0, np.nan), axis=0)
    if spec.max_weight:
        filled = weights.fillna(0.0)
        weights = cap_weights(filled, spec.max_weight).where(weights.notna())

    detail = pd.DataFrame(detail_rows)
    n_hold = weights.notna().sum(axis=1)
    n_hold = n_hold[n_hold > 0]
    return RotationPlan(weights=weights, selected_sectors=sel, momentum=mom,
                        holding_detail=detail, n_holdings=n_hold)


def sector_momentum_ic(sector_ret: pd.DataFrame, lookback: int = 60,
                       skip: int = 5, rebalance: str = "M",
                       min_sectors: int = 20) -> pd.Series:
    """板块动量的**月度 RankIC**：独立于回测机制的直接检验

    在每个调仓日 t：
        打分 = 过去 lookback 日（跳过最近 skip 日）的板块收益
        目标 = 该板块从 t 到下一个调仓日的收益
        IC   = 打分与目标收益的横截面秩相关（Spearman）

    正 IC 说明动量有效（强者恒强），负 IC 说明是反转。
    这个指标不涉及选股、费用、制度约束，所以能干净地回答
    "板块动量到底有没有预测力"，避免把回测里的其它噪声误当成结论。
    """
    if sector_ret.empty:
        return pd.Series(dtype=float)
    idx = sector_index(sector_ret)
    reb = list(rebalance_dates(sector_ret.index, rebalance))
    rows = {}
    for i, t in enumerate(reb[:-1]):
        nxt = reb[i + 1]
        pos = idx.index.get_loc(t)
        if pos < skip + lookback:
            continue
        score = idx.iloc[pos - skip] / idx.iloc[pos - skip - lookback] - 1.0
        fwd_pos = idx.index.get_indexer([nxt], method="nearest")[0]
        fwd = idx.iloc[fwd_pos] / idx.iloc[pos] - 1.0
        df = pd.DataFrame({"s": score, "f": fwd}).dropna()
        if len(df) < min_sectors:
            continue
        rows[t] = df["s"].rank().corr(df["f"].rank())      # 秩相关，无需 scipy
    return pd.Series(rows, dtype=float).sort_index()


def sector_holding_history(plan: RotationPlan) -> pd.DataFrame:
    """把调仓明细整理成"日期 × 板块 = 持有只数"的表，便于看轮动轨迹"""
    d = plan.holding_detail
    if d is None or d.empty:
        return pd.DataFrame()
    t = d.pivot_table(index="date", columns="sector", values="n_candidates",
                      aggfunc="sum").fillna(0)
    return t


def plan_summary(plan: RotationPlan) -> str:
    """轮动计划的人话摘要"""
    n = plan.n_holdings
    L = ["=" * 74, "板块轮动计划", "=" * 74]
    if n.empty:
        return "\n".join(L + ["  没有产生任何调仓", "=" * 74])
    L.append(f"  调仓次数    : {len(n)}")
    L.append(f"  平均持股数  : {n.mean():.1f}  (最少 {int(n.min())}, 最多 {int(n.max())})")
    L.append(f"  空仓调仓日  : {int((n == 0).sum())}")
    if not plan.holding_detail.empty:
        top = plan.holding_detail.groupby("sector").size().sort_values(ascending=False)
        L.append(f"  被选中过的板块数: {len(top)}")
        L.append("  —— 最常被选中的板块 Top10 ——")
        for sec, cnt in top.head(10).items():
            L.append(f"    {sec:<10} {cnt:>3} 次")
    L.append("=" * 74)
    return "\n".join(L)
