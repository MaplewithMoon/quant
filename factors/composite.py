# -*- coding: utf-8 -*-
"""因子合成：把多个单因子合成一个综合打分

为什么要合成
------------
单因子的 IC 通常只有 0.03~0.05，且稳定性有限。把多个**低相关**的因子合成，
可以在不显著增加风险的情况下提升 ICIR。

合成方式
--------
    equal      等权（各自标准化后取平均）—— 最稳健，不容易过拟合权重
    ic_weight  按 |IC 均值| 加权
    icir_weight 按 ICIR 加权 —— 更看重稳定性而非单纯预测力

流程
----
    1) 每个因子先按 direction 统一方向（都变成"越大越看多"）
    2) 逐日横截面 z-score 标准化（可选缩尾）
    3) 按权重加权平均

⚠️ 权重必须用**训练段**的 IC 计算，用全样本 IC 加权会引入前视。
   `composite_score(..., ic_stats=...)` 允许你传入训练段统计量；
   不传则用全样本（探索阶段可用，正式评估必须用训练段）。
"""
from typing import Dict

import numpy as np
import pandas as pd

from .base import FACTORS, get_factor
from .evaluation import ic_series, ic_stats
from .panel import forward_returns, adjusted_close


def _standardize(f: pd.DataFrame, mask: pd.DataFrame = None,
                 clip: float = 3.0) -> pd.DataFrame:
    x = f.where(mask) if mask is not None else f
    mu = x.mean(axis=1)
    sd = x.std(axis=1, ddof=0)
    z = x.sub(mu, axis=0).div(sd.replace(0, np.nan), axis=0)
    return z.clip(-clip, clip) if clip else z


def composite_score(panel: dict, names: list, method: str = "equal",
                    panel_mask: pd.DataFrame = None, clip: float = 3.0,
                    ic_stats_map: Dict[str, dict] = None,
                    return_detail: bool = False):
    """合成多个因子

    参数:
        panel:         load_panel 的产出
        names:         因子名列表
        method:        'equal' / 'ic_weight' / 'icir_weight'
        panel_mask:    股票池掩码（只在池内做横截面标准化）
        ic_stats_map:  {因子名: ic_stats 的 dict}；不传则用全样本 IC（有前视，仅探索用）
        return_detail: True 时额外返回各因子的标准化值与权重

    返回:
        score（宽表）或 (score, detail)
    """
    if not names:
        raise ValueError("因子列表为空")

    zs, dirs = {}, {}
    for nm in names:
        fac = get_factor(nm)
        raw = fac.compute(panel)
        z = _standardize(raw, panel_mask, clip=clip)
        # 统一方向：direction=-1 的因子取负，使"越大越看多"对所有因子成立
        zs[nm] = z * (1 if fac.direction > 0 else -1)
        dirs[nm] = fac.direction

    # ---- 权重 ----
    if method == "equal":
        weights = {nm: 1.0 / len(names) for nm in names}
    else:
        stats = {}
        if ic_stats_map:
            stats = {nm: ic_stats_map[nm] for nm in names if nm in ic_stats_map}
        if not stats:
            close = adjusted_close(panel)
            fwd = forward_returns(close, periods=1, lag=1)
            for nm in names:
                # 用统一方向后的 z 值算 IC，这样 IC 应为正
                ic = ic_series(zs[nm], fwd, min_count=20)
                stats[nm] = ic_stats(ic)
        if method == "ic_weight":
            raw_w = {nm: abs(stats.get(nm, {}).get("IC均值", 0.0) or 0.0) for nm in names}
        elif method == "icir_weight":
            raw_w = {nm: abs(stats.get(nm, {}).get("ICIR", 0.0) or 0.0) for nm in names}
        else:
            raise ValueError(f"未知合成方式 {method!r}")
        tot = sum(raw_w.values())
        weights = ({nm: 1.0 / len(names) for nm in names} if tot <= 0
                   else {nm: w / tot for nm, w in raw_w.items()})

    score = None
    for nm, w in weights.items():
        part = zs[nm] * w
        score = part if score is None else score.add(part, fill_value=0)
    score = score.reindex(index=panel["close"].index, columns=panel["close"].columns)

    if return_detail:
        return score, {"weights": weights, "z": zs, "directions": dirs,
                       "method": method}
    return score


def correlation_matrix(panel: dict, names: list, mask: pd.DataFrame = None,
                       min_count: int = 20) -> pd.DataFrame:
    """因子间的横截面相关（按日计算再取均值）

    合成前应确认因子间相关性不过高（>0.7 说明信息重复）。

    min_count: 当日有效股票数少于该值则不参与统计。若整个样本没有一天达标，
               返回的相关系数为 NaN——调用方应显式检查，不要拿 NaN 去比较大小。
    """
    zs = {nm: _standardize(get_factor(nm).compute(panel) *
                           (1 if get_factor(nm).direction > 0 else -1), mask)
          for nm in names}
    mat = pd.DataFrame(np.eye(len(names)), index=names, columns=names)
    for i, a in enumerate(names):
        for b in names[i + 1:]:
            x, y = zs[a], zs[b]
            # ⚠️ 不同因子的列空间可能不同（如 bp 依赖 daily_basic，覆盖的股票比
            # vol_60 少）。直接 `x.notna() & y.notna()` 会按列取并集，缺失列被填成
            # NaN，bool 掩码被提升成 float64 —— 后续用它做布尔索引会抛
            # "Cannot mask with non-boolean array containing NA / NaN values"。
            # 先对齐到公共列空间，掩码就是纯 bool。
            cols = x.columns.union(y.columns)
            xx = x.reindex(columns=cols)
            yy = y.reindex(columns=cols)
            m = (xx.notna() & yy.notna()).to_numpy(dtype=bool)
            xv = xx.to_numpy(dtype=float)
            yv = yy.to_numpy(dtype=float)
            cs = []
            for k, d in enumerate(xx.index):
                mk = m[k]
                if int(mk.sum()) < min_count:
                    continue
                xs, ys = xv[k][mk], yv[k][mk]
                if xs.std() == 0 or ys.std() == 0:
                    continue
                cs.append(float(np.corrcoef(xs, ys)[0, 1]))
            v = float(np.nanmean(cs)) if cs else np.nan
            mat.loc[a, b] = mat.loc[b, a] = v
    return mat
