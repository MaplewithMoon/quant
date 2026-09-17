# -*- coding: utf-8 -*-
"""因子有效性门禁（T1·③）

【要解决什么问题】
A8 说的是：`ocf_to_np` / `gm_yoy_chg` 这类因子**样本外 IC 显著，但多空为负、
分层单调性≈0**，目前只靠人工看分层才发现。而项目里已有的选因子规则
（`factors.fundamental.combine_by_rule`）三个条件**全是样本内**：

    ok = (内p < 0.05) & (内多空年化 > 0) & (内单调性 > 0.5)

样本外**没有任何自动检查**。

【四个条件必须**各自独立**返回，不能揉成一个布尔】
报告里要能说清"死在哪个条件上"，否则只会得到一个"不合格"而不知从何改起。

    1. 样本内外同号       —— 符号翻转是最强的失效信号
    2. 分块 t 值          —— 重叠窗口会让朴素 t 虚高数倍，必须用不重叠分块
    3. 样本外分层单调性    —— IC 显著但分层不单调 = 赚不到钱
    4. 样本量下限          —— 样本太少的"显著"没有意义

【两条设计约束（来自实践）】
    - **不静默剔除**：不合格因子保留在结果里但**标红**。静默剔除会造出
      "被审查过的幸存者因子库"，与股票池的幸存者偏差是同构的病。
    - **留豁免通道**：`--no-gate` 可跳过，但报告头部必须显示
      "未通过 / 未执行门禁"，否则大家会绕开它，等于没建。

【分块方式】
按**自然年或半年**切，而不是任意切分 —— 任意切分会引入新的自由度
（"换个分块长度显著性就变了"），而年度切分是外生给定的。
"""
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

# 门槛
MIN_ABS_IC = 0.02          # |IC| 下限
MIN_ABS_T = 2.0            # |分块 t| 下限
MIN_MONO = 0.5             # 分层单调性下限（秩相关）
MIN_OBS = 20               # 每期横截面最少股票数
MIN_PERIODS = 24           # 因子序列最少期数（约两年月度）


def _rank_corr(a: pd.Series, b: pd.Series) -> float:
    """Spearman（先排名再 Pearson，避免 scipy 依赖）"""
    d = pd.concat([a, b], axis=1).dropna()
    if len(d) < 3:
        return np.nan
    x, y = d.iloc[:, 0].rank(), d.iloc[:, 1].rank()
    if x.std() == 0 or y.std() == 0:
        return np.nan
    return float(x.corr(y))


def _ic_series(factor: pd.DataFrame, fwd: pd.DataFrame,
               min_count: int = MIN_OBS) -> pd.Series:
    out = {}
    for d in factor.index.intersection(fwd.index):
        f, r = factor.loc[d], fwd.loc[d]
        m = f.notna() & r.notna()
        if int(m.sum()) < min_count:
            continue
        v = _rank_corr(f[m], r[m])
        if np.isfinite(v):
            out[d] = v
    return pd.Series(out, dtype=float).sort_index()


def _block_t(ic: pd.Series, block: int) -> float:
    """**不重叠**分块的 IC 均值 t 值

    重叠窗口下朴素 t 会虚高好几倍（样本高度自相关）。这里每 `block` 期取一个
    观测再算 t。
    """
    if len(ic) < block * 6:
        return np.nan
    d = ic.iloc[::block]
    n = len(d)
    sd = float(d.std(ddof=1))
    if n < 4 or sd == 0:
        return np.nan
    return float(d.mean() / (sd / np.sqrt(n)))


def _layering(factor: pd.DataFrame, fwd: pd.DataFrame,
              n_q: int = 3, min_count: int = MIN_OBS) -> float:
    """分层的单调性（档位均值与前瞻收益的秩相关）"""
    rows = []
    for d in factor.index.intersection(fwd.index):
        f, r = factor.loc[d], fwd.loc[d]
        m = f.notna() & r.notna()
        if int(m.sum()) < min_count:
            continue
        q = pd.qcut(f[m].rank(method="first"), n_q, labels=False,
                    duplicates="drop")
        g = r[m].groupby(q).mean()
        if len(g) < n_q:
            continue
        rows.append(g)
    if len(rows) < 6:
        return np.nan
    tab = pd.DataFrame(rows)
    return _rank_corr(pd.Series(tab.columns.astype(float)),
                      tab.mean(axis=0).reset_index(drop=True))


def gate_check(factor: pd.DataFrame, fwd_ret: pd.DataFrame,
               split_point=None, horizon: int = 20,
               label: str = "") -> Dict:
    """对单个因子做门槛检查，**四个条件各自独立返回**

    参数:
        factor:    因子宽表（index=日期, columns=股票）
        fwd_ret:   前瞻收益宽表（同形状）—— 用 `factors.panel.forward_returns`
        split_point: 样本内外分界；None = 用前 60% 作样本内

    返回:
        {
          "factor": label,
          "status": "PASS" | "WARN" | "FAIL" | "SKIP",
          "conditions": {名称: {"ok": bool|None, "value": ..., "why": str}},
          "reasons": [未通过的原因...],
        }
        `ok=None` 表示样本不足**无法判定**（既不算通过也不算失败，
        但必须在报告里显示为"未判定"，不能假装通过）。
    """
    res = {"factor": label, "status": "SKIP", "conditions": {}, "reasons": []}
    if factor is None or fwd_ret is None or factor.empty:
        res["reasons"].append("无因子数据")
        return res

    ic_all = _ic_series(factor, fwd_ret)
    if len(ic_all) < MIN_PERIODS:
        res["status"] = "SKIP"
        res["reasons"].append(
            f"有效期数 {len(ic_all)} < {MIN_PERIODS}，无法判定")
        res["conditions"]["样本量"] = {"ok": None, "value": len(ic_all),
                                       "why": "期数不足"}
        return res

    # 分界：默认前 60%
    if split_point is None:
        cut = ic_all.index[int(len(ic_all) * 0.6)]
    else:
        cut = pd.Timestamp(split_point)
    ic_in, ic_out = ic_all[ic_all.index < cut], ic_all[ic_all.index >= cut]

    # ① 样本内外同号
    s_in, s_out = np.sign(ic_in.mean()), np.sign(ic_out.mean())
    ok_sign = bool(np.isfinite(s_in) and np.isfinite(s_out) and s_in == s_out)
    res["conditions"]["样本内外同号"] = {
        "ok": ok_sign, "value": (float(ic_in.mean()), float(ic_out.mean())),
        "why": "" if ok_sign else f"符号翻转 内{s_in:+.0f} vs 外{s_out:+.0f}"}

    # ② 分块 t（按自然年切 -> 用 12 期作为块长，月度因子即一年）
    t_block = _block_t(ic_all, block=12)
    ok_t = bool(np.isfinite(t_block) and abs(t_block) >= MIN_ABS_T)
    res["conditions"]["分块t"] = {
        "ok": ok_t, "value": t_block,
        "why": "" if ok_t else f"|t|={t_block:.2f} < {MIN_ABS_T}"}

    # ③ 样本外分层单调性
    mono = _layering(factor.loc[factor.index >= cut],
                     fwd_ret.loc[fwd_ret.index >= cut])
    ok_mono = bool(np.isfinite(mono) and abs(mono) >= MIN_MONO)
    res["conditions"]["样本外分层单调性"] = {
        "ok": ok_mono, "value": mono,
        "why": "" if ok_mono else f"单调性 {mono:.2f} < {MIN_MONO}"
                                   f"（IC 显著但分层不单调 = 赚不到钱）"}

    # ④ |IC| 下限
    ok_ic = bool(np.isfinite(ic_all.mean()) and abs(ic_all.mean()) >= MIN_ABS_IC)
    res["conditions"]["|IC|下限"] = {
        "ok": ok_ic, "value": float(ic_all.mean()),
        "why": "" if ok_ic else f"|IC|={abs(ic_all.mean()):.3f} < {MIN_ABS_IC}"}

    reasons = [c["why"] for c in res["conditions"].values() if c["why"]]
    n_fail = sum(1 for c in res["conditions"].values() if c["ok"] is False)
    res["reasons"] = reasons
    res["ic_all"] = float(ic_all.mean())
    res["ic_in"] = float(ic_in.mean()) if len(ic_in) else np.nan
    res["ic_out"] = float(ic_out.mean()) if len(ic_out) else np.nan
    res["status"] = "PASS" if n_fail == 0 else ("WARN" if n_fail == 1 else "FAIL")
    return res


def gate_summary(reports: List[Dict]) -> Dict:
    """把多个因子的门禁结果汇总成报告头部要用的 {status, summary, reasons}

    ⚠️ **不剔除不合格因子**：它们只被标记。静默剔除会造出"被审查过的
    幸存者因子库"，与股票池的幸存者偏差是同构的病。
    """
    reports = [r for r in (reports or []) if r]
    if not reports:
        return {}
    n_fail = sum(1 for r in reports if r["status"] == "FAIL")
    n_warn = sum(1 for r in reports if r["status"] == "WARN")
    n_skip = sum(1 for r in reports if r["status"] == "SKIP")
    bad = [r for r in reports if r["status"] in ("FAIL", "WARN")]
    status = "FAIL" if n_fail else ("WARN" if n_warn else "PASS")
    summary = (f"{len(reports)} 个因子：{len(reports) - n_fail - n_warn - n_skip}"
               f" 通过 / {n_warn} 警告 / {n_fail} 不合格 / {n_skip} 未判定")
    reasons = [f"{r['factor']}: {'、'.join(r['reasons'])}" for r in bad[:6]]
    return {"status": status, "summary": summary, "reasons": reasons,
            "reports": reports, "n_fail": n_fail, "n_warn": n_warn,
            "n_skip": n_skip}


__all__ = ["gate_check", "gate_summary", "MIN_ABS_IC", "MIN_ABS_T",
           "MIN_MONO", "MIN_OBS", "MIN_PERIODS"]
