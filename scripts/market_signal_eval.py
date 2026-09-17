# -*- coding: utf-8 -*-
"""市场级信号有没有预测力？—— 用数据说话，而不是假设

`factors/market.py` 提供的四个信号（两融、北向、股指期货基差、期权 PCR）都是
业界常提的择时指标。但"常提"不等于"在本库的数据与区间上有效"。这个脚本做
最小但诚实的检验：

  1. **单信号 IC**：信号（已 PIT lag）与指数未来 N 日收益的秩相关
  2. **t 值**：IC 序列的 Newey-West 粗略 t（重叠窗口会高估显著性，所以用
     **不重叠**的抽样做一次对照）
  3. **分层**：按信号分 3 档，看各档未来的年化收益 —— 单调性比 IC 更直观
  4. **样本内外**：2018-2022 / 2023- 两段分别看，避免"只在某一段有效"

结论无论好坏都照实打印。若某个信号没有预测力，脚本会明说，并给出
"不要把它接进仓位管理"的建议。

用法:
    python scripts/market_signal_eval.py
    python scripts/market_signal_eval.py --start 2018-01-01 --end 2026-09-16
"""
import argparse
import sys

sys.path.insert(0, ".")

import numpy as np
import pandas as pd

from database.loader import load_index_daily
from factors.evaluation import _corr          # rank+Pearson 版 Spearman，不依赖 scipy
from factors.market import market_signals

SIGNAL_DESC = {
    "margin_chg": "两融余额20日变化率（杠杆资金）",
    "north_net": "北向资金20日均值（外资流向）",
    "IF_basis": "沪深300股指期货基差",
    "IC_basis": "中证500股指期货基差",
    "IM_basis": "中证1000股指期货基差",
    "pcr_vol": "期权认沽认购比（成交量）",
    "pcr_oi": "期权认沽认购比（持仓量）",
}
# 方向先验：基差为正（升水）偏多；PCR 高（买沽多）偏空 -> 取负
SIGN_PRIOR = {"pcr_vol": -1, "pcr_oi": -1}


def _rank_ic(x, y):
    """秩相关（Spearman），样本不足返回 nan"""
    d = pd.concat([x, y], axis=1).dropna()
    if len(d) < 30:
        return np.nan, 0
    return _corr(d.iloc[:, 0], d.iloc[:, 1]), len(d)


def _ic_tstat(sig: pd.Series, fwd: pd.Series, block: int) -> float:
    """**不重叠**分块的 IC 均值 t 值

    重叠窗口（如用未来 20 日收益、每天算一次 IC）会让样本高度自相关，
    朴素 t 值能虚高好几倍。这里每 `block` 天取一个观测，做一次对照。
    """
    d = pd.concat([sig, fwd], axis=1).dropna()
    if len(d) < block * 20:
        return np.nan
    d = d.iloc[::block]
    x, y = d.iloc[:, 0], d.iloc[:, 1]
    if x.std() == 0 or y.std() == 0:
        return np.nan
    r = x.corr(y)                       # 分块后样本独立，用 Pearson 即可
    n = len(d)
    return float(r * np.sqrt((n - 2) / max(1e-12, 1 - r ** 2)))


def main():
    ap = argparse.ArgumentParser(description="市场级信号的预测力检验")
    ap.add_argument("--start", default="2018-01-01")
    ap.add_argument("--end", default=None)
    ap.add_argument("--horizon", type=int, default=20,
                    help="未来收益窗口（交易日），默认 20")
    ap.add_argument("--index", default="000300.SH")
    ap.add_argument("--split", default="2023-01-01", help="样本内外分界")
    args = ap.parse_args()

    sig = market_signals(args.start, args.end)
    if sig.empty:
        print("没取到任何市场级信号")
        return 1
    print(f"信号表: {sig.shape[0]} 个交易日 × {sig.shape[1]} 列  "
          f"({str(sig.index.min())[:10]} ~ {str(sig.index.max())[:10]})")

    idx = load_index_daily(args.index, args.start, args.end)
    if idx.empty:
        print(f"取不到指数 {args.index} 行情")
        return 1
    close = idx.set_index("trade_date")["close"].sort_index()
    fwd = close.shift(-args.horizon) / close - 1.0
    print(f"基准: {args.index}  {len(close)} 个交易日，未来 {args.horizon} 日收益\n")

    rows = []
    for c in sig.columns:
        if c in ("margin_rzrq",):       # 绝对水平，不是信号
            continue
        s = sig[c] * SIGN_PRIOR.get(c, 1)
        ic_all, n = _rank_ic(s, fwd)
        t_block = _ic_tstat(s, fwd, args.horizon)
        in_s = s[s.index < pd.Timestamp(args.split)]
        out_s = s[s.index >= pd.Timestamp(args.split)]
        ic_in, _ = _rank_ic(in_s, fwd.reindex(in_s.index))
        ic_out, _ = _rank_ic(out_s, fwd.reindex(out_s.index))
        rows.append({"信号": c, "说明": SIGNAL_DESC.get(c, ""), "样本": n,
                     "IC": ic_all, "分块t": t_block,
                     "IC(样本内)": ic_in, "IC(样本外)": ic_out})
    tab = pd.DataFrame(rows)
    print("=" * 104)
    print("【一、IC 与显著性】IC>0 表示信号越大、未来收益越高（已按先验调过方向）")
    print("=" * 104)
    print(tab.to_string(index=False, float_format=lambda v: f"{v:+.3f}"))

    # 分层：分 3 档看未来年化
    print("\n" + "=" * 104)
    print(f"【二、三档分层】按信号分位分档，各档未来 {args.horizon} 日年化收益")
    print("=" * 104)
    hdr = f"{'信号':<10} {'低档':>10} {'中档':>10} {'高档':>10} {'高-低':>10}  单调?"
    print(hdr)
    mono = {}
    for c in [x for x in sig.columns if x != "margin_rzrq"]:
        s = (sig[c] * SIGN_PRIOR.get(c, 1)).dropna()
        f = fwd.reindex(s.index)
        d = pd.concat([s.rename("s"), f.rename("f")], axis=1).dropna()
        if len(d) < 90:
            print(f"{c:<10} 样本不足（{len(d)}）")
            continue
        q = d["s"].quantile([1 / 3, 2 / 3]).values
        grp = np.digitize(d["s"], q)
        ann = {}
        for g, name in zip((0, 1, 2), ("低", "中", "高")):
            v = d["f"][grp == g]
            # 未来 h 日收益 -> 年化（每 h 天一个观测，一年约 252/h 个）
            ann[name] = float((1 + v.mean()) ** (252.0 / args.horizon) - 1) if len(v) else np.nan
        diff = ann["高"] - ann["低"]
        mono[c] = (ann["低"], ann["中"], ann["高"])
        if ann["低"] <= ann["中"] <= ann["高"]:
            ok = "递增"
        elif ann["低"] >= ann["中"] >= ann["高"]:
            ok = "递减"
        else:
            ok = "否"
        print(f"{c:<10} {ann['低']:>+9.1%} {ann['中']:>+9.1%} {ann['高']:>+9.1%} "
              f"{diff:>+9.1%}  {ok}")

    # 结论
    print("\n" + "=" * 104)
    print("【三、结论】门槛：全样本与样本内 |IC| >= 0.05、分块 |t| >= 2、样本内外同号")
    print("=" * 104)
    usable = []
    for r in rows:
        c = r["信号"]
        if np.isnan(r["IC"]) or np.isnan(r["分块t"]):
            print(f"  ⚠️  {c:<10} 样本不足，无法判断")
            continue
        cond_ic = abs(r["IC"]) >= 0.05
        cond_t = abs(r["分块t"]) >= 2.0
        cond_is = (not np.isnan(r["IC(样本内)"])) and abs(r["IC(样本内)"]) >= 0.05
        cond_oos = (not np.isnan(r["IC(样本外)"])) and \
                   (np.sign(r["IC(样本外)"]) == np.sign(r["IC"]))
        # 样本内也要过门槛：只在样本外有效的信号，等于「回测期看着好」，
        # 无法区分是真规律还是这段时间的风格恰好如此。
        ok = cond_ic and cond_t and cond_is and cond_oos
        why = []
        if not cond_ic:
            why.append(f"全样本|IC|={abs(r['IC']):.3f}<0.05")
        if not cond_is:
            why.append(f"样本内|IC|={abs(r['IC(样本内)']):.3f}<0.05")
        if not cond_t:
            why.append(f"|分块t|={abs(r['分块t']):.2f}<2")
        if not cond_oos:
            why.append("样本外符号不一致")
        print(f"  {'✅' if ok else '❌'} {c:<10} " + ("可用" if ok else "不满足：" + "、".join(why)))
        if ok:
            usable.append(c)
    if usable:
        print(f"\n  可用信号: {', '.join(usable)}")
        print("  -> 可以用 factors.market.exposure_from_signal + apply_exposure 接进仓位管理")
    else:
        print("\n  没有任何信号达到门槛。**不要**把它们接进仓位管理 —— 那只会增加"
              "参数与过拟合风险，不会提高收益。")
    print("\n说明：因子/信号在这里是市场级单序列检验，样本量远小于横截面因子，"
          "\n      所以门槛只是「有没有资格继续看」，不是「已验证有效」。")
    return 0


if __name__ == "__main__":
    sys.exit(main())
