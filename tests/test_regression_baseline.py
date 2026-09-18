# -*- coding: utf-8 -*-
"""回归基准（T4·D4）：固定数据 + 固定策略 + 固定参数 → 结果哈希必须稳定

【为什么需要】
重构可以**悄悄**改变回测结果。这个项目已经有过教训：
涨跌停价修一次、`margin` 去重一次，"同样的代码"跑出来的数字就不一样了。
没有基线时，你无法区分"这次重构没影响"和"这次重构把结果改了 3%"。

【做法】
用**合成数据**（不依赖 `db/`，CI 能跑）+ 固定权重，跑一次组合回测，
把结果的**内容哈希**（净值曲线 6 位小数 + 交易笔数 + 账目差额）存进
`tests/_baseline.json`。测试断言当前跑出来的哈希与基线一致。

⚠️ 合成数据是刻意的：
  - 不依赖 db/ → CI 里也跑得动，真正的"每次提交都拦"
  - 完全确定（固定 seed）→ 哈希不会因为数据更新而漂移，
    所以**哈希一变就一定意味着代码变了**，信号干净

不一致时**不要**直接改基线了事 —— 先搞清楚是修 bug 还是引入了 bug，
在提交信息里写明为什么变。基线文件是给人看的证据，不是自动更新的缓存。
"""
import hashlib
import json
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

BASELINE = Path(__file__).resolve().parent / "_baseline.json"


def _panel(n_days=252, codes=("600000", "000001", "300750", "002415", "601318"),
           seed=42):
    """确定性合成行情（固定 seed，与数据更新无关）"""
    rng = np.random.default_rng(seed)
    idx = pd.bdate_range("2023-01-03", periods=n_days)
    ret = rng.normal(0.0004, 0.015, (n_days, len(codes)))
    close = pd.DataFrame(10 * np.exp(np.cumsum(ret, axis=0)),
                         index=idx, columns=list(codes))
    vol = pd.DataFrame(rng.uniform(1e7, 5e7, (n_days, len(codes))),
                       index=idx, columns=list(codes))
    return {"close": close, "open": close.shift(1).fillna(close.iloc[0]),
            "high": close * 1.01, "low": close * 0.99, "volume": vol,
            "limit_up": close * 1.10, "limit_down": close * 0.90}


def _weights(panel, every=20):
    idx = panel["close"].index
    tw = pd.DataFrame(np.nan, index=idx, columns=panel["close"].columns)
    for d in idx[::every]:
        tw.loc[d] = 1.0 / panel["close"].shape[1]
    return tw


def _run():
    """固定参数跑一次组合回测，返回结果对象"""
    from backtest.multi_engine import PortfolioBacktestEngine

    panel = _panel()
    eng = PortfolioBacktestEngine(initial_capital=1_000_000,
                                  commission=0.0001, stamp_duty=0.0005,
                                  slippage=0.001, fill_timing="next_open")
    return eng.run(panel, _weights(panel), audit_lookahead=False)


def _hash(res) -> str:
    """结果内容哈希 —— 单一实现，避免与 analytics.snapshot 漂移"""
    from analytics.snapshot import result_hash
    return result_hash(res)


def test_regression_baseline_matches():
    """**核心**：结果哈希必须与基线一致

    不一致 = 有人（很可能是我）改了会影响回测结果的代码。
    请先判断是"修了 bug"还是"引入 bug"，再决定是否更新基线。
    """
    cur = _hash(_run())
    if not BASELINE.exists():
        BASELINE.write_text(json.dumps(
            {"hash": cur, "note": "首次生成；改动影响回测结果时需人工确认后更新"},
            ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"[NEW] 已生成回归基线 {cur}（请提交 tests/_baseline.json）")
        return
    saved = json.loads(BASELINE.read_text(encoding="utf-8"))
    assert cur == saved.get("hash"), (
        f"回测结果与基线**不一致**！\n"
        f"  基线 {saved.get('hash')}（{saved.get('note', '')}）\n"
        f"  当前 {cur}\n"
        f"这说明本次改动**改变了回测结果**。请先查清是修 bug 还是引入 bug，\n"
        f"在提交信息里写明原因，再更新 tests/_baseline.json。")
    print(f"[OK] 回归基准一致：{cur}")


def test_regression_hash_is_sensitive():
    """**反向验证**：改一个参数，哈希必须变

    如果改了参数哈希还不变，说明这个基线没有区分力（形同虚设）。
    """
    from backtest.multi_engine import PortfolioBacktestEngine

    panel = _panel()
    tw = _weights(panel)
    a = PortfolioBacktestEngine(slippage=0.001).run(panel, tw,
                                                    audit_lookahead=False)
    b = PortfolioBacktestEngine(slippage=0.0).run(panel, tw,
                                                  audit_lookahead=False)
    assert _hash(a) != _hash(b), \
        "把滑点从 1bp 改成 0 结果哈希却没变 —— 基线没有区分力"
    # 同一参数重复跑必须完全一致（否则基线会自己抖动）
    c = PortfolioBacktestEngine(slippage=0.001).run(panel, tw,
                                                    audit_lookahead=False)
    assert _hash(a) == _hash(c), "同样的参数两次跑出的哈希不同 —— 结果不确定"
    print("[OK] 基线有区分力：改滑点哈希变；同参数重复跑哈希不变")


def test_snapshot_records_provenance():
    """E4：快照必须记下 commit / 数据指纹 / 参数 / 结果哈希"""
    from analytics import snapshot as snap

    tmp = Path(__file__).resolve().parent.parent / ".deps" / "_test_snap"
    tmp.mkdir(parents=True, exist_ok=True)
    for f in tmp.glob("*"):
        f.unlink()
    res = _run()
    p = snap.write_snapshot(res, tmp, label="回归基准", params={"seed": 42},
                            extra={"days": 252})
    assert p.exists() and p.name == snap.META_NAME
    meta = json.loads(p.read_text(encoding="utf-8"))
    for k in ("time", "commit", "fingerprint", "params", "result_hash"):
        assert k in meta, f"快照缺字段 {k}"
    assert meta["params"]["seed"] == 42 and meta["days"] == 252
    assert meta["result_hash"] == _hash(res), "快照里的 result_hash 与结果不符"
    print(f"[OK] 结果快照：commit={meta['commit'] or '-'} "
          f"hash={meta['result_hash']} 参数/指纹已记录")


def test_stale_detection_flags_changed_data():
    """E4 的关键能力：**报出哪些结论的数据前提已经变了**"""
    import json as _json

    from analytics import snapshot as snap

    tmp = Path(__file__).resolve().parent.parent / ".deps" / "_test_snap2"
    tmp.mkdir(parents=True, exist_ok=True)
    for f in tmp.glob("*"):
        f.unlink()

    # 造一个"数据指纹与当前不同"的快照 -> 必须被报为过期
    old = {"daily": {"exists": True, "n_files": 1, "max_mtime": 1.0},
           "valuation": {"exists": True, "n_files": 1, "max_mtime": 1.0},
           "adjust": {"exists": True, "n_files": 1, "max_mtime": 1.0},
           "limit": {"exists": True, "n_files": 1, "max_mtime": 1.0},
           "suspend": {"exists": True, "n_files": 1, "max_mtime": 1.0}}
    (tmp / snap.META_NAME).write_text(_json.dumps(
        {"label": "旧结论", "time": "2026-01-01 00:00:00", "commit": "abc123",
         "fingerprint": old, "result_hash": "deadbeef"},
        ensure_ascii=False), encoding="utf-8")
    items = snap.stale_snapshots(tmp)
    if not snap.data_fingerprint():
        print("[SKIP] 无 db/（指纹为空），跳过过期检测")
        return
    assert items, "数据指纹变了却没报过期 —— 这个机制就没用了"
    assert "旧结论" in snap.format_stale(items)
    assert items[0]["changed"], "没说明是哪些数据集变了"
    print(f"[OK] 过期检测：报出「{items[0]['label']}」，"
          f"变化的数据集 {items[0]['changed'][:3]}")


if __name__ == "__main__":
    test_regression_baseline_matches()
    test_regression_hash_is_sensitive()
    test_snapshot_records_provenance()
    test_stale_detection_flags_changed_data()
    print("\n全部回归基准 / 结果快照测试通过")
