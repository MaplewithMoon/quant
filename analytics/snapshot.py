# -*- coding: utf-8 -*-
"""结果快照与过期检测（T3·E4）

【要解决什么问题】
这个项目已经**两次**因为数据修复而结论变化：

    - 涨跌停价修复：115 万行被改写（创业板 2020-08-24 前误用 ±20%）
    - `frozen/margin` 整行重复：`sum(rzye)` 被放大 2.5 倍，`margin_chg` 的
      IC 从 −0.020 变成 −0.057

但**没有任何机制知道哪些结论过期了**。旧报告还躺在 `results/` 里，
数字看着挺像那么回事，没人知道它是在旧数据上跑出来的。

【做法】
每份结果旁落一个快照 JSON：

    commit      跑出这份结果的代码版本
    fingerprint 当时依赖的**数据集指纹**（文件数 + 最新 mtime）
    params      参数
    result_hash 结果本身的内容哈希（净值曲线 + 交易笔数）
    time        时间

`stale_snapshots()` 拿当前指纹去比对，直接回答"**哪些结果已经过期**"。

⚠️ 指纹口径与 `database.provenance.PANEL_DEPS` 一致（面板缓存用同一套），
不另起一套 —— 两套口径必然漂移。
"""
import hashlib
import json
import subprocess
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional

META_NAME = "_snapshot.json"


def git_commit() -> str:
    """当前 commit（取不到返回空字符串，不抛）"""
    try:
        r = subprocess.run(["git", "rev-parse", "--short", "HEAD"],
                           capture_output=True, text=True, timeout=10)
        return (r.stdout or "").strip() if r.returncode == 0 else ""
    except Exception:
        return ""


def result_hash(res) -> str:
    """结果内容哈希 —— 净值曲线 + 交易笔数 + 账目差额

    用**四舍五入到 6 位**的净值序列，避免浮点末位噪声让哈希乱跳
    （那样这个机制会因为"假变化"太多而没人看）。
    """
    h = hashlib.sha256()
    eq = getattr(res, "equity", None)
    if eq is not None and len(eq):
        try:
            vals = [f"{float(x):.6f}" for x in eq.to_numpy()]
            h.update(",".join(vals).encode())
            h.update(str(list(eq.index.astype(str))[:1]).encode())
            h.update(str(list(eq.index.astype(str)[-1:])).encode())
        except Exception:
            h.update(str(len(eq)).encode())
    tr = getattr(res, "trades", None)
    if tr is not None:
        h.update(f"|n={len(tr)}".encode())
    h.update(f"|gap={getattr(res, 'ledger_gap', 0.0):.10f}".encode())
    return h.hexdigest()[:16]


def data_fingerprint() -> Dict:
    """当前数据指纹（与面板缓存同一套口径）"""
    try:
        from database.provenance import dataset_fingerprint
        return dataset_fingerprint()
    except Exception:
        return {}


def write_snapshot(res, outdir, label: str = "", params: dict = None,
                   extra: dict = None) -> Path:
    """把结果快照写到 `<outdir>/_snapshot.json`（原子写）"""
    outdir = Path(outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    meta = {
        "label": label,
        "time": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "commit": git_commit(),
        "fingerprint": data_fingerprint(),
        "params": _jsonable(params or {}),
        "result_hash": result_hash(res),
        "defects": [getattr(d, "key", str(d))
                    for d in (getattr(res, "defects", None) or [])],
        "gate": (getattr(res, "gate", None) or {}).get("status", ""),
        "fill_timing": getattr(res, "fill_timing", ""),
    }
    if extra:
        meta.update(_jsonable(extra))
    path = outdir / META_NAME
    tmp = path.with_name(f"{path.name}.tmp")
    tmp.write_text(json.dumps(meta, ensure_ascii=False, indent=2),
                   encoding="utf-8")
    import os
    os.replace(tmp, path)
    return path


def _jsonable(d):
    if isinstance(d, dict):
        return {str(k): _jsonable(v) for k, v in d.items()}
    if isinstance(d, (list, tuple)):
        return [_jsonable(v) for v in d]
    if isinstance(d, (str, int, float, bool)) or d is None:
        return d
    return str(d)


def stale_snapshots(root="results") -> List[Dict]:
    """扫描 root 下所有快照，报出**已过期**的（数据指纹变了）

    返回 [{dir, label, time, commit, changed:[变化的数据集], saved_result_hash}]
    这是"哪些结论不再可信"的直接答案。
    """
    from database.provenance import fingerprint_changed
    cur = data_fingerprint()
    out = []
    for p in sorted(Path(root).rglob(META_NAME)):
        try:
            meta = json.loads(p.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            continue
        saved = meta.get("fingerprint")
        if not saved:
            out.append({"dir": str(p.parent), "label": meta.get("label", ""),
                        "time": meta.get("time", ""), "commit": meta.get("commit", ""),
                        "changed": ["<无指纹记录，无法判定，按过期处理>"],
                        "saved_result_hash": meta.get("result_hash", "")})
            continue
        ch = fingerprint_changed(saved, cur)
        if ch:
            out.append({"dir": str(p.parent), "label": meta.get("label", ""),
                        "time": meta.get("time", ""), "commit": meta.get("commit", ""),
                        "changed": ch,
                        "saved_result_hash": meta.get("result_hash", "")})
    return out


def format_stale(items: List[Dict]) -> str:
    if not items:
        return "所有结果快照的数据指纹与当前一致（无过期结论）。"
    L = [f"⚠ 有 {len(items)} 份结果的**数据已变化，结论可能过期**："]
    for it in items:
        # label 必须打印：目录名往往看不出这是什么结论，label 才认得出来
        head = it.get("label") or it["dir"]
        L.append(f"  · {head}")
        L.append(f"      {it['dir']}")
        L.append(f"      跑于 {it['time']} @{it['commit'] or '?'}"
                 f"  变化的数据集: {', '.join(it['changed'])}")
    L.append("  处理：重跑这些结果，或在结论里注明其数据前提已变。")
    return "\n".join(L)


__all__ = ["META_NAME", "git_commit", "result_hash", "data_fingerprint",
           "write_snapshot", "stale_snapshots", "format_stale"]
