# -*- coding: utf-8 -*-
"""告警（T3·E5）—— 「失败了要出声」

【为什么这一条最要紧】
2026-09 那次日常更新：跑了 1.5 小时 → 一个损坏分片被 `except` 吞掉只打印一行
→ 重建清洗层崩掉。**全程没有任何人收到通知**，是用户把 traceback 贴过来才发现的。

数据更新是无人值守的定时任务。脚本层面已经做得很扎实（原子写、断点、非零退出码），
但**没人知道它失败了** = 等于没做。

【设计原则】
1. **永不抛异常**：告警本身出问题不能把主流程带崩。所有路径都包 try/except。
2. **一定落盘**：至少写 `logs/alerts.jsonl`。外部推送（webhook）是尽力而为 ——
   没有配就只落盘 + 打印，不能因为"没配 webhook"就不告警。
3. **区分级别**：`error` 需要人处理，`warn` 只是提醒，`info` 供留痕。
4. **带上下文**：光说"失败了"没用，要带上是哪个数据集/哪只标的/什么原因。

配置（可选）：
    ALERT_WEBHOOK   形如 https://open.feishu.cn/... 或任意接受 JSON POST 的地址
                    没配就只落盘 + 打印
"""
import json
import os
import sys
from datetime import datetime
from pathlib import Path

ALERT_LOG = Path(__file__).resolve().parent.parent / "logs" / "alerts.jsonl"
LEVEL_MARK = {"error": "⛔", "warn": "⚠", "info": "ℹ"}


def _now() -> str:
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")


def alert(level: str, title: str, detail: str = "", **ctx) -> dict:
    """记一条告警。**永不抛异常。**

    返回落盘的记录（便于测试断言）；连落盘都失败时返回 dict 但仍不抛。
    """
    rec = {"time": _now(), "level": level, "title": title,
           "detail": str(detail)[:2000], "ctx": ctx}
    # 1) 打印（无人值守时也会进 stdout/日志）
    try:
        mark = LEVEL_MARK.get(level, "?")
        line = f"{mark} [{level.upper()}] {title}"
        if detail:
            line += f" —— {str(detail)[:200]}"
        print(line, file=sys.stderr if level == "error" else sys.stdout,
              flush=True)
    except Exception:
        pass
    # 2) 落盘（这是"一定发生"的那部分）
    try:
        ALERT_LOG.parent.mkdir(parents=True, exist_ok=True)
        with open(ALERT_LOG, "a", encoding="utf-8") as fh:
            fh.write(json.dumps(rec, ensure_ascii=False) + "\n")
    except Exception:
        pass
    # 3) 外部推送（尽力而为；没配就跳过，**不因为没配就不告警**）
    _push(rec)
    return rec


def error(title, detail="", **ctx):
    return alert("error", title, detail, **ctx)


def warn(title, detail="", **ctx):
    return alert("warn", title, detail, **ctx)


def info(title, detail="", **ctx):
    return alert("info", title, detail, **ctx)


def _push(rec: dict):
    """可选的外部推送。任何失败都吞掉 —— 告警不能反过来弄挂主流程。"""
    url = os.environ.get("ALERT_WEBHOOK", "").strip()
    if not url:
        return
    try:
        import ssl
        import urllib.request
        body = json.dumps(
            {"msg_type": "text",
             "content": {"text": f"[{rec['level'].upper()}] {rec['title']}\n"
                                 f"{rec['detail'][:500]}\n{rec['time']}"}},
            ensure_ascii=False).encode("utf-8")
        req = urllib.request.Request(url, data=body, method="POST")
        req.add_header("Content-Type", "application/json")
        ctx = ssl.create_default_context()
        urllib.request.urlopen(req, timeout=8, context=ctx)
    except Exception:
        pass


def recent(n: int = 20, level: str = None) -> list:
    """读回最近 n 条（供人工检查 / 测试）"""
    if not ALERT_LOG.exists():
        return []
    out = []
    try:
        for ln in ALERT_LOG.read_text(encoding="utf-8").splitlines():
            ln = ln.strip()
            if not ln:
                continue
            try:
                r = json.loads(ln)
            except ValueError:
                continue
            if level is None or r.get("level") == level:
                out.append(r)
    except OSError:
        return []
    return out[-n:]


def summarize(since_hours: int = 24) -> str:
    """最近一段时间的告警摘要（给日报/巡检用）"""
    from datetime import timedelta
    cut = datetime.now() - timedelta(hours=since_hours)
    rows = []
    for r in recent(1000):
        try:
            t = datetime.strptime(r["time"], "%Y-%m-%d %H:%M:%S")
        except (KeyError, ValueError):
            continue
        if t >= cut:
            rows.append(r)
    if not rows:
        return f"最近 {since_hours} 小时无告警。"
    n_err = sum(1 for r in rows if r.get("level") == "error")
    L = [f"最近 {since_hours} 小时：{len(rows)} 条告警（error {n_err}）"]
    for r in rows[-10:]:
        L.append(f"  {LEVEL_MARK.get(r.get('level'), '?')} {r['time']} "
                 f"{r['title']}")
    return "\n".join(L)


__all__ = ["alert", "error", "warn", "info", "recent", "summarize",
           "ALERT_LOG"]
