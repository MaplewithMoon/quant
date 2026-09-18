# -*- coding: utf-8 -*-
"""告警测试（T3·E5）—— 「失败了要出声」

【为什么这条必须测】
B19 那次事故：日常更新跑了 1.5 小时 → 损坏分片被 `except` 吞掉只打印一行
→ 重建清洗层崩掉，**全程没人收到通知**。数据更新是无人值守任务，
没有告警 = 脚本写得再扎实也等于没做。

两条最要紧的性质：
  1. **告警自身出问题不能把主流程带崩**（永不抛异常）
  2. **一定落盘**：没配 webhook 也必须告警，不能"因为没配就不告警"
"""
import json
import os
import sys
from datetime import datetime
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))


def _isolate():
    """把告警日志指到项目内的临时文件（不用 tmp_path：受限环境会 PermissionError）"""
    import uuid

    import utils.alert as al
    d = Path(__file__).resolve().parent.parent / ".deps" / "_test_alert"
    d.mkdir(parents=True, exist_ok=True)
    al.ALERT_LOG = d / f"alerts_{uuid.uuid4().hex[:8]}.jsonl"
    return al


def test_alert_writes_to_disk_with_context():
    """**核心**：告警必须落盘，且带上可定位的上下文

    只喊"失败了"没用 —— 要能看出是哪个数据集/哪只标的/什么原因。
    """
    al = _isolate()
    al.error("日常更新失败：清洗层不完整", "3 个分片读不出来", n_bad=3, year=2026)
    al.warn("跳过了损坏分片", "daily_raw/601089", n=1)
    rows = al.recent(10)
    assert len(rows) == 2, rows
    r0 = rows[0]
    assert r0["level"] == "error" and "清洗层" in r0["title"]
    assert r0["ctx"] == {"n_bad": 3, "year": 2026}, r0["ctx"]
    assert r0["detail"] == "3 个分片读不出来"
    datetime.strptime(r0["time"], "%Y-%m-%d %H:%M:%S")   # 时间必须可解析
    assert al.recent(10, level="warn")[0]["level"] == "warn"
    print(f"[OK] 告警落盘并带上下文：{[r['title'][:18] for r in rows]}")


def test_alert_never_raises():
    """**关键**：告警自己坏了也不能弄挂主流程"""
    al = _isolate()
    # 让写盘必定失败
    orig = al.ALERT_LOG
    al.ALERT_LOG = Path("Z:/definitely/not/writable/x.jsonl")
    try:
        rec = al.error("写盘失败也要能返回", "测试")
        assert isinstance(rec, dict) and rec["level"] == "error"
    finally:
        al.ALERT_LOG = orig
    # 不可序列化的上下文也不能炸（_jsonable 之外还有兜底）
    rec2 = al.warn("带奇怪上下文", "", obj=object())
    assert isinstance(rec2, dict)
    print("[OK] 告警永不抛异常（写盘失败 / 不可序列化上下文都能兜住）")


def test_alert_works_without_webhook():
    """没配 webhook 也必须告警 —— 不能"因为没配就不告警\""""
    al = _isolate()
    os.environ.pop("ALERT_WEBHOOK", None)
    al.error("没有 webhook 也要落盘", "x")
    assert al.recent(1), "没配 webhook 就丢了告警"
    # 配了但地址不可达，也不能影响落盘
    os.environ["ALERT_WEBHOOK"] = "http://127.0.0.1:9/nope"
    try:
        al.error("推送失败也要落盘", "y")
        assert len(al.recent(10)) == 2
    finally:
        os.environ.pop("ALERT_WEBHOOK", None)
    print("[OK] 无 webhook / webhook 不可达时，告警仍然落盘")


def test_summarize_reports_recent_errors():
    """摘要要能一眼看出"最近有没有出事\""""
    al = _isolate()
    assert "无告警" in al.summarize(24)
    al.error("出事了", "细节")
    s = al.summarize(24)
    assert "1 条告警" in s and "error 1" in s and "出事了" in s, s
    assert "无告警" in al.summarize(0)      # 0 小时窗口内没有
    print(f"[OK] 告警摘要：{s.splitlines()[0]}")


def test_daily_update_calls_alert():
    """日常更新必须真的接了告警（不能只是模块存在）"""
    src = (Path(__file__).resolve().parent.parent
           / "scripts" / "daily_update.py").read_text(encoding="utf-8")
    for needle in ("from utils.alert import",
                   "alert_error(", "alert_warn(",
                   "except BaseException"):
        assert needle in src, f"daily_update.py 缺少 {needle}"
    # 至少要有：损坏分片跳过(warn)、清洗层不完整(error)、校验不过(error)、
    # 异常退出(error)
    assert src.count("alert_error(") >= 3, "error 级告警接得太少"
    print("[OK] daily_update.py 已接告警（warn + 3 处 error + 异常兜底）")


if __name__ == "__main__":
    test_alert_writes_to_disk_with_context()
    test_alert_never_raises()
    test_alert_works_without_webhook()
    test_summarize_reports_recent_errors()
    test_daily_update_calls_alert()
    print("\n全部告警测试通过")
