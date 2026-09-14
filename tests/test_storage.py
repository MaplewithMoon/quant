# -*- coding: utf-8 -*-
"""数据层可靠性测试：原子写 / 断点容错 / 磁盘保护不可被吞 / 下载失败不误标完成"""
import json
import os
import shutil
import sys
import tempfile
from pathlib import Path

import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from database.storage import Storage, DiskFullError
from database.config import FrozenWriteError
from database.downloader.tushare_client import TushareCallError, TushareClient


def _tmpdir():
    """在工作区内建临时目录

    注意：不用 tempfile.mkdtemp —— 它以 0o700 创建目录，acl 不含本环境的
    沙箱身份，导致"能在里面建文件却建不了子目录"。用普通 mkdir 继承父目录 ACL。
    """
    import uuid
    base = Path(__file__).resolve().parent.parent / ".deps" / "_test_store"
    base.mkdir(parents=True, exist_ok=True)
    d = base / f"store_{uuid.uuid4().hex[:8]}"
    d.mkdir(parents=True, exist_ok=True)
    return d


def _df():
    return pd.DataFrame({"code": ["000001"], "trade_date": [pd.Timestamp("2024-01-02")],
                         "close": [10.0]})


# ============================================================
# 原子写
# ============================================================
def test_save_is_atomic_on_failure():
    """写入过程中失败，不得留下损坏的正式文件"""
    d = _tmpdir()
    try:
        st = Storage(root=d)
        orig = pd.DataFrame.to_parquet

        def boom(self, path, *a, **kw):
            # 先写一半再失败，模拟进程被 kill / 磁盘满
            Path(path).write_bytes(b"half-written-garbage")
            raise OSError("模拟写入中断")

        pd.DataFrame.to_parquet = boom
        try:
            try:
                st.save(_df(), year=2024, code="000001")
                raise AssertionError("应当抛出异常")
            except OSError:
                pass
        finally:
            pd.DataFrame.to_parquet = orig

        target = d / "year=2024" / "000001.parquet"
        assert not target.exists(), "失败后不应留下正式文件"
        leftovers = list(d.rglob("*.tmp*"))
        assert not leftovers, f"不应残留临时文件: {leftovers}"
        print("[OK] 原子写：写入中断后无正式文件、无残留临时文件")
    finally:
        shutil.rmtree(d, ignore_errors=True)


def test_save_writes_via_replace():
    """正常写入应产生可读文件，且不存在 .tmp 残留"""
    d = _tmpdir()
    try:
        st = Storage(root=d)
        p = st.save(_df(), year=2024, code="000001")
        assert p.exists()
        got = pd.read_parquet(p)
        assert len(got) == 1 and got["close"].iloc[0] == 10.0
        assert not list(d.rglob("*.tmp*"))
        print("[OK] 原子写正常路径：文件可读、无临时文件残留")
    finally:
        shutil.rmtree(d, ignore_errors=True)


def test_save_skips_existing_unless_force():
    d = _tmpdir()
    try:
        st = Storage(root=d)
        st.save(pd.DataFrame({"a": [1]}), year=2024, code="X")
        st.save(pd.DataFrame({"a": [2]}), year=2024, code="X")          # 跳过
        assert pd.read_parquet(d / "year=2024" / "X.parquet")["a"].iloc[0] == 1
        st.save(pd.DataFrame({"a": [3]}), year=2024, code="X", force=True)
        assert pd.read_parquet(d / "year=2024" / "X.parquet")["a"].iloc[0] == 3
        print("[OK] 断点续跑语义：已存在则跳过，force=True 才覆盖")
    finally:
        shutil.rmtree(d, ignore_errors=True)


# ============================================================
# 断点容错与缓存
# ============================================================
def test_checkpoint_recovers_from_corruption():
    """断电写坏断点文件后，不能每次运行都抛 JSONDecodeError"""
    d = _tmpdir()
    try:
        st = Storage(root=d)
        st.mark_done("000001")
        st.checkpoint_file.write_text("{损坏的JSON", encoding="utf-8")

        st2 = Storage(root=d)          # 新实例，清掉缓存
        cp = st2.load_checkpoint()
        assert cp == {}, f"损坏的断点应被当作空: {cp}"
        assert st2.checkpoint_file.with_suffix(".json.corrupt").exists(), \
            "损坏文件应被备份而不是直接丢弃"
        # 损坏后仍可正常写入
        st2.mark_done("000002")
        assert "000002" in json.loads(st2.checkpoint_file.read_text(encoding="utf-8"))["done"]
        print("[OK] 断点损坏自愈：返回空 + 备份为 .corrupt + 后续可正常写入")
    finally:
        shutil.rmtree(d, ignore_errors=True)


def test_checkpoint_cache_avoids_reread():
    d = _tmpdir()
    try:
        st = Storage(root=d)
        st.mark_done("A")
        first = st.load_checkpoint()
        assert first is st.load_checkpoint(), "应命中缓存（同一对象）"
        st.save_checkpoint({"done": ["B"]})
        assert st.load_checkpoint()["done"] == ["B"]
        st.mark_done("C")
        assert st.load_checkpoint()["done"] == ["B", "C"]
        print("[OK] 断点缓存：同一实例内不重复读盘，写入后缓存同步更新")
    finally:
        shutil.rmtree(d, ignore_errors=True)


# ============================================================
# 磁盘保护不可被吞
# ============================================================
def test_disk_full_error_cannot_be_swallowed():
    """DiskFullError 必须能穿透下载器里遍地都是的 except Exception"""
    assert not issubclass(DiskFullError, Exception), \
        "DiskFullError 应继承 BaseException，否则会被 except Exception 静默吞掉"
    assert issubclass(DiskFullError, BaseException)

    # 模拟下载器的兜底写法
    caught = None
    try:
        try:
            raise DiskFullError("磁盘满了")
        except Exception as e:          # noqa: BLE001 —— 这正是要验证的写法
            caught = e
    except DiskFullError:
        pass
    assert caught is None, "except Exception 不应捕获 DiskFullError"

    d = _tmpdir()
    try:
        st = Storage(root=d)
        st.assert_disk_ok()             # 当前磁盘充足，不应抛
        print("[OK] DiskFullError 穿透 except Exception（磁盘保护不会被静默吞掉）")
    finally:
        shutil.rmtree(d, ignore_errors=True)


# ============================================================
# 下载失败 vs 无数据
# ============================================================
def test_tushare_client_raises_after_retries():
    """重试耗尽必须抛异常，绝不能返回 None（返回 None 会被当成'无数据'永久标记完成）"""
    c = TushareClient()
    calls = {"n": 0}

    class FakePro:
        def daily(self, **kw):
            calls["n"] += 1
            raise RuntimeError("最多每分钟调用200次")

    c.get_pro = lambda: FakePro()
    c._wait = lambda: None              # 跳过真实限流等待
    try:
        c.call("daily", max_retries=2, ts_code="000001.SZ")
        raise AssertionError("重试耗尽应当抛异常")
    except TushareCallError as e:
        assert "重试" in str(e)
    assert calls["n"] == 3, f"应尝试 3 次（1+2 次重试），实际 {calls['n']}"
    print(f"[OK] tushare 重试耗尽抛 TushareCallError（尝试 {calls['n']} 次，不再返回 None）")


def test_tushare_client_returns_empty_not_none():
    """接口正常返回空 -> 空 DataFrame（这才是'确实没有数据'）"""
    c = TushareClient()

    class FakePro:
        def daily(self, **kw):
            return None

    c.get_pro = lambda: FakePro()
    c._wait = lambda: None
    df = c.call("daily", ts_code="000001.SZ")
    assert df is not None and df.empty, "应返回空 DataFrame 而非 None"
    print("[OK] 接口正常返回空 -> 空 DataFrame（可与'失败'区分开）")


def test_tushare_rate_limit_on_every_retry():
    """每次重试都应重新限流（旧实现只在进循环前限流一次）"""
    c = TushareClient()
    waits = {"n": 0}
    calls = {"n": 0}

    class FakePro:
        def daily(self, **kw):
            calls["n"] += 1
            raise RuntimeError("网络错误")

    c.get_pro = lambda: FakePro()
    c._wait = lambda: waits.__setitem__("n", waits["n"] + 1)
    import time as _t
    orig = _t.sleep
    _t.sleep = lambda s: None
    try:
        try:
            c.call("daily", max_retries=2)
        except TushareCallError:
            pass
    finally:
        _t.sleep = orig
    assert waits["n"] == calls["n"] == 3, f"限流应调用 3 次，实际 {waits['n']}"
    print(f"[OK] 重试不绕限流：3 次尝试对应 3 次限流调用")


# ============================================================
# frozen 只读保护仍然有效
# ============================================================
def test_frozen_guard_still_works():
    st = Storage("daily_raw")           # 未传 allow_frozen
    assert st.layer == "frozen"
    try:
        st.save(_df(), year=2024, code="000001")
        raise AssertionError("写入 frozen 应当被拒绝")
    except FrozenWriteError:
        pass
    st2 = Storage("daily_raw", allow_frozen=True)
    st2.assert_writable()               # 不抛
    print("[OK] frozen 只读保护仍生效；下载器路径 allow_frozen=True 可写")


if __name__ == "__main__":
    test_save_is_atomic_on_failure()
    test_save_writes_via_replace()
    test_save_skips_existing_unless_force()
    test_checkpoint_recovers_from_corruption()
    test_checkpoint_cache_avoids_reread()
    test_disk_full_error_cannot_be_swallowed()
    test_tushare_client_raises_after_retries()
    test_tushare_client_returns_empty_not_none()
    test_tushare_rate_limit_on_every_retry()
    test_frozen_guard_still_works()
    print("\n全部数据层可靠性测试通过")
