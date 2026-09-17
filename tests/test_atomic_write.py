# -*- coding: utf-8 -*-
"""原子写与损坏文件处理测试

背景（为什么这个必须测）
------------------------
2026-09-17 的一次日常更新把整个任务打挂了，根因是一个 **1570 字节的半截
parquet**：`frozen/daily_raw/year=2026/601089.parquet`。头部魔数 `PAR1` 还在，
尾部变成了零字节 —— 典型的"写一半被中断"。

而它之所以能一直烂在那里、直到重建清洗层才炸，是因为**每一层都在帮它隐瞒**：

  1. `daily_update.py::upsert` 直接 `merged.to_parquet(最终路径)`，非原子写；
  2. 下次更新读这个文件抛异常，被 `except Exception` 吞掉，只打印一行
     `{code} 失败: ...`，混在 5,556 只股票的进度条里根本看不见；
  3. 更糟的是，如果当时把"读不出来"当成"文件不存在"，就会把这次 15 天的
     窗口当成全部历史写进去 —— 这只股票**多年的数据被静默删除**，
     而且看起来完全正常。

所以这里测三件事：原子写真的原子、写坏了要能发现、坏文件不能被当成空文件。
"""
import os
import sys

import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))


def _tmpdir():
    """项目内临时目录（不用 pytest 的 tmp_path：受限环境会 PermissionError）"""
    import uuid

    from pathlib import Path
    base = Path(__file__).resolve().parent.parent / ".deps" / "_test_atomic"
    base.mkdir(parents=True, exist_ok=True)
    d = base / f"t_{uuid.uuid4().hex[:8]}"
    d.mkdir(parents=True, exist_ok=True)
    return d


def test_is_valid_parquet_detects_truncated_file():
    """**回归**：截断文件的头部魔数还在，只查头会漏掉 —— 必须查尾部

    真实坏文件长这样：`head=b'PAR1'`、`tail=b'\\x00\\x00\\x00\\x00'`。
    这正是第一版检查会漏掉的情况（只看头就以为文件是好的）。
    """
    from database.storage import is_valid_parquet

    tmp = _tmpdir()
    good = tmp / "good.parquet"
    pd.DataFrame({"a": [1, 2, 3]}).to_parquet(good, index=False)
    assert is_valid_parquet(good), "正常文件被误判为损坏"

    # 构造截断文件：保留头 4 字节，其余清零
    raw = good.read_bytes()
    trunc = tmp / "trunc.parquet"
    trunc.write_bytes(raw[:4] + b"\x00" * (max(8, len(raw) // 2) - 4))
    assert not is_valid_parquet(trunc), \
        "截断文件（头是 PAR1、尾是零字节）未被检出 —— 这正是 601089 的形态"

    # 尾魔数被破坏
    bad_tail = tmp / "badtail.parquet"
    bad_tail.write_bytes(raw[:-4] + b"\x00\x00\x00\x00")
    assert not is_valid_parquet(bad_tail), "尾魔数被破坏未被检出"

    # 太小的文件
    tiny = tmp / "tiny.parquet"
    tiny.write_bytes(b"PAR1")
    assert not is_valid_parquet(tiny), "4 字节文件不该被当成合法 parquet"
    # 不存在的文件
    assert not is_valid_parquet(tmp / "nope.parquet")
    print("[OK] 损坏检测：截断/尾魔数坏/过小/不存在 都能识别（头尾都查）")


def test_find_corrupt_parquet_lists_only_bad_files():
    """扫描函数只报坏文件，好文件一个都不能误报"""
    from database.storage import find_corrupt_parquet

    tmp = _tmpdir()
    for i in range(3):
        pd.DataFrame({"a": [i]}).to_parquet(tmp / f"ok{i}.parquet", index=False)
    (tmp / "bad.parquet").write_bytes(b"PAR1" + b"\x00" * 200)

    bad = find_corrupt_parquet(tmp, "*.parquet")
    names = sorted(p.name for p in bad)
    assert names == ["bad.parquet"], f"应只报 bad.parquet，实得 {names}"
    print("[OK] 扫描：3 个好文件 + 1 个坏文件 -> 只报出坏的那个")


def test_atomic_to_parquet_write_and_overwrite():
    """原子写：能写、能覆盖、不留临时文件"""
    from database.storage import atomic_to_parquet

    tmp = _tmpdir()
    p = tmp / "a.parquet"
    atomic_to_parquet(pd.DataFrame({"a": [1, 2]}), p)
    assert pd.read_parquet(p)["a"].tolist() == [1, 2]
    atomic_to_parquet(pd.DataFrame({"a": [9]}), p)
    assert pd.read_parquet(p)["a"].tolist() == [9], "覆盖写失败"
    leftovers = [f.name for f in tmp.iterdir() if ".tmp" in f.name]
    assert not leftovers, f"残留临时文件: {leftovers}"
    print("[OK] 原子写：写入 / 覆盖 / 无临时文件残留")


def test_atomic_write_failure_keeps_old_file():
    """**最关键的一条**：校验失败时**旧文件必须还在**

    这是原子写相对"直接写最终路径"的核心价值 —— 写坏了不会把已有的好数据
    一起毁掉。旧文件还在，就还有救。
    """
    import database.storage as st

    tmp = _tmpdir()
    p = tmp / "keep.parquet"
    st.atomic_to_parquet(pd.DataFrame({"a": [1, 2, 3]}), p)
    before = pd.read_parquet(p)["a"].tolist()

    orig = st._verify_parquet
    try:
        def boom(_path):
            raise RuntimeError("模拟校验失败")
        st._verify_parquet = boom
        try:
            st.atomic_to_parquet(pd.DataFrame({"a": [99]}), p)
            raise AssertionError("校验失败却没有抛异常")
        except RuntimeError:
            pass
    finally:
        st._verify_parquet = orig

    assert pd.read_parquet(p)["a"].tolist() == before, \
        "写失败把旧文件毁了 —— 原子写失去意义"
    leftovers = [f.name for f in tmp.iterdir() if ".tmp" in f.name]
    assert not leftovers, f"失败后残留临时文件: {leftovers}"
    print("[OK] 原子写失败：旧文件完好、临时文件已清理（不会毁掉已有数据）")


def test_upsert_refuses_to_overwrite_when_existing_file_corrupt():
    """**回归**：旧分片损坏时必须**拒绝写入**，绝不能当成空文件

    如果当成空文件，就会把"这次增量窗口"（默认 15 天）当成全部历史写进去，
    这只股票多年的数据被静默删除 —— 比留一个坏文件糟糕得多：
    坏文件至少还会报错，静默截断不会。
    """
    import uuid

    from database.config import FROZEN_ROOT
    import scripts.daily_update as du

    code = f"_t{uuid.uuid4().hex[:6]}"
    part = FROZEN_ROOT / "daily_raw" / "year=2026"
    try:
        part.mkdir(parents=True, exist_ok=True)
    except OSError:
        print("[SKIP] frozen 不可写，跳过 upsert 损坏保护回归")
        return
    path = part / f"{code}.parquet"
    path.write_bytes(b"PAR1" + b"\x00" * 100)          # 伪造一个半截文件

    df = pd.DataFrame({"code": [code], "trade_date": pd.to_datetime(["2026-09-17"]),
                       "close": [10.0]})
    try:
        du.upsert("daily_raw", code, df)
        raise AssertionError("旧分片损坏时 upsert 竟然写了进去（会截断历史）")
    except du.CorruptParquetError:
        pass
    finally:
        assert path.exists(), "损坏文件被删掉了 —— 应保留以便隔离取证"
        path.unlink()
    assert (du.CORRUPT_FOUND == {("daily_raw", code)}), \
        f"损坏未被记录，结尾就不会报警：{du.CORRUPT_FOUND}"
    print("[OK] upsert：旧分片损坏时拒绝写入并记录（不会把 15 天窗口当全部历史）")


def test_quarter_ranges_helper_is_used_by_downloader():
    """两融的季度区间回归（顺带确认抽出来的函数还在被用）"""
    from database.downloader.other import QUARTER_ENDS, quarter_ranges

    for _, e in QUARTER_ENDS:
        assert e in ("0331", "0630", "0930", "1231"), f"非季末: {e}"
    rs = quarter_ranges(2024, 2024)
    assert len(rs) == 4 and rs[0][0] == "20240101" and rs[-1][1] == "20241231"
    print("[OK] 两融季度区间仍是季末闭合（download_frozen_tushare 与它同源）")


if __name__ == "__main__":
    test_is_valid_parquet_detects_truncated_file()
    test_find_corrupt_parquet_lists_only_bad_files()
    test_atomic_to_parquet_write_and_overwrite()
    test_atomic_write_failure_keeps_old_file()
    test_upsert_refuses_to_overwrite_when_existing_file_corrupt()
    test_quarter_ranges_helper_is_used_by_downloader()
    print("\n全部原子写/损坏处理测试通过")
