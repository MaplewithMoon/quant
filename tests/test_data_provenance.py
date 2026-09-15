# -*- coding: utf-8 -*-
"""数据指纹与派生缓存失效测试

背景（为什么这是个真问题）
--------------------------
面板缓存 `.cache/panel/<start>_<end>/` 原先**只以区间为键**。数据被重建之后
缓存仍然命中，回测拿着旧数据算出结论 —— 静默污染全部研究结论。

本项目刚修完涨跌停价，115 万行被改写（创业板 2020-08-24 前误用 ±20%），
所有 2017–2020 的回测结论都受它影响；如果缓存不失效，重跑也发现不了。

所以缓存元数据里必须记下**所依赖数据集的指纹**。
"""
import os
import sys

import pandas as pd

from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))


def _fake_fp(mtime=1000.0, n=10):
    return {ds: {"exists": True, "n_files": n, "max_mtime": mtime, "generated": 1}
            for ds in ("daily", "valuation", "adjust", "limit", "suspend")}


def test_fingerprint_changed_detects_mtime_and_count():
    """指纹比对：mtime / 文件数任一变化都要算"变了" """
    from database.provenance import fingerprint_changed

    a = _fake_fp()
    assert fingerprint_changed(a, _fake_fp()) == [], "完全相同不应报变化"

    b = _fake_fp()
    b["limit"]["max_mtime"] += 1
    assert fingerprint_changed(a, b) == ["limit"], "mtime 变化未检出"

    c = _fake_fp()
    c["daily"]["n_files"] += 1
    assert fingerprint_changed(a, c) == ["daily"], "文件数变化未检出"

    # 旧缓存没有指纹字段 -> 必须全部判为变化（宁可重算，不可用旧数据）
    assert set(fingerprint_changed(None, a)) == set(a), "无指纹时未整体作废"

    # 新增依赖数据集也要作废
    d = _fake_fp()
    d["newdata"] = {"exists": True, "n_files": 1, "max_mtime": 1.0}
    assert "newdata" in fingerprint_changed(a, d)
    print("[OK] 指纹比对：mtime/文件数/缺失指纹/新增依赖 都能检出变化")


def _tmpdir():
    """在工作区内建临时目录

    注意：**不用 tempfile.mkdtemp** —— 它以 0o700 创建目录，acl 不含本环境的
    沙箱身份，导致"能在里面建文件却建不了子目录"（WinError 5）。
    用普通 mkdir 继承父目录 ACL。（与 tests/test_storage.py 同因同解）
    """
    import uuid

    base = Path(__file__).resolve().parent.parent / ".deps" / "_test_prov"
    base.mkdir(parents=True, exist_ok=True)
    d = base / f"prov_{uuid.uuid4().hex[:8]}"
    d.mkdir(parents=True, exist_ok=True)
    return d


def test_panel_cache_roundtrip_and_invalidation(tmp_path):
    """面板缓存：指纹一致则命中；底层数据一变则拒绝复用"""
    import backtest.panel_data as pdata
    import database.provenance as prov

    idx = pd.bdate_range("2024-01-02", "2024-01-31")
    panel = {
        "close": pd.DataFrame(1.0, index=idx, columns=["600000", "000001"]),
        "open": pd.DataFrame(1.0, index=idx, columns=["600000", "000001"]),
    }
    cache_dir = str(tmp_path / "panel")
    start, end = "2024-01-01", "2024-01-31"

    orig = prov.dataset_fingerprint
    try:
        prov.dataset_fingerprint = lambda *a, **k: _fake_fp(mtime=1000.0)
        pdata._write_panel_cache(panel, cache_dir, start, end)

        # ① 指纹未变 -> 命中
        got = pdata._read_panel_cache(cache_dir, start, end)
        assert got is not None, "指纹未变时缓存应命中"
        assert list(got["close"].columns) == ["600000", "000001"]
        assert len(got["close"]) == len(idx)

        # ② 底层数据变了（limit 被重建）-> 必须作废
        prov.dataset_fingerprint = lambda *a, **k: _fake_fp(mtime=2000.0)
        assert pdata._read_panel_cache(cache_dir, start, end) is None, \
            "底层数据已更新，缓存必须失效（否则回测静默用旧数据）"

        # ③ 旧缓存没有指纹字段 -> 也必须作废
        prov.dataset_fingerprint = orig
        meta_f = os.path.join(cache_dir, pdata._cache_key(start, end), "_meta.json")
        import json
        meta = json.load(open(meta_f, encoding="utf-8"))
        meta.pop("data_fingerprint", None)
        json.dump(meta, open(meta_f, "w", encoding="utf-8"))
        assert pdata._read_panel_cache(cache_dir, start, end) is None, \
            "无指纹的历史缓存必须作废（不能假定它是新的）"
    finally:
        prov.dataset_fingerprint = orig
    print("[OK] 面板缓存：指纹一致命中 / 数据变更失效 / 无指纹旧缓存作废")


def test_year_globs_only_lists_existing_partitions(tmp_path):
    """分区 glob 只能列**真实存在**的分区

    这是"停牌面板静默为空"那个 bug 的根因回归：旧实现按区间硬拼 year=YYYY，
    对静态数据集（只有 year=2005 占位）会拼出不存在的路径 -> DuckDB 抛
    IOException -> 被 except Exception 吞掉 -> 数据静默变空。
    """
    from database.config import NON_ANNUAL_YEAR, year_globs

    # 静态数据集：只有占位分区，任何区间都必须落到它上面
    static = tmp_path / "static"
    (static / f"year={NON_ANNUAL_YEAR}").mkdir(parents=True)
    (static / f"year={NON_ANNUAL_YEAR}" / "a.parquet").write_bytes(b"x")
    g = year_globs(static, 2024, 2025)
    assert f"year={NON_ANNUAL_YEAR}" in g, f"静态数据集应忽略区间，实得 {g}"
    assert "year=2024" not in g, "不得拼出不存在的分区"

    # 年度数据集：只取区间内且存在的
    ann = tmp_path / "ann"
    for y in (2020, 2021, 2024):
        (ann / f"year={y}").mkdir(parents=True)
        (ann / f"year={y}" / "a.parquet").write_bytes(b"x")
    g = year_globs(ann, 2020, 2021)
    assert "year=2020" in g and "year=2021" in g and "year=2024" not in g, g

    # 区间内一个分区都没有 -> 回退到全部（交给 WHERE 过滤），不能是空或报错
    g = year_globs(ann, 1990, 1995)
    assert g != "[]" and "year=2020" in g, f"区间无分区时应回退，实得 {g}"

    # 数据集不存在 -> 空列表，调用方须自行判断（不能拼出非法路径）
    assert year_globs(tmp_path / "nope", 2024, 2025) == "[]"

    # 已经是年份分区 -> 只补 *.parquet
    g = year_globs(ann / "year=2024", 2020, 2021)
    assert g.count("year=2024") == 1 and "year=*" not in g, g
    print("[OK] year_globs：静态占位分区/年度区间/区间为空回退/数据集缺失 均正确")


def test_suspend_panel_actually_loads():
    """**回归**：停牌面板必须真的读到数据，且只在交易日记为停牌

    旧实现两处都错（glob 拼错 + trade_date 是 VARCHAR），异常被吞，
    于是每一次组合回测的 suspended 都是空表 —— 停牌股照常买卖。
    """
    import backtest.panel_data as pdata
    from database.config import FROZEN_ROOT

    sus_dir = FROZEN_ROOT / "suspend"
    if not sus_dir.exists():
        print("[SKIP] 无 frozen/suspend（数据未下载），跳过停牌面板回归")
        return
    out = pdata.load_status_panels("2024-01-01", "2024-06-30")
    assert "suspended" in out, \
        "停牌面板为空 —— 停牌股会被当成正常可交易（历史 bug 复现）"
    s = out["suspended"]
    assert s.shape[0] > 0 and s.shape[1] > 0, f"停牌面板形状异常 {s.shape}"
    assert s.values.any(), "停牌面板全是 False，说明没读到任何停牌记录"
    # 真实管线会走 align_to：缺失（NaN=未停牌）补 False 并转 bool
    s = pdata.align_to(out, s.index, s.columns)["suspended"]
    assert (s.dtypes == bool).all(), f"align_to 后必须是 bool，实得 {set(s.dtypes)}"
    # 停牌只发生在交易日（S 记录本身只罗列交易日）
    assert all(d.weekday() < 5 for d in s.index), "停牌面板出现了周末，日期解析有问题"
    # 复牌日（suspend_type='R'）不应被当成停牌日
    from database.config import connect_duckdb, year_globs
    con = connect_duckdb()
    try:
        r = con.execute(f"""
            SELECT count(*) FROM read_parquet({year_globs(sus_dir, 2024, 2024)})
            WHERE suspend_type = 'R'
              AND strptime(trade_date,'%Y%m%d')::DATE
                  BETWEEN DATE '2024-01-01' AND DATE '2024-06-30'
        """).fetchone()[0]
    finally:
        con.close()
    if r:
        print(f"  （区间内有 {r} 条复牌记录，已确认未被计入停牌）")
    print(f"[OK] 停牌面板真实加载：{s.shape[0]} 交易日 × {s.shape[1]} 只，"
          f"{int(s.values.sum()):,} 个停牌标记")



def test_panel_deps_cover_everything_load_panel_reads():
    """依赖清单必须覆盖 load_panel + load_status_panels 真正读的数据集

    漏一个数据集 = 那条数据变了缓存却不失效，等于白做。
    """
    import inspect

    import backtest.panel_data as pdata
    from database.provenance import PANEL_DEPS

    src = inspect.getsource(pdata.load_status_panels)
    assert "dir_of(" in src and "limit" in src, "load_status_panels 应读 limit"
    assert "suspend" in src, "load_status_panels 应读 suspend"

    import factors.panel as fp
    src2 = inspect.getsource(fp.load_panel)
    for ds in ("daily", "adjust", "valuation"):
        assert ds in src2, f"load_panel 应读 {ds}"

    for ds in ("daily", "adjust", "valuation", "limit", "suspend"):
        assert ds in PANEL_DEPS, f"PANEL_DEPS 缺少 {ds}（缓存会漏判失效）"
    print(f"[OK] 缓存依赖清单覆盖面板实际读取的全部数据集：{list(PANEL_DEPS)}")


def test_parse_tushare_date_handles_mixed_formats():
    """**回归**：tushare 日期列里混着 'YYYYMMDD' 和 'YYYY-MM-DD HH:MM:SS'

    实测 `frozen/holders.ann_date` 约 1.75% 的行是带时间的格式。
    读取方若写死 `format="%Y%m%d"`，这些行会静默变 NaT 并被 dropna 丢掉。
    另外 pandas>=2 用 `pd.to_datetime(series)` 不带 format 时会**按第一个元素
    定格式**，混格式下后面的会被判成 NaT —— 必须 format="mixed"。
    """
    import pandas as pd
    from database.dates import is_date_column, parse_tushare_date

    s = pd.Series(["20230428", "2025-06-03 16:01:59", None, "", "nan",
                   "2024-01-02", "2023/05/06"])
    got = parse_tushare_date(s)
    assert got.iloc[0] == pd.Timestamp("2023-04-28")
    assert got.iloc[1] == pd.Timestamp("2025-06-03"), "带时间的格式未解析"
    assert pd.isna(got.iloc[2]) and pd.isna(got.iloc[3]) and pd.isna(got.iloc[4])
    assert got.iloc[5] == pd.Timestamp("2024-01-02"), \
        "同一批里混两种格式时后者被漏掉（pandas 按首元素定格式的坑）"
    assert got.iloc[6] == pd.Timestamp("2023-05-06")
    # 标量输入
    assert parse_tushare_date("2025-06-03 16:01:59") == pd.Timestamp("2025-06-03")
    # 列名判定必须严格后缀匹配：`update_flag` 含 "date" 但不是日期列
    assert is_date_column("ann_date") and is_date_column("trade_date")
    assert is_date_column("report_period")
    assert not is_date_column("update_flag"), "update_flag 被误判成日期列"
    assert not is_date_column("time_deposits")
    print("[OK] tushare 日期容错解析：YYYYMMDD / 带时间 / 斜杠 / 混格式 / 列名判定")


if __name__ == "__main__":
    import shutil

    test_fingerprint_changed_detects_mtime_and_count()
    test_parse_tushare_date_handles_mixed_formats()
    td = _tmpdir()
    try:
        test_panel_cache_roundtrip_and_invalidation(td)
        test_year_globs_only_lists_existing_partitions(td)
    finally:
        shutil.rmtree(td, ignore_errors=True)
    test_suspend_panel_actually_loads()
    test_panel_deps_cover_everything_load_panel_reads()
    print("\n全部数据指纹/缓存失效测试通过")
