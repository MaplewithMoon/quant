# -*- coding: utf-8 -*-
"""CLI 冒烟测试：每个脚本的 `--help` 必须能正常跑完

【为什么需要这个文件】
T0 那一版给 `portfolio_backtest.py` 加风控参数时，**重复定义了 `--max-weight`**
（第 75 行的组合构建上限 + 新加的风控上限），argparse 直接抛
`ArgumentError: argument --max-weight: conflicting option string`。

结果：这个脚本的 `--help` **坏了好几个提交都没人发现** ——
因为它不在任何测试的覆盖范围里，而单元测试都直接调库、不碰 CLI。

这类 bug（参数重复、import 顺序错、模块级副作用）用一条 `--help` 就能全抓出来，
成本极低。所以：**所有 `scripts/*.py` 的 `--help` 都必须能跑通。**

`--help` 是安全的：argparse 打印后以 0 退出，不会真的跑回测、不写文件、不联网。
"""
import os
import subprocess
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# 有 CLI 的脚本 = 源码里出现 `ArgumentParser`。
# **不用手工 SKIP 名单**：手写的名单会随脚本增删漂移，而且漏掉一个就会
# 把"没有 argparse 的脚本"误判成失败 —— 第一版就是这么错的：
#   backtest_demo.py 只 print 不解析参数 -> 报"没有打印 usage"
#   run_tests.py 的 --help 被忽略 -> **真的跑起全套测试**，120 秒超时
def _scripts():
    from pathlib import Path
    d = Path(__file__).resolve().parent.parent / "scripts"
    out = []
    for p in sorted(d.glob("*.py")):
        if p.name.startswith("_"):
            continue
        try:
            if "ArgumentParser" in p.read_text(encoding="utf-8", errors="ignore"):
                out.append(p)
        except OSError:
            continue
    return out


def test_all_scripts_help_runs():
    """每个脚本的 --help 必须退出码 0 且打印 usage"""
    bad = []
    for p in _scripts():
        env = dict(os.environ)
        env["PYTHONUTF8"] = "1"
        env["PYTHONIOENCODING"] = "utf-8"
        try:
            r = subprocess.run([sys.executable, str(p), "--help"],
                               capture_output=True, text=True, encoding="utf-8",
                               errors="replace", timeout=120, env=env,
                               cwd=str(p.parent.parent))
        except Exception as e:
            bad.append((p.name, f"{type(e).__name__}: {e}"))
            continue
        out = (r.stdout or "") + (r.stderr or "")
        if r.returncode != 0:
            tail = " / ".join(out.strip().splitlines()[-2:])[:220]
            bad.append((p.name, f"退出码 {r.returncode}: {tail}"))
        elif "usage" not in out.lower():
            bad.append((p.name, "没有打印 usage"))
    assert not bad, "以下脚本的 --help 跑不通：\n" + "\n".join(
        f"  {n}: {why}" for n, why in bad)
    print(f"[OK] {len(_scripts())} 个脚本的 --help 全部正常")


def test_no_duplicate_cli_flags():
    """**回归**：不允许重复定义同名参数

    重复定义会让 argparse 在 `parse_args`/`--help` 时抛
    `conflicting option string`。上面那条已经能间接抓到，这里再显式钉一次，
    让失败信息直指"哪个脚本、哪个参数"。
    """
    from pathlib import Path

    import re
    dups = []
    for p in _scripts():
        txt = p.read_text(encoding="utf-8", errors="ignore")
        flags = re.findall(r'add_argument\(\s*"(--[a-zA-Z0-9\-]+)"', txt)
        seen, dup = set(), set()
        for f in flags:
            if f in seen:
                dup.add(f)
            seen.add(f)
        if dup:
            dups.append((p.name, sorted(dup)))
    assert not dups, "以下脚本重复定义了 CLI 参数：\n" + "\n".join(
        f"  {n}: {d}" for n, d in dups)
    print(f"[OK] {len(_scripts())} 个脚本无重复 CLI 参数")


def test_execution_args_available_on_engine_scripts():
    """用了引擎的回测脚本必须能拿到统一执行参数（T1·② 的回归）

    新脚本容易"忘了接执行参数"，导致零滑点/无冲击模型而结果偏乐观。
    """
    import subprocess

    from pathlib import Path
    root = Path(__file__).resolve().parent.parent
    need = ("backtest.py", "portfolio_backtest.py", "multifactor_backtest.py",
            "sector_rotation_backtest.py", "optimize.py")
    missing = []
    for name in need:
        p = root / "scripts" / name
        if not p.exists():
            continue
        env = dict(os.environ, PYTHONUTF8="1", PYTHONIOENCODING="utf-8")
        r = subprocess.run([sys.executable, str(p), "--help"],
                           capture_output=True, text=True, encoding="utf-8",
                           errors="replace", timeout=120, env=env, cwd=str(root))
        out = (r.stdout or "") + (r.stderr or "")
        for flag in ("--fill-timing", "--slippage", "--impact-model"):
            if flag not in out:
                missing.append(f"{name} 缺 {flag}")
    assert not missing, "执行参数缺失（会造成零摩擦/偏乐观结果）：\n  " + \
        "\n  ".join(missing)
    print(f"[OK] {len(need)} 个回测脚本都暴露了统一的执行参数")


if __name__ == "__main__":
    test_all_scripts_help_runs()
    test_no_duplicate_cli_flags()
    test_execution_args_available_on_engine_scripts()
    print("\n全部 CLI 冒烟测试通过")
