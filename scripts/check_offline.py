# -*- coding: utf-8 -*-
"""离线校验：把数据目录指到不存在的位置，跑"本该不依赖数据"的测试

【为什么需要这个脚本】
同一类 CI 失败已经出现两次：

  1. `test_market.py` 读真实 ETF 数据，本地过、CI 不过（已加存在性守卫）
  2. `test_resumption_and_suspension.py` 是**纯合成**用例，却隐式依赖
     `frozen/calendar` —— CI 里日历为空，`resumption_windows` 恒返回全 False，
     断言全废。**本地有 db 所以一直是绿的。**

第 2 类更危险：它看上去完全不该依赖本机数据，肉眼审不出来。

所以：**任何跑得动的测试都必须能在没有 `db/` 的环境下通过**（要么有守卫跳过，
要么把日历/主表这类依赖做成参数注入）。这个脚本把这件事变成可重复的操作，
不必等 CI 报错。

⚠️ 每个测试文件跑在**独立子进程**里（见 `_offline_child.py` 的说明）：
同一进程里"先 import 再改路径"是无效的，会误报。

用法：
    python scripts/check_offline.py                 # 默认集合
    python scripts/check_offline.py tests/test_xxx.py ...
"""
import json
import shutil
import subprocess
import sys
import uuid
from pathlib import Path

sys.path.insert(0, ".")
if __name__ == "__main__":
    from utils.console import force_utf8_stdout
    force_utf8_stdout()

# 默认检查这些"应该完全不依赖 db"的模块
DEFAULT = (
    "tests/test_result_report.py",
    "tests/test_style.py",
    "tests/test_gate_and_c5.py",
    "tests/test_defects.py",
    "tests/test_reliability_guards.py",
    "tests/test_resumption_and_suspension.py",   # **曾踩坑**
    "tests/test_limit_rules.py",                 # **曾踩坑**
    "tests/test_market_rules.py",
    "tests/test_atomic_write.py",
    "tests/test_cli.py",
)
MARK = "__OFFLINE_RESULT__"


def main(paths=None):
    root = Path(__file__).resolve().parent.parent
    paths = [p for p in (paths or DEFAULT) if (root / p).exists()]
    fake = root / f"__no_such_db_{uuid.uuid4().hex[:6]}__"
    child = root / "scripts" / "_offline_child.py"

    print("=" * 74)
    print(f"离线校验：{len(paths)} 个测试文件，数据目录指向不存在的路径（模拟 CI）")
    print("=" * 74)
    fail, ran = [], 0
    try:
        for p in paths:
            r = subprocess.run(
                [sys.executable, str(child), p, str(fake)],
                capture_output=True, text=True, encoding="utf-8",
                errors="replace", timeout=300, cwd=str(root),
                env={**__import__("os").environ, "PYTHONUTF8": "1",
                     "PYTHONIOENCODING": "utf-8"})
            line = next((ln for ln in (r.stdout or "").splitlines()
                         if ln.startswith(MARK)), None)
            if line is None:
                fail.append((p, "<no-result>",
                             ((r.stderr or r.stdout or "")[-200:]).replace("\n", " ")))
                continue
            res = json.loads(line[len(MARK):])
            ran += res["ran"]
            nf = len(res["fails"])
            for n, why in res["fails"]:
                fail.append((p, n, why))
            status = "✓" if not nf else "✗"
            tail = f"，{nf} 失败" if nf else ""
            print(f"  {status} {Path(p).name}: {res['ran']} 通过{tail}")
    finally:
        shutil.rmtree(fake, ignore_errors=True)

    print()
    if fail:
        print(f"✗ {len(fail)} 个失败（隐式依赖 db/，CI 会红）：")
        for p, n, why in fail:
            print(f"    {Path(p).name}::{n}\n        {why}")
        print()
        print("修法二选一：① 加存在性守卫，数据不在时干净跳过；")
        print("            ② 把日历/主表这类依赖做成**参数注入**（推荐 ——")
        print("               合成用例本就该与真实数据完全解耦）")
        return 1
    print(f"✓ {ran} 个测试在无 db/ 环境下全部通过")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:] or None))
