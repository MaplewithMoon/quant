# -*- coding: utf-8 -*-
"""运行全部单元测试（也可用 pytest）"""
import sys
import io
import os
sys.path.insert(0, ".")
sys.path.insert(0, "tests")
if __name__ == "__main__":
    # 只在直接运行时切编码：模块顶层替换 sys.stdout 是有副作用的 import，
    # 会破坏 pytest 的输出捕获（详见 utils/console.py）
    from utils.console import force_utf8_stdout
    force_utf8_stdout()

import subprocess
import sys

# 自动发现，而不是硬编码清单。
# 硬编码清单会**静默漏掉**新加的测试文件（`tests/test_market.py` 加进来时就
# 没被发现，`run_tests.py` 一路绿灯但那个文件根本没跑）—— 而 pytest/CI 用的是
# 自动发现，两边结果不一致，本地"全绿"就变得不可信。
def _discover():
    from pathlib import Path
    d = Path(__file__).resolve().parent.parent / "tests"
    return sorted(f"tests/{p.name}" for p in d.glob("test_*.py"))


TESTS = _discover()


def main():
    print("=" * 60)
    print("运行全部单元测试")
    print("=" * 60)
    failures = 0
    for t in TESTS:
        print(f"\n>>> {t}")
        r = subprocess.run([sys.executable, t], capture_output=True, text=True, encoding="utf-8", errors="replace")
        print(r.stdout)
        if r.returncode != 0:
            print(r.stderr[-800:])
            failures += 1
    print("=" * 60)
    if failures:
        print(f"{failures} 个测试文件失败")
        sys.exit(1)
    print("全部测试通过")


if __name__ == "__main__":
    main()
