# -*- coding: utf-8 -*-
"""离线校验的子进程入口：在**进程启动时**就把数据目录指向不存在的路径

由 `scripts/check_offline.py` 调用，每个测试文件一个独立进程 ——
与 CI（`run_tests.py` 逐文件跑子进程）保持一致。

⚠️ 为什么必须独立进程：在同一个进程里"先 import 项目模块、再改路径"是**无效**的
—— `database.loader` 之类的模块在 import 时就把根路径抓成了自己的常量，
之后再改 `database.config` 影响不到它们。第一版 harness 就是这么误报的：
把 `test_market_rules.py` 判成隐式依赖 db，而 CI 里它其实是好的。
"""
import importlib.util
import json
import sys
from pathlib import Path


def main():
    test_path = sys.argv[1]
    fake = Path(sys.argv[2])          # 不存在的目录
    sys.path.insert(0, ".")

    # ⚠️ 必须在**任何**项目模块被 import 之前改掉。
    # 两个常量都要改：有的模块用 FROZEN_ROOT，有的用 DB_ROOT 拼路径。
    import database.config as cfg
    cfg.DB_ROOT = fake
    cfg.FROZEN_ROOT = fake / "frozen"

    spec = importlib.util.spec_from_file_location(Path(test_path).stem, test_path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = mod
    try:
        spec.loader.exec_module(mod)
    except Exception as e:
        print("__OFFLINE_RESULT__" + json.dumps(
            {"ran": 0, "fails": [("<import>", f"{type(e).__name__}: {e}")]}))
        return 0

    fails, ran = [], 0
    for n in [x for x in dir(mod) if x.startswith("test_")]:
        fn = getattr(mod, n)
        if not callable(fn):
            continue
        try:
            fn()
            ran += 1
        except Exception as e:
            fails.append((n, f"{type(e).__name__}: {str(e)[:160]}"))
    print("__OFFLINE_RESULT__" + json.dumps({"ran": ran, "fails": fails}))
    return 0


if __name__ == "__main__":
    sys.exit(main())
