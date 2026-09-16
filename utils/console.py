# -*- coding: utf-8 -*-
"""控制台编码辅助（**只应在脚本入口调用**）

【为什么要有这个模块】
    Windows 控制台默认是 cp936/GBK，脚本里打印中文会抛 `UnicodeEncodeError`。
    历史上 19 个脚本各自在**模块顶层**写了

        sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

    这是有副作用的 import：任何 `import scripts.xxx` 都会替换调用方的
    `sys.stdout`。在 pytest 下后果是灾难性的 —— 被替换掉的旧 wrapper 被回收时
    会**关掉底层 buffer**（那正是 pytest 捕获输出用的临时文件），于是会话在
    收集完测试之后直接崩溃：

        ValueError: I/O operation on closed file.
        （collected 112 items → 一个测试都没跑，退出码 1）

    CI 在 ubuntu/windows 两个平台都是这么挂的。所以：
      - **不要**在模块顶层调用本函数
      - 只在 `if __name__ == "__main__":` 里调用
"""
import io
import sys

__all__ = ["force_utf8_stdout"]


def force_utf8_stdout(line_buffering: bool = True) -> bool:
    """把 stdout/stderr 切到 UTF-8，返回是否真的切换了

    满足以下任一条件时**不做任何事**（宁可不切，也不要弄坏调用方）：
      - 不在 Windows 上（Linux/macOS 默认就是 UTF-8）
      - 流已经被重定向成没有 `.buffer` 的对象（pytest 捕获、StringIO 等）
      - 当前流已经是 UTF-8
    """
    if sys.platform != "win32":
        return False
    changed = False
    for name in ("stdout", "stderr"):
        stream = getattr(sys, name, None)
        buf = getattr(stream, "buffer", None)
        if buf is None:
            # pytest 的捕获对象、io.StringIO 等：没有 buffer，强行替换会
            # 报 AttributeError 或破坏捕获，直接跳过
            continue
        enc = (getattr(stream, "encoding", "") or "").lower().replace("-", "")
        if enc == "utf8":
            continue
        try:
            setattr(sys, name, io.TextIOWrapper(buf, encoding="utf-8",
                                                line_buffering=line_buffering))
            changed = True
        except (ValueError, OSError):
            # 流已关闭等异常：不因为"改个编码"把脚本搞挂
            continue
    return changed
