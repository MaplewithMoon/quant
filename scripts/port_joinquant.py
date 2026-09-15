# -*- coding: utf-8 -*-
"""把聚宽原始策略文件转成可在本项目运行的版本

做法：**策略主体一行不改**，只替换文件头的 import 段。
这样策略逻辑与原版逐行一致，回测差异只可能来自数据与撮合，便于归因。

被替换掉的导入：
    from jqdata import *        -> from joinquant import *
    from jqdata import finance  -> 去掉（审计意见筛选默认关闭）
    from jqfactor import *      -> 去掉（未使用）
    from six import BytesIO     -> from io import BytesIO
另外策略 2 默认读聚宽上传的 tiny_index.csv（微盘股指数），本库没有这个文件，
改为使用代码里本来就写好的 399101.XSHE 分支 —— 这是**唯一一处逻辑改动**。
"""
import re
import sys
from pathlib import Path

ROOT = Path(r"D:\quant")
SRC = {
    "v1": ROOT / "动态持仓 止损线9 单价不超过50-Clone.py",
    "v2": ROOT / "添加399101可选-Clone.py",
}
OUT = ROOT / "strategies" / "jq"

HEADER = '''# -*- coding: utf-8 -*-
"""{title}

【移植说明】
    本文件由 scripts/port_joinquant.py 自动生成：
    **策略主体与原版逐行一致**，只替换了文件头的 import。
        from jqdata import *     -> from joinquant import *
        from jqfactor import *   -> 删除（未使用）
        from six import BytesIO  -> from io import BytesIO
    数据与撮合的近似见 docs/聚宽策略移植报告.md。

    原文件：{src}
    来源  ：{url}
"""
from io import BytesIO  # noqa: F401
from joinquant import *  # noqa: F401,F403

'''

# 需要从策略 2 里做的一处行为改动：微盘指数文件不存在，改用 399101 分支
V2_PATCH = [
    ("g.index = '800007.choice'",
     "g.index = '399101.XSHE'  # [移植改动] 微盘股指数 tiny_index.csv 本库没有，"
     "改用代码里自带的 399101 分支"),
]


def port(key: str) -> Path:
    src = SRC[key]
    text = src.read_text(encoding="utf-8")
    lines = text.splitlines()
    # 文件头的注释块（前若干行 # 开头）里提取标题/来源
    title, url = src.stem, ""
    for ln in lines[:8]:
        if "标题：" in ln:
            title = ln.split("标题：")[-1].strip()
        if "joinquant.com/post/" in ln:
            m = re.search(r"https://\S+", ln)
            url = m.group(0) if m else ""
    # 去掉所有 import 行
    body = []
    for ln in lines:
        s = ln.strip()
        if (s.startswith("from jqdata") or s.startswith("from jqfactor")
                or s.startswith("from six") or s.startswith("import six")):
            continue
        body.append(ln)
    out_text = HEADER.format(title=title, src=src.name, url=url) + "\n".join(body) + "\n"
    if key == "v2":
        for a, b in V2_PATCH:
            if a not in out_text:
                print(f"  [警告] 未找到待替换片段: {a}")
            out_text = out_text.replace(a, b)
    OUT.mkdir(parents=True, exist_ok=True)
    dst = OUT / (("guojiu_smallcap" if key == "v1" else "guojiu_smallcap_v2") + ".py")
    dst.write_text(out_text, encoding="utf-8")
    return dst


if __name__ == "__main__":
    for k in ("v1", "v2"):
        p = port(k)
        n = len(p.read_text(encoding="utf-8").splitlines())
        print(f"  {SRC[k].name}  ->  {p.relative_to(ROOT)}  ({n} 行)")
