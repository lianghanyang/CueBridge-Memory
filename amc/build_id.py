"""BUILD id（2026-11-02）

为什么单独成模块：**app.py 和 service.py 都要用**（缓存键要用它），
而 app.py 已经 import service ⇒ 从 service 里 import app 会循环。

计算范围：`amc/**/*.py`（**递归**）—— 2026-11-02 修过的坑：
原来只 `os.listdir(amc/)`，漏掉 `amc/profile/*.py`，导致改了 profile 后 BUILD 不变。
"""
from __future__ import annotations

import hashlib
import os


def compute_build() -> str:
    h = hashlib.sha1()
    try:
        base = os.path.dirname(os.path.abspath(__file__))
        files = []
        for root, _dirs, names in os.walk(base):
            for n in names:
                if n.endswith(".py"):
                    files.append(os.path.join(root, n))
        for fp in sorted(files):
            h.update(open(fp, "rb").read())
    except Exception:
        return "unknown"
    return h.hexdigest()[:12]


BUILD = compute_build()
