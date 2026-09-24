# -*- coding: utf-8 -*-
"""批量为本仓库所有「同名 Markdown + PDF」生成目录书签。

用法：
    python scripts/add_all_bookmarks.py [--dry]

    --dry  只预览，不写入

说明：
    扫描仓库根目录下所有 .md 文件，若存在同名 .pdf，则调用
    add_pdf_bookmarks.py 为其生成书签。
"""
import os
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
SCRIPT = os.path.join(HERE, "add_pdf_bookmarks.py")


def main():
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

    dry = "--dry" in sys.argv
    pairs = []
    for name in sorted(os.listdir(ROOT)):
        if not name.endswith(".md"):
            continue
        md = os.path.join(ROOT, name)
        pdf = os.path.join(ROOT, name[:-3] + ".pdf")
        if os.path.exists(pdf):
            pairs.append((md, pdf))

    if not pairs:
        print("未找到「同名 Markdown + PDF」文件对。")
        return

    print(f"共找到 {len(pairs)} 对文件。\n")
    for md, pdf in pairs:
        cmd = [sys.executable, SCRIPT, md, pdf]
        if dry:
            cmd.append("--dry")
        subprocess.run(cmd, check=False)
        print()


if __name__ == "__main__":
    main()
