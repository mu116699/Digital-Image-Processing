# -*- coding: utf-8 -*-
"""将 Markdown 文件转换为带目录书签的 PDF。

流程：
1. Markdown → HTML（Python markdown 库 + mermaid.js + KaTeX CDN）
2. HTML → PDF（headless Edge/Chrome，--virtual-time-budget 等待 JS 渲染）
3. PDF 添加书签目录（add_pdf_bookmarks.py）

用法：
    python scripts/md_to_pdf.py [markdown_file]

    不带参数时默认处理 deep-learning-papers-principles.md
    带参数时处理指定的 Markdown 文件，如：
    python scripts/md_to_pdf.py morphological-image-processing.md
"""
import markdown
import re
import os
import sys
import subprocess
import argparse
from pathlib import Path

try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass

ROOT = Path(__file__).resolve().parent.parent

# ── 解析命令行参数 ───────────────────────────────────────────────────────────
parser = argparse.ArgumentParser(description="Markdown → PDF（带书签）")
parser.add_argument(
    "md_file",
    nargs="?",
    default="deep-learning-papers-principles.md",
    help="Markdown 文件名（位于仓库根目录），默认: deep-learning-papers-principles.md",
)
args = parser.parse_args()

MD_FILE = ROOT / args.md_file
if not MD_FILE.exists():
    print(f"错误：文件不存在: {MD_FILE}")
    sys.exit(1)

STEM = MD_FILE.stem
HTML_FILE = ROOT / f"{STEM}.html"
PDF_FILE = ROOT / f"{STEM}.pdf"

# ── CSS ──────────────────────────────────────────────────────────────────────
# 标题字号（pt）须与 add_pdf_bookmarks.py 的 SIZE_TO_LEVEL 一致：
#   h1=36  h2=28  h3=24  h4=20
CSS = r"""
@page {
    margin: 20mm 15mm 20mm 15mm;
    size: A4;
}
* { box-sizing: border-box; }
body {
    font-family: "Microsoft YaHei", "PingFang SC", "Noto Sans CJK SC", sans-serif;
    font-size: 11pt;
    line-height: 1.7;
    color: #333;
    margin: 0;
    padding: 0;
}

/* ── 标题 ── */
h1 {
    font-size: 36pt;
    color: #1a1a2e;
    text-align: center;
    border-bottom: 3px solid #16213e;
    padding-bottom: 12px;
    margin: 0 0 20px 0;
}
h2 {
    font-size: 28pt;
    color: #16213e;
    border-bottom: 2px solid #0f3460;
    padding-bottom: 8px;
    margin: 30px 0 15px 0;
    page-break-before: always;
}
h2:first-of-type { page-break-before: avoid; }
h3 {
    font-size: 24pt;
    color: #0f3460;
    margin: 25px 0 10px 0;
    page-break-after: avoid;
}
h4 {
    font-size: 20pt;
    color: #1a1a2e;
    margin: 20px 0 8px 0;
    page-break-after: avoid;
}

/* ── 段落 ── */
p { margin: 8px 0; text-align: justify; }

/* 粗体段落标记（核心思想、关键设计等）独占一行时加大加粗 */
p > strong:only-child {
    display: block;
    font-size: 13pt;
    color: #0f3460;
    margin-top: 12px;
    margin-bottom: 4px;
}

/* ── 列表 ── */
ul, ol { margin: 8px 0; padding-left: 2em; }
li { margin: 4px 0; }

/* ── 代码 ── */
code {
    font-family: "Consolas", "Courier New", monospace;
    font-size: 10pt;
    background-color: #f0f0f0;
    padding: 1px 4px;
    border-radius: 3px;
}
pre {
    background-color: #f8f8f8;
    border: 1px solid #e0e0e0;
    border-radius: 4px;
    padding: 12px;
    overflow-x: auto;
    page-break-inside: avoid;
    margin: 12px 0;
}
pre code {
    background-color: transparent;
    padding: 0;
    font-size: 9.5pt;
    line-height: 1.5;
}

/* ── Mermaid 图 ── */
pre.mermaid, .mermaid {
    text-align: center;
    background-color: white;
    border: none;
    padding: 10px 0;
}
.mermaid svg { max-width: 100%; height: auto; }

/* ── 表格 ── */
table {
    border-collapse: collapse;
    width: 100%;
    margin: 15px 0;
    font-size: 10pt;
    page-break-inside: avoid;
}
th, td { border: 1px solid #ccc; padding: 6px 10px; text-align: left; }
th { background-color: #e8e8e8; font-weight: bold; }
tr:nth-child(even) { background-color: #fafafa; }

/* ── 引用 ── */
blockquote {
    border-left: 4px solid #0f3460;
    background-color: #f5f5fa;
    padding: 10px 15px;
    margin: 15px 0;
    color: #555;
}

/* ── 链接 / 分隔线 ── */
a { color: #0f3460; text-decoration: none; }
hr { border: none; border-top: 1px solid #ddd; margin: 20px 0; }

/* ── 打印优化 ── */
h2, h3, h4 { page-break-after: avoid; }
table, pre, blockquote { page-break-inside: avoid; }
"""

# ── HTML 模板 ────────────────────────────────────────────────────────────────
HTML_TEMPLATE = r"""<!DOCTYPE html>
<html lang="zh-CN">
<head>
    <meta charset="UTF-8">
    <title>__TITLE__</title>
    <style>__CSS__</style>
</head>
<body>
__BODY__

<!-- KaTeX -->
<link rel="stylesheet" href="https://cdn.jsdelivr.net/npm/katex@0.16.9/dist/katex.min.css">
<script src="https://cdn.jsdelivr.net/npm/katex@0.16.9/dist/katex.min.js"></script>
<script src="https://cdn.jsdelivr.net/npm/katex@0.16.9/dist/contrib/auto-render.min.js"></script>
<!-- Mermaid -->
<script src="https://cdn.jsdelivr.net/npm/mermaid@10/dist/mermaid.min.js"></script>

<script>
document.addEventListener('DOMContentLoaded', function() {
    // KaTeX 渲染
    if (typeof renderMathInElement !== 'undefined') {
        renderMathInElement(document.body, {
            delimiters: [
                {left: '$$', right: '$$', display: true},
                {left: '$', right: '$', display: false}
            ],
            throwOnError: false
        });
    }
    // Mermaid 渲染
    if (typeof mermaid !== 'undefined') {
        mermaid.initialize({ startOnLoad: false, theme: 'default' });
        mermaid.run();
    }
});
</script>
</body>
</html>
"""


# ── 转换函数 ────────────────────────────────────────────────────────────────

def preprocess_markdown(md_text):
    """确保独占一行的 **粗体** 标记后面有空行，使其成为独立段落。"""
    return re.sub(
        r'^(\*\*[^\n]+\*\*)\n(?!\n)',
        r'\1\n\n',
        md_text,
        flags=re.MULTILINE,
    )


def md_to_html(md_text):
    """Markdown → HTML，并将 mermaid 代码块转为 <pre class="mermaid">。"""
    md = markdown.Markdown(extensions=[
        'fenced_code',
        'tables',
        'attr_list',
        'def_list',
    ])
    html = md.convert(md_text)
    # <pre><code class="language-mermaid">…</code></pre> → <pre class="mermaid">…</pre>
    html = re.sub(
        r'<pre><code class="language-mermaid">(.*?)</code></pre>',
        r'<pre class="mermaid">\1</pre>',
        html,
        flags=re.DOTALL,
    )
    return html


def find_browser():
    """查找可用的浏览器（Edge 优先于 Chrome）。"""
    candidates = [
        r"C:\Program Files (x86)\Microsoft\Edge\Application\msedge.exe",
        r"C:\Program Files\Microsoft\Edge\Application\msedge.exe",
        r"C:\Program Files\Google\Chrome\Application\chrome.exe",
        r"C:\Program Files (x86)\Google\Chrome\Application\chrome.exe",
    ]
    for p in candidates:
        if os.path.exists(p):
            return p
    return None


def main():
    # 1. 读取 Markdown
    print(f"[1/4] 读取: {MD_FILE.name}")
    md_text = MD_FILE.read_text(encoding="utf-8")
    md_text = preprocess_markdown(md_text)

    # 2. 转换为 HTML
    print("[2/4] 转换 Markdown → HTML ...")
    html_body = md_to_html(md_text)
    # 从 Markdown 第一行 # 标题 提取文档标题
    title_match = re.search(r'^#\s+(.+?)\s*$', md_text, re.MULTILINE)
    doc_title = title_match.group(1) if title_match else STEM
    full_html = (HTML_TEMPLATE
                 .replace("__CSS__", CSS)
                 .replace("__TITLE__", doc_title)
                 .replace("__BODY__", html_body))
    HTML_FILE.write_text(full_html, encoding="utf-8")
    print(f"      HTML 已保存: {HTML_FILE.name} ({len(full_html) // 1024} KB)")

    # 3. 查找浏览器并生成 PDF
    browser = find_browser()
    if not browser:
        print("错误：未找到 Edge 或 Chrome，请安装后重试。")
        sys.exit(1)
    print(f"[3/4] 生成 PDF（{Path(browser).name}，等待 Mermaid + KaTeX 渲染）...")

    html_uri = HTML_FILE.as_uri()
    cmd = [
        browser,
        '--headless',
        '--disable-gpu',
        '--no-sandbox',
        '--disable-extensions',
        '--hide-scrollbars',
        '--no-pdf-header-footer',
        f'--print-to-pdf={PDF_FILE}',
        '--virtual-time-budget=180000',
        html_uri,
    ]
    try:
        subprocess.run(cmd, check=True, timeout=600)
    except subprocess.CalledProcessError as e:
        print(f"      PDF 生成失败: {e}")
        sys.exit(1)
    except subprocess.TimeoutExpired:
        print("      PDF 生成超时（5 分钟）")
        sys.exit(1)

    if not PDF_FILE.exists():
        print("      错误：PDF 文件未生成")
        sys.exit(1)
    size_mb = PDF_FILE.stat().st_size / (1024 * 1024)
    print(f"      PDF 已保存: {PDF_FILE.name} ({size_mb:.1f} MB)")

    # 4. 添加书签
    print("[4/4] 添加 PDF 书签目录 ...")
    bookmark_script = ROOT / "scripts" / "add_pdf_bookmarks.py"
    try:
        result = subprocess.run(
            [sys.executable, str(bookmark_script), str(MD_FILE), str(PDF_FILE)],
            check=True,
            timeout=120,
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
        )
        # 打印书签脚本输出
        if result.stdout:
            for line in result.stdout.strip().splitlines():
                print(f"      {line}")
    except subprocess.CalledProcessError as e:
        print(f"      书签添加失败（PDF 已生成，可手动添加）:")
        if e.stdout:
            for line in e.stdout.strip().splitlines()[:10]:
                print(f"      {line}")
    except subprocess.TimeoutExpired:
        print("      书签添加超时（PDF 已生成）")

    print(f"\n✅ 完成！PDF: {PDF_FILE}")


if __name__ == "__main__":
    main()
