# -*- coding: utf-8 -*-
"""为笔记 PDF 添加目录书签（PDF outline / bookmarks）。

思路：
1. 从同名 Markdown 源文件解析 1~4 级标题，得到「干净的标题文本 + 层级」。
2. 从 PDF 中按**行**识别标题：先按 y 坐标把文本片段聚成行，
   再判定该行的标题层级（36=h1, 28=h2, 24=h3, 20=h4）。
   标题里可能混入不同字号的片段：
   - 行内代码为标题字号的 0.85 倍（h3 里的 `xxx` 为 20.4，h4 里的 `xxx` 为 17.0）；
   - 行内公式约为标题字号的 1.21 倍（h4 里的 $R_b$ 为 24.2，下标 b 为 16.9）。
   因此判定规则为：**优先取行内最大字号**（行内代码比标题小，不影响最大值）；
   若最大字号不是基准字号，说明是行内公式把字号撑大了，
   此时**回退到字符数最多的字号**（公式通常比标题文字短）。
   两种情形举例：
   - h3「6.1 composite_curve」：max=24.0（基准）→ h3；
   - h4「1. $R_b$：理论动态范围」：max=24.2（非基准）→ 回退众数 20.0 → h4。
3. 两者按顺序对齐，用归一化文本校验；标题在 PDF 中折行时自动合并续行。
4. 用 pypdf 把标题树写入 PDF 的 outline。

依赖：
    pip install pypdf

用法：
    python scripts/add_pdf_bookmarks.py <markdown> <pdf> [--dry]

    --dry  只预览前 15 个书签，不写入文件

示例：
    python scripts/add_pdf_bookmarks.py cnn-transformer-notes.md cnn-transformer-notes.pdf
    python scripts/add_pdf_bookmarks.py feature-descriptors-notes.md feature-descriptors-notes.pdf --dry

注意：
    - 脚本会**原地**更新 PDF（先写 <pdf>.tmp 再替换），重复运行不会叠加旧书签。
    - Windows 控制台默认 GBK，标题含生僻字时需先执行
      `[Console]::OutputEncoding=[Text.Encoding]::UTF8`，否则打印可能乱码。
    - 字号约定（h1=36, h2=28, h3=24, h4=20）需与 Markdown → PDF 的导出样式一致，
      若导出工具不同，请相应调整 SIZE_TO_LEVEL。
"""
import io
import os
import re
import sys
import unicodedata

from pypdf import PdfReader, PdfWriter

# 字号 -> 标题层级
# 原始约定（wkhtmltopdf 等工具）：h1=36, h2=28, h3=24, h4=20
# headless Chrome/Edge 会将 CSS pt 放大 4/3 倍（96/72 DPI 转换）：
#   h1=48, h2=37.3, h3=32, h4=26.7
SIZE_TO_LEVEL = {
    36.0: 1, 28.0: 2, 24.0: 3, 20.0: 4,   # wkhtmltopdf
    48.0: 1, 37.3: 2, 32.0: 3, 26.7: 4,   # headless Chrome/Edge
}
# 同一行内文本片段的 y 坐标容差
Y_TOLERANCE = 3.0

# CJK 部首补充区（U+2E80~U+2EFF）的兼容字符：NFKC 不会归一化，需手工映射
RADICAL_MAP = str.maketrans({
    "⻆": "角", "⻅": "见", "⻋": "车", "⻓": "长", "⻔": "门",
    "⻘": "青", "⻜": "飞", "⻢": "马", "⻥": "鱼", "⻦": "鸟",
    "⻩": "黄", "⻮": "齿", "⻰": "龙", "⻳": "龟", "⻄": "西",
    "⻉": "贝", "⻒": "金", "⻝": "食", "⻟": "飠", "⻤": "鬼",
    "⻣": "骨", "⻚": "页", "⻛": "风", "⻞": "食", "⻡": "首",
    "⻢": "马", "⻧": "卤", "⻪": "黾", "⻫": "齐", "⻬": "齐",
})


def norm(s: str) -> str:
    """归一化：NFKC（还原康熙部首等兼容字符）+ 手工映射 CJK 部首补充区 + 去空白。"""
    s = unicodedata.normalize("NFKC", s)
    s = s.translate(RADICAL_MAP)
    return "".join(s.split())


def clean_md_title(text: str) -> str:
    """去掉 Markdown 行内标记，得到纯文本标题。"""
    text = re.sub(r"\[([^\]]*)\]\([^)]*\)", r"\1", text)  # 链接
    text = text.replace("**", "").replace("__", "")
    text = text.replace("`", "")
    return text.strip()


def parse_md_headings(md_path: str):
    """返回 [(level, title), ...]。

    跳过围栏代码块（``` 或 ~~~）和缩进代码块内的行，
    避免将代码中的 # 注释误识别为 Markdown 标题。
    """
    out = []
    in_fence = False
    fence_marker = None
    with io.open(md_path, encoding="utf-8") as f:
        for line in f:
            stripped = line.strip()
            # 检测围栏代码块的开始/结束
            if stripped.startswith("```") or stripped.startswith("~~~"):
                marker = stripped[:3]
                if not in_fence:
                    in_fence = True
                    fence_marker = marker
                elif marker == fence_marker:
                    in_fence = False
                    fence_marker = None
                continue
            if in_fence:
                continue
            # 跳过缩进代码块（4+ 空格或 tab 开头，且非列表项）
            if (line.startswith("    ") or line.startswith("\t")) and not re.match(r'^\s*[-*+\d]', line):
                continue
            m = re.match(r"^(#{1,4})\s+(.+?)\s*$", line)
            if m:
                out.append((len(m.group(1)), clean_md_title(m.group(2))))
    return out


def line_font_size(runs):
    """判定该行的标题字号（见模块 docstring 第 2 条）。

    优先取行内最大字号；若它不是基准字号（被行内公式撑大），
    则回退到字符数最多的字号。
    """
    max_fs = max(r[1] for r in runs)
    if max_fs in SIZE_TO_LEVEL:
        return max_fs
    counts = {}
    for _, fs, t in runs:
        counts[fs] = counts.get(fs, 0) + len(t)
    return max(counts.items(), key=lambda kv: (kv[1], kv[0]))[0]


def extract_pdf_headings(pdf_path: str):
    """返回 [(page_index0, level, text), ...]。

    先按 y 坐标把文本片段聚成行，再取行内众数字号判定标题层级。
    """
    reader = PdfReader(pdf_path)
    result = []
    for pno, page in enumerate(reader.pages):
        runs = []

        def visitor(text, cm, tm, font_dict, font_size):
            if not text.strip():
                return
            runs.append((tm[5], tm[4], round(font_size, 1), text))

        page.extract_text(visitor_text=visitor)

        # 按 y 升序（PDF 中 y 向下增大）聚成行
        lines = []
        for y, x, fs, t in sorted(runs, key=lambda r: (r[0], r[1])):
            for ln in lines:
                if abs(ln["y"] - y) <= Y_TOLERANCE:
                    ln["runs"].append((x, fs, t))
                    break
            else:
                lines.append({"y": y, "runs": [(x, fs, t)]})

        for ln in lines:
            fs = line_font_size(ln["runs"])
            if fs in SIZE_TO_LEVEL:
                text = "".join(r[2] for r in sorted(ln["runs"], key=lambda r: r[0]))
                if text.strip():
                    result.append((pno, SIZE_TO_LEVEL[fs], text))
    return result


def build_outline(md_headings, pdf_headings):
    """把 md 标题与 pdf 标题对齐，返回 [(level, title, page_index0), ...]。

    以 md 标题为准逐项消费 pdf 标题；若某个标题在 PDF 中折成多行，
    则把后续行拼接进来，直到与 md 标题完全一致（拼接不成功则不合并）。
    """
    n_pdf = len(pdf_headings)
    items = []
    mismatches = 0
    i = 0
    for md_level, md_title in md_headings:
        if i >= n_pdf:
            print(f"  ! PDF 标题已用尽，无法定位：{md_title}")
            break
        page, _pdf_level, pdf_text = pdf_headings[i]

        # 折行标题：向后拼接续行，仅当能拼出与 md 完全一致的文本时才合并
        acc = pdf_text
        if norm(acc) != norm(md_title):
            j = i
            while j + 1 < n_pdf and norm(md_title).startswith(norm(acc)):
                j += 1
                acc += pdf_headings[j][2]
                if norm(acc) == norm(md_title):
                    i = j
                    break

        if norm(acc) != norm(md_title):
            mismatches += 1
            if mismatches <= 10:
                print(f"  ! 第 {i} 项不匹配：\n      md : {md_title}\n      pdf: {norm(acc)}")
        items.append((md_level, md_title, page))
        i += 1

    if i < n_pdf:
        print(f"  ! PDF 中还有 {n_pdf - i} 个标题未被使用")
    if mismatches:
        print(f"  ! 共 {mismatches} 项文本不匹配（仍按顺序对齐）")
    return items


def write_bookmarks(pdf_path: str, items, out_path: str):
    reader = PdfReader(pdf_path)
    writer = PdfWriter()
    # import_outline=False：丢弃源 PDF 已有的书签，保证重复运行不会叠加
    writer.append(reader, import_outline=False)

    stack = []  # [(level, outline_ref)]
    for level, title, page in items:
        while stack and stack[-1][0] >= level:
            stack.pop()
        parent = stack[-1][1] if stack else None
        ref = writer.add_outline_item(title, page, parent=parent)
        stack.append((level, ref))

    with open(out_path, "wb") as f:
        writer.write(f)


def main():
    # Windows 控制台默认 GBK，标题里的生僻字/兼容字符会导致打印报错
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass
    args = [a for a in sys.argv[1:] if not a.startswith("--")]
    dry = "--dry" in sys.argv
    if len(args) < 2:
        print(__doc__)
        sys.exit(1)
    md_path, pdf_path = args[0], args[1]
    print(f"处理：{os.path.basename(pdf_path)}")
    md_headings = parse_md_headings(md_path)
    pdf_headings = extract_pdf_headings(pdf_path)
    print(f"  md 标题 {len(md_headings)} 个，pdf 标题 {len(pdf_headings)} 个")
    items = build_outline(md_headings, pdf_headings)

    if dry:
        print("  [dry-run] 前 15 个书签：")
        for level, title, page in items[:15]:
            print(f"    {'  ' * (level - 1)}L{level} p{page + 1}  {title}")
        return

    tmp = pdf_path + ".tmp"
    write_bookmarks(pdf_path, items, tmp)
    os.replace(tmp, pdf_path)
    print(f"  已写入 {len(items)} 个书签 -> {os.path.basename(pdf_path)}")


if __name__ == "__main__":
    main()
