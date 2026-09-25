"""校验 Markdown 文档中所有 Mermaid 代码块的语法结构。"""
import re
import sys
from pathlib import Path

MD = Path(__file__).resolve().parent.parent / "deep-learning-papers-principles.md"

VALID_TYPES = (
    "flowchart", "graph", "sequenceDiagram", "classDiagram",
    "stateDiagram", "erDiagram", "gantt", "pie", "mindmap",
)

text = MD.read_text(encoding="utf-8")
blocks = re.findall(r"```mermaid\r?\n(.*?)```", text, re.S)
print(f"找到 {len(blocks)} 个 mermaid 块")

issues = 0
for i, code in enumerate(blocks, 1):
    lines = [ln for ln in code.splitlines() if ln.strip()]
    if not lines:
        print(f"[{i}] 空块")
        issues += 1
        continue
    head = lines[0].strip()
    if not head.startswith(VALID_TYPES):
        print(f"[{i}] 非法图类型: {head!r}")
        issues += 1

    # 括号配对（忽略引号内的内容）
    stripped = re.sub(r'"[^"]*"', '""', code)
    for op, cl, name in (("[", "]", "方括号"), ("(", ")", "圆括号"), ("{", "}", "花括号")):
        if stripped.count(op) != stripped.count(cl):
            print(f"[{i}] {name}不配对: {stripped.count(op)} vs {stripped.count(cl)}")
            issues += 1
    if code.count('"') % 2:
        print(f"[{i}] 引号不配对")
        issues += 1

    # subgraph / end 配对
    sg = len(re.findall(r"(?m)^\s*subgraph\b", code))
    en = len(re.findall(r"(?m)^\s*end\s*$", code))
    if sg != en:
        print(f"[{i}] subgraph/end 不配对: {sg} vs {en}")
        issues += 1

    # 保留字作为节点 ID
    for kw in ("end", "graph", "subgraph", "class", "click", "style"):
        if re.search(rf"(?m)^\s*{kw}\s*[\[\(\{{]", code):
            print(f"[{i}] 保留字 {kw!r} 被用作节点 ID")
            issues += 1

print(f"\n发现问题数: {issues}")
sys.exit(1 if issues else 0)