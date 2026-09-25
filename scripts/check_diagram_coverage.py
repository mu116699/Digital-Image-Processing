"""检查文档中每个「核心思想」段落是否紧跟 Mermaid 架构图。"""
import re
from pathlib import Path

MD = Path(__file__).resolve().parent.parent / "deep-learning-papers-principles.md"
lines = MD.read_text(encoding="utf-8").splitlines()

# 记录所有 ### 与 #### 标题行号
headings = [(i, ln) for i, ln in enumerate(lines) if ln.startswith("### ") or ln.startswith("#### ")]
missing = []
total = 0

for idx, (i, ln) in enumerate(headings):
    end = headings[idx + 1][0] if idx + 1 < len(headings) else len(lines)
    body = "\n".join(lines[i:end])
    if "**核心思想**" not in body:
        continue
    total += 1
    # 核心思想 之后到 关键设计 之间是否有 mermaid
    m = re.search(r"\*\*核心思想\*\*(.*?)(?=\*\*关键设计\*\*|\Z)", body, re.S)
    if not m or "```mermaid" not in m.group(1):
        missing.append((i + 1, ln))

print(f"含「核心思想」的小节总数: {total}")
print(f"缺少架构图的小节数: {len(missing)}\n")
for ln_no, title in missing:
    print(f"  行 {ln_no}: {title}")