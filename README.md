# Digital-Image-Processing

数字图像处理学习笔记与知识速查仓库，涵盖颜色空间理论、色度子采样、图像描述子、CNN / Transformer 原理以及 JPEG / JPEG 2000 压缩原理等主题。每份笔记均提供 Markdown 源文件与对应的 PDF 版本。

## 文档目录

| 文档 | 说明 |
|---|---|
| [颜色空间理论清单](color-space-theory-checklist.md) | 从定义、数学原理、物理基础到使用范围，系统梳理颜色空间：光谱与三刺激假设、RGB/XYZ/Lab/YCbCr 等坐标体系及其适用场景。 |
| [YUV 家族与色度子采样](yuv-family-chroma-subsampling.md) | 厘清 YUV、YPbPr、YIQ、YCbCr 四个"亮度-色差"色彩空间的关系与用途，详解 4:4:4 / 4:2:2 / 4:2:0 / 4:1:1 采样实现，并区分 Y'（luma）与 Y（luminance）。 |
| [图像描述子综述](feature-descriptors-notes.md) | 按"全局 → 局部 → 纹理 → 学习型 → 颜色"脉络组织，区分手工设计与学习型两大分支，覆盖 SIFT、SURF、ORB、HOG、LBP、VLAD、SuperPoint 等描述子的原理、分类与应用；每个公式附逐项符号解释，并给出各描述子的特征提取处理步骤与作用。 |
| [CNN 与 Transformer 原理详解](cnn-transformer-notes.md) | 从卷积到注意力，系统讲解 CNN 与 Transformer 的数学原理、关键组件、反向传播、架构演进与高效注意力，并给出两者统一视角及其与特征描述子的关系；每个公式附逐项符号解释。 |
| [JPEG 与 JPEG 2000 压缩原理详解](jpeg-jpeg2000-knowledge-reference.md) | 独立知识速查文件：JPEG 的 DCT、量化、ZigZag、霍夫曼/算术编码、文件结构与伪影，以及 JPEG 2000 的小波变换、EBCOT、渐进性与 ROI，并附对比选型。 |

> 各文档的 PDF 版本与同名 Markdown 文件位于同一目录，便于离线阅读与打印。
> 所有 PDF 均已写入**目录书签（outline）**，可在阅读器的书签面板中按层级跳转。

## 工具脚本

`scripts/` 目录下提供 PDF 目录书签生成工具。

| 脚本 | 说明 |
|---|---|
| [`scripts/add_pdf_bookmarks.py`](scripts/add_pdf_bookmarks.py) | 为单个 PDF 生成目录书签：从同名 Markdown 解析 1~4 级标题，再按 PDF 字号（h1=36 / h2=28 / h3=24 / h4=20）识别标题行，两者对齐后写入 PDF outline。 |
| [`scripts/add_all_bookmarks.py`](scripts/add_all_bookmarks.py) | 批量处理：扫描仓库根目录下所有「同名 Markdown + PDF」文件对，逐个生成书签。 |

### 依赖

```bash
pip install pypdf
```

### 用法

```bash
# 单个文件（--dry 只预览前 15 个书签，不写入）
python scripts/add_pdf_bookmarks.py cnn-transformer-notes.md cnn-transformer-notes.pdf
python scripts/add_pdf_bookmarks.py feature-descriptors-notes.md feature-descriptors-notes.pdf --dry

# 批量处理全部文件对
python scripts/add_all_bookmarks.py
python scripts/add_all_bookmarks.py --dry
```

### 说明

- 脚本**原地**更新 PDF（先写 `<pdf>.tmp` 再替换），重复运行不会叠加旧书签。
- 标题在 PDF 中折行时会自动合并续行；行内代码 / 行内公式导致的字号混排也能正确判定层级。
- Windows 控制台默认 GBK，标题含生僻字时建议先执行
  `[Console]::OutputEncoding=[Text.Encoding]::UTF8`，否则打印可能乱码。
- 字号约定需与 Markdown → PDF 的导出样式一致；若导出工具不同，请调整脚本中的 `SIZE_TO_LEVEL`。

## 许可说明

本项目采用 [MIT License](LICENSE) 授权。

Copyright (c) 2026 tianheju

你可以自由地使用、复制、修改、合并、发布、分发、再授权和/或销售本项目的副本，但需在副本或实质性部分中保留上述版权声明与许可声明。本软件按"原样"提供，不附带任何形式的明示或暗示担保。
