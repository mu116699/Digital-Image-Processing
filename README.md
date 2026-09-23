# Digital-Image-Processing

数字图像处理学习笔记与知识速查仓库，涵盖颜色空间理论、色度子采样、图像描述子以及 JPEG / JPEG 2000 压缩原理等主题。每份笔记均提供 Markdown 源文件与对应的 PDF 版本。

## 文档目录

| 文档 | 说明 |
|---|---|
| [颜色空间理论清单](color-space-theory-checklist.md) | 从定义、数学原理、物理基础到使用范围，系统梳理颜色空间：光谱与三刺激假设、RGB/XYZ/Lab/YCbCr 等坐标体系及其适用场景。 |
| [YUV 家族与色度子采样](yuv-family-chroma-subsampling.md) | 厘清 YUV、YPbPr、YIQ、YCbCr 四个"亮度-色差"色彩空间的关系与用途，详解 4:4:4 / 4:2:2 / 4:2:0 / 4:1:1 采样实现，并区分 Y'（luma）与 Y（luminance）。 |
| [图像描述子综述](feature-descriptors-notes.md) | 按"全局 → 局部 → 纹理 → 学习型 → 颜色"脉络组织，区分手工设计与学习型两大分支，覆盖 SIFT、SURF、ORB、HOG、LBP、VLAD、SuperPoint 等描述子的原理、分类与应用。 |
| [JPEG 与 JPEG 2000 压缩原理详解](jpeg-jpeg2000-knowledge-reference.md) | 独立知识速查文件：JPEG 的 DCT、量化、ZigZag、霍夫曼/算术编码、文件结构与伪影，以及 JPEG 2000 的小波变换、EBCOT、渐进性与 ROI，并附对比选型。 |

> 各文档的 PDF 版本与同名 Markdown 文件位于同一目录，便于离线阅读与打印。

## 许可说明

本项目采用 [MIT License](LICENSE) 授权。

Copyright (c) 2026 tianheju

你可以自由地使用、复制、修改、合并、发布、分发、再授权和/或销售本项目的副本，但需在副本或实质性部分中保留上述版权声明与许可声明。本软件按"原样"提供，不附带任何形式的明示或暗示担保。
