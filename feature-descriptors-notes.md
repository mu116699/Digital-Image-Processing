# 图像描述子（Feature Descriptors）综述：原理、分类与应用

> **关于参考书的一点说明**
>
> 本文目标是按 *Computer Vision Metrics: Survey, Taxonomy, and Analysis*（Scott Krig, Apress）
> 这类 survey 的通用脉络来组织描述子内容，即以
> **全局 → 局部 → 纹理 → 学习型 → 颜色** 为主线，并区分
> **手工设计（hand-crafted）与学习型（learned）** 两大分支。
>
> 该书未能获取原文，因此本文**不引用其具体页码、章节号和原文表述**，
> 只采用这一领域公认的分类框架与数学定义。所有公式与算法描述均来自公开的
> 原始论文与标准定义（SIFT、SURF、ORB、BRIEF、BRISK、FREAK、HOG、LBP、
> GLOH、Shape Context、MSER、Census、VLAD、Fisher Vector、SuperPoint 等）。

---

## 目录

1. 描述子的基本概念
2. 描述子的分类体系（Taxonomy）
3. 数学原理基础
4. 关键点检测子（Detectors）
5. 局部描述子（Local Descriptors）逐类详解
6. 全局描述子（Global Descriptors）
7. 纹理描述子（Texture Descriptors）
8. 颜色描述子（Color Descriptors）
9. 形状与轮廓描述子（Shape / Contour Descriptors）
10. 光谱与子空间描述子（Spectral / Basis / Subspace）
11. 学习型与深度描述子（Learning-based）
12. 立体匹配与光流描述子
13. 视频与时序描述子
14. 匹配、索引与检索
15. 评价指标与基准数据集
16. 应用方向
17. 选型与设计权衡
18. 速查表
19. 结论

---

## 1. 描述子的基本概念

### 1.1 什么叫描述子

**描述子（descriptor）** 是把图像中某一局部或全局区域的内容压缩成一个
固定维度数值向量的函数。它的作用是让“图像块”变成一个可比较、可匹配、
可检索的数学对象。

形式化地说，描述子是一个映射：

$$
f: \mathcal{P}(I) \rightarrow \mathbb{R}^{d} \quad \text{或} \quad \{0,1\}^{d}
$$

其中：

- $I$ 是图像（灰度或彩色）
- $\mathcal{P}(I)$ 是图像上的一个区域或一个点邻域
- $d$ 是描述子的维度
- 浮点描述子取值于 $\mathbb{R}^{d}$，二进制描述子取值于 $\{0,1\}^{d}$

描述子本身通常是**向量**；它表达的是"这块图像长什么样"，而不是"这块图像在哪"。

### 1.2 检测子、关键点与描述子的关系

一个完整的局部特征系统通常由三部分组成：

| 环节 | 作用 | 输出 | 典型方法 |
|---|---|---|---|
| 检测（detection） | 找出图像中"值得描述"的位置 | 关键点坐标 $(x,y)$、尺度 $\sigma$、方向 $\theta$ | Harris、FAST、DoG、MSER |
| 描述（description） | 把关键点邻域编码成向量 | $d$ 维向量 | SIFT、SURF、ORB、BRIEF |
| 匹配（matching） | 比较两个描述子的相似度 | 对应关系 | L2、Hamming、NNDR、RANSAC |

**检测子（detector）** 与 **描述子（descriptor）** 是两个可以独立替换的模块。
例如 ORB = oFAST（检测）+ rBRIEF（描述），两者来自不同源头。

### 1.3 描述子的数学本质

从数学角度看，描述子的构造基本都遵循同一个模式：

$$
\text{descriptor} = \mathcal{N}\Big( \mathcal{Q}\big( \mathcal{F}(I, \Omega) \big) \Big)
$$

其中：

- $\Omega$：局部区域（点邻域、仿射归一化区域、区域检测结果）
- $\mathcal{F}$：特征提取算子（梯度、滤波响应、强度比较、矩、频谱）
- $\mathcal{Q}$：量化/编码（直方图分箱、二值化、投影、聚类分配）
- $\mathcal{N}$：归一化（L1/L2 归一化、均值-方差归一化、幂律归一化）

不同的描述子，差别主要在于这四步的具体选择。

### 1.4 好描述子应具备的性质

| 性质 | 含义 | 数学体现 |
|---|---|---|
| 可重复性（repeatability） | 同一物理点在变换后仍被检测到 | 检测子在同一点上重复命中 |
| 可区分性（distinctiveness） | 不同区域产生不同描述子 | 描述子空间内类间距离大 |
| 紧致性（compactness） | 占用空间小、匹配快 | 维度 $d$ 小 |
| 不变性（invariance） | 对某种变换不敏感 | $f(T(I)) = f(I)$ |
| 等变性（equivariance） | 随变换而一致变化 | $f(T(I)) = T'(f(I))$ |
| 鲁棒性（robustness） | 对噪声、压缩、光照稳定 | 扰动下 $\lVert \Delta f \rVert$ 小 |
| 效率（efficiency） | 计算与匹配开销低 | 可用积分图、位运算、LSH |

其中不变性是最核心的设计目标，常见有：

- **平移不变**：通过对局部区域相对坐标编码
- **旋转不变**：通过主方向分配 + 旋转归一化
- **尺度不变**：通过尺度空间极值检测 + 尺度归一化窗口
- **仿射不变**：通过二阶矩矩阵做仿射归一化
- **光照不变**：通过梯度、强度差、归一化、截断
- **视角/3D 不变**：最难，通常只能做到有限视角内鲁棒

---

## 2. 描述子的分类体系（Taxonomy）

### 2.1 按空间范围分

**全局描述子（global descriptors）**

- 对整幅图像产生一个向量
- 典型：颜色直方图、GIST、GLCM、Hu 矩、BoW、Fisher Vector、CNN 全局池化
- 优点：紧凑、适合整图检索与分类
- 缺点：无空间定位能力，遮挡敏感

**局部描述子（local descriptors）**

- 对关键点邻域产生一个向量
- 典型：SIFT、SURF、ORB、BRIEF、BRISK、FREAK、LBP
- 优点：可匹配、可定位、对遮挡相对鲁棒
- 缺点：数量多，需要匹配与几何验证

### 2.2 按设计方式分

**手工设计（hand-crafted）**

- 由人根据图像结构先验设计
- 代表：SIFT、SURF、HOG、LBP、Gabor、Shape Context、MSER

**学习型（learned / learned embeddings）**

- 参数由数据训练得到
- 代表：PCA-SIFT、LDA 描述子、Random Ferns、L2-Net、HardNet、SuperPoint、DELF

### 2.3 按数学构造来源分（本文主线）

| 构造家族 | 数学工具 | 代表方法 |
|---|---|---|
| 梯度/方向直方图 | 一阶微分 + 加权投票 | SIFT、SURF、GLOH、HOG、DAISY |
| 强度比较/二进制 | 比较算子 + 位运算 | BRIEF、ORB、BRISK、FREAK、LATCH |
| 矩与正交基 | 矩展开、正交多项式 | Hu、Zernike、Legendre、Tchebichef |
| 纹理/滤波响应 | 卷积滤波组 | LBP、Gabor、MR8、GLCM、Tamura、小波 |
| 光谱/子空间 | FFT、DCT、PCA、LDA、ICA | Fourier 描述子、DCT、Eigenfaces、Fisherfaces |
| 形状/轮廓 | 边界参数化、极坐标分箱 | Shape Context、链码、CSS、Fourier 形状 |
| 区域/分割 | 连通域、稳定性 | MSER、EBR、IBR |
| 学习/嵌入 | 度量学习、深度网络 | Fisher Vector、VLAD、NetVLAD、SuperPoint |

### 2.4 按不变性能力分

| 能力等级 | 含义 | 代表 |
|---|---|---|
| 仅平移 | 只对位置不敏感 | 直方图类、LBP（原型） |
| 平移+旋转 | 加方向归一化 | SIFT、SURF、ORB（rBRIEF）、BRISK |
| 平移+旋转+尺度 | 加尺度空间 | SIFT、SURF、KAZE、AKAZE |
| 平移+旋转+尺度+仿射 | 加仿射归一化 | Harris-Affine、Hessian-Affine、MSER+ |
| 宽基线/视角 | 学习型或专用设计 | SuperPoint、D2-Net、视角不变特征 |

### 2.5 按数据类型与匹配方式分

**浮点型描述子**

- 存储在 $\mathbb{R}^d$
- 匹配用欧氏距离 $L_2$ 或余弦相似度
- 代表：SIFT(128)、SURF(64/128)、GLOH(128)、HOG

**二进制描述子**

- 存储在 $\{0,1\}^d$
- 匹配用汉明距离（XOR + popcount）
- 代表：BRIEF(128–512)、ORB(256)、BRISK(512)、FREAK(512)、LATCH(256)

二进制描述子的核心优势：匹配极快。

$$
D_{Hamming}(\mathbf{a},\mathbf{b}) = \sum_{i=1}^{d} [a_i \neq b_i] = \mathrm{popcount}(\mathbf{a} \oplus \mathbf{b})
$$

### 2.6 按应用任务分

- 匹配/配准类：SIFT、SURF、ORB、SuperPoint
- 检测/分类类：HOG、LBP、Haar-like、DPM
- 检索类：BoW、VLAD、Fisher Vector、R-MAC、DELF
- 识别类：LBPH、Eigenfaces、Fisherfaces、深度嵌入
- 立体/稠密类：Census、NCC、DAISY

---

## 3. 数学原理基础

### 3.1 尺度空间理论（Scale Space）

尺度不变性的数学基础是**尺度空间**。图像在不同尺度下的表示由高斯卷积给出：

$$
L(x,y,\sigma) = G(x,y,\sigma) * I(x,y)
$$

其中：

$$
G(x,y,\sigma) = \frac{1}{2\pi\sigma^2} e^{-\frac{x^2+y^2}{2\sigma^2}}
$$

尺度空间满足扩散方程（这一性质称为半群性质）：

$$
\frac{\partial L}{\partial \sigma} = \sigma \nabla^2 L
$$

**高斯拉普拉斯（LoG）** 用于斑点检测：

$$
\nabla^2 G = \frac{\partial^2 G}{\partial x^2} + \frac{\partial^2 G}{\partial y^2}
$$

尺度归一化 LoG 的极值给出特征尺度：

$$
\sigma^2 \nabla^2 L(x,y,\sigma)
$$

**高斯差分（DoG）** 是对 LoG 的高效近似：

$$
D(x,y,\sigma) = \big( G(x,y,k\sigma) - G(x,y,\sigma) \big) * I(x,y)
$$

$$
\approx (k-1)\sigma^2 \nabla^2 L
$$

这就是 SIFT 能在多尺度上检测极值的关键。

**非线性尺度空间（KAZE/AKAZE）** 用非线性扩散替代高斯：

$$
\frac{\partial L}{\partial t} = \mathrm{div}\big( c(x,y,t)\,\nabla L \big)
$$

其中 $c$ 由梯度自适应控制（Perona–Malik 型 diffusivity），从而在保边的同时建立尺度。

### 3.2 梯度与局部结构

一阶梯度：

$$
I_x = \frac{\partial I}{\partial x},\quad I_y = \frac{\partial I}{\partial y}
$$

梯度幅值与方向：

$$
m = \sqrt{I_x^2 + I_y^2},\quad \theta = \mathrm{atan2}(I_y, I_x)
$$

**自相关矩阵（结构张量）**：

$$
M = \sum_{(x,y)\in W} w(x,y)
\begin{bmatrix}
I_x^2 & I_x I_y \\
I_x I_y & I_y^2
\end{bmatrix}
$$

特征值 $\lambda_1,\lambda_2$ 决定局部结构类型：

| 特征值关系 | 结构类型 |
|---|---|
| $\lambda_1,\lambda_2$ 都小 | 平坦区域 |
| 一大一小 | 边缘 |
| 都大 | 角点 |

**Harris 响应**：

$$
R = \det(M) - k\,(\mathrm{tr}\,M)^2
$$

**Shi–Tomasi 响应**：

$$
R = \min(\lambda_1, \lambda_2)
$$

**Hessian 矩阵**（用于斑点/DoH）：

$$
H = \begin{bmatrix} I_{xx} & I_{xy} \\ I_{xy} & I_{yy} \end{bmatrix},\quad
\det(H) = I_{xx}I_{yy} - I_{xy}^2
$$

### 3.3 直方图编码与量化

方向直方图是最常见的编码方式。给定方向集合 $\{\theta_i\}$ 与权重 $\{w_i\}$：

$$
h_b = \sum_i w_i \cdot \mathbb{1}\big[ \theta_i \in \text{bin}_b \big]
$$

SIFT 使用**三线性插值**把样本分配到相邻的子区域和相邻的方向箱，避免分箱边界效应：

$$
w = w_{sub} \cdot w_{ori}
$$

其中 $w_{sub}$、$w_{ori}$ 都是在相邻格点上的线性权重。

归一化的作用：

- L2 归一化：抑制对比度变化
- 截断（clip 0.2）后再次归一化：抑制主导梯度
- L1 + 开方（RootSIFT）：改善分布稳定性

**RootSIFT** 的做法：

$$
\mathbf{v}_{root} = \sqrt{\frac{\mathbf{v}_{L1}}{\lVert \mathbf{v} \rVert_1}}
$$

### 3.4 二进制测试与汉明度量

二进制描述子由一组**强度比较**构成：

$$
\tau(I; \mathbf{p}, \mathbf{q}) =
\begin{cases}
1, & I(\mathbf{p}) < I(\mathbf{q}) \\
0, & \text{否则}
\end{cases}
$$

一个 $d$ 位描述子：

$$
f_d(I) = \sum_{i=1}^{d} 2^{i-1} \tau(I; \mathbf{p}_i, \mathbf{q}_i)
$$

采样模式 $\{(\mathbf{p}_i,\mathbf{q}_i)\}$ 决定了描述子的性能：

- BRIEF：随机采样，无旋转不变性
- ORB：学习式选择去相关样本对，并把采样模式按主方向旋转
- BRISK：同心圆采样，长短对分开使用
- FREAK：视网膜式采样，带级联

### 3.5 矩与正交基

**几何矩**：

$$
m_{pq} = \sum_{x}\sum_{y} x^p y^q I(x,y)
$$

**中心矩**：

$$
\mu_{pq} = \sum_{x}\sum_{y} (x-\bar{x})^p (y-\bar{y})^q I(x,y)
$$

**归一化中心矩**：

$$
\eta_{pq} = \frac{\mu_{pq}}{\mu_{00}^{1+\frac{p+q}{2}}}
$$

**Hu 七不变矩**由 $\eta_{pq}$ 组合而成，对平移、旋转、尺度不变：

$$
\begin{aligned}
\phi_1 &= \eta_{20} + \eta_{02} \\
\phi_2 &= (\eta_{20}-\eta_{02})^2 + 4\eta_{11}^2 \\
\phi_3 &= (\eta_{30}-3\eta_{12})^2 + (3\eta_{21}-\eta_{03})^2 \\
&\dots
\end{aligned}
$$

**Zernike 矩**：在单位圆盘上以正交多项式为基：

$$
Z_{nm} = \frac{n+1}{\pi}\int_0^{2\pi}\!\!\int_0^1 \overline{V_{nm}(\rho,\theta)}\, f(\rho,\theta)\,\rho\, d\rho\, d\theta
$$

正交性使得矩之间信息不冗余，且可以做到旋转不变（取模）。

### 3.6 滤波与纹理

**Gabor 滤波**：

$$
g(x,y;\lambda,\theta,\psi,\sigma,\gamma) =
\exp\!\left(-\frac{x'^2+\gamma^2 y'^2}{2\sigma^2}\right)
\exp\!\left(i\Big(2\pi\frac{x'}{\lambda}+\psi\Big)\right)
$$

其中

$$
x' = x\cos\theta + y\sin\theta,\quad y' = -x\sin\theta + y\cos\theta
$$

**LBP（局部二值模式）**：

$$
\mathrm{LBP}_{P,R} = \sum_{p=0}^{P-1} s(g_p - g_c)\, 2^p,\quad
s(x) = \begin{cases}1, & x \ge 0\\ 0, & x<0\end{cases}
$$

**均匀模式**：0/1 跳变不超过 2 次的模式，占自然图像绝大多数，因此
$\mathrm{LBP}^{u2}$ 用 $P(P-1)+3$ 个箱代替 $2^P$ 个箱，大幅降维。

**GLCM（灰度共生矩阵）**：给定偏移 $\delta$，统计灰度对 $(i,j)$ 的出现次数：

$$
P_\delta(i,j) = \#\{(x,y) : I(x,y)=i,\ I(x+\delta_x, y+\delta_y)=j\}
$$

Haralick 特征从 $P_\delta$ 派生：

$$
\text{Contrast} = \sum_{i,j}(i-j)^2 P(i,j)
$$

$$
\text{Energy} = \sum_{i,j} P(i,j)^2,\quad
\text{Homogeneity} = \sum_{i,j}\frac{P(i,j)}{1+|i-j|}
$$

$$
\text{Correlation} = \sum_{i,j}\frac{(i-\mu_i)(j-\mu_j)P(i,j)}{\sigma_i\sigma_j}
$$

### 3.7 子空间与统计学习

**PCA（Eigenfaces）**：对中心化数据求协方差矩阵特征向量：

$$
C = \frac{1}{N}\sum_{i=1}^{N}(\mathbf{x}_i-\bar{\mathbf{x}})(\mathbf{x}_i-\bar{\mathbf{x}})^T
$$

保留前 $k$ 个最大特征值对应的特征向量作为投影基。

**LDA（Fisherfaces）**：最大化类间散度与类内散度之比：

$$
W = \arg\max_W \frac{|W^T S_B W|}{|W^T S_W W|}
$$

**GMM 与 Fisher Vector**：用高斯混合建模局部描述子分布，再对参数求梯度：

$$
\mathcal{G}_\mu = \frac{1}{N\sqrt{\pi_k}}\sum_{i}\gamma_i(k)\frac{\mathbf{x}_i-\boldsymbol{\mu}_k}{\boldsymbol{\sigma}_k}
$$

**VLAD（Vector of Locally Aggregated Descriptors）**：

$$
\mathbf{v}_k = \sum_{i:\,\mathrm{NN}(\mathbf{x}_i)=c_k} (\mathbf{x}_i - c_k)
$$

再做幂律归一化与 L2 归一化。

**NetVLAD**：把硬分配替换为软分配：

$$
\bar{a}_k(\mathbf{x}_i) = \frac{e^{\mathbf{w}_k^T \mathbf{x}_i + b_k}}{\sum_{k'} e^{\mathbf{w}_{k'}^T \mathbf{x}_i + b_{k'}}}
$$

### 3.8 不变性的构造手段

| 不变性 | 构造手段 | 数学操作 |
|---|---|---|
| 平移 | 使用相对坐标/局部邻域 | 以关键点为中心建立坐标系 |
| 旋转 | 估计主方向并旋转采样 | 梯度方向直方图峰值 $\theta^*$ |
| 尺度 | 尺度空间极值 + 尺度归一化窗口 | $\sigma^2\nabla^2 L$ 极值 |
| 仿射 | 二阶矩矩阵归一化 | $\Sigma^{-1/2}$ 变换到单位圆 |
| 光照对比度 | 归一化 | L2/L1 归一化、clip |
| 光照偏移 | 差分 | 梯度、强度比较 |
| 视角 | 多视角数据或学习 | 度量学习、注意力匹配 |

**主方向估计**（SIFT）：

$$
\theta^* = \arg\max_\theta \sum_i w_i\, \mathbb{1}[\theta_i \in \text{bin}(\theta)]
$$

其中权重为梯度幅值乘以高斯窗。

### 3.9 相似度度量与匹配

**欧氏距离**：

$$
d_2(\mathbf{a},\mathbf{b}) = \lVert \mathbf{a}-\mathbf{b} \rVert_2
$$

**余弦相似度**：

$$
\mathrm{cos}(\mathbf{a},\mathbf{b}) = \frac{\mathbf{a}\cdot\mathbf{b}}{\lVert\mathbf{a}\rVert\lVert\mathbf{b}\rVert}
$$

**最近邻距离比（NNDR / Lowe ratio test）**：

$$
\frac{d(\mathbf{a}, \mathbf{b}_{1})}{d(\mathbf{a}, \mathbf{b}_{2})} < \rho
$$

其中 $\mathbf{b}_1$ 是最近邻、$\mathbf{b}_2$ 是次近邻，$\rho$ 常取 0.6–0.8。

**几何验证**：用 RANSAC 拟合单应/基础矩阵，剔除外点：

$$
H^* = \arg\max_H \sum_i \mathbb{1}\big[ \lVert \mathbf{x}'_i - H\mathbf{x}_i \rVert < \epsilon \big]
$$

---

## 4. 关键点检测子（Keypoint Detectors）

### 4.1 角点检测

| 检测子 | 原理 | 特点 |
|---|---|---|
| Moravec | 各方向灰度差最小值的局部极大 | 最早，无旋转不变性 |
| Harris–Stephens | 结构张量 + 响应函数 | 旋转不变，光照鲁棒 |
| Shi–Tomasi | $\min(\lambda_1,\lambda_2)$ | 更稳定的跟踪角点 |
| Förstner | 协方差矩阵 + 圆度 | 精度高，用于配准 |
| Kitchen–Rosenfeld | 梯度方向变化率 | 计算简单 |
| FAST | 圆周 Bresenham 测试 | 极快，无尺度不变 |
| AGAST | FAST 的自适应泛化 | 更快的 FAST 变体 |
| oFAST | FAST + 强度质心方向 | ORB 的检测子 |

**Harris 推导**：对邻域做一阶泰勒展开：

$$
I(x+\Delta x, y+\Delta y) \approx I(x,y) + I_x \Delta x + I_y \Delta y
$$

平方误差和：

$$
E(\Delta x,\Delta y) = \sum_W \big( I_x \Delta x + I_y \Delta y \big)^2 = \mathbf{d}^T M \mathbf{d}
$$

对 $M$ 做特征值分析，则得到角点判据。

**FAST 判据**：圆形半径 3 的 16 像素圆周上，若有连续 $N$（通常 9 或 12）个像素
都比中心亮/暗超过阈值 $t$，则判为角点：

$$
\exists\, \text{连续 } N \text{ 个 } p:\ |I(p)-I(c)| > t
$$

**强度质心方向**（oFAST）：

$$
m_{pq} = \sum_{x,y} x^p y^q I(x,y),\quad
\theta = \mathrm{atan2}(m_{01}, m_{10})
$$

### 4.2 斑点与区域检测

| 检测子 | 原理 | 特点 |
|---|---|---|
| LoG | 尺度归一化高斯拉普拉斯极值 | 尺度不变 |
| DoG | 相邻高斯差 | SIFT 的核心 |
| DoH | Hessian 行列式 | SURF 的基础 |
| SIFT 检测子 | DoG 极值 + 亚像素定位 + 边缘剔除 | 尺度/旋转不变 |
| SURF 检测子 | 积分图 + 盒滤波 Hessian | 速度快 |
| BRISK 检测子 | AGAST + 尺度空间 | 快速多尺度 |
| CenSurE / STAR | 多尺度中心环绕滤波 | 近似尺度不变 |
| MSER | 阈值扫描 + 连通域稳定性 | 仿射不变强 |
| EBR / IBR | 边缘/强度极值区域 | 早期区域检测 |
| KAZE / AKAZE | 非线性尺度空间 | 保边、重复性好 |
| SUSAN | 最小核同值区 | 角点+边缘统一框架 |
| Hessian-Laplace | Hessian 定位 + LoG 尺度 | 更稳定的尺度选择 |
| Harris-Laplace | Harris 定位 + LoG 尺度 | 经典仿射不变检测 |
| Harris-Affine | 迭代仿射归一化 | 仿射不变 |
| Hessian-Affine | Hessian 定位 + 仿射归一化 | 仿射不变 |

**MSER 稳定性判据**：随阈值 $\theta$ 扫描，连通域面积变化率取局部极小：

$$
q(i) = \frac{|A(i+\Delta) - A(i-\Delta)|}{A(i)}
$$

### 4.3 边缘检测子

| 检测子 | 原理 |
|---|---|
| Roberts | 2×2 差分算子 |
| Sobel | 3×3 平滑+差分 |
| Prewitt | 均匀加权差分 |
| Scharr | 更精确的旋转对称差分 |
| Laplacian | 二阶微分零交叉 |
| LoG / Marr–Hildreth | 平滑后二阶微分 |
| Canny | 高斯平滑 + 非极大抑制 + 双阈值滞后 |
| 结构化边缘（SE） | 随机森林预测边缘 |

**Canny 的非极大抑制**：沿梯度方向比较，只保留局部极大：

$$
\text{keep if } m(x,y) \ge m(x \pm \delta\cos\theta,\ y \pm \delta\sin\theta)
$$

---

## 5. 局部描述子（Local Descriptors）逐类详解

### 5.1 梯度方向直方图类

#### SIFT（Scale-Invariant Feature Transform, 128 维）

**流程**：

1. **尺度空间极值**：$D(x,y,\sigma)$ 在 3D 邻域（含相邻尺度）中比较
2. **亚像素定位**：泰勒展开求极值偏移

$$
\hat{\mathbf{x}} = -\left(\frac{\partial^2 D}{\partial \mathbf{x}^2}\right)^{-1} \frac{\partial D}{\partial \mathbf{x}}
$$

3. **边缘剔除**：主曲率比判据

$$
\frac{(\mathrm{tr}\,H)^2}{\det H} < \frac{(r+1)^2}{r},\quad r \approx 10
$$

4. **方向分配**：36 箱梯度方向直方图，峰值为主方向，达到峰值 80% 的方向作辅方向
5. **描述子**：16×16 邻域内 4×4 子区域，每子区域 8 方向 → 128 维
6. **归一化**：L2 归一化 → clip 0.2 → 再 L2 归一化

**SIFT 的数学核心**是"梯度方向统计 + 局部空间分块 + 全局归一化"。

#### SURF（Speeded-Up Robust Features, 64/128 维）

用盒滤波近似 Hessian：

$$
\det(H_{approx}) = D_{xx}D_{yy} - (0.9\,D_{xy})^2
$$

**方向分配**：在半径 $6\sigma$ 的圆域内计算 Haar 小波响应 $d_x, d_y$，
用 $60^\circ$ 滑动窗口求和，最大值方向为主方向。

**描述子**：4×4 子区域，每区域统计

$$
\big( \textstyle\sum d_x,\ \sum d_y,\ \sum |d_x|,\ \sum |d_y| \big) \rightarrow 64
$$

扩展版本加入 $\sum d_y$ 的符号 → 128 维。

#### GLOH（Gradient Location and Orientation Histogram, 128 维）

- 采用**对数极坐标**分箱：3 个半径 × 8 个角度 = 17 个空间箱
- 每箱 16 个方向箱 → 272 维
- 用 PCA 降到 128 维
- 比 SIFT 更具区分性，但计算更贵

#### DAISY（200 维）

- 面向**稠密**匹配设计
- 在每个采样点周围用高斯加权方向直方图
- 用可分离卷积快速计算
- 适合宽基线立体匹配

#### HOG（Histogram of Oriented Gradients）

**流程**：

1. 计算梯度幅值与方向
2. 把图像划分为 cell（如 8×8），每 cell 统计 9 个方向箱（0–180°）
3. 若干 cell 组成 block（如 2×2），对 block 内特征做 **L2-Hys** 归一化
4. 拼接所有 block 特征

**L2-Hys 归一化**：

$$
\mathbf{v} \leftarrow \frac{\mathbf{v}}{\sqrt{\lVert\mathbf{v}\rVert_2^2 + \epsilon^2}},\quad
v_i \leftarrow \min(v_i, 0.2),\quad
\text{再归一化}
$$

HOG 是行人检测（HOG + SVM）与 DPM 的核心。

#### 相关变体

| 名称 | 特点 |
|---|---|
| PCA-SIFT | 对梯度块做 PCA 降维到 20 维 |
| RootSIFT | SIFT 描述子 L1 归一化后开方 |
| RIFT | 旋转不变特征变换，用环形分箱 |
| CSIFT | 颜色不变 SIFT |
| OpponentSIFT | 在对立色通道上计算 SIFT |
| DSP-SIFT | 尺度池化，提升匹配性能 |
| PHOG | 金字塔 HOG，用于形状/场景 |
| HOG3D | 时空梯度直方图 |
| PIFF | 像素级特征融合的类 HOG 全局特征 |

### 5.2 二进制描述子类

#### BRIEF（Binary Robust Independent Elementary Features, 128–512 bit）

$$
f(I) = \sum_{i} 2^{i-1}\,\tau(I;\mathbf{p}_i,\mathbf{q}_i)
$$

- 采样对从高斯分布抽取
- **无旋转不变性、无尺度不变性**，对模糊较敏感
- 匹配用汉明距离，极快

#### ORB（Oriented FAST and Rotated BRIEF, 256 bit）

1. **oFAST** 检测 + 强度质心方向
2. **Steered BRIEF**：把采样模式按方向 $\theta$ 旋转

$$
R_\theta = \begin{bmatrix}\cos\theta & -\sin\theta\\ \sin\theta & \cos\theta\end{bmatrix}
$$

3. **rBRIEF**：用贪心搜索在训练集上选出去相关、方差大的 256 个测试对
4. 加入金字塔实现尺度不变性

ORB 是 ORB-SLAM 的基石，兼顾速度与质量。

#### BRISK（Binary Robust Invariant Scalable Keypoints, 512 bit）

- **采样模式**：同心圆（半径按指数增长），共 60 个采样点
- **短对**（$<\delta_{max}$）：用于估计方向
- **长对**（$>\delta_{min}$）：用于构造描述子
- 512 bit，匹配快

#### FREAK（Fast Retina Keypoint, 512 bit）

- **视网膜式采样**：中心密集、外围稀疏，模拟人眼感受野
- 通过**级联比较**逐步筛选，前面若干位就能筛掉大量候选
- 用 45 个感受野之间的比较构成二进制串

#### 其他二进制描述子

| 名称 | 特点 |
|---|---|
| LATCH | 学习式三像素比较，抗噪更好 |
| LIOP | 局部强度序模式，利用像素排序 |
| BOLD | 自适应在线学习二值测试 |
| BinBoost | Boosting 学习二进制哈希 |
| AKAZE-MLDB | 非线性尺度空间 + 改进 LDB |
| KAZE 描述子 | 非线性尺度 + 类 SURF 浮点描述子 |
| D-BRIEF | 面向稠密的 BRIEF 变体 |

### 5.3 区域与仿射描述子

| 名称 | 特点 |
|---|---|
| MSER 区域 | 稳定性极强的仿射不变区域 |
| Harris-Affine 区域 | 迭代估计仿射形状 |
| Hessian-Affine 区域 | Hessian 版本 |
| EBR / IBR | 早期极值区域 |
| 局部强度序 | 序统计描述 |

这些区域经过仿射归一化后，再送入 SIFT/SURF 描述子，即可获得仿射不变特征。

### 5.4 矩型局部描述子

| 名称 | 特点 |
|---|---|
| 图像矩 | 可构造平移/旋转/尺度不变 |
| Zernike 矩 | 正交基，旋转不变，重建能力强 |
| 伪 Zernike 矩 | 放宽正交性，数值更稳 |
| Legendre 矩 | 正交多项式，在矩形区域定义 |
| Tchebichef / Krawtchouk 矩 | 离散正交矩，适合数字图像 |
| 小波矩 | 多分辨率 + 矩 |

---

## 6. 全局描述子（Global Descriptors）

### 6.1 GIST / 空间包络

对人类场景感知的近似建模：

1. 对图像做多尺度、多方向的 Gabor 滤波
2. 把每张响应图划成 4×4 网格
3. 对每个网格取平均能量

维度 = 尺度数 × 方向数 × 网格数（典型 512）。

GIST 用于场景分类与快速图像检索。

### 6.2 全局颜色统计

- 颜色直方图
- 颜色矩（均值、方差、偏度）
- 颜色聚合向量（CCV）
- 颜色相关图（correlogram）

### 6.3 全局纹理统计

- GLCM / Haralick
- Tamura 特征：粗糙度、对比度、方向性、线状性、规则性、粗略度
- 小波能量
- MR8 滤波组响应直方图

### 6.4 全局形状描述

- Hu 矩
- Zernike 矩
- Fourier 描述子
- 骨架图

### 6.5 全局聚合描述子

- **BoW（Bag of Words）**：局部描述子聚类成词袋，统计词频
- **Fisher Vector**：GMM 梯度聚合
- **VLAD**：残差聚合
- **CNN 全局池化**：VGG/ResNet 特征 + average/max/GeM/R-MAC 池化

---

## 7. 纹理描述子（Texture Descriptors）

| 类别 | 方法 | 原理 |
|---|---|---|
| 统计型 | GLCM、GLRLM、LBP、灰度梯度 | 统计像素对/游程/局部模式 |
| 滤波型 | Gabor、MR8、小波、可操纵滤波 | 多方向多尺度响应能量 |
| 模型型 | 马尔可夫随机场、分形维数 | 用概率模型描述纹理生成 |
| 变换型 | 傅里叶能量、Gabor 谱、小波 | 频域能量分布 |
| 结构型 | 形态学、纹理单元、Textons | 重复基元的排列 |
| 感知型 | Tamura | 按人类视觉描述词汇 |

**LBP 的常用变体**：

| 变体 | 特点 |
|---|---|
| $\mathrm{LBP}^{u2}$ | 均匀模式，降维 |
| $\mathrm{LBP}^{ri}$ | 旋转不变 |
| $\mathrm{LBP}^{riu2}$ | 旋转不变均匀模式 |
| CLBP | 加入幅值与中心像素信息 |
| LBP-HF | 傅里叶域的旋转不变 LBP |
| CS-LBP | 中心对称比较，维度更低 |
| LBP-TOP | 时空三维 LBP，用于视频 |
| 多尺度 LBP | 多半径组合 |

**Textons**：先用滤波组响应聚类得到纹理基元，再用基元直方图表示纹理。

---

## 8. 颜色描述子（Color Descriptors）

### 8.1 颜色直方图族

- RGB / HSV / Lab 三通道直方图
- 联合直方图 vs 边缘分布
- 加空间信息的直方图（分块、环形）

### 8.2 颜色矩

一阶矩（均值）、二阶矩（标准差）、三阶矩（偏度）：

$$
\mu_k = \frac{1}{N}\sum_i p_{ik},\quad
\sigma_k = \left(\frac{1}{N}\sum_i (p_{ik}-\mu_k)^2\right)^{1/2},\quad
s_k = \left(\frac{1}{N}\sum_i (p_{ik}-\mu_k)^3\right)^{1/3}
$$

### 8.3 颜色结构描述

- **CCV（Color Coherence Vector）**：区分相干/非相干像素
- **Color Correlogram**：颜色对的空间相关性
- **Color Names**：把颜色映射到 11 个语言色名

### 8.4 颜色 SIFT 家族

在颜色空间中重复 SIFT 流程：

| 描述子 | 颜色空间 |
|---|---|
| RGB-SIFT | 各 RGB 通道分别做 SIFT |
| OpponentSIFT | 对立色空间 |
| HSV-SIFT / HueSIFT | HSV |
| rgSIFT | 归一化 rg 色度 |
| C-SIFT | 色度不变量 |
| Transformed Color SIFT | 颜色变换不变 |

### 8.5 颜色描述子的匹配

- 直方图交集
- $\chi^2$ 距离

$$
d_{\chi^2}(\mathbf{h},\mathbf{g}) = \frac{1}{2}\sum_i \frac{(h_i-g_i)^2}{h_i+g_i}
$$

- EMD（Earth Mover's Distance）

---

## 9. 形状与轮廓描述子（Shape / Contour Descriptors）

### 9.1 轮廓参数化

**链码（Freeman Chain Code）**：用 8 方向编码边界。

**轮廓点复数表示**：

$$
z(k) = x(k) + i\,y(k)
$$

**Fourier 描述子**：对 $z(k)$ 做 DFT：

$$
Z(u) = \sum_{k=0}^{N-1} z(k) e^{-i 2\pi uk / N}
$$

性质：

- 平移只改变 $Z(0)$（去掉即可不变）
- 旋转只改变相位
- 尺度只改变幅值
- 取 $|Z(u)|$ 得到旋转不变描述子

### 9.2 曲率与尺度空间

**曲率尺度空间（CSS）**：

$$
\kappa(u,\sigma) = \frac{\dot{x}\ddot{y}-\ddot{x}\dot{y}}{(\dot{x}^2+\dot{y}^2)^{3/2}}
$$

随 $\sigma$ 增大，曲率零交叉点从细节到整体逐层出现，形成"尺度树"。

**转向函数（Turning Function）**：把边界累积转角作为弧长的函数，
对旋转、平移不变，对尺度可归一化。

### 9.3 形状上下文（Shape Context）

对形状上一点 $p_i$，统计其余 $n-1$ 个点的相对位置分布：

- 对数极坐标分箱：12 个角度 × 5 个半径 = 60 箱
- 得到 60 维直方图

**匹配代价**：

$$
C_{ij} = \frac{1}{2}\sum_{k=1}^{K}\frac{(h_i(k)-h_j(k))^2}{h_i(k)+h_j(k)}
$$

用匈牙利算法求最优匹配。

### 9.4 其他形状描述子

| 名称 | 原理 |
|---|---|
| Hu 矩 | 七个不变矩组合 |
| Zernike 矩 | 正交基，旋转不变 |
| 距离变换 / 中轴 | 骨架描述 |
| Chamfer 距离 | 边缘点距离匹配 |
| 多边形近似 | Douglas–Peucker |
| B 样条 | 平滑轮廓参数化 |
| 骨架图 | 拓扑结构描述 |
| 形状上下文 | 点分布直方图 |
| 内距离（Inner Distance） | 基于内部测地距离 |

---

## 10. 光谱与子空间描述子（Spectral / Basis / Subspace）

### 10.1 变换域

| 变换 | 用途 |
|---|---|
| DFT | 频域能量、平移不变 |
| DCT | 压缩、能量集中 |
| 小波（Haar、Daubechies） | 多分辨率纹理/形状 |
| Gabor 族 | 方向-频率分解 |
| 可操纵滤波 | 任意方向的可调响应 |
| 梅林/傅里叶-梅林 | 尺度和旋转不变配准 |

### 10.2 子空间方法

| 方法 | 原理 | 应用 |
|---|---|---|
| PCA | 最大方差投影 | Eigenfaces、降维 |
| LDA | 最大类间/类内比 | Fisherfaces |
| ICA | 统计独立分量 | 盲源分离 |
| NMF | 非负分解 | 部件化表示 |
| 稀疏编码 | 稀疏线性组合 | 图像分类 |
| 字典学习 | 学习过完备基 | 去噪、分类 |
| 谱图（Spectral Graph） | 图拉普拉斯特征 | 形状匹配 |

---

## 11. 学习型与深度描述子（Learning-based）

### 11.1 经典学习型描述子

| 名称 | 学习方式 |
|---|---|
| PCA-SIFT | 对 SIFT 梯度块做 PCA |
| LDA 描述子 | 用判别分析投影 |
| Random Ferns | 随机蕨分类器 |
| Random Forests 描述 | 决策森林输出 |
| Boosting 描述子 | AdaBoost 选择特征 |
| 度量学习描述子 | 三元组损失学习距离 |
| DeepCompare / DeepDesc | 浅层孪生网络 + 相似度损失 |
| MatchNet | 孪生网络 + 全连接度量 |

### 11.2 深度局部描述子

| 名称 | 核心思想 |
|---|---|
| L2-Net | 逐块归一化，改善匹配分布 |
| HardNet | 在 batch 内挖掘最难的负样本 |
| SOSNet | 引入二阶相似度正则 |
| SuperPoint | 自监督：MagicPoint + 单应变换自适应，输出检测+256 维描述子 |
| SuperGlue | 注意力图神经网络 + 最优传输匹配 |
| D2-Net | 联合检测与描述，单次前向 |
| R2D2 | 可重复性与可靠性联合学习 |
| DELF | 语义特征 + 注意力池化 |
| DELG | 全局与局部联合，统一检索与匹配 |
| LF-Net | 全监督学习检测与描述 |
| SIFT + 学习式筛选 | 学习式后处理/筛选 |

**HardNet 的困难负样本挖掘**：

$$
\mathcal{L} = \frac{1}{N}\sum_i \max\big(0,\ 1 + d(\mathbf{a}_i,\mathbf{p}_i) - d(\mathbf{a}_i, \mathbf{n}^*_i)\big)
$$

其中 $\mathbf{n}^*$ 是批次内最近的非匹配描述子。

**SuperPoint 的伪标签自监督**：对图像施加随机单应变换，把关键点位置相应地变换，
用形状一致的方式生成热力图监督信号。

### 11.3 深度全局与检索描述子

| 名称 | 特点 |
|---|---|
| 全连接层特征（FC7 等） | 早期迁移学习特征 |
| 全局平均池化（GAP） | 紧凑全局表示 |
| GeM 池化 | 广义均值池化 |

$$
\text{GeM} = \left( \frac{1}{|\mathcal{X}|}\sum_{x\in\mathcal{X}} x^p \right)^{1/p}
$$

| R-MAC | 多区域最大激活聚合 |
| NetVLAD | 可微 VLAD 层 |
| Fisher Vector + CNN | CNN 特征 + FV 编码 |
| BEBLID | 提升式二进制描述子 |

### 11.4 学习型描述子的关键损失

- 对比损失（contrastive）
- 三元组损失（triplet）
- 铰链/边界挖掘（hard mining）
- 排序损失（ranking）
- 归一化相关 / 相似度匹配损失（NCC loss）

---

## 12. 立体匹配与光流描述子

### 12.1 稠密立体匹配代价

| 方法 | 原理 |
|---|---|
| SAD | 绝对差之和 |
| SSD | 平方差之和 |
| NCC | 归一化互相关 |
| Census | 强度比较位串 + 汉明距离 |
| BT（Birchfield–Tomasi） | 抗采样、对幅值不敏感 |
| 互信息（MI） | 统计依赖度，适跨模态 |
| AD-Census | 组合代价 |
| 半全局匹配（SGM） | 代价聚合 + 一致性检查 |

**Census 变换**：

$$
C(p) = \bigotimes_{\mathbf{q}\in \mathcal{N}(p)} \big[ I(\mathbf{q}) > I(p) \big]
$$

匹配用汉明距离。

### 12.2 光流

**Lucas–Kanade**（局部，亮度恒定假设）：

$$
I(x+u, y+v) \approx I(x,y) + I_x u + I_y v
$$

最小化：

$$
E(u,v) = \sum_W \big( I_x u + I_y v + I_t \big)^2
$$

解：

$$
\begin{bmatrix}u\\v\end{bmatrix} =
\left(\sum_W \begin{bmatrix}I_x^2 & I_xI_y\\ I_xI_y & I_y^2\end{bmatrix}\right)^{-1}
\sum_W \begin{bmatrix}I_xI_t\\ I_yI_t\end{bmatrix}
$$

**Horn–Schunck**（全局，加平滑正则）：

$$
E = \iint \big( I_x u + I_y v + I_t \big)^2 + \alpha^2 \big( \lVert \nabla u \rVert^2 + \lVert \nabla v \rVert^2 \big)\, dx\,dy
$$

---

## 13. 视频与时序描述子

| 名称 | 原理 |
|---|---|
| HOG3D | 时空梯度方向直方图 |
| HOF | 光流方向直方图 |
| MBH | 运动边界直方图 |
| iDT | 改进稠密轨迹 + 四种描述子融合 |
| Dense Trajectories | 稠密采样点轨迹 + 轨迹形状描述 |
| LBP-TOP | 三正交平面 LBP |
| 3D SIFT | 时空 SIFT |
| C3D / I3D / Two-Stream | 深度时空特征（学习型） |
| TimeSformer / VideoMAE | 基于 Transformer 的视频表示 |

---

## 14. 匹配、索引与检索

### 14.1 匹配策略

- 暴力匹配（brute force）
- 最近邻 + 距离比（NNDR）
- 交叉检验（cross-check / mutual NN）
- 比率检验 + 几何验证（RANSAC）
- 互信息/相关（跨模态）

### 14.2 近似最近邻索引

| 方法 | 原理 |
|---|---|
| KD-Tree | 递归空间划分 |
| FLANN | 自动选算法与参数 |
| LSH | 哈希碰撞近似近邻 |
| Hamming 索引 | 二进制位串分块 |
| 层次 K-means | 词袋倒排 |
| 乘积量化（PQ） | 子空间分解 + 码本 |
| 倒排文件（IVF） | 聚类桶 + 倒排 |
| HNSW | 分层可导航小世界图 |

### 14.3 检索流程

1. 提取局部描述子
2. 量化到视觉词 / 聚合（BoW、VLAD、FV）
3. 归一化（幂律、L2、白化）
4. 倒排索引 + 候选排序
5. 几何重排序（RANSAC、空间验证）

---

## 15. 评价指标与基准数据集

### 15.1 指标

| 指标 | 含义 |
|---|---|
| Repeatability | 变换后同一物理点被重复检测的比例 |
| Recall / Precision | 匹配查全/查准 |
| 匹配得分（matching score） | 正确匹配比例 |
| mAP | 检索平均精度均值 |
| 定位误差 | 匹配点像素误差 |
| 描述子距离分布 | 正/负样本可分性 |
| 计算耗时 / 内存 | 效率代价 |

### 15.2 常用数据集与基准

| 名称 | 用途 |
|---|---|
| Oxford VGG Affine | 仿射不变性评测 |
| HPatches | 局部描述子现代基准 |
| Brown / UBC | 宽基线匹配 |
| EVD | 视角变化评测 |
| MVS / PhotoTourism | 3D 重建匹配 |
| Oxford RobotCar | 长时视觉定位 |
| ImageNet / Places | 全局表示与检索 |
| Oxford5k / Paris6k / ROxford / RParis | 图像检索 |
| KITTI / EuRoC | 视觉里程计、SLAM |

---

## 16. 应用方向

### 16.1 几何与三维

| 应用 | 常用描述子 |
|---|---|
| 图像拼接/全景 | SIFT、SURF、ORB |
| 运动恢复结构（SfM） | SIFT（COLMAP 等） |
| 立体视觉与深度 | Census、AD-Census、DAISY、SGM |
| 多视图立体 | DAISY、PatchMatch + 描述子 |
| 相机标定/配准 | Harris、SIFT、棋盘角点 |
| 图像配准（医学/遥感） | SIFT、MI、形状上下文 |

### 16.2 定位与机器人

| 应用 | 常用描述子 |
|---|---|
| SLAM | ORB（ORB-SLAM）、SuperPoint |
| 视觉里程计 | FAST + 光流、ORB |
| 回环检测 | BoW、NetVLAD、DELG |
| 重定位 | NetVLAD、全局描述子 |
| 视觉导航 | 全局描述子 + 局部匹配 |

### 16.3 检测与识别

| 应用 | 常用描述子 |
|---|---|
| 行人检测 | HOG + SVM、DPM |
| 人脸检测 | Haar-like + AdaBoost、深度 |
| 通用目标检测 | CNN 特征（发展自 HOG 思路） |
| 人脸识别 | LBPH、Eigenfaces、Fisherfaces、深度嵌入 |
| 表情/年龄 | LBP、Gabor、深度 |
| 字符/OCR | HOG、LBP、投影特征、深度 |
| 文档表格/版面 | 线段、布局特征、深度布局模型 |

### 16.4 检索与分类

| 应用 | 常用描述子 |
|---|---|
| 图像检索 | BoW、VLAD、Fisher Vector、DELF/DELG |
| 近重复检测 | 全局描述子 + 哈希 |
| 场景分类 | GIST、CNN 全局特征 |
| 纹理分类 | LBP、Gabor、GLCM、Textons |
| 商品/商标检索 | 局部描述子 + 聚合 |
| 视频检索 | iDT、深度时空特征 |

### 16.5 跟踪与视频

| 应用 | 常用描述子 |
|---|---|
| 目标跟踪 | LBP、HOG、相关滤波（KCF/MOSSE）、深度特征 |
| 动作识别 | HOG3D、HOF、MBH、iDT、I3D |
| 行人重识别 | 颜色直方图 + 深度嵌入 |
| 手势/姿态 | 形状上下文、骨架 + 深度 |

### 16.6 工业与其他

| 应用 | 常用描述子 |
|---|---|
| 表面缺陷检测 | LBP、GLCM、Gabor |
| 布料/材料纹理 | LBP、Tamura、小波 |
| 遥感变化检测 | SIFT、全局统计、深度 |
| 医学图像分析 | LBP、Gabor、形状、深度 |
| 生物特征（指纹/虹膜） | 方向场、Gabor、纹理编码 |
| 增强现实（AR） | ORB、SuperPoint、平面跟踪 |
| 文档扫描矫正 | 边缘 + 线段 + 单应 |

---

## 17. 选型与设计权衡

### 17.1 选择维度

| 需求 | 推荐 |
|---|---|
| 最高匹配精度、可离线 | SIFT、RootSIFT、GLOH、SuperPoint |
| 实时性需求高 | ORB、BRISK、FREAK、AKAZE |
| 极小内存 | BRIEF/ORB（二进制）、LBP |
| 强仿射变化 | MSER + SIFT、Harris-Affine |
| 非刚性形变 | Shape Context、内距离 |
| 纹理分类 | LBP、Gabor、GLCM、Textons |
| 目标检测（经典） | HOG、Haar-like |
| 大规模检索 | VLAD、Fisher Vector、NetVLAD、深度全局 |
| 宽基线/难场景 | SuperPoint + SuperGlue、D2-Net |
| 跨模态 | 互信息、学习型嵌入 |

### 17.2 主要权衡

| 权衡 | 说明 |
|---|---|
| 维度 vs 速度 | 128 维 SIFT 精度高但慢；256 bit ORB 快但精度略低 |
| 不变性 vs 区分性 | 越不变往往越不具区分性 |
| 手工 vs 学习 | 学习型精度高，但依赖数据与算力 |
| 稀疏 vs 稠密 | 稀疏可匹配，稠密可建图 |
| 检测 vs 描述强制解耦 | 耦合（如 D2-Net）可联合优化 |
| 归一化强度 | 过强丢失信息，过弱受光照影响 |

### 17.3 设计检查清单

- 需要哪些不变性？（旋转/尺度/仿射/光照/视角）
- 匹配距离用 L2 还是 Hamming？
- 需要多少关键点？稠密还是稀疏？
- 是否可容忍检测失败？
- 是否需要固定维度以便索引？
- 是否需要 GPU 加速？
- 是否有标注数据可做学习？

---

## 18. 速查表

### 18.1 主要描述子一览

| 描述子 | 维度 | 类型 | 不变性 | 主要应用 |
|---|---|---|---|---|
| SIFT | 128 | 浮点 | 尺度、旋转 | 匹配、SfM、拼接 |
| RootSIFT | 128 | 浮点 | 尺度、旋转 | 匹配（提升版） |
| PCA-SIFT | 20–36 | 浮点 | 尺度、旋转 | 快速匹配 |
| SURF | 64/128 | 浮点 | 尺度、旋转 | 实时匹配 |
| GLOH | 128 | 浮点 | 尺度、旋转 | 高精度匹配 |
| DAISY | 200 | 浮点 | 尺度、旋转（弱） | 稠密立体 |
| HOG | 可变 | 浮点 | 光照（弱） | 行人检测 |
| PHOG | 可变 | 浮点 | 光照（弱） | 形状/场景 |
| BRIEF | 128–512 | 二进制 | 无 | 快速匹配 |
| ORB | 256 | 二进制 | 尺度、旋转 | SLAM、实时 |
| BRISK | 512 | 二进制 | 尺度、旋转 | 实时匹配 |
| FREAK | 512 | 二进制 | 尺度、旋转 | 移动端 |
| AKAZE | 486 | 二进制 | 尺度、旋转 | 鲁棒匹配 |
| LATCH | 256 | 二进制 | 无（可加） | 稠密匹配 |
| LBP | 59/256 | 整数/直方图 | 旋转（变体） | 纹理、人脸 |
| Gabor | 可变 | 浮点 | 方向/频率 | 纹理 |
| GLCM/Haralick | 4–14 | 浮点 | 无 | 纹理分类 |
| Tamura | 6 | 浮点 | 无 | 纹理感知 |
| Shape Context | 60 | 浮点 | 平移、尺度 | 形状匹配 |
| Fourier 描述子 | 可变 | 浮点 | 平移、旋转、尺度 | 轮廓匹配 |
| Hu 矩 | 7 | 浮点 | 平移、旋转、尺度 | 形状识别 |
| Zernike 矩 | 可变 | 浮点 | 旋转 | 形状识别 |
| MSER | 区域 | 区域 | 仿射 | 区域匹配、文字 |
| GIST | 512 | 浮点 | 无 | 场景分类 |
| BoW | 词袋 | 稀疏 | 继承局部 | 检索 |
| VLAD | $K\times d$ | 浮点 | 继承局部 | 检索 |
| Fisher Vector | $2Kd$ | 浮点 | 继承局部 | 检索/分类 |
| NetVLAD | $K\times d$ | 浮点 | 学习 | 地点识别 |
| Census | 位串 | 二进制 | 幅值 | 立体匹配 |
| SuperPoint | 256 | 浮点 | 学习 | SLAM、匹配 |
| SuperGlue | 匹配网络 | 学习 | 学习 | 宽基线匹配 |
| DELF | 1024 | 浮点 | 学习 | 检索 |
| DELG | 全局+局部 | 学习 | 学习 | 统一检索 |
| GeM/R-MAC | 可变 | 浮点 | 学习 | 检索 |

### 18.2 分类脉络速记

```
描述子
├─ 全局
│   ├─ 颜色：直方图、矩、CCV、correlogram
│   ├─ 纹理：GLCM、Tamura、LBP 直方图、Gabor 能量
│   ├─ 形状：Hu、Zernike、Fourier
│   ├─ 场景：GIST
│   └─ 聚合：BoW、VLAD、Fisher、CNN 全局池化
└─ 局部
    ├─ 梯度直方图：SIFT、SURF、GLOH、HOG、DAISY
    ├─ 二进制：BRIEF、ORB、BRISK、FREAK、LATCH
    ├─ 区域/仿射：MSER、Harris-Affine、Hessian-Affine
    ├─ 矩：Zernike、Legendre、Tchebichef
    ├─ 形状：Shape Context、链码、CSS、转向函数
    ├─ 纹理：LBP、Gabor、MR8、Textons
    └─ 学习型：L2-Net、HardNet、SuperPoint、D2-Net、R2D2
```

---

## 19. 结论

描述子的核心思想可以浓缩为三步：

1. **选一个区域**——用检测子找出稳定、可重复的位置或区域
2. **编码一个向量**——用梯度、比较、矩、滤波、频谱等算子把内容量化
3. **做一次归一化**——用归一化、主方向、尺度、仿射等手段获得不变性

不同描述子的差别，本质上就是：

- 用什么算子 $\mathcal{F}$
- 用什么量化 $\mathcal{Q}$
- 用什么归一化 $\mathcal{N}$
- 想要哪种不变性
- 是手工设计还是从数据中学习

从历史脉络看，描述子经历了三个阶段：

- **手工几何阶段**：Harris、SIFT、SURF、HOG、LBP、Shape Context
- **二进制与实时阶段**：BRIEF、ORB、BRISK、FREAK、AKAZE
- **学习与深度阶段**：Fisher/VLAD → HardNet、SuperPoint、SuperGlue、D2-Net、DELF/DELG

但即使到了深度学习时代，经典描述子的设计思想（尺度空间、主方向归一化、
方向直方图、二值比较、聚合编码）依然是理解现代特征表示的重要基础，
也是许多工业场景中低延迟、低功耗方案的首选。