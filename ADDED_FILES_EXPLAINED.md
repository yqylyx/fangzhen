# 新增文件详解：Phys-HPGT 深度学习项目

这份文档专门解释我这次为仓库新增的深度学习工程文件。目标是让你快速知道：

- 每个文件负责什么
- 里面的核心类 / 函数是怎么工作的
- 文件之间怎么衔接
- 后续如果你要继续改，应该优先改哪里

本次新增内容围绕一个完整的深度学习项目展开，任务是：

- 电力系统暂态稳定诊断二分类：`稳定 / 失稳`
- 风险回归：输出 `[0,1]` 风险分数
- 支持可变时间长度 `T`
- 支持可变通道数：电压通道 `Nv`、功角通道 `Ng`
- 支持缺失值 / 截断，通过 mask 显式处理
- 支持 early-exit 在线推理
- 支持 TorchScript / ONNX 导出

---

## 1. `requirements.txt`

### 作用

这个文件定义项目运行所需的 Python 依赖，方便直接安装环境。

### 包含的核心依赖

- `torch>=2.2,<2.6`
- `numpy`
- `scipy`
- `pandas`
- `einops`
- `torchmetrics`
- `tqdm`
- `pyyaml`
- `matplotlib`
- `onnx`

### 设计考虑

- `torch` 是主框架。
- `einops` 用来做 patch 展开和张量重排，代码可读性明显更好。
- `torchmetrics` 负责 F1、Recall、AUC、Confusion Matrix 等评估。
- `onnx` 负责导出。
- `matplotlib` 只负责保存图，不依赖 seaborn，保持环境简洁。

### 可选依赖

文件里还留了注释形式的可选包：

- `torch_geometric`
- `onnxruntime`
- `captum`
- `accelerate`

也就是说核心项目不强依赖这些包，但如果你要增强图网络、解释性或者多卡训练，可以再装。

---

## 2. `configs/default.yaml`

### 作用

这是整个项目的默认配置文件，训练、评估、导出都会从这里读参数。

### 主要配置块

#### `seed / project_name / output_dir`

控制实验随机种子、项目名和输出目录。

#### `data`

定义数据读取方式：

- `root: npy_jobs`
- `file_glob`
- `sentinel: -99999`
- `normalize: robust`
- `train/val/test` 切分比例
- `split_file`
- `respect_existing_split_dirs`
- dataloader workers 等

这块的设计重点是兼容你当前仓库的 `.npy` 数据目录，同时允许以后替换成新的数据目录。

#### `model`

定义 Phys-HPGT 的核心结构参数：

- `patch_len`
- `d_model`
- `n_layers`
- `n_graph_layers`
- `n_heads`
- `dropout`
- `temporal_encoder`
- `graph_type`
- `graph_topk`
- `n_exit`
- `early_exit`
- `use_freq_branch`

这里决定模型容量、patch 粒度、图注意力方式和 early-exit 数量。

#### `train`

定义训练超参：

- `batch_size`
- `epochs`
- `lr`
- `weight_decay`
- `early_stopping_patience`
- `amp`
- `grad_clip`
- `device`
- `compile`

#### `loss`

定义多任务损失权重和类别不平衡设置：

- `w_cls`
- `w_risk`
- `lambda_phys`
- `lambda_con`
- `focal_gamma`
- `class_weight_unstable`
- `auto_class_weight`

#### `physics`

定义物理启发特征里的阈值，例如：

- `v_warn`
- `v_danger`
- `angle_warn_deg`
- `angle_danger_deg`

#### `eval`

定义评估阈值和 early stopping 监控指标组合权重。

#### `stream`

定义流式推理步长和 early-exit 置信度阈值。

#### `export`

定义 TorchScript 和 ONNX 输出路径与 opset。

### 为什么要独立配置文件

这样做的好处是：

- 后续换数据不用改代码
- 调参更方便
- 可以很容易复制出 `debug.yaml`、`small.yaml`、`large.yaml` 等实验配置

---

## 3. `src/__init__.py`

### 作用

这是 `src` 包的初始化文件，让 `src` 可以被 `python -m src.train` 这种方式作为模块运行。

### 价值

它本身很小，但很重要。没有它，`src` 不会被视为一个规范 Python 包，模块导入会更容易出问题。

---

## 4. `src/data/__init__.py`

### 作用

对数据层的主要接口做统一导出：

- `NPYTransientDataset`
- `collate_fn`

### 价值

这样训练脚本里可以直接写：

```python
from .data import NPYTransientDataset, collate_fn
```

而不用从更深层路径导入，结构更干净。

---

## 5. `src/data/dataset.py`

这是数据层最关键的文件之一。

### 核心目标

把单个 `.npy` 暂态样本读成统一格式，并在 batch 组装时处理：

- 时间维 padding
- 通道维 padding
- 缺失值识别
- 各种 mask 生成

### 主要内容

#### `_load_npy_dict(path)`

读取 `.npy` 文件，并确保里面保存的是一个 `dict`。

兼容 `np.save(..., allow_pickle=True)` 保存出来的 object ndarray 包装形式。

#### `_find_first(data, *keys)`

用于兼容不同字段名，比如：

- `t` / `times`
- `V` / `voltages`
- `delta` / `angles`
- `y` / `label`

这一步是为了兼容你仓库原有数据，而不是强迫你重新转换数据格式。

#### `_ensure_2d(array, t_len=None)`

保证输入张量最终是二维 `[T, C]`。

如果某些单通道样本是 `[T]`，会自动扩成 `[T,1]`。

#### `_replace_invalid(arr, sentinel)`

把缺失值统一处理：

- `NaN`
- `-99999`

输出：

- 清洗后的数组
- 对应的有效性 mask

#### `_safe_scale` / `_normalize`

做鲁棒归一化。

这里没有直接对全数组粗暴做标准化，而是先看 valid mask，只对有效值估计中位数和尺度，避免缺失值污染统计量。

默认使用 robust normalization，而不是简单 z-score，这是为了提升异常值和缺失值场景下的稳定性。

---

### `class NPYTransientDataset`

#### `__init__`

负责：

- 确定数据根目录
- 保存 split 名称
- 保存 sentinel 和归一化方式
- 解析文件列表

一个重要细节是这里兼容了两种 `file_list`：

- 相对路径
- 已经从外部 `rglob` 出来的路径

我后面还专门修了一次路径重复拼接的问题，避免出现 `npy_jobs/npy_jobs/...` 这种错误。

#### `peek_label(idx)`

这是一个轻量辅助函数，用来快速查看标签，主要给训练脚本估算类别权重时用。

这样就不用先完整构造一个 batch 才知道标签分布。

#### `__getitem__(idx)`

这是最核心的函数。

它做的事包括：

1. 读取单个 `.npy`
2. 自动兼容新旧字段名
3. 读取 `t/V/delta/y/meta`
4. 缺失值转成 `NaN` 并生成：
   - `mask_V`
   - `mask_delta`
5. 对有效值做鲁棒归一化，得到：
   - `V`
   - `delta`
6. 保留原始零填充版本：
   - `raw_V`
   - `raw_delta`
7. 生成：
   - `ch_mask_V`
   - `ch_mask_delta`
   - `time_mask`

最终它返回的是一个字典，既包含模型真正用的归一化输入，也包含构造物理损失时需要的“原始值版本”。

这是一个很关键的设计：

- 模型前向用归一化值更稳定
- 物理特征计算必须尽量基于原始物理量，否则阈值就失去意义

---

### `collate_fn(batch)`

这个函数负责把一个 batch 的样本对齐成统一尺寸。

#### 它解决的问题

不同样本可能有：

- 不同时间长度 `T`
- 不同电压通道数 `Nv`
- 不同功角通道数 `Ng`

所以必须做 padding。

#### 具体做法

- 时间维 pad 到 batch 内最大 `T`
- 电压通道 pad 到 batch 内最大 `Nv`
- 功角通道 pad 到 batch 内最大 `Ng`
- padding 值统一填 0
- 用 mask 标记真实位置和 padding 位置

#### 输出

统一输出：

- `V`
- `delta`
- `raw_V`
- `raw_delta`
- `mask_V`
- `mask_delta`
- `ch_mask_V`
- `ch_mask_delta`
- `time_mask`
- `y`
- `files`
- `meta`

### 这个文件的整体价值

这个文件基本上决定了“数据能不能正确喂给模型”。如果后续训练效果不稳定，数据层永远是应该优先检查的地方之一。

---

## 6. `src/models/__init__.py`

### 作用

对模型包的主模型做统一导出：

- `PhysHPGT`

方便外部统一导入。

---

## 7. `src/models/phys_hpgt.py`

这是整个项目最核心的模型文件。

它实现了 Phys-HPGT 主体结构。

---

### 7.1 文件整体结构

这个文件里我没有把所有模块拆成很多小文件，而是先集中在一个文件中，目的是：

- 便于整体阅读
- 便于调试模型 shape
- 便于你后面快速定位 forward 路径

等你后面稳定下来，如果想进一步工程化，可以再拆成：

- `patch.py`
- `temporal.py`
- `graph.py`
- `fusion.py`
- `heads.py`

---

### 7.2 辅助函数

#### `masked_mean`

对带 mask 的张量做均值池化。

这是这个模型里最常用的工具函数之一，因为：

- patch pooling 要用
- channel pooling 要用
- graph pooling 要用
- 频域摘要也要用

#### `sinusoidal_positional_encoding`

生成标准 Transformer 风格的位置编码，用在 patch 序列上。

---

### 7.3 `GraphNorm`

这是对图节点维度做归一化的模块。

#### 作用

因为图注意力阶段的输入是通道节点 embedding，如果不同样本通道数不同、节点统计差异很大，直接做 attention 很容易不稳定。

`GraphNorm` 做的是：

- 按节点维度统计均值 / 方差
- 只在有效节点上归一化
- 再乘可学习缩放参数

#### 为什么不是直接 LayerNorm

因为 LayerNorm 是对特征维归一化，而这里我们更关心“不同节点集合”的统计漂移，所以单独做一个 GraphNorm 更合适。

---

### 7.4 `ChannelPatchEmbed`

这是 channel-independent patch 编码器。

#### 输入输出

输入：

- `x: [B,T,C]`
- `mask: [B,T,C]`

输出：

- `token: [B,C,P,D]`
- `patch_mask: [B,C,P]`

#### 核心思路

1. 把时间维 pad 到 `patch_len` 整数倍
2. 重排成 `[B,C,P,patch_len]`
3. 对 patch 内无效位置做 mask
4. 统计 patch 有效比例
5. 若 patch 有效比例太低，则该 patch 视为无效
6. 把 patch 投影到 `d_model`

#### 为什么叫 channel-independent

因为每个通道先独立沿时间维切 patch，再共享同一个线性投影层编码。这样不会一开始就把不同物理通道混在一起。

这个设计对“不同系统、不同通道数”的泛化更友好。

---

### 7.5 `TemporalTransformerBlock`

这是时序 Transformer 编码块。

#### 组成

- `LayerNorm`
- `MultiheadAttention`
- 残差连接
- FFN
- 残差连接

#### mask 支持

这里显式使用了 patch mask，并把空序列做了兜底处理，避免 attention 因为全被 mask 掉而报错。

#### 为什么返回 attention

如果 `return_attn=True`，它会返回多头注意力权重，供后面的解释性接口使用。

---

### 7.6 `TCNResidualBlock`

这是 TCN 模式下的时序编码块。

#### 作用

当你把配置切到：

```yaml
temporal_encoder: tcn
```

模型就不再用 Transformer，而改用 dilated convolution 建模 patch 序列。

#### 结构

- `Conv1d`
- GELU
- Dropout
- 第二层 `Conv1d`
- 残差连接

#### 为什么保留 TCN 选项

因为某些时序任务里：

- TCN 更轻
- 更稳定
- 小数据下更不容易过拟合

所以我没有把结构锁死在 Transformer 上。

---

### 7.7 `GraphAttentionBlock`

这是跨通道图注意力模块。

#### 节点定义

节点就是所有通道：

- 电压通道节点
- 功角通道节点

拼成一个统一的节点集合。

#### 核心功能

- 多头图自注意力
- 支持 `dense_attn`
- 支持 `sparse_topk`
- GraphNorm
- FFN
- residual

#### `dense_attn`

所有节点两两连接，相当于稠密图。

#### `sparse_topk`

每个节点只保留 top-k 邻居，近似成稀疏图，有利于控制复杂度，也更符合“局部强耦合”的物理直觉。

#### 注意力返回

如果 `return_attn=True`，这里也会返回图注意力矩阵，供可视化使用。

---

### 7.8 `DecisionHead`

每个 early-exit 都有一个独立的决策头。

#### 输出

- `cls`: `[B,2]`
- `risk`: `[B,1]`

#### 为什么分类和回归分成两个头

因为：

- 二分类更关注边界判决
- 风险回归更关注连续程度

把它们拆成两个头，可以让共享 backbone 学到更多通用表征，同时不把两个任务完全混成一回事。

---

### 7.9 `PhysHPGT`

这是整个模型主类。

#### `__init__`

初始化阶段主要做了这些事：

1. 构建 patch 编码器
2. 构建 modality embedding
3. 根据配置构建：
   - Transformer 时序编码器
   - 或 TCN 时序编码器
4. 构建图注意力层
5. 构建 RiskToken 和跨模态门控融合层
6. 可选频域分支
7. 构建多级 early-exit 头
8. 构建 contrastive embedding 投影层

---

### 7.10 `_encode_temporal`

这是时序编码统一入口。

#### 功能

- 给 patch 序列加位置编码
- 根据配置走 Transformer 或 TCN
- 返回编码后的 token
- 如果需要，返回 attention

#### 为什么做统一入口

这样主 `forward` 不需要关心底层是 Transformer 还是 TCN，只需要拿到统一输出即可。

---

### 7.11 `_spectral_feature`

这是频域摘要分支。

#### 正常训练 / 普通推理时

使用 `torch.fft.rfft` 提取频域幅值摘要。

#### ONNX 导出时

我专门做了兼容处理：

- ONNX 导出时关闭频域分支
- 避免 FFT / 动态池化造成导出失败

这部分是后面我实际调试导出链路时补的，不是纸面设计。

---

### 7.12 `_fuse_modalities`

这是跨模态融合函数。

#### 输入

- 电压节点 embedding
- 功角节点 embedding
- 节点 mask
- 原始输入 `V / delta`

#### 过程

1. 分别聚合电压节点和功角节点
2. 聚合全局图节点表示
3. 与 `RiskToken` 一起生成门控输入
4. 计算 gate
5. 做 gated fusion
6. 若开启频域分支，则再叠加频域摘要

#### 融合形式

核心形式就是：

```python
fused = gate * pooled_v + (1 - gate) * pooled_d
```

### 为什么这样设计

因为不同故障 / 拓扑下，电压和功角的重要性会变化，门控融合比固定拼接更灵活。

---

### 7.13 `_resolve_exit_states`

这个函数负责把中间决策状态整理成固定数量的 early-exit。

#### 背景

真正 backbone 产生的中间状态数量，未必正好等于配置里的 `n_exit`。

所以这里会：

- 不够时做 refinement 扩展
- 太多时均匀抽样

这样最终始终保证返回 `n_exit` 个出口。

---

### 7.14 `forward(...)`

这是模型的主计算路径。

#### 主要步骤

1. 处理 `ch_mask`
2. 电压和功角分别做 patch embedding
3. 加 modality embedding
4. 分别做 temporal encoding
5. 对 patch 序列做 pooling 得到 channel node embedding
6. 拼接成总图节点
7. 先做一次融合，得到第一个决策状态
8. 逐层通过 graph attention block
9. 每层图编码后再做一次融合
10. 生成多个 exit 的分类和风险输出
11. 生成 contrastive embedding
12. 若要求解释性输出，返回 attention / gate / channel importance

#### 返回值

至少包含：

- `logits_list`
- `risk_list`
- `emb`

可选包含：

- `attn`
- `hidden`

### 为什么这是当前项目最重要的文件

因为它定义了模型能力边界。你后面想提升精度、做消融实验、换图结构、换融合方式，最主要都是改这里。

---

## 8. `src/models/losses.py`

这是训练目标定义文件。

它把分类、风险回归、物理正则、监督对比学习整合到一起。

---

### 8.1 `_cfg_get`

一个小工具函数，用于从嵌套配置字典中安全取值。

这样 loss 模块不需要依赖整个配置对象的具体类型。

---

### 8.2 `FocalLoss`

#### 作用

解决类别不平衡问题，重点关注难分类样本。

#### 为什么这里需要它

暂态稳定诊断里“失稳”样本通常更重要，而且经常类别不平衡。

普通交叉熵可能更偏向多数类，Focal Loss 可以缓解这个问题。

---

### 8.3 `supervised_contrastive_loss`

#### 作用

对 embedding 做监督式对比学习。

#### 核心思路

- 同类别样本拉近
- 异类别样本拉远
- 如果提供风险分数，还会进一步按风险距离调整 pair 权重
- `hard_neg=True` 时，会更强调难负样本

#### 为什么这对项目有价值

因为单纯做分类头，有时 backbone 学到的是“够分就行”的边界特征；而监督式对比学习会迫使 embedding 空间本身更有结构。

---

### 8.4 物理辅助函数

#### `_masked_mean`

和模型里的版本类似，用在物理特征构造阶段。

#### `_masked_spread`

计算带 mask 的功角 spread。

#### `_angle_to_deg`

自动猜测功角单位是否可能是弧度，如果数值范围偏小，会转换到度。

这个设计是为了兼容“输入功角单位未显式声明”的场景。

---

### 8.5 `physics_risk_features(...)`

这是物理一致性分支的核心。

#### 输入

- `V`
- `delta`
- `mask_V`
- `mask_delta`

#### 输出

返回一个字典，里面包括：

- `risk_target`
- `phys_label`
- `low_v_ratio`
- `danger_v_ratio`
- `tail_low_ratio`
- `angle_peak`
- `angle_tail`
- `angle_speed`
- `missing_ratio`

#### 作用

它并不依赖外部额外标注，而是直接从输入波形构造“物理启发风险目标”。

也就是说，风险回归头有了一个更物理化的监督信号，而不只是跟着分类头走。

#### 为什么重要

因为这让模型不仅学“是不是失稳”，还学“风险程度和哪些物理现象相关”。

---

### 8.6 `physics_consistency_loss(...)`

#### 作用

约束：

- 分类头的失稳概率
- 风险回归头的输出
- 物理启发风险目标

三者尽量一致。

#### 组成

- 分类概率和物理风险目标的 MSE
- 风险回归和物理风险目标的 smooth L1
- 分类概率和物理标签的一致性 BCE

这样做的效果是：

- 防止 risk head 变成一个“随便输出”的附属分支
- 提高模型输出和物理直觉的一致性

---

### 8.7 `total_loss(...)`

这是总损失整合函数。

#### 它包含四部分

- 分类损失 `cls_loss`
- 风险回归损失 `risk_loss`
- 物理一致性损失 `phys_loss`
- 对比学习损失 `con_loss`

#### 最终加权方式

由配置控制：

- `w_cls`
- `w_risk`
- `lambda_phys`
- `lambda_con`

#### 一个我实际修过的细节

在小 batch 烟雾测试时，对比损失数值可能出现极小负值，我把它夹到了 `>=0`，让训练日志更稳定，也避免看起来反直觉。

---

## 9. `src/utils/__init__.py`

### 作用

统一导出可视化函数：

- `save_attention_heatmap`
- `save_waveforms`

---

## 10. `src/utils/vis.py`

这是解释性和图像输出文件。

### 设计目标

不是做交互式复杂可视化，而是做：

- 能稳定保存 PNG
- 不容易把 VSCode / 远程环境卡死
- 足够支撑训练后分析

---

### 10.1 `_to_numpy`

把 `Tensor / ndarray / 其他数组类型` 统一转为 numpy，简化画图接口。

---

### 10.2 `_extract_attention_map`

从模型返回的 attention 字典中提取适合可视化的二维热力图。

#### 支持的来源

- 图注意力
- 时序注意力
- 通道重要性

如果 attention 维度太高，会自动做均值归约。

---

### 10.3 `save_attention_heatmap(...)`

保存注意力热力图。

#### 关键设计

- 限制 `max_channels`
- 限制 `max_patches`
- 默认 `dpi=120`

这些都是为了避免图片过大导致：

- VSCode 卡顿
- 远程桌面打开慢
- PNG 体积过大

---

### 10.4 `save_waveforms(...)`

保存电压和功角波形图。

#### 输出

- 上半部分：电压
- 下半部分：功角

#### 关键设计

- 限制最大通道数
- 支持 `downsample`
- 保存后及时 `plt.close`

这里我还按你的要求把一个典型保存示例写进了注释，方便后续排查绘图慢的问题。

---

## 11. `src/infer_stream.py`

这是在线 / 流式推理接口。

### 核心类：`StreamInfer`

它的目标是让模型不是只能“整段样本离线预测”，而是可以按时间步不断更新。

---

### 11.1 `__init__`

初始化时保存：

- 模型
- `patch_len`
- `step`
- `exit_threshold`
- `device`
- `max_steps`

同时建立 ring buffer：

- `V_buffer`
- `delta_buffer`
- `mask_V_buffer`
- `mask_delta_buffer`

---

### 11.2 `_prepare_step`

把每个时刻输入统一成一维 tensor。

兼容：

- numpy
- python list
- torch tensor

---

### 11.3 `_normalize_masks`

兼容不同 mask 输入格式：

- `None`
- `dict`
- `(mask_v, mask_d)`
- 一个平坦拼接 mask

这样上层调用更灵活，不会被单一接口形式卡住。

---

### 11.4 `_build_batch`

把 ring buffer 里的历史时刻拼成模型可接受的 batch：

- `[1,T,Nv]`
- `[1,T,Ng]`
- mask
- channel mask
- time mask

---

### 11.5 `update(...)`

每来一个时间步就调用一次。

#### 做的事情

1. 整理输入
2. 整理 mask
3. 把当前时刻写入 ring buffer
4. 如果还没到 `step`，返回 `None`
5. 如果到达预测步点，则调用 `predict()`

---

### 11.6 `predict()`

这是流式决策主逻辑。

#### 做的事情

1. 用当前 buffer 构造一个 batch
2. 跑模型
3. 遍历所有 exit
4. 看是否有某一级置信度超过 `exit_threshold`
5. 若超过，则提前退出
6. 否则使用最后一级结果

#### 返回

- `cls`
- `prob`
- `risk`
- `exit_level`
- `explanation_stub`

其中 `explanation_stub` 里带了：

- 每一级 exit 的置信度
- 每一级 exit 的失稳概率
- 通道重要性
- 融合门控信息

这给后续做在线解释预留了接口。

---

## 12. `src/train.py`

这是训练、验证、测试的总入口文件。

它把数据、模型、损失、评估、checkpoint 和可视化都串起来了。

---

### 12.1 顶部初始化

这里我加了：

```python
os.environ.setdefault("MPLCONFIGDIR", ...)
```

#### 原因

我在本机实际运行时遇到了 matplotlib 默认缓存目录权限问题，所以把缓存定向到了工作区里的 `.matplotlib_cache`，避免训练 / 评估 / 画图时报权限错误。

这不是纸面代码，而是根据实际运行问题加的修复。

---

### 12.2 配置与通用工具函数

包括：

- `load_config`
- `seed_everything`
- `ensure_dir`
- `save_json`

这些负责实验可复现和结果落盘。

---

### 12.3 数据切分函数

#### `discover_files`

扫描数据文件。

#### `build_splits`

优先逻辑是：

1. 如果有 `train / val / test` 子目录，优先使用
2. 否则若已有 `split_file`，直接读取
3. 否则自动按 `8/1/1` 切分并保存 split json

#### `truncate_split`

用于调试时限制样本数。

#### `build_datasets`

构造 train/val/test 三个 `NPYTransientDataset`。

#### `build_loader`

构造 DataLoader。

---

### 12.4 设备和类别权重

#### `get_device`

根据配置和 `torch.cuda.is_available()` 自动选择设备。

#### `estimate_class_weight`

根据训练集标签分布自动估算失稳类权重。

这样你就不用每次手动猜 `class_weight_unstable`。

---

### 12.5 评估指标和混淆矩阵

#### `save_confusion_matrix_png`

保存 confusion matrix PNG。

#### `binary_auc_np`

当 `torchmetrics` 不可用时，手动算一个二分类 AUC。

#### `compute_metrics_manual`

手动计算：

- macro/micro F1
- unstable precision/recall/F1
- AUC
- confusion matrix

#### `compute_metrics_torchmetrics`

如果装了 `torchmetrics`，优先用官方实现。

### 为什么保留 manual fallback

这样环境不完整时也不至于整个评估流程直接挂掉。

---

### 12.6 `train_one_epoch(...)`

#### 功能

单轮训练。

#### 流程

1. `model.train()`
2. batch 搬到设备上
3. 基于 `raw_V / raw_delta` 构造物理特征
4. 前向
5. 计算总损失
6. AMP / 非 AMP 反向传播
7. 梯度裁剪
8. 优化器更新
9. 记录 loss 指标

#### 输出

返回这一轮平均：

- `loss`
- `loss_cls`
- `loss_risk`
- `loss_phys`
- `loss_contrastive`
- `lr`

---

### 12.7 `eval_one_epoch(...)`

#### 功能

单轮验证 / 测试。

#### 除了 loss，还额外做

- 收集所有 `y_true`
- 收集最终出口的失稳概率
- 统计平均推理延迟
- 计算所有评估指标

#### 为什么只用最后一个 exit 做评估

因为离线评估通常更关心模型最完整的决策结果；online/streaming 场景才重点看提前退出。

---

### 12.8 `monitor_score(...)`

把多个指标组合成 early stopping 的监控分数：

- unstable_f1
- unstable_recall
- auc

### 为什么这样做

因为这个任务里不能只盯总体准确率，失稳类召回通常更重要。

---

### 12.9 checkpoint 相关函数

- `save_checkpoint`
- `maybe_load_checkpoint`

支持：

- 保存 `last.pt`
- 保存 `best_by_f1_unstable.pt`
- 断点续训

---

### 12.10 `dump_attention_examples(...)`

这个函数会从验证 / 测试集里抽取少量样本，保存：

- attention heatmap
- waveforms

这样训练完成后可以快速看模型关注了哪里。

---

### 12.11 `build_model(...)`

按配置构建 `PhysHPGT`。

如果配置允许，还可以使用 `torch.compile`。

---

### 12.12 `main()`

主入口，负责整个训练和评估流程。

#### 主要步骤

1. 解析参数
2. 读取配置
3. 固定随机种子
4. 构建数据集和 dataloader
5. 自动估计类权重
6. 构建模型、优化器、GradScaler
7. 若需要，加载 checkpoint
8. 若是 `--eval-only`，直接评估
9. 否则进入训练循环
10. 每轮训练后在 val 上评估
11. 根据监控分数决定是否更新 best checkpoint
12. 触发 early stopping 后结束
13. 加载 best 模型并在 test 上评估
14. 保存最终指标和图

### 这个文件的重要性

如果你后面要：

- 改训练策略
- 加 warmup / scheduler
- 改监控指标
- 改验证逻辑

主要都在这个文件里动。

---

## 13. `export/export_onnx.py`

这是模型导出文件。

### 支持导出

- TorchScript
- ONNX

并带一个 `onnxruntime` demo。

---

### 13.1 顶部 `sys.path` 处理

我加了项目根目录注入：

```python
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
```

#### 原因

这是我实际运行时修的。

直接执行：

```bash
python export/export_onnx.py
```

时，Python 默认不一定能正确找到 `src` 包，所以这里显式把根目录加入导入路径。

---

### 13.2 `ExportWrapper`

封装原始模型，只保留导出真正需要的输出：

- `logits`
- `risk`

这样 TorchScript / ONNX 导出的 graph 更干净，不会把解释性输出也一起塞进去。

---

### 13.3 `load_config`

读取 YAML 配置。

---

### 13.4 `load_model`

这个函数我做了一个很重要的导出兼容处理：

- 导出时默认把 `use_freq_branch=False`

#### 为什么

因为我实际测试时，频域分支会导致 ONNX 导出失败或兼容性变差。

所以这里采取了“导出兼容模式”：

- 训练 / 普通推理：保留频域分支
- 导出 ONNX：关闭频域分支

如果加载 checkpoint，会忽略 `freq_proj` 的不匹配权重。

---

### 13.5 `make_dummy_inputs`

生成一组导出时用的 dummy 输入。

包括：

- `V`
- `delta`
- `mask_V`
- `mask_delta`
- `ch_mask_V`
- `ch_mask_delta`
- `time_mask`

---

### 13.6 `export_torchscript`

导出 TorchScript。

#### 一个我实际修过的细节

我把：

```python
check_trace=False
```

加进去了。

#### 原因

原始动态模型在 trace 校验阶段会因为动态图路径导致对比失败，但实际导出文件本身是可用的。

这是一个为了工程可落地做的折中。

---

### 13.7 `export_onnx`

导出 ONNX。

#### 关键点

定义了动态轴：

- `T`
- `Nv`
- `Ng`

也就是满足你提出的“动态时间 / 动态通道数”要求。

---

### 13.8 `run_onnx_demo`

如果环境里装了 `onnxruntime`，这个函数会：

1. 加载导出的 ONNX
2. 自动读取模型真实输入名
3. 构造 feed
4. 预热
5. 多次运行统计平均延迟

#### 一个我实际修过的细节

一开始 demo 会因为 `time_mask` 在导出图中被优化掉，导致 runtime 报：

- `Invalid input name: time_mask`

所以我把它改成了：

- 动态读取 `session.get_inputs()`
- 只给 runtime 真正存在的输入

这个修复让 demo 真正跑通了。

---

### 13.9 `main()`

串起整个导出流程：

- 读配置
- 加载模型
- 导出 TorchScript
- 导出 ONNX
- 可选运行 onnxruntime demo

---

## 14. `README.md`

### 作用

我重写了 README，让仓库除了原有规则型脚本之外，也能完整说明这个深度学习项目怎么用。

### 主要内容

README 覆盖了：

- 项目目标
- 环境安装
- 数据格式说明
- 项目结构
- 训练命令
- 评估命令
- 导出命令
- 流式推理示例
- 调参与 debug 建议
- ONNX / 在线推理说明

### 为什么 README 很重要

因为你后续不可能每次都翻源码。一个好的 README 本身就是“项目的第一层接口”。

---

## 15. 文件之间的调用关系

可以把这套新增工程理解成下面这条链路：

### 数据流

`configs/default.yaml`

-> 被 `src/train.py` / `export/export_onnx.py` 读取

`src/data/dataset.py`

-> 把 `.npy` 样本转成带 mask 的 batch

`src/models/phys_hpgt.py`

-> 接收 batch，输出 `logits_list / risk_list / emb / attn`

`src/models/losses.py`

-> 基于模型输出 + 原始物理量构造总损失

`src/train.py`

-> 执行训练 / 验证 / 测试 / checkpoint / 可视化保存

`src/infer_stream.py`

-> 在在线场景中用 ring buffer 调用模型，实现 early-exit

`src/utils/vis.py`

-> 保存 attention heatmap 和波形图

`export/export_onnx.py`

-> 导出 TorchScript / ONNX，并做 onnxruntime 推理 demo

---

## 16. 我在实现过程中实际修过的问题

这些不是“静态写代码”就结束了，而是在真实运行验证中发现并修掉的：

### 1. 数据集路径重复拼接

问题：

- 外部给 `file_list` 时，如果已经带了 `npy_jobs/...` 前缀，数据集初始化里会再次拼接 root

修复：

- 先判断路径自身是否存在，再决定是否拼 root

### 2. matplotlib 缓存目录权限问题

问题：

- 当前机器上默认 `~/.matplotlib` 不可写，训练 / help / 画图时会报权限错误

修复：

- 在 `train.py` 和 `vis.py` 里把 `MPLCONFIGDIR` 定向到工作区 `.matplotlib_cache`

### 3. `export/export_onnx.py` 无法直接导入 `src`

问题：

- 直接脚本运行时 `src` 包不在搜索路径里

修复：

- 手动注入项目根目录到 `sys.path`

### 4. TorchScript trace 校验失败

问题：

- 动态图模型在 `torch.jit.trace` 的校验环节容易出现 graph diff

修复：

- `check_trace=False`

### 5. ONNX 导出与频域分支冲突

问题：

- 频域摘要分支在导出时容易触发算子兼容问题

修复：

- 导出脚本默认关闭 `use_freq_branch`

### 6. ONNXRuntime demo 输入名不匹配

问题：

- 图优化后某些输入被裁掉，静态手写 feed 会报错

修复：

- 动态读取 `session.get_inputs()`

### 7. 对比学习损失出现极小负值

问题：

- 在极小 batch 烟雾测试时出现轻微负值

修复：

- `clamp_min(0.0)`

---

## 17. 哪些文件是你后续最可能修改的

如果你后面继续迭代，我建议优先关注这些文件：

### 最常改

- `configs/default.yaml`
- `src/models/phys_hpgt.py`
- `src/models/losses.py`
- `src/train.py`

### 如果数据格式变化

- `src/data/dataset.py`

### 如果你更关注在线推理

- `src/infer_stream.py`

### 如果你更关注部署

- `export/export_onnx.py`

---

## 18. 当前实现状态总结

这次新增的文件不是只有“框架”，而是已经打通了主链路：

- 数据读取可运行
- 模型前向可运行
- loss 可运行
- 训练 / 验证函数可运行
- 流式推理可运行
- TorchScript 导出可运行
- ONNX 导出可运行
- onnxruntime demo 可运行

也就是说，这是一套已经具备最小工程闭环的实现，而不是只有接口没有行为的空壳。

---

## 19. 建议你下一步怎么用

如果你准备真正开始实验，推荐顺序是：

1. 先直接跑

```bash
python -m src.train --config configs/default.yaml
```

2. 看训练输出和 `runs/phys_hpgt` 下的指标文件

3. 如果失稳召回偏低，优先调整：

- `class_weight_unstable`
- `lambda_phys`
- `lambda_con`
- `exit_threshold`

4. 如果显存吃紧，优先调整：

- `batch_size`
- `patch_len`
- `d_model`
- `n_layers`

5. 如果你准备正式部署，再重点看：

- `src/infer_stream.py`
- `export/export_onnx.py`

---

如果你愿意，我下一步还可以继续帮你把这份文档再细化成“按函数逐段解释版”，或者直接把每个核心文件再补上中文行内注释。
