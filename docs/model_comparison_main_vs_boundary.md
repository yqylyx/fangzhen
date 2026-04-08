# 主工程 `src/` 与 `tsa_boundary_detection/` 模型异同详解

这份文档专门对比仓库里的两套模型：

- 主工程：`src/`
- 边界检测变体：`tsa_boundary_detection/`

核心结论先说：

1. 两者的**主干网络结构高度相似**，前 5 个阶段基本可以看成同一套骨架。
2. 两者**不只是最后一层不同**。
3. 真正变化的是整套任务定义、标签来源、输出形式、损失函数、阈值策略和评估逻辑。

如果要一句话概括：

- `src/` 是“暂态稳定诊断模型”
- `tsa_boundary_detection/` 是“边界样本检测模型”

---

## 1. 两个模型分别是干什么的

### 1.1 主工程 `src/`

主工程做的是**稳定 / 失稳诊断**，并额外回归一个风险值。

对应说明：

- 模型头定义在 [src/models/phys_hpgt.py](/Users/yqy/PycharmProjects/PaperProject/src/models/phys_hpgt.py#L226)
- 多任务损失定义在 [src/models/losses.py](/Users/yqy/PycharmProjects/PaperProject/src/models/losses.py#L189)

主工程的输出是：

- 二分类 logits：稳定 / 失稳
- 连续风险值：`risk in [0, 1]`
- 对比学习嵌入：`emb`

代码里最直接的地方是：

```python
class DecisionHead(nn.Module):
    ...
    def forward(self, x: Tensor) -> tuple[Tensor, Tensor]:
        return self.cls(x), self.risk(x)
```

见 [src/models/phys_hpgt.py](/Users/yqy/PycharmProjects/PaperProject/src/models/phys_hpgt.py#L226)。

---

### 1.2 边界检测变体 `tsa_boundary_detection/`

边界检测变体做的是**boundary / non-boundary** 判断。

这里的 `boundary` 不是“失稳”，而是“边界样本、临界样本、值得重点关注的样本”。

对应说明：

- README 定义见 [tsa_boundary_detection/README.md](/Users/yqy/PycharmProjects/PaperProject/tsa_boundary_detection/README.md#L1)
- 模型头定义见 [tsa_boundary_detection/src/models/phys_hpgt.py](/Users/yqy/PycharmProjects/PaperProject/tsa_boundary_detection/src/models/phys_hpgt.py#L221)

它的输出是：

- 单个 boundary logit
- `sigmoid(logit)` 后得到 `p_boundary`
- 辅助嵌入 `emb`

最直接的代码是：

```python
class BoundaryHead(nn.Module):
    ...
    def forward(self, x: Tensor) -> Tensor:
        return self.boundary(x).squeeze(-1)
```

见 [tsa_boundary_detection/src/models/phys_hpgt.py](/Users/yqy/PycharmProjects/PaperProject/tsa_boundary_detection/src/models/phys_hpgt.py#L221)。

---

## 2. 相同点：两者的主干阶段为什么看起来几乎一样

如果只看 `forward()` 主线，两者的结构几乎平行。

主工程主线：

- [src/models/phys_hpgt.py](/Users/yqy/PycharmProjects/PaperProject/src/models/phys_hpgt.py#L454)

边界检测主线：

- [tsa_boundary_detection/src/models/phys_hpgt.py](/Users/yqy/PycharmProjects/PaperProject/tsa_boundary_detection/src/models/phys_hpgt.py#L435)

两者共同的结构阶段是：

1. 输入 `V / delta / mask`
2. `patch_embed` 做局部时间切片
3. `Temporal Encoder` 做通道内时序建模
4. `masked_mean` 池化成 `v_nodes / d_nodes`
5. 拼成 `all_nodes`
6. `GraphAttentionBlock` 做跨通道交互
7. 模态融合 + 频域增强
8. 多出口 early-exit
9. 输出最终任务头结果 + `emb`

这也是为什么你看到它们的流程图阶段很像。

### 2.1 输入和 patch embedding 相同

两套模型都先处理：

- 电压序列 `V`
- 功角序列 `delta`
- 对应掩码 `mask_V / mask_delta`

然后都进入：

```python
v_tokens, v_patch_mask = self.patch_embed(V, mask_V)
d_tokens, d_patch_mask = self.patch_embed(delta, mask_delta)
```

主工程见 [src/models/phys_hpgt.py](/Users/yqy/PycharmProjects/PaperProject/src/models/phys_hpgt.py#L480)  
边界版见 [tsa_boundary_detection/src/models/phys_hpgt.py](/Users/yqy/PycharmProjects/PaperProject/tsa_boundary_detection/src/models/phys_hpgt.py#L461)

---

### 2.2 通道内时序建模相同

两者都先把每个通道单独作为一条时序处理，而不是一开始就做跨通道混合。

代码结构基本一致：

```python
v_tokens, temporal_attn_v = self._encode_temporal(...)
d_tokens, temporal_attn_d = self._encode_temporal(...)
```

主工程见 [src/models/phys_hpgt.py](/Users/yqy/PycharmProjects/PaperProject/src/models/phys_hpgt.py#L486)  
边界版见 [tsa_boundary_detection/src/models/phys_hpgt.py](/Users/yqy/PycharmProjects/PaperProject/tsa_boundary_detection/src/models/phys_hpgt.py#L467)

这说明两者都保留了“先建模单通道动态，再建模跨通道关系”的设计哲学。

---

### 2.3 节点化、图注意力和频域增强相同

两者都把 patch 序列压成节点，再进入图注意力：

```python
v_nodes = masked_mean(v_tokens, v_patch_mask, dim=2)
d_nodes = masked_mean(d_tokens, d_patch_mask, dim=2)
all_nodes = torch.cat([v_nodes, d_nodes], dim=1)
```

主工程见 [src/models/phys_hpgt.py](/Users/yqy/PycharmProjects/PaperProject/src/models/phys_hpgt.py#L490)  
边界版见 [tsa_boundary_detection/src/models/phys_hpgt.py](/Users/yqy/PycharmProjects/PaperProject/tsa_boundary_detection/src/models/phys_hpgt.py#L471)

之后两者都复用：

- `GraphAttentionBlock`
- `fusion_proj`
- `freq_proj`
- `exit_refiners`
- `emb_proj`

所以如果只问“网络骨架是不是差不多”，答案是：**是，差不多到几乎是一套母体复制出来的变体**。

---

## 3. 关键不同点：并不只是最后阶段判断不一样

下面这些差异都很关键。

---

### 3.1 任务语义不同

这是第一层、也是最根本的变化。

主工程：

- 标签语义：`0=stable, 1=unstable`
- 任务目标：暂态稳定诊断

边界检测：

- 标签语义：`0=non-boundary, 1=boundary`
- 任务目标：边界样本筛查

也就是说，边界版不是在判断“系统会不会失稳”，而是在判断“这个样本是不是位于判别边界附近、是不是临界工况”。

这在 README 中写得很明确，见 [tsa_boundary_detection/README.md](/Users/yqy/PycharmProjects/PaperProject/tsa_boundary_detection/README.md#L5)。

---

### 3.2 输出头不同

主工程用的是 `DecisionHead`：

```python
self.exit_heads = nn.ModuleList([DecisionHead(d_model, dropout=dropout) for _ in range(n_exit)])
```

见 [src/models/phys_hpgt.py](/Users/yqy/PycharmProjects/PaperProject/src/models/phys_hpgt.py#L353)

每个出口都输出两个东西：

- 分类 logits
- 风险值

而边界版用的是 `BoundaryHead`：

```python
self.exit_heads = nn.ModuleList([BoundaryHead(d_model, dropout=dropout) for _ in range(n_exit)])
```

见 [tsa_boundary_detection/src/models/phys_hpgt.py](/Users/yqy/PycharmProjects/PaperProject/tsa_boundary_detection/src/models/phys_hpgt.py#L340)

它每个出口只输出一个：

- boundary logit

因此在 `forward()` 最终输出里也不同：

主工程：

```python
outputs = {
    "logits_list": logits_list,
    "risk_list": risk_list,
    "emb": emb,
}
```

见 [src/models/phys_hpgt.py](/Users/yqy/PycharmProjects/PaperProject/src/models/phys_hpgt.py#L530)

边界版：

```python
outputs = {
    'logits_list': logits_list,
    'emb': emb,
}
```

见 [tsa_boundary_detection/src/models/phys_hpgt.py](/Users/yqy/PycharmProjects/PaperProject/tsa_boundary_detection/src/models/phys_hpgt.py#L508)

这不只是“最后判断标签名字改了”，而是**输出空间本身变了**。

---

### 3.3 融合阶段的 token 语义不同

这个差异比较细，但很重要。

主工程在融合阶段引入的是：

```python
self.risk_token = nn.Parameter(torch.randn(1, d_model) * 0.02)
...
risk_context = pooled_nodes + self.risk_token.expand_as(pooled_nodes)
```

见 [src/models/phys_hpgt.py](/Users/yqy/PycharmProjects/PaperProject/src/models/phys_hpgt.py#L318) 和 [src/models/phys_hpgt.py](/Users/yqy/PycharmProjects/PaperProject/src/models/phys_hpgt.py#L424)

边界版则改成：

```python
self.task_token = nn.Parameter(torch.randn(1, d_model) * 0.02)
...
task_context = pooled_nodes + self.task_token.expand_as(pooled_nodes)
```

见 [tsa_boundary_detection/src/models/phys_hpgt.py](/Users/yqy/PycharmProjects/PaperProject/tsa_boundary_detection/src/models/phys_hpgt.py#L305) 和 [tsa_boundary_detection/src/models/phys_hpgt.py](/Users/yqy/PycharmProjects/PaperProject/tsa_boundary_detection/src/models/phys_hpgt.py#L405)

这意味着：

- 主工程这里的融合语义偏“风险感知”
- 边界版这里的融合语义偏“任务感知”

所以即使主干外形很像，融合时注入的任务先验已经不同了。

---

### 3.4 损失函数体系不同

这是第二个最本质的变化。

#### 主工程：多任务损失

主工程的 `total_loss()` 由四部分组成：

1. 分类损失 `cls_loss`
2. 风险回归损失 `risk_loss`
3. 物理一致性损失 `phys_loss`
4. 对比学习损失 `con_loss`

代码见 [src/models/losses.py](/Users/yqy/PycharmProjects/PaperProject/src/models/losses.py#L189)。

而且主工程会先构造物理风险特征：

```python
phys_feat = physics_risk_features(...)
```

对应 [src/models/losses.py](/Users/yqy/PycharmProjects/PaperProject/src/models/losses.py#L102)。

#### 边界版：单任务边界损失

边界版只保留 boundary 二分类损失：

```python
criterion = build_boundary_loss(cfg, device=y.device)
losses = [criterion(logits.view(-1), y) for logits in logits_list]
total = torch.stack(losses).mean()
```

见 [tsa_boundary_detection/src/models/losses.py](/Users/yqy/PycharmProjects/PaperProject/tsa_boundary_detection/src/models/losses.py#L52)

可选损失只有：

- `BCEWithLogitsLoss`
- `BoundaryFocalLoss`

见 [tsa_boundary_detection/src/models/losses.py](/Users/yqy/PycharmProjects/PaperProject/tsa_boundary_detection/src/models/losses.py#L38)。

所以两者训练信号完全不是一个复杂度：

- 主工程：多任务、物理启发、风险约束
- 边界版：纯 boundary 判别

---

### 3.5 标签来源不同

这是边界版最关键的“数据定义差异”。

#### 主工程

主工程直接从 `.npy` 样本内部读取标签：

```python
y = int(find_first(sample, 'y', 'label'))
```

见 [src/data/dataset.py](/Users/yqy/PycharmProjects/PaperProject/src/data/dataset.py#L133)

#### 边界版

边界版不是用 `.npy` 自带标签，而是用一个外部 CSV 构建边界标签索引：

```python
label_index = BoundaryLabelIndex.from_csv(csv_path, root)
```

见 [tsa_boundary_detection/src/runtime.py](/Users/yqy/PycharmProjects/PaperProject/tsa_boundary_detection/src/runtime.py#L173)

`BoundaryLabelIndex` 会把 CSV 里的样本键映射到具体 `.npy` 文件，再决定某个样本是否是边界样本。核心逻辑见：

- 读取 CSV：[tsa_boundary_detection/src/data/label_builder.py](/Users/yqy/PycharmProjects/PaperProject/tsa_boundary_detection/src/data/label_builder.py#L97)
- 解析匹配策略：[tsa_boundary_detection/src/data/label_builder.py](/Users/yqy/PycharmProjects/PaperProject/tsa_boundary_detection/src/data/label_builder.py#L129)
- 给单个样本打边界标签：[tsa_boundary_detection/src/data/label_builder.py](/Users/yqy/PycharmProjects/PaperProject/tsa_boundary_detection/src/data/label_builder.py#L217)

这意味着边界版的数据监督来自：

- 一个“人工确认边界样本清单”
- 而不是原始系统的稳定性标签

这已经不是“同一任务换了个 head”，而是**监督目标本身变了**。

---

### 3.6 训练集切分策略不同

主工程的 train/val 切分只是普通源域内部切分：

- 逻辑见 [src/train.py](/Users/yqy/PycharmProjects/PaperProject/src/train.py#L95)

边界版则专门做了**分层切分**，尽量让 train/val 中 boundary / non-boundary 比例都保留：

```python
def split_train_val_stratified(files, labels, val_ratio, seed):
    ...
```

见 [tsa_boundary_detection/src/runtime.py](/Users/yqy/PycharmProjects/PaperProject/tsa_boundary_detection/src/runtime.py#L102)

这是因为 boundary 样本往往更稀少，普通随机切分更容易把 val 切坏。

---

### 3.7 类别不平衡处理不同

主工程使用的是：

- 失稳类权重 `class_weight_unstable`
- Focal loss

见 [src/models/losses.py](/Users/yqy/PycharmProjects/PaperProject/src/models/losses.py#L197)

边界版则显式估计 boundary 正样本权重，并可选 `WeightedRandomSampler`：

- `pos_weight` 估计见 [tsa_boundary_detection/src/runtime.py](/Users/yqy/PycharmProjects/PaperProject/tsa_boundary_detection/src/runtime.py#L156)
- sampler 构造见 [tsa_boundary_detection/src/runtime.py](/Users/yqy/PycharmProjects/PaperProject/tsa_boundary_detection/src/runtime.py#L145)
- 损失构造见 [tsa_boundary_detection/src/models/losses.py](/Users/yqy/PycharmProjects/PaperProject/tsa_boundary_detection/src/models/losses.py#L38)

也就是说，边界版在训练时更直接地围绕“正样本稀缺”这个问题做设计。

---

### 3.8 阈值策略不同

主工程默认评估阈值就是配置里的固定值，通常是 `0.5`：

- 见 [src/eval.py](/Users/yqy/PycharmProjects/PaperProject/src/eval.py#L138)

边界版则明确不会把阈值固定死，而是在 `36-val` 上做阈值搜索：

```python
selection = select_best_threshold(...)
best_threshold = float(selection.threshold)
```

见 [tsa_boundary_detection/src/train.py](/Users/yqy/PycharmProjects/PaperProject/tsa_boundary_detection/src/train.py#L161)

评估时也会保存：

- `p_boundary`
- `pred_boundary_label`
- `threshold_used`

见 [tsa_boundary_detection/src/eval.py](/Users/yqy/PycharmProjects/PaperProject/tsa_boundary_detection/src/eval.py#L53)

这说明边界版从设计上就不是“固定分类器”，而是“概率筛查器 + 验证集定阈值”。

---

### 3.9 评估输出不同

主工程评估重点是：

- `accuracy`
- `precision`
- `recall`
- `f1`
- `auc`
- confusion matrix

边界版除了这些，还强调：

- `real_boundary_count`
- `pred_boundary_count`
- `tp/fp/fn/tn`
- `count_error`
- `predicted_boundary` 样本图

评估代码见 [tsa_boundary_detection/src/eval.py](/Users/yqy/PycharmProjects/PaperProject/tsa_boundary_detection/src/eval.py#L104) 和 [tsa_boundary_detection/src/eval.py](/Users/yqy/PycharmProjects/PaperProject/tsa_boundary_detection/src/eval.py#L125)。

这说明边界版更像一个“样本筛查器 / 候选发现器”，而不只是最终分类器。

---

## 4. 一个简化对照表

| 维度 | 主工程 `src/` | 边界检测 `tsa_boundary_detection/` |
|---|---|---|
| 主任务 | 稳定 / 失稳诊断 | 边界 / 非边界检测 |
| 主干结构 | 双模态 + patch + temporal + graph + fusion | 基本相同 |
| token 语义 | `risk_token` | `task_token` |
| 输出头 | 分类 logits + 风险值 | boundary logit |
| 输出字段 | `logits_list`, `risk_list`, `emb` | `logits_list`, `emb` |
| 损失 | 分类 + 风险回归 + 物理一致性 + 对比学习 | BCE/Focal 边界损失 |
| 标签来源 | `.npy` 自带 `y/label` | 外部 `boundary CSV` |
| 切分方式 | 普通源域切分 | boundary 分层切分 |
| 阈值 | 通常固定 0.5 | 在 `36-val` 上搜索 |
| 评估侧重点 | 诊断性能 | 边界样本筛查性能 |

---

## 5. 最后回答你的原问题

如果你的问题是：

> 这两种模型阶段是不是类似，只有最后阶段判断不一样？

那么更准确的回答应该是：

**前 5 个阶段的网络骨架是非常相似的，但差别绝不只是最后一个判断头。**

真正不同的是整套任务系统：

- 学什么标签
- 输出什么结果
- 用什么损失训练
- 怎么处理类别不平衡
- 阈值怎么选
- 评估时看什么指标

也就是说：

- 从“结构图外形”看，它们很像
- 从“机器学习问题定义”看，它们已经是两种不同模型

---

## 6. 如果你接下来还想继续看什么

接下来你可以继续让我补其中一种：

1. 只写“主干网络层面”的并排对照图
2. 只写“训练与评估逻辑”的并排对照
3. 直接继续整理成 PPT 里能用的“模型对比页”

如果你要，我下一步可以继续把这份文档再升级成：

- 带流程图的版本
- 带代码片段高亮的版本
- 或者一页 PPT 讲稿版
