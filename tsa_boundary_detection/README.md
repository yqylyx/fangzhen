# Phys-HPGT Boundary Detection Variant

这个目录是在现有时序分类项目框架上改造出来的“边界样本检测”版本，不是重新从零写的新项目。

任务定义已经从原来的 `stable / unstable` 改成：

- `boundary = 1`
- `non_boundary = 0`

它保留了原项目的核心设计思想：

- 支持可变时间长度 `T`
- 支持可变电压通道数 `Nv`
- 支持可变功角通道数 `Ng`
- batch 内继续使用 `padding + mask`
- 电压分支和功角分支仍然分开编码
- 模型不写死输入维度，仍然依赖 mask-aware pooling / aggregation
- 双分支特征融合、通道弱绑定的思路继续保留

## 实验协议

边界检测协议固定为跨系统设置：

- Train: 只使用 `36data`
- Val: 从 `36data` 内部按分层方式切分
- Test-OOD-1: `37data`
- Test-OOD-2: `74data`

也就是说：

- `37data` 和 `74data` 不参与任何训练
- 阈值选择也只能基于 `36-val`
- 最终阈值固定后，再用于 `37-test` 和 `74-test`

## 边界标签来源

边界标签来自 `npy_jobs` 目录下的人工确认 CSV：

- 默认配置：`../npy_jobs/boundary_suspicious_samples_index.csv`

当前实现使用如下原则生成 `boundary_label`：

1. 样本出现在边界 CSV 中，则 `boundary_label = 1`
2. 样本不在边界 CSV 中，则 `boundary_label = 0`

重要前提假设：

- 当前 CSV 被视为“完整的已确认边界样本清单”
- 因此，不在 CSV 中的样本会被视为 `non-boundary`

### 匹配逻辑

为避免重名误匹配，标签索引会优先使用更精确的信息：

1. 如果 CSV 带有 `file` 或 `source_abs_path`，优先使用样本相对路径精确匹配
2. 否则使用 `dataset_name + file_name` 联合匹配
3. 如果 CSV 不包含数据集字段，才退化到仅 `file_name` 匹配

注意：仓库里的 `.npy` 文件名在不同数据集之间、甚至同一数据集的不同子目录之间都可能重名。
因此当 CSV 只提供不充分的键时，代码会直接报错要求使用更强的联合匹配，而不会默默错配。

## 输出内容

训练和评估会分别对以下集合输出结果：

- `36-val`
- `37-test`
- `74-test`

每个集合都会单独保存：

- 指标 JSON
- 样本级预测 CSV
- 仅保留预测为 boundary 的 CSV
- 数量统计 CSV
- confusion matrix PNG

主要指标包括：

- Precision
- Recall
- F1
- PR-AUC
- ROC-AUC
- Confusion Matrix

同时会额外保存每个集合的数量统计：

- `real_boundary_count`
- `pred_boundary_count`
- `tp`
- `fp`
- `fn`
- `tn`
- `count_error = pred_boundary_count - real_boundary_count`

## 阈值选择

边界检测不会把分类阈值固定成 `0.5`。

当前默认策略：

- 在 `36-val` 上搜索最优阈值
- 默认目标是 `maximize F1`
- 最终保存 `best_threshold`
- 再把这个阈值固定用于 `37-test` 和 `74-test`

相关输出文件位于：

- `metrics/threshold_selection_best.json`
- `metrics/threshold_selection_final.json`

## 类别不平衡处理

当前默认损失为：

- `BCEWithLogitsLoss(pos_weight=...)`

其中 `pos_weight` 默认根据训练集边界样本比例自动估计。

另外代码也支持：

- 可选 `WeightedRandomSampler`
- 可选 `focal loss`

这些都可以在 `config.yaml` 中切换。

## 可视化输出

如果在配置中启用 `visualization.enabled: true`，评估时会导出边界样本图像，默认保存：

- `predicted_boundary`
- `tp`
- `fp`
- `fn`

图里会显示：

- 文件名
- 数据集名
- true boundary label
- predicted boundary label
- `p_boundary`

## 运行方式

建议先进入本目录再运行：

```bash
cd tsa_boundary_detection
python -m src.train --config config.yaml
```

只评估已有 checkpoint：

```bash
cd tsa_boundary_detection
python -m src.eval --config config.yaml --checkpoint runs/boundary_train36_val36_test37_74/checkpoints/best_by_36val_f1.pt
```

## 目录说明

核心文件：

- `src/data/dataset.py`
  复用了原始 `.npy` 读取、padding、mask、可变维度处理逻辑，并把标签替换成 `boundary_label`
- `src/data/label_builder.py`
  从边界 CSV 构建标签索引，并负责冲突检测与日志统计
- `src/models/phys_hpgt.py`
  延续原始双分支主干，只把任务头改成 boundary 单日志输出
- `src/models/losses.py`
  改成边界二分类损失
- `src/train.py`
  负责 36 训练 / 36-val 阈值选择 / 37 和 74 OOD 测试
- `src/eval.py`
  负责独立评估、指标导出、样本级预测导出和数量统计导出
- `config.yaml`
  默认实验配置

## 与原项目的关系

这是一个“基于现有框架改造出来的边界检测版本”，不是一个全新的无关工程。
原先可变维度输入、mask、双分支建模的设计全部保留，只把任务目标和评估导出改成了 boundary detection。
