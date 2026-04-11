# Boundary Scan Tool

这个增强后的 `boundary_scan_tool` 用于边界样本补全和主动发现。

当前目标不是立即训练最终的边界分类模型，而是先尽可能把边界样本找全，尤其是那些容易漏掉的样本：

- 稳定侧边界样本
- 失稳侧边界样本
- 中间临界边界样本
- 只有 1~2 条功角通道振荡特别强，但整体仍然看起来比较稳定的样本

## 当前工具定位

这个工具现在服务的是：

- 边界样本补全
- 边界样本主动发现
- 高召回候选挖掘

它不会把 CSV 外的所有样本直接当成最终负类。

## 输入数据

默认扫描：

- `npy_jobs/36data`
- `npy_jobs/37data`
- `npy_jobs/74data`

每个 `.npy` 样本通常包含这些字段：

- `angles`
- `voltages`
- `times`
- `label`

## seed 边界集

工具默认读取：

- `npy_jobs/boundary_suspicious_samples_index.csv`

并把它视为已确认的 seed 边界样本集。

匹配优先级是：

1. `file`
2. `source_abs_path`
3. `sample_name + dataset_name`

被匹配到的样本会标记为 `is_seed_boundary = 1`。剩余样本则作为“未确认样本”，继续作为主动发现的搜索空间。

## 为什么现在要分三类候选

单一 `boundary_score` 很容易向某一类样本偏移。在实际中这会造成：

- 稳定侧边界被失稳侧样本淹没
- 局部强振荡样本被更明显的全局失效掩盖

所以工具现在会输出至少这三类候选：

- `stable_side_boundary_candidate`
- `unstable_side_boundary_candidate`
- `central_ambiguous_candidate`

同时也会保留：

- `obvious_stable`
- `obvious_unstable`

## 特征增强

工具在保留原有电压和功角特征的基础上，新增了两组重点特征。

### 1. 局部强振荡特征

例如：

- `tail_amp_top1`
- `tail_amp_top2`
- `tail_amp_top1_ratio`
- `tail_amp_top2_ratio`
- `large_amp_channel_count_20`
- `large_amp_channel_count_30`
- `amp_std_across_channels`
- `amp_gini_like`
- `top1_minus_median_amp`
- `decay_ratio_top1`
- `decay_ratio_mean`

这一组主要用来找出“只有少数功角通道特别危险”的样本。

### 2. 稳定侧临界特征

例如：

- `voltage_rebound_instability_score`
- `tail_low_voltage_reentry_count`
- `spread_reentry_count`
- `tail_sign_change_density`
- `oscillation_persistence_score`

这一组主要用来找出“最后虽然还稳定，但过程非常临界”的样本。

## 评分输出

现在工具不再只输出一个笼统的分数，而是同时给出：

- `stable_side_score`
- `unstable_side_score`
- `central_ambiguous_score`
- `seed_similarity_score`
- `overall_candidate_score`

其中 `seed_similarity_score` 是基于标准化后的特征向量，与 seed 边界样本集进行最近邻相似度计算，并加入同数据集偏置。

## 可视化

图像布局现在不再把文字框叠在曲线上。

现在固定为：

- 左上：全部电压曲线
- 左中：全部相对功角曲线
- 左下：辅助统计曲线，例如 `spread_t` 和 `min_voltage_t`
- 右侧：独立信息面板

这样人工看图会更直接。

## 主要输出

默认输出到：

- `results/boundary_scan/`

关键文件包括：

- `all_samples_boundary_scores.csv`
- `suspicious_samples_topk.csv`
- `new_boundary_candidates_topk.csv`
- `per_dataset_candidate_summary.csv`
- `stable_side_boundary_candidates_topk.csv`
- `unstable_side_boundary_candidates_topk.csv`
- `central_ambiguous_boundary_candidates_topk.csv`
- `plots/36/*.png`
- `plots/37/*.png`
- `plots/74/*.png`
- `report.html`
- `report.md`
- `errors.csv`

## 运行

```bash
python boundary_scan_tool/boundary_scan.py --input_dir npy_jobs --output_dir results/boundary_scan
```

快速 smoke test：

```bash
python boundary_scan_tool/boundary_scan.py --input_dir npy_jobs --output_dir results/boundary_scan --max_files 50
```

## 推荐工作流

1. 先运行这个工具，发现新候选。
2. 先打开 `report.html` 和 `new_boundary_candidates_topk.csv` 对照复查。
3. 结合图和自动原因摘要，人工确认哪些样本应当并入边界集。
4. 把新确认的样本合并进 seed CSV。
5. 再跑下一轮补全。
6. 等边界集基本补全后，再训练最终边界分类模型。
