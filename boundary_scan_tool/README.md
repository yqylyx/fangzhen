# Boundary Scan Tool

这是一个独立的边界样本筛选与可视化工具，与主模型训练代码解耦，可以单独运行。

它的目的不是训练边界分类器，也不是让你去 CSV 里逐条手工改标签，而是：

- 自动筛查可疑边界样本
- 自动识别稳定侧边界样本
- 自动识别高电压但接近快失稳的稳定侧边界样本
- 自动识别失稳侧边界样本
- 自动生成电压/功角曲线图
- 自动生成可视化报告
- 方便人工查看临界工况、弱阻尼工况、过渡工况

## 独立性说明

本工具的所有核心文件都位于 `boundary_scan_tool/` 目录中，不依赖现有项目里的自定义训练模块。
你可以把整个目录单独放进项目中使用。

## 依赖

- Python 3
- numpy
- pandas
- matplotlib
- pyyaml

示例安装：

```bash
pip install numpy pandas matplotlib pyyaml
```

## 目录结构

```text
boundary_scan_tool/
├── boundary_scan.py
├── feature_extract.py
├── visualization.py
├── report_builder.py
├── config.yaml
└── README.md
```

## 运行方式

默认以项目根目录下的 `npy_jobs` 作为输入根目录，扫描其中的 `36data`、`37data`、`74data`。

如果你不想写命令行参数，直接运行 `boundary_scan_tool/boundary_scan.py` 也可以，脚本会默认读取项目根目录下的 `npy_jobs`，并把结果输出到 `results/boundary_scan/`。

```bash
python boundary_scan_tool/boundary_scan.py --input_dir npy_jobs --output_dir results/boundary_scan
```

也可以指定配置文件：

```bash
python boundary_scan_tool/boundary_scan.py \
  --input_dir npy_jobs \
  --output_dir results/boundary_scan \
  --config boundary_scan_tool/config.yaml
```

如果你只想快速验证流程，可以限制扫描数量：

```bash
python boundary_scan_tool/boundary_scan.py --input_dir npy_jobs --output_dir results/boundary_scan --max_files 50
```

## 输出内容

每次重新运行时，脚本都会先清空当前输出目录中的旧结果，再写入本次新的 CSV、图和报告。

运行后会在输出目录下生成：

- `all_samples_boundary_scores.csv`
  所有样本的特征、边界分数和侧边界标记总表
- `suspicious_samples_topk.csv`
  Top-K 最可疑样本索引表，会自动排除明显失稳样本
- `plots/<dataset_name>/...png`
  Top-K 可疑样本的电压/功角曲线图
- `report.html`
  可直接打开浏览的可疑样本报告
- `errors.csv`
  扫描或绘图失败的样本记录

## 当前工作流

1. 运行脚本扫描所有样本并提取特征。
2. 查看 `report.html`，快速浏览 Top-K 可疑样本。
3. 重点查看稳定侧边界和失稳侧边界分组。
4. 结合图和关键指标，人工识别值得进一步标注或建模的临界工况。

## 新增的侧边界识别逻辑

### 稳定侧边界

稳定标签样本中，如果表现出以下特征，就会被提升为稳定侧边界候选：

- 功角尾段仍持续振荡，但没有明显滑极
- 电压尾段频繁接近低压危险区
- 即使电压整体仍较高，只要功角展宽和角速度已经逼近快失稳区，也会被单独识别
- 功角展宽、角速度、电压恢复特征共同落入中间风险带

### 失稳侧边界

失稳标签样本中，如果表现出以下特征，就会被提升为失稳侧边界候选：

- 已经失稳，但并非完全崩溃
- 电压没有彻底塌陷，而是处在反复震荡或局部失稳状态
- 功角和电压指标更像“部分失稳、临界失稳、反复振荡”而不是极端崩溃

## 建议人工查看时重点关注

- 功角后段是否长时间振荡但未滑极
- 电压后段是否反复接近 `0.85 ~ 0.9 p.u.`
- 高电压样本里是否已经出现功角快速展宽、角速度升高但尚未掉压
- 样本是否既不像明显稳定，也不像明显失稳
- 稳定标签样本里是否存在潜在风险工况
- 失稳标签样本里是否存在未完全崩溃的边界工况
- 是否表现为弱阻尼、慢衰减、临界工况

## 边界分数说明

工具默认使用纯特征规则进行打分，不依赖深度学习框架。

当前分数由三部分组成：

1. 原始边界分数：基于电压和功角特征的高斯接近度计算。
2. 明显稳定/明显失稳惩罚：对两端样本降分，其中明显失稳样本会直接从 Top-K 和报告候选中排除。
3. 侧边界加权：
   - 对稳定侧边界，增加“持续振荡但未滑极”的加权
   - 对高电压快失稳样本，增加“高电压但功角动态已经危险”的加权
   - 对失稳侧边界，增加“反复震荡但未完全崩溃”的加权

关键参数都在 `config.yaml` 中可调，包括：

- `tail_window_sec`
- `final_window_sec`
- `invalid_value`
- `relative_angle_mode`
- 原始边界分数的高斯项参数
- 稳定侧边界阈值与 bonus 权重
- 高电压快失稳阈值与 bonus 权重
- 失稳侧边界阈值与 bonus 权重

## 备注

- 当前默认对功角使用“减去每个时刻中位数”的相对化方式。
- 遇到坏文件时不会让全流程崩掉，失败记录会写入 `errors.csv`。
- 图中会自动显示关键指标、边界侧别以及 side signal，便于你直接看图判断是否像稳定侧或失稳侧边界工况。
