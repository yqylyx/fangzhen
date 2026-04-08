# Phys-HPGT Cross-System OOD

这个项目实现了一个可训练、可评估、可在线推理的 PyTorch 框架，用于电力系统暂态稳定诊断。

- 主任务：稳定 / 失稳二分类
- 辅助任务：风险回归 `risk in [0, 1]`
- 输入支持：可变时间长度 `T`、可变电压通道数 `Nv`、可变功角通道数 `Ng`
- 缺失值 / 截断：统一通过 `padding + mask` 处理
- 模型：Phys-HPGT，采用 channel-agnostic 编码，不强绑定通道编号

## 当前默认实验协议

当前默认协议已经切换为 **train on 74data, validate/test on 36data and 37data**：

- Train: 只使用 `npy_jobs/74data`
- Val: 从 `74data` 内部按固定随机种子切出一部分，用于 early stopping
- OOD test 1: 整个 `npy_jobs/36data`
- OOD test 2: 整个 `npy_jobs/37data`

这不是普通随机混合切分，而是 **cross-system generalization / OOD evaluation**。
真正难点在于跨系统语义漂移，而不是张量 shape 本身。

## 数据目录

```text
npy_jobs/
  36data/
  37data/
  74data/
```

每个样本是一个 `.npy` 文件，内容为 `dict`，兼容以下字段：

```python
{
  "t": float32[T],
  "V": float32[T, Nv],
  "delta": float32[T, Ng],
  "y": int64,
  "meta": {...}
}
```

同时兼容旧字段名：

- `t` 或 `times`
- `V` 或 `voltages`
- `delta` 或 `angles`
- `y` 或 `label`

## 归一化与预处理

- 归一化统计量只从 `74data-train` 拟合
- `74data-val` 只使用训练集统计量
- `36data` 和 `37data` 只做 `transform`，绝不参与拟合
- 电压和功角分开归一化，不混用统计量
- 功角默认预处理是 `initial_then_mean`

## 核心文件

- `src/data/dataset.py`
  读取 `.npy`、识别缺失值、生成 mask、返回 `dataset_name`，并在 `collate_fn` 中完成时间维和通道维 padding。
- `src/data/normalization.py`
  只用训练域拟合 `TransientNormalizer`，并保存为 `scaler_<train_tag>train.json`。
- `src/models/phys_hpgt.py`
  主模型实现，包含 patch 编码、Transformer/TCN 时序编码、图注意力、跨模态融合、early-exit、风险回归。
- `src/models/losses.py`
  Focal Loss、监督式对比学习、物理风险特征、物理一致性损失、总损失组合。
- `src/train.py`
  训练入口。当前默认是 `74 训练 / 74 内验证 / 36 和 37 外部验证`，并且协议已做成可配置，不再写死 36/37/74。
- `src/eval.py`
  独立评估入口，会根据当前配置自动识别训练域和外部测试域。
- `src/infer_stream.py`
  在线流式推理接口。
- `src/utils/vis.py`
  保存注意力热图和波形图。
- `export/export_onnx.py`
  导出 TorchScript / ONNX。

## 最小运行命令

训练：

```bash
python -m src.train --config configs/config.yaml
```

Windows 直接指定解释器：

```bash
C:\Users\hp-pc01\.conda\envs\npycheck\python.exe -m src.train --config configs/config.yaml
```

只评估：

```bash
python -m src.eval --config configs/config.yaml --checkpoint runs/experiments/train74_val74_ood36_37/checkpoints/best_by_74val_f1.pt
```

导出 TorchScript / ONNX：

```bash
python export/export_onnx.py --config configs/config.yaml --checkpoint runs/experiments/train74_val74_ood36_37/checkpoints/best_by_74val_f1.pt --demo
```

## 输出目录

默认输出目录：`runs/experiments/train74_val74_ood36_37`

```text
runs/experiments/train74_val74_ood36_37/
  checkpoints/
    best_by_74val_f1.pt
    last.pt
  metrics/
    history.json
    metrics_74val.json
    metrics_36test.json
    metrics_37test.json
  predictions/
    predictions_74val.csv
    predictions_36test.csv
    predictions_37test.csv
  plots/
    confusion_74val.png
    confusion_36test.png
    confusion_37test.png
  normalization/
    scaler_74train.json
    scaler_summary.json
  splits/
    cross_system_protocol.json
```

日志会明确打印：

- `In-domain validation on 74data`
- `OOD test on 36data`
- `OOD test on 37data`

## 当前配置文件

默认配置在 `configs/config.yaml`，关键字段是：

```yaml
data:
  train_dir: 74data
  test_dirs:
    - 36data
    - 37data
```

如果你之后想再切回别的协议，只需要改这两个字段，不需要再改训练代码。


更多实验目录说明见 uns/experiments/EXPERIMENTS.md。
