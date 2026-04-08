# Experiment Index

当前仓库里已经整理成两套可对比的 cross-system 实验目录。

## 1. Train on 36, validate on 36, OOD on 37 and 74

目录：`runs/experiments/train36_val36_ood37_74`

这套结果来自之前已经跑完的实验：

- Train: `36data`
- Val: `36data` 内部分割
- OOD test: `37data`
- OOD test: `74data`

关键文件：

- `checkpoints/best_by_36val_f1.pt`
- `metrics/metrics_36val.json`
- `metrics/metrics_37test.json`
- `metrics/metrics_74test.json`

## 2. Train on 74, validate on 74, OOD on 36 and 37

目录：`runs/experiments/train74_val74_ood36_37`

这套目录已经准备好，当前默认配置会把新训练结果输出到这里：

- Train: `74data`
- Val: `74data` 内部分割
- OOD test: `36data`
- OOD test: `37data`

训练完成后，关键文件会是：

- `checkpoints/best_by_74val_f1.pt`
- `metrics/metrics_74val.json`
- `metrics/metrics_36test.json`
- `metrics/metrics_37test.json`

## 当前默认训练命令

```bash
C:\Users\hp-pc01\.conda\envs\npycheck\python.exe -m src.train --config configs/config.yaml
```

## 当前默认评估命令

```bash
C:\Users\hp-pc01\.conda\envs\npycheck\python.exe -m src.eval --config configs/config.yaml --checkpoint runs/experiments/train74_val74_ood36_37/checkpoints/best_by_74val_f1.pt
```
