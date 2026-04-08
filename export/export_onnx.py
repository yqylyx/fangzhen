from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np
import torch
import yaml

from src.models.phys_hpgt import PhysHPGT


class ExportWrapper(torch.nn.Module):
    def __init__(self, model: torch.nn.Module) -> None:
        super().__init__()
        self.model = model

    def forward(
        self,
        V: torch.Tensor,
        delta: torch.Tensor,
        mask_V: torch.Tensor,
        mask_delta: torch.Tensor,
        ch_mask_V: torch.Tensor,
        ch_mask_delta: torch.Tensor,
        time_mask: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        outputs = self.model(
            V,
            delta,
            mask_V,
            mask_delta,
            ch_mask_V,
            ch_mask_delta,
            time_mask,
            return_attn=False,
        )
        return outputs["logits_list"][-1], outputs["risk_list"][-1]


def load_config(path: str | Path) -> dict[str, Any]:
    with Path(path).open("r", encoding="utf-8") as fh:
        return yaml.safe_load(fh) or {}


def load_model(cfg: dict[str, Any], checkpoint: str | None, device: torch.device) -> torch.nn.Module:
    export_cfg = dict(cfg["model"])
    export_cfg["use_freq_branch"] = False
    model = PhysHPGT(**export_cfg).to(device)
    if checkpoint:
        ckpt = torch.load(checkpoint, map_location=device)
        state = ckpt["model"]
        filtered = {k: v for k, v in state.items() if not k.startswith("freq_proj")}
        missing, unexpected = model.load_state_dict(filtered, strict=False)
        if missing:
            print(f"export note: missing keys ignored -> {missing}")
        if unexpected:
            print(f"export note: unexpected keys ignored -> {unexpected}")
    model.eval()
    return model


def make_dummy_inputs(cfg: dict[str, Any], device: torch.device) -> tuple[torch.Tensor, ...]:
    patch_len = int(cfg["model"].get("patch_len", 16))
    T = patch_len * 4
    Nv = 12
    Ng = 6
    V = torch.randn(1, T, Nv, device=device)
    delta = torch.randn(1, T, Ng, device=device)
    mask_V = torch.ones(1, T, Nv, dtype=torch.bool, device=device)
    mask_delta = torch.ones(1, T, Ng, dtype=torch.bool, device=device)
    ch_mask_V = torch.ones(1, Nv, dtype=torch.bool, device=device)
    ch_mask_delta = torch.ones(1, Ng, dtype=torch.bool, device=device)
    time_mask = torch.ones(1, T, dtype=torch.bool, device=device)
    return V, delta, mask_V, mask_delta, ch_mask_V, ch_mask_delta, time_mask


def export_torchscript(wrapper: torch.nn.Module, example_inputs: tuple[torch.Tensor, ...], save_path: Path) -> None:
    save_path.parent.mkdir(parents=True, exist_ok=True)
    traced = torch.jit.trace(wrapper, example_inputs, strict=False, check_trace=False)
    traced.save(str(save_path))


def export_onnx(wrapper: torch.nn.Module, example_inputs: tuple[torch.Tensor, ...], save_path: Path, opset: int) -> None:
    save_path.parent.mkdir(parents=True, exist_ok=True)
    input_names = ["V", "delta", "mask_V", "mask_delta", "ch_mask_V", "ch_mask_delta", "time_mask"]
    output_names = ["logits", "risk"]
    dynamic_axes = {
        "V": {1: "T", 2: "Nv"},
        "delta": {1: "T", 2: "Ng"},
        "mask_V": {1: "T", 2: "Nv"},
        "mask_delta": {1: "T", 2: "Ng"},
        "ch_mask_V": {1: "Nv"},
        "ch_mask_delta": {1: "Ng"},
        "time_mask": {1: "T"},
        "logits": {0: "B"},
        "risk": {0: "B"},
    }
    torch.onnx.export(
        wrapper,
        example_inputs,
        str(save_path),
        input_names=input_names,
        output_names=output_names,
        dynamic_axes=dynamic_axes,
        opset_version=opset,
    )


def run_onnx_demo(onnx_path: Path, example_inputs: tuple[torch.Tensor, ...]) -> None:
    try:
        import onnxruntime as ort
    except Exception as exc:
        print(f"onnxruntime not available, skip demo: {exc}")
        return

    providers = ["CPUExecutionProvider"]
    session = ort.InferenceSession(str(onnx_path), providers=providers)
    all_inputs = {
        name: tensor.detach().cpu().numpy()
        for name, tensor in zip(["V", "delta", "mask_V", "mask_delta", "ch_mask_V", "ch_mask_delta", "time_mask"], example_inputs)
    }
    required_names = [item.name for item in session.get_inputs()]
    feed = {name: all_inputs[name] for name in required_names}

    for _ in range(3):
        _ = session.run(None, feed)
    start = time.perf_counter()
    runs = 20
    for _ in range(runs):
        logits, risk = session.run(None, feed)
    latency = (time.perf_counter() - start) * 1000.0 / runs
    print(f"onnxruntime logits shape={logits.shape}, risk shape={risk.shape}, avg_latency_ms={latency:.3f}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Export Phys-HPGT to TorchScript and ONNX")
    parser.add_argument("--config", type=str, default="configs/config.yaml")
    parser.add_argument("--checkpoint", type=str, default=None)
    parser.add_argument("--onnx-path", type=str, default=None)
    parser.add_argument("--torchscript-path", type=str, default=None)
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--demo", action="store_true")
    args = parser.parse_args()

    cfg = load_config(args.config)
    device = torch.device(args.device if args.device.startswith("cuda") and torch.cuda.is_available() else "cpu")
    model = load_model(cfg, args.checkpoint, device)
    wrapper = ExportWrapper(model).to(device)
    example_inputs = make_dummy_inputs(cfg, device)

    onnx_path = Path(args.onnx_path or cfg["export"].get("onnx_path", "export/phys_hpgt.onnx"))
    torchscript_path = Path(args.torchscript_path or cfg["export"].get("torchscript_path", "export/phys_hpgt.ts"))
    opset = int(cfg["export"].get("opset", 17))

    export_torchscript(wrapper, example_inputs, torchscript_path)
    export_onnx(wrapper, example_inputs, onnx_path, opset)
    print(f"saved torchscript -> {torchscript_path}")
    print(f"saved onnx -> {onnx_path}")

    if args.demo:
        run_onnx_demo(onnx_path, example_inputs)


if __name__ == "__main__":
    main()





