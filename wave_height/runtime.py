from __future__ import annotations

import hashlib
import json
import random
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from .data import WaveDataset, height_array
from .metrics import RegressionMetrics
from .model import WaveHeightNet


def choose_device(name: str) -> torch.device:
    if name == "auto":
        name = "cuda" if torch.cuda.is_available() else (
            "mps" if torch.backends.mps.is_available() else "cpu")
    if name == "cuda" and not torch.cuda.is_available():
        raise ValueError("CUDA is unavailable. Install a CUDA PyTorch build or pass --device cpu/auto.")
    if name == "mps" and not torch.backends.mps.is_available():
        raise ValueError("MPS is unavailable. Pass --device cpu.")
    return torch.device(name)


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def seed_worker(worker_id: int) -> None:
    seed = torch.initial_seed() % 2**32
    np.random.seed(seed)
    random.seed(seed)


def make_loader(dataset: WaveDataset, batch_size: int, workers: int,
                shuffle: bool, device: torch.device, seed: int = 42) -> DataLoader:
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle,
                      num_workers=workers, pin_memory=device.type == "cuda",
                      persistent_workers=False, worker_init_fn=seed_worker,
                      generator=torch.Generator().manual_seed(seed))


def manifest_hash(path: str | Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def mean_height(dataset: WaveDataset) -> torch.Tensor:
    total = np.zeros(dataset.size, dtype=np.float64)
    for pair in dataset.pairs:
        total += height_array(dataset.root / pair["target"], dataset.size)
    return torch.from_numpy((total / len(dataset)).astype(np.float32))[None, None]


@torch.inference_mode()
def evaluate_loader(model: WaveHeightNet, loader: DataLoader, device: torch.device,
                    baseline: torch.Tensor | None = None) -> dict:
    model.eval()
    metrics = RegressionMetrics()
    baseline_metrics = RegressionMetrics() if baseline is not None else None
    if baseline is not None:
        baseline = baseline.to(device)
    for batch in loader:
        rgb, target = batch["rgb"].to(device), batch["target"].to(device)
        pred = model(rgb)
        metrics.update(pred, target)
        if baseline_metrics is not None:
            baseline_metrics.update(baseline.expand_as(target), target)
    result = {"model": metrics.compute()}
    if baseline_metrics is not None:
        result["train_mean_map_baseline"] = baseline_metrics.compute()
        base_mae = result["train_mean_map_baseline"]["mae"]
        result["mae_improvement_over_mean_percent"] = (
            100 * (1 - result["model"]["mae"] / base_mae) if base_mae > 0 else None)
    return result


def atomic_torch_save(value: dict, path: Path) -> None:
    temporary = path.with_suffix(".tmp")
    torch.save(value, temporary)
    temporary.replace(path)


def load_model(path: str | Path, device: torch.device) -> tuple[WaveHeightNet, dict]:
    checkpoint = torch.load(path, map_location="cpu", weights_only=True)
    if checkpoint.get("architecture") != "resnet18_unet_height_v1":
        raise ValueError("Unsupported checkpoint architecture")
    model = WaveHeightNet(pretrained=False)
    model.load_state_dict(checkpoint["model"])
    model.to(device).eval()
    return model, checkpoint


def write_json(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
                    encoding="utf-8")

