"""Train RGB -> G/255 surface height with held-out validation and a mean-map baseline."""
from __future__ import annotations

import argparse
import json
import math
import time
from pathlib import Path

import torch

from .data import WaveDataset
from .metrics import RegressionMetrics, regression_loss
from .model import WaveHeightNet
from .runtime import (atomic_torch_save, choose_device, evaluate_loader, make_loader,
                      manifest_hash, mean_height, seed_everything, write_json)

DEFAULTS = dict(manifest="artifacts/manifest.json", data_root=None, output="runs/wave_height",
                height=320, width=384, batch_size=8, epochs=60, lr=2e-4,
                weight_decay=1e-4, workers=4, device="auto", amp=True, pretrained=True,
                seed=42, patience=12, gradient_weight=.5, horizontal_flip=False,
                limit_train=None, limit_val=None, max_steps=None, resume=None)


def parse_args() -> argparse.Namespace:
    probe = argparse.ArgumentParser(add_help=False)
    probe.add_argument("--config", type=Path)
    config_args, _ = probe.parse_known_args()
    settings = dict(DEFAULTS)
    if config_args.config:
        overrides = json.loads(config_args.config.read_text(encoding="utf-8"))
        unknown = set(overrides) - set(settings)
        if unknown:
            raise ValueError(f"Unknown config keys: {unknown}")
        settings.update(overrides)
    parser = argparse.ArgumentParser(description=__doc__, parents=[probe])
    for key in ["manifest", "data_root", "output", "device", "resume"]:
        parser.add_argument("--" + key.replace("_", "-"), default=settings[key])
    for key in ["height", "width", "batch_size", "epochs", "workers", "seed", "patience",
                "limit_train", "limit_val", "max_steps"]:
        parser.add_argument("--" + key.replace("_", "-"), type=int, default=settings[key])
    for key in ["lr", "weight_decay", "gradient_weight"]:
        parser.add_argument("--" + key.replace("_", "-"), type=float, default=settings[key])
    for key in ["amp", "pretrained", "horizontal_flip"]:
        parser.add_argument("--" + key.replace("_", "-"), action=argparse.BooleanOptionalAction,
                            default=settings[key])
    return parser.parse_args()


def train(args: argparse.Namespace) -> dict:
    if min(args.height, args.width) < 32:
        raise ValueError("Training dimensions must be at least 32")
    if args.batch_size < 1 or args.epochs < 1 or args.workers < 0 or args.patience < 0:
        raise ValueError("Invalid training settings")
    if args.max_steps is not None and args.max_steps < 1:
        raise ValueError("--max-steps must be positive")
    if args.lr <= 0 or args.gradient_weight < 0 or args.weight_decay < 0:
        raise ValueError("Invalid optimizer/loss settings")
    seed_everything(args.seed)
    # Avoid excessive CPU threads for small image batches, particularly on laptops.
    torch.set_num_threads(min(torch.get_num_threads(), 8))
    device = choose_device(args.device)
    output = Path(args.output)
    if not args.resume and ((output / "last.pt").exists() or (output / "history.jsonl").exists()):
        raise FileExistsError(f"Run already exists: {output}. Use --resume or a new --output.")
    output.mkdir(parents=True, exist_ok=True)
    size = (args.height, args.width)
    train_data = WaveDataset(args.manifest, "train", size, args.data_root, augment=True,
                             horizontal_flip=args.horizontal_flip, limit=args.limit_train)
    val_data = WaveDataset(args.manifest, "val", size, args.data_root, limit=args.limit_val)
    signature = manifest_hash(args.manifest)
    saved = torch.load(args.resume, map_location="cpu", weights_only=True) if args.resume else None
    if saved:
        if saved["manifest_sha256"] != signature:
            raise ValueError("Resume manifest differs from the original run.")
        for key in ["height", "width", "epochs", "lr", "weight_decay", "gradient_weight",
                    "limit_train", "limit_val", "max_steps", "horizontal_flip", "batch_size"]:
            if saved["config"][key] != getattr(args, key):
                raise ValueError(f"Resume requires the original {key} setting.")
    model = WaveHeightNet(pretrained=args.pretrained and saved is None).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
    use_amp = args.amp and device.type == "cuda"
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)
    baseline = saved["mean_height"] if saved else mean_height(train_data)
    np_path = output / "train_mean_height.npy"
    import numpy as np
    np.save(np_path, baseline.numpy()[0, 0])
    start_epoch, best, bad_epochs = 0, math.inf, 0
    if saved:
        model.load_state_dict(saved["model"])
        optimizer.load_state_dict(saved["optimizer"])
        scheduler.load_state_dict(saved["scheduler"])
        scaler.load_state_dict(saved["scaler"])
        start_epoch, best, bad_epochs = saved["epoch"] + 1, saved["best_val_mae"], saved["bad_epochs"]
        if start_epoch >= args.epochs:
            raise ValueError("Checkpoint has already completed the requested epochs.")
        # Allow a resumed run in a new output directory to retain its best-so-far model.
        if not (output / "best.pt").exists():
            previous_best = Path(args.resume).parent / "best.pt"
            if previous_best.exists():
                best_saved = torch.load(previous_best, map_location="cpu", weights_only=True)
                if best_saved["manifest_sha256"] != signature:
                    raise ValueError("The previous best checkpoint belongs to a different dataset.")
                atomic_torch_save(best_saved, output / "best.pt")
            else:
                raise FileNotFoundError("Resume needs the best.pt from the original run directory.")
    config = {k: v for k, v in vars(args).items() if k != "config"}
    write_json(output / "config.json", config)
    write_json(output / "target_encoding.json", train_data.manifest["target"])
    train_loader = make_loader(train_data, args.batch_size, args.workers, True, device, args.seed)
    val_loader = make_loader(val_data, args.batch_size, args.workers, False, device, args.seed)
    print(f"Device={device}, AMP={use_amp}, train={len(train_data)}, val={len(val_data)}, size={size}",
          flush=True)
    if args.limit_train or args.limit_val or args.max_steps:
        print("LIMITED RUN: pipeline check only; not a full-dataset training result.", flush=True)
    summary = None
    for epoch in range(start_epoch, args.epochs):
        started = time.monotonic()
        # Per-epoch seeds make epoch-boundary resume reproducible with a fixed worker count.
        seed_everything(args.seed + epoch)
        train_loader.generator.manual_seed(args.seed + epoch)
        model.train()
        train_metrics, loss_sum, batches = RegressionMetrics(), 0., 0
        for step, batch in enumerate(train_loader):
            if args.max_steps is not None and step >= args.max_steps:
                break
            rgb = batch["rgb"].to(device, non_blocking=True)
            target = batch["target"].to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=use_amp):
                pred = model(rgb)
                loss = regression_loss(pred.float(), target, args.gradient_weight)
            if not torch.isfinite(loss):
                raise FloatingPointError(f"Non-finite loss at epoch {epoch+1}, step {step}")
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0, error_if_nonfinite=not use_amp)
            scaler.step(optimizer)
            scaler.update()
            train_metrics.update(pred, target)
            loss_sum += float(loss.item())
            batches += 1
            if (step + 1) % 50 == 0:
                print(f"Epoch {epoch+1}/{args.epochs}, batch {step+1}/{len(train_loader)}, "
                      f"loss={loss_sum/batches:.5f}", flush=True)
        validation = evaluate_loader(model, val_loader, device, baseline)
        scheduler.step()
        score = validation["model"]["mae"]
        improved = score < best
        best = min(best, score)
        bad_epochs = 0 if improved else bad_epochs + 1
        summary = {"epoch": epoch + 1, "loss": loss_sum / batches,
                   "train": train_metrics.compute(), "val": validation,
                   "lr_next": optimizer.param_groups[0]["lr"],
                   "seconds": time.monotonic() - started, "best_val_mae": best}
        checkpoint = {"architecture": "resnet18_unet_height_v1", "model": model.state_dict(),
                      "optimizer": optimizer.state_dict(), "scheduler": scheduler.state_dict(),
                      "scaler": scaler.state_dict(), "epoch": epoch, "best_val_mae": best,
                      "bad_epochs": bad_epochs, "mean_height": baseline,
                      "manifest_sha256": signature, "target": train_data.manifest["target"],
                      "config": config, "limited_run": bool(args.limit_train or args.limit_val
                                                              or args.max_steps)}
        atomic_torch_save(checkpoint, output / "last.pt")
        if improved:
            atomic_torch_save(checkpoint, output / "best.pt")
        with (output / "history.jsonl").open("a", encoding="utf-8") as stream:
            stream.write(json.dumps(summary, allow_nan=False) + "\n")
        write_json(output / "summary.json", summary)
        print(json.dumps(summary, allow_nan=False), flush=True)
        if args.patience and bad_epochs >= args.patience:
            print(f"Early stopping after {args.patience} epochs without improvement.", flush=True)
            break
    return summary


if __name__ == "__main__":
    train(parse_args())

