"""Evaluate stored-height predictions without per-image rescaling or alignment."""
from __future__ import annotations

import argparse
from pathlib import Path

import torch

from .data import WaveDataset
from .preview import evaluation_preview
from .runtime import (choose_device, evaluate_loader, load_model, make_loader,
                      manifest_hash, write_json)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True, type=Path)
    parser.add_argument("--manifest", default="artifacts/manifest.json", type=Path)
    parser.add_argument("--data-root", type=Path)
    parser.add_argument("--split", choices=["val", "test"], default="test")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--limit", type=int)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--preview-count", type=int, default=6)
    args = parser.parse_args()
    torch.set_num_threads(min(torch.get_num_threads(), 8))
    device = choose_device(args.device)
    model, checkpoint = load_model(args.checkpoint, device)
    if manifest_hash(args.manifest) != checkpoint["manifest_sha256"]:
        raise ValueError("Manifest differs from the training manifest. Use its original split file.")
    size = (checkpoint["config"]["height"], checkpoint["config"]["width"])
    dataset = WaveDataset(args.manifest, args.split, size, args.data_root, limit=args.limit)
    loader = make_loader(dataset, args.batch_size, args.workers, False, device)
    result = evaluate_loader(model, loader, device, checkpoint.get("mean_height"))
    result.update(split=args.split, checkpoint=str(args.checkpoint), evaluation_size=list(size),
                  limited_training=checkpoint["limited_run"], limited_evaluation=args.limit is not None)
    destination = args.output or args.checkpoint.parent / f"{args.split}_metrics.json"
    write_json(destination, result)
    if args.preview_count > 0 and checkpoint.get("mean_height") is not None:
        evaluation_preview(model, dataset, device, checkpoint["mean_height"],
                           destination.with_suffix(".jpg"), args.preview_count)
    import json
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()

