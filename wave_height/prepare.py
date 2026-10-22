"""Audit paired PNGs and build a chronological split with purge gaps."""
from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw


def discover_root(path: Path) -> Path:
    path = path.resolve()
    candidates = [p for p in [path, *sorted(path.iterdir())]
                  if p.is_dir() and (p / "Original").is_dir() and (p / "Depth").is_dir()]
    if len(candidates) != 1:
        raise ValueError("Specify the one directory containing Original/ and Depth/ with --data-root.")
    return candidates[0]


def index_images(directory: Path, prefix: str) -> dict[int, Path]:
    result = {}
    for path in sorted(directory.iterdir()):
        if path.suffix.lower() != ".png":
            continue
        match = re.fullmatch(rf"{prefix}(\d+)", path.stem, re.IGNORECASE)
        if not match:
            raise ValueError(f"Unexpected PNG name: {path}")
        frame_id = int(match.group(1))
        if frame_id in result:
            raise ValueError(f"Duplicate numeric frame ID: {path}")
        result[frame_id] = path
    if not result:
        raise ValueError(f"No {prefix} PNG images in {directory}")
    return result


def split_ids(ids: list[int], gap: int, train_fraction: float = .7,
              val_fraction: float = .15) -> dict[str, list[int]]:
    if gap < 0 or not (0 < train_fraction < 1 and 0 < val_fraction < 1
                       and train_fraction + val_fraction < 1):
        raise ValueError("Invalid gap or split fractions")
    boundary1 = int(len(ids) * train_fraction)
    boundary2 = int(len(ids) * (train_fraction + val_fraction))
    splits = {"train": ids[:boundary1], "val": ids[boundary1 + gap:boundary2],
              "test": ids[boundary2 + gap:]}
    if any(not values for values in splits.values()):
        raise ValueError("Too few pairs for this split/gap. Reduce --gap.")
    return splits


def make_preview(root: Path, pairs: list[dict], output: Path) -> None:
    indices = np.linspace(0, len(pairs) - 1, min(5, len(pairs)), dtype=int)
    tile_w, tile_h = 368, 309
    canvas = Image.new("RGB", (tile_w * 3, (tile_h + 26) * len(indices)), "#171c24")
    draw = ImageDraw.Draw(canvas)
    for row, i in enumerate(indices):
        pair = pairs[int(i)]
        with Image.open(root / pair["original"]) as source:
            rgb = source.convert("RGB")
        with Image.open(root / pair["target"]) as source:
            raw = source.convert("RGB")
            green = raw.getchannel("G").convert("RGB")
        for col, (im, label) in enumerate([(rgb, "RGB"), (raw, "Stored target"),
                                         (green, "G channel: brighter = higher")]):
            im.thumbnail((tile_w, tile_h))
            x, y = col * tile_w, row * (tile_h + 26)
            draw.text((x + 4, y + 5), f'{pair["id"]}  {label}', fill="white")
            canvas.paste(im, (x, y + 26))
    canvas.save(output)


def prepare(data_root: Path, output: Path, gap: int = 50) -> dict:
    root = discover_root(data_root)
    originals = index_images(root / "Original", "Original")
    targets = index_images(root / "Depth", "Depth")
    if originals.keys() != targets.keys():
        raise ValueError(f"Unpaired IDs: missing target={sorted(originals.keys()-targets.keys())[:20]}, "
                         f"missing original={sorted(targets.keys()-originals.keys())[:20]}")
    ids = sorted(originals)
    splits = split_ids(ids, gap)
    split_by_id = {i: name for name, values in splits.items() for i in values}
    histogram = np.zeros(256, dtype=np.int64)
    dimensions, modes, pairs, nonzero_rb, pixels = {}, {}, [], 0, 0
    sampled_differences, previous = [], None
    for n, frame_id in enumerate(ids):
        original, target = originals[frame_id], targets[frame_id]
        # verify() checks every PNG's integrity, including RGB PNGs not fully used here.
        with Image.open(original) as im:
            size = im.size
            im.verify()
        with Image.open(target) as im:
            target_size, mode = im.size, im.mode
            im.verify()
        if size != target_size:
            raise ValueError(f"Pair size mismatch: {original.name}, {target.name}")
        if mode != "RGB":
            raise ValueError(f"Expected 8-bit RGB label, got {mode}: {target}")
        with Image.open(target) as im:
            values = np.asarray(im)
        histogram += np.bincount(values[..., 1].ravel(), minlength=256)
        nonzero_rb += int(np.count_nonzero(values[..., [0, 2]]))
        pixels += int(values.shape[0] * values.shape[1])
        key = f"{size[0]}x{size[1]}"
        dimensions[key] = dimensions.get(key, 0) + 1
        modes[mode] = modes.get(mode, 0) + 1
        if n < 40:
            thumb = np.asarray(Image.fromarray(values[..., 1]).resize((92, 77)), dtype=np.float32)
            if previous is not None:
                sampled_differences.append(float(np.abs(thumb - previous).mean() / 255))
            previous = thumb
        pairs.append({"id": frame_id, "original": original.relative_to(root).as_posix(),
                      "target": target.relative_to(root).as_posix(),
                      "split": split_by_id.get(frame_id, "purged")})
        if (n + 1) % 500 == 0:
            print(f"Audited {n + 1}/{len(ids)} pairs", flush=True)
    used = np.flatnonzero(histogram)
    manifest = {"version": 1, "dataset_root": str(root), "target": {
        "quantity": "water_surface_height", "channel": 1, "channel_name": "G",
        "divisor": 255.0, "brighter_is_higher": True,
        "units": "normalized_stored_value", "metric_mapping": None,
        "zero_is_valid": True,
        "note": "Not camera distance. Global physical height mapping is unknown."},
        "split_policy": {"method": "chronological", "purge_frames_after_boundary": gap,
                         "note": "Assumes numeric IDs follow time. No independent-scene guarantee."},
        "pairs": pairs}
    report = {"pairs": len(pairs), "id_range": [ids[0], ids[-1]], "dimensions": dimensions,
              "target_modes": modes, "nonzero_red_blue_values": nonzero_rb,
              "green_range": [int(used[0]), int(used[-1])],
              "green_mean": float(np.dot(np.arange(256), histogram) / pixels),
              "green_histogram": histogram.tolist(),
              "fraction_at_zero": float(histogram[0] / pixels),
              "fraction_at_255": float(histogram[255] / pixels),
              "mean_adjacent_target_mae_first_40": float(np.mean(sampled_differences))
                  if sampled_differences else None,
              "split_counts": {name: sum(p["split"] == name for p in pairs)
                               for name in ["train", "val", "test", "purged"]},
              "target_quantity": "water_surface_height", "physical_mapping_known": False}
    output.mkdir(parents=True, exist_ok=True)
    (output / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    (output / "dataset_report.json").write_text(
        json.dumps(report, indent=2) + "\n", encoding="utf-8")
    make_preview(root, pairs, output / "dataset_preview.jpg")
    print(json.dumps({k: v for k, v in report.items() if k != "green_histogram"}, indent=2))
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, default=Path("."))
    parser.add_argument("--output", type=Path, default=Path("artifacts"))
    parser.add_argument("--gap", type=int, default=50)
    args = parser.parse_args()
    prepare(args.data_root, args.output, args.gap)


if __name__ == "__main__":
    main()

