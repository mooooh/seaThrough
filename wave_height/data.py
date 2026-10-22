from __future__ import annotations

import json
import random
from pathlib import Path

import numpy as np
import torch
from PIL import Image, ImageEnhance, ImageFilter
from torch.utils.data import Dataset

MEAN = torch.tensor([.485, .456, .406]).view(3, 1, 1)
STD = torch.tensor([.229, .224, .225]).view(3, 1, 1)


def read_manifest(path: str | Path, data_root: str | Path | None = None) -> tuple[dict, Path]:
    manifest = json.loads(Path(path).read_text(encoding="utf-8"))
    target = manifest["target"]
    if target["quantity"] != "water_surface_height" or target["channel"] != 1:
        raise ValueError("This pipeline expects G-channel surface height labels.")
    if target["divisor"] != 255.0 or target["brighter_is_higher"] is not True:
        raise ValueError("Unsupported label encoding; expected G/255, brighter is higher.")
    ids, paths = set(), set()
    for pair in manifest["pairs"]:
        if pair["split"] not in {"train", "val", "test", "purged"}:
            raise ValueError(f'Unknown split: {pair["split"]}')
        if pair["id"] in ids:
            raise ValueError(f'Duplicate pair ID: {pair["id"]}')
        ids.add(pair["id"])
        for key in ("original", "target"):
            if pair[key] in paths:
                raise ValueError(f'Duplicate image path: {pair[key]}')
            paths.add(pair[key])
    root = Path(data_root or manifest["dataset_root"]).resolve()
    if not root.is_dir():
        raise FileNotFoundError(f"Dataset directory missing: {root}. Pass --data-root on a new PC.")
    return manifest, root


def rgb_tensor(image: Image.Image, size: tuple[int, int]) -> torch.Tensor:
    # size is (height, width). No crop: every pixel keeps the corresponding target location.
    image = image.convert("RGB").resize((size[1], size[0]), Image.Resampling.BILINEAR)
    values = np.asarray(image, dtype=np.float32).copy() / 255.0
    tensor = torch.from_numpy(values.transpose(2, 0, 1))
    return (tensor - MEAN) / STD


def height_array(path: Path, size: tuple[int, int]) -> np.ndarray:
    with Image.open(path) as image:
        if image.mode != "RGB":
            raise ValueError(f"Expected RGB label: {path}")
        image = image.getchannel("G").convert("F").resize(
            (size[1], size[0]), Image.Resampling.BILINEAR)
        return np.asarray(image, dtype=np.float32).copy() / 255.0


class WaveDataset(Dataset):
    def __init__(self, manifest: str | Path, split: str, size: tuple[int, int],
                 data_root: str | Path | None = None, augment: bool = False,
                 horizontal_flip: bool = False, limit: int | None = None):
        self.manifest, self.root = read_manifest(manifest, data_root)
        self.pairs = [p for p in self.manifest["pairs"] if p["split"] == split]
        if limit is not None:
            if limit <= 0:
                raise ValueError("limit must be positive")
            self.pairs = self.pairs[:limit]
        if not self.pairs:
            raise ValueError(f"Empty {split} split")
        self.size, self.augment, self.horizontal_flip = size, augment, horizontal_flip

    def __len__(self) -> int:
        return len(self.pairs)

    def __getitem__(self, index: int) -> dict:
        pair = self.pairs[index]
        with Image.open(self.root / pair["original"]) as source:
            rgb = source.convert("RGB")
        target = height_array(self.root / pair["target"], self.size)
        if self.augment:
            # Photometric changes affect RGB alone; geometry must remain paired.
            rgb = ImageEnhance.Brightness(rgb).enhance(random.uniform(.8, 1.2))
            rgb = ImageEnhance.Contrast(rgb).enhance(random.uniform(.85, 1.15))
            rgb = ImageEnhance.Color(rgb).enhance(random.uniform(.85, 1.15))
            if random.random() < .1:
                rgb = rgb.filter(ImageFilter.GaussianBlur(random.uniform(.1, .6)))
            if self.horizontal_flip and random.random() < .5:
                rgb = rgb.transpose(Image.Transpose.FLIP_LEFT_RIGHT)
                target = target[:, ::-1].copy()
        return {"rgb": rgb_tensor(rgb, self.size),
                "target": torch.from_numpy(target[None].copy()), "id": pair["id"]}

