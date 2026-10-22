from pathlib import Path

import numpy as np
import torch
from PIL import Image, ImageDraw


@torch.inference_mode()
def evaluation_preview(model, dataset, device, baseline: torch.Tensor, output: Path,
                       count: int = 6) -> None:
    model.eval()
    count = min(count, len(dataset))
    indices = np.linspace(0, len(dataset) - 1, count, dtype=int)
    tile_w = 256
    tile_h = round(tile_w * dataset.size[0] / dataset.size[1])
    canvas = Image.new("RGB", (tile_w * 5, (tile_h + 24) * count), "#171c24")
    draw = ImageDraw.Draw(canvas)
    baseline_image = Image.fromarray(np.rint(baseline[0, 0].numpy() * 255).astype(np.uint8))
    for row, index in enumerate(indices):
        sample = dataset[int(index)]
        prediction = model(sample["rgb"][None].to(device))[0, 0].float().cpu().numpy()
        target = sample["target"][0].numpy()
        with Image.open(dataset.root / dataset.pairs[int(index)]["original"]) as source:
            rgb = source.convert("RGB")
        error = np.clip(np.abs(prediction - target) / .25, 0, 1)
        heat = np.zeros((*error.shape, 3), dtype=np.uint8)
        heat[..., 0] = np.rint(error * 255).astype(np.uint8)
        tiles = [(rgb, f'RGB {sample["id"]}'),
                 (Image.fromarray(np.rint(target * 255).astype(np.uint8)), "Target height [0,1]"),
                 (Image.fromarray(np.rint(prediction * 255).astype(np.uint8)), "Predicted height [0,1]"),
                 (baseline_image, "Train mean height [0,1]"),
                 (Image.fromarray(heat), "Absolute error: red >= 0.25")]
        for col, (tile, title) in enumerate(tiles):
            x, y = col * tile_w, row * (tile_h + 24)
            draw.text((x + 4, y + 5), title, fill="white")
            canvas.paste(tile.convert("RGB").resize((tile_w, tile_h)), (x, y + 24))
    canvas.save(output)

