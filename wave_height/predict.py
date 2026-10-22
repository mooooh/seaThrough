"""Predict surface height from an RGB image, image directory, or optional video input."""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch
from PIL import Image, ImageDraw
from torch.nn import functional as F

from .data import rgb_tensor
from .runtime import choose_device, load_model, write_json


@torch.inference_mode()
def predict_image(model, image: Image.Image, size: tuple[int, int], device) -> np.ndarray:
    batch = rgb_tensor(image, size)[None].to(device)
    value = model(batch)
    # Restore input dimensions; this interpolates output and does not recover extra detail.
    value = F.interpolate(value, size=(image.height, image.width), mode="bilinear", align_corners=False)
    return value[0, 0].float().cpu().numpy().clip(0, 1)


def save_prediction(image: Image.Image, value: np.ndarray, output: Path,
                    stem: str, height_range: tuple[float, float] | None = None) -> None:
    np.save(output / f"{stem}_height_normalized.npy", value)
    Image.fromarray(np.rint(value * 65535).astype(np.uint16)).save(output / f"{stem}_height16.png")
    grey = np.rint(value * 255).astype(np.uint8)
    green = np.zeros((*grey.shape, 3), dtype=np.uint8)
    green[..., 1] = grey
    Image.fromarray(green).save(output / f"{stem}_height_green.png")
    # Fixed [0,1] display scale for every frame; no per-frame min/max normalization.
    width, height = image.size
    canvas = Image.new("RGB", (width * 2, height + 26), "#171c24")
    canvas.paste(image.convert("RGB"), (0, 26))
    canvas.paste(Image.fromarray(grey).convert("RGB"), (width, 26))
    draw = ImageDraw.Draw(canvas)
    draw.text((6, 6), "Input RGB", fill="white")
    draw.text((width + 6, 6), "Predicted surface height [0,1]: brighter = higher", fill="white")
    canvas.save(output / f"{stem}_preview.jpg")
    if height_range is not None:
        low, high = height_range
        np.save(output / f"{stem}_height_world_units.npy", low + value * (high - low))


def process_video(model, source: Path, output: Path, size, device, height_range, every: int) -> None:
    try:
        import cv2
    except ImportError as error:
        raise ImportError("Video needs optional dependency: pip install opencv-python") from error
    capture = cv2.VideoCapture(str(source))
    if not capture.isOpened():
        raise ValueError(f"Cannot open video: {source}")
    fps = capture.get(cv2.CAP_PROP_FPS)
    frames, saved = 0, 0
    try:
        while True:
            ok, frame = capture.read()
            if not ok:
                break
            if frames % every == 0:
                rgb = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
                value = predict_image(model, rgb, size, device)
                save_prediction(rgb, value, output, f"frame_{frames:08d}", height_range)
                saved += 1
                if saved % 50 == 0:
                    print(f"Saved {saved} predictions (source frame {frames})", flush=True)
            frames += 1
    finally:
        capture.release()
    if not saved:
        raise ValueError("Video contained no readable frames")
    write_json(output / "video_info.json", {"source": str(source.resolve()), "source_fps": fps,
                                           "every": every, "frames_read": frames,
                                           "frames_saved": saved})


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True, type=Path)
    parser.add_argument("--input", required=True, type=Path)
    parser.add_argument("--output", type=Path, default=Path("outputs/prediction"))
    parser.add_argument("--device", default="auto")
    parser.add_argument("--every", type=int, default=1, help="Video frame stride")
    parser.add_argument("--height-range", type=float, nargs=2, metavar=("LOW", "HIGH"),
                        help="Only supply if global linear G-to-world-height mapping is known")
    args = parser.parse_args()
    if args.every < 1:
        raise ValueError("--every must be positive")
    if args.height_range and not (np.isfinite(args.height_range).all()
                                  and args.height_range[0] < args.height_range[1]):
        raise ValueError("--height-range needs finite LOW < HIGH")
    if not args.input.exists():
        raise FileNotFoundError(args.input)
    # Avoid silently mixing results from different runs.
    if args.output.exists() and any(args.output.iterdir()):
        raise FileExistsError(f"Output is not empty: {args.output}. Choose a new --output.")
    args.output.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(min(torch.get_num_threads(), 8))
    device = choose_device(args.device)
    model, checkpoint = load_model(args.checkpoint, device)
    size = (checkpoint["config"]["height"], checkpoint["config"]["width"])
    encoding = dict(checkpoint["target"])
    encoding.update(checkpoint=str(args.checkpoint.resolve()), limited_training=checkpoint["limited_run"],
                    png16_scale="pixel/65535 = normalized height; original labels have 8-bit precision",
                    supplied_world_height_range=args.height_range,
                    world_height_unit="same as user-supplied range; not automatically meters")
    write_json(args.output / "encoding.json", encoding)
    extensions = {".png", ".jpg", ".jpeg", ".bmp", ".webp", ".tif", ".tiff"}
    if args.input.is_file() and args.input.suffix.lower() not in extensions:
        process_video(model, args.input, args.output, size, device, args.height_range, args.every)
    else:
        sources = [args.input] if args.input.is_file() else sorted(
            p for p in args.input.iterdir() if p.suffix.lower() in extensions)
        if not sources:
            raise ValueError("No input images")
        names = [p.stem for p in sources]
        if len(set(names)) != len(names):
            raise ValueError("Input images have duplicate stems; rename to avoid output collisions.")
        for index, source in enumerate(sources):
            with Image.open(source) as image:
                rgb = image.convert("RGB")
            value = predict_image(model, rgb, size, device)
            save_prediction(rgb, value, args.output, source.stem, args.height_range)
            if (index + 1) % 50 == 0 or len(sources) == 1:
                print(f"Predicted {index+1}/{len(sources)}: {source.name}", flush=True)


if __name__ == "__main__":
    main()

