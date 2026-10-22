import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
import torch
from PIL import Image

from wave_height.data import WaveDataset, height_array, read_manifest
from wave_height.metrics import RegressionMetrics, regression_loss
from wave_height.model import WaveHeightNet
from wave_height.predict import predict_image, save_prediction
from wave_height.prepare import index_images, split_ids
from wave_height.runtime import mean_height


class LabelAndSplitTests(unittest.TestCase):
    def test_green_channel_and_zero_preserved(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "label.png"
            rgb = np.zeros((8, 8, 3), dtype=np.uint8)
            rgb[..., 0], rgb[..., 2] = 255, 200
            rgb[:, 4:, 1] = 255
            Image.fromarray(rgb).save(path)
            label = height_array(path, (8, 8))
            np.testing.assert_array_equal(label[:, :4], 0.)
            np.testing.assert_array_equal(label[:, 4:], 1.)

    def test_numeric_pair_index_and_duplicates(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            for name in ["Original10.png", "Original2.png", ".DS_Store"]:
                (root / name).touch()
            self.assertEqual(sorted(index_images(root, "Original")), [2, 10])
            (root / "Original0002.png").touch()
            with self.assertRaisesRegex(ValueError, "Duplicate"):
                index_images(root, "Original")

    def test_temporal_gap_and_disjoint_splits(self):
        splits = split_ids(list(range(1, 5001)), 50)
        self.assertEqual({k: len(v) for k, v in splits.items()},
                         {"train": 3500, "val": 700, "test": 700})
        self.assertEqual(splits["val"][0] - splits["train"][-1], 51)
        self.assertEqual(splits["test"][0] - splits["val"][-1], 51)
        for a, b in [("train", "val"), ("train", "test"), ("val", "test")]:
            self.assertFalse(set(splits[a]) & set(splits[b]))

    def test_mean_baseline_uses_only_training_targets_and_flip_is_paired(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            pairs = []
            for n, (value, split) in enumerate([(0, "train"), (128, "train"), (255, "val")]):
                rgb = np.zeros((8, 8, 3), dtype=np.uint8)
                rgb[..., 1] = value
                Image.fromarray(rgb).save(root / f"target{n}.png")
                Image.fromarray(rgb).save(root / f"rgb{n}.png")
                pairs.append(dict(id=n, original=f"rgb{n}.png", target=f"target{n}.png", split=split))
            target = dict(quantity="water_surface_height", channel=1, divisor=255., brighter_is_higher=True)
            manifest = root / "manifest.json"
            manifest.write_text(json.dumps(dict(dataset_root=str(root), target=target, pairs=pairs)))
            dataset = WaveDataset(manifest, "train", (8, 8))
            np.testing.assert_allclose(mean_height(dataset).numpy(), 64 / 255, atol=1e-7)
            patterned = np.zeros((8, 8, 3), dtype=np.uint8)
            patterned[:, 4:, 1] = 255
            Image.fromarray(patterned).save(root / "target0.png")
            Image.fromarray(patterned).save(root / "rgb0.png")
            dataset = WaveDataset(manifest, "train", (8, 8), augment=True, horizontal_flip=True)
            with patch("wave_height.data.random.random", return_value=0.), \
                 patch("wave_height.data.random.uniform", return_value=1.):
                sample = dataset[0]
            np.testing.assert_array_equal(sample["target"].numpy()[0, :, :4], 1.)
            np.testing.assert_array_equal(sample["target"].numpy()[0, :, 4:], 0.)
            self.assertGreater(float(sample["rgb"][1, :, :3].mean()),
                               float(sample["rgb"][1, :, 5:].mean()))
            pairs[2]["original"] = pairs[0]["original"]
            manifest.write_text(json.dumps(dict(dataset_root=str(root), target=target, pairs=pairs)))
            with self.assertRaisesRegex(ValueError, "Duplicate image"):
                read_manifest(manifest)


class RegressionTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(2)

    def test_loss_preserves_absolute_values_and_accepts_zero(self):
        target = torch.zeros(2, 1, 32, 48)
        self.assertEqual(float(regression_loss(target, target, .5)), 0.)
        pred = torch.full_like(target, .2, requires_grad=True)
        loss = regression_loss(pred, target, .5)
        self.assertAlmostEqual(float(loss.detach()), .2, places=6)
        loss.backward()
        self.assertTrue(torch.isfinite(pred.grad).all())
        self.assertGreater(float(pred.grad.abs().sum()), 0.)

    def test_metrics_count_pixels_across_unequal_batches(self):
        metrics = RegressionMetrics()
        metrics.update(torch.ones(2, 1, 8, 8), torch.zeros(2, 1, 8, 8))
        metrics.update(torch.zeros(1, 1, 8, 8), torch.zeros(1, 1, 8, 8))
        result = metrics.compute()
        self.assertAlmostEqual(result["mae"], 2 / 3)
        self.assertAlmostEqual(result["rmse"], (2 / 3)**.5)
        self.assertEqual(result["images"], 3)

    def test_network_handles_odd_sizes_and_backpropagation(self):
        model = WaveHeightNet(pretrained=False).train()
        image = torch.rand(1, 3, 65, 97)
        pred = model(image)
        self.assertEqual(tuple(pred.shape), (1, 1, 65, 97))
        self.assertTrue(((pred >= 0) & (pred <= 1)).all())
        regression_loss(pred, torch.rand_like(pred), .5).backward()
        grads = [p.grad for p in model.parameters() if p.grad is not None]
        self.assertTrue(all(torch.isfinite(g).all() for g in grads))

    def test_export_uses_fixed_scale_and_restores_input_size(self):
        model = WaveHeightNet(pretrained=False).eval()
        rgb = Image.new("RGB", (117, 81), "blue")
        result = predict_image(model, rgb, (64, 96), torch.device("cpu"))
        self.assertEqual(result.shape, (81, 117))
        value = np.full((81, 117), .25, dtype=np.float32)
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory)
            save_prediction(rgb, value, output, "sample", (-2., 2.))
            np.testing.assert_array_equal(np.load(output / "sample_height_normalized.npy"), value)
            np.testing.assert_array_equal(np.load(output / "sample_height_world_units.npy"), -1.)
            with Image.open(output / "sample_height16.png") as image:
                np.testing.assert_array_equal(np.asarray(image), 16384)


if __name__ == "__main__":
    unittest.main()

