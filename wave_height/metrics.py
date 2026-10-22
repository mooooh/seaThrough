import math

import torch
from torch.nn import functional as F


def gradient_error(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    return ((pred[..., 1:, :] - pred[..., :-1, :]
             - target[..., 1:, :] + target[..., :-1, :]).abs().mean()
            + (pred[..., :, 1:] - pred[..., :, :-1]
               - target[..., :, 1:] + target[..., :, :-1]).abs().mean()) / 2


def regression_loss(pred: torch.Tensor, target: torch.Tensor, gradient_weight: float) -> torch.Tensor:
    edge = pred.new_zeros(())
    for scale in (1, 2, 4):
        p = F.avg_pool2d(pred, scale) if scale > 1 else pred
        t = F.avg_pool2d(target, scale) if scale > 1 else target
        edge = edge + gradient_error(p, t) / 3
    return F.l1_loss(pred, target) + gradient_weight * edge


class RegressionMetrics:
    def __init__(self):
        self.abs_sum, self.square_sum, self.count = 0., 0., 0
        self.edge_sum, self.images = 0., 0

    def update(self, pred: torch.Tensor, target: torch.Tensor):
        error = (pred.detach().float() - target.detach().float())
        self.abs_sum += float(error.abs().sum().item())
        self.square_sum += float(error.square().sum().item())
        self.count += error.numel()
        self.edge_sum += float(gradient_error(pred.float(), target.float()).item()) * pred.shape[0]
        self.images += pred.shape[0]

    def compute(self) -> dict:
        if not self.count:
            raise ValueError("No samples were evaluated")
        return {"mae": self.abs_sum / self.count,
                "rmse": math.sqrt(self.square_sum / self.count),
                "gradient_mae": self.edge_sum / self.images,
                "mae_in_8bit_code_values": self.abs_sum / self.count * 255,
                "images": self.images, "units": "normalized_stored_height"}

