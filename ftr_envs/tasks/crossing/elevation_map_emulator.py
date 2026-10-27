"""Make Isaac's exact heightmap look like the elevation map a deployed policy actually receives.

Isaac crops the observation window out of the terrain's numpy height grid: exact geometry,
razor-sharp edges, every gap as deep as it really is. The Gazebo/real pipeline instead fuses
lidar returns with elevation_mapping and then runs grid_map filters (fused_pipeline.yaml: a
12 cm max-then-min closing, OpenCV inpainting), so obstacles come out slightly grown, edges
smeared over a few cells and narrow gaps filled. A policy -- or the velocity controller, which
is trained on Gazebo maps -- that only ever saw the sharp version meets a different input
distribution when deployed.

This module is the whole emulation, in-process and batched over all envs, no ROS:

    dilate (max over k_dilate cells) -> erode (min over k_erode cells) -> Gaussian blur (sigma)
    -> + height_offset (the fused map sits ~1 cm above the true surface)

applied to a window `margin` cells wider than the observation on every side (so the filters
see real terrain past the edge instead of padding), then cropped back. The parameters are fit
against paired Gazebo windows (misc/map_emulation/fit_emulator.py: true-geometry window vs the
recorded /elevation_map_gt_filtered at the same GT pose).

There is deliberately no update-rate hold here. The deployed map is WORLD-fixed and the
policy node re-crops it at the current pose every control step; only the map's content is
up to one map period old, which matters only for terrain seen for the first time -- and the
window's terrain has been in lidar range for metres. Holding the robot-centric window between
updates would instead freeze it in the robot frame and drag obstacles along with the robot.
"""
import math

import torch
import torch.nn.functional as F


class ElevationMapEmulator:
    def __init__(self, k_dilate: int = 1, k_erode: int = 1, blur_sigma: float = 0.0, height_offset: float = 0.0):
        for k in (k_dilate, k_erode):
            if k < 1 or k % 2 == 0:
                raise ValueError(f"morphology kernels must be odd and >= 1, got {k}")
        self.k_dilate = k_dilate
        self.k_erode = k_erode
        self.blur_sigma = float(blur_sigma)
        self.height_offset = float(height_offset)
        self._blur_r = int(math.ceil(3.0 * self.blur_sigma)) if self.blur_sigma > 0 else 0
        self._kernel = None

    @property
    def margin(self) -> int:
        """Cells of context needed on every side of the observation window."""
        return self.k_dilate // 2 + self.k_erode // 2 + self._blur_r

    def _gauss(self, device, dtype):
        if self._kernel is None or self._kernel.device != device or self._kernel.dtype != dtype:
            x = torch.arange(-self._blur_r, self._blur_r + 1, device=device, dtype=dtype)
            g = torch.exp(-0.5 * (x / self.blur_sigma) ** 2)
            self._kernel = g / g.sum()
        return self._kernel

    def __call__(self, hm: torch.Tensor) -> torch.Tensor:
        """hm: [N, H + 2m, W + 2m] (m = self.margin) -> [N, H, W]."""
        x = hm.unsqueeze(1)
        if self.k_dilate > 1:
            x = F.max_pool2d(F.pad(x, [self.k_dilate // 2] * 4, mode="replicate"), self.k_dilate, stride=1)
        if self.k_erode > 1:
            x = -F.max_pool2d(F.pad(-x, [self.k_erode // 2] * 4, mode="replicate"), self.k_erode, stride=1)
        if self._blur_r:
            g = self._gauss(x.device, x.dtype)
            r = self._blur_r
            x = F.conv2d(F.pad(x, [0, 0, r, r], mode="replicate"), g.view(1, 1, -1, 1))
            x = F.conv2d(F.pad(x, [r, r, 0, 0], mode="replicate"), g.view(1, 1, 1, -1))
        m = self.margin
        x = x.squeeze(1) + self.height_offset
        return x[:, m: x.shape[1] - m, m: x.shape[2] - m] if m else x
