# -*- coding: utf-8 -*-
"""
====================================
@File Name ：torch.py
@Time ： 2024/10/12 PM8:50
@Program IDE ：PyCharm
@Create by Author ： hongchuan zhang
====================================

"""
import torch


def add_noise(data, std):
    size = data.size()
    noise = torch.normal(0, std=std, size=size, device=data.device)
    return data + noise


def _random_run_lengths(total: int, min_size: int, max_size: int) -> list[int]:
    sizes = []
    remaining = total
    while remaining > 0:
        s = min(int(torch.randint(min_size, max_size + 1, (1,))), remaining)
        # avoid leaving a tail shorter than min_size
        if 0 < remaining - s < min_size:
            s = remaining
        sizes.append(s)
        remaining -= s
    return sizes


def add_patch_noise(data, std, min_size=2, max_size=5):
    """Add blocky patch noise: the (h, w) grid is tiled into rectangular
    patches with side lengths in [min_size, max_size]; each env draws an
    independent N(0, std) value per patch, broadcast over the whole patch."""
    batch, h, w = data.shape
    row_sizes = _random_run_lengths(h, min_size, max_size)
    col_sizes = _random_run_lengths(w, min_size, max_size)

    patch_values = torch.normal(
        0, std=std, size=(batch, len(row_sizes), len(col_sizes)), device=data.device
    )
    noise = torch.repeat_interleave(patch_values, torch.tensor(row_sizes, device=data.device), dim=1)
    noise = torch.repeat_interleave(noise, torch.tensor(col_sizes, device=data.device), dim=2)
    return data + noise


def rand_range(range_, *args, **kargs):
    """

    :param range_: (_min, _max) value
    :param args: same torch.rand
    :param kargs: same tor.rand
    :return:
    """
    return torch.rand(*args, **kargs) * (
            range_[1] - range_[0]
    ) + range_[0]
