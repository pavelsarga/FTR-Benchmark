# -*- coding: utf-8 -*-
"""
====================================
@File Name ：terrain_cfg.py
@Time ： 2024/9/29 PM6:16
@Program IDE ：PyCharm
@Create by Author ： hongchuan zhang
====================================

"""
import json
import os
import traceback
from math import floor
from pathlib import Path
from functools import partial

import cv2
import numpy as np
import yaml


class MapHelper:
    def __init__(self, lower, upper, cell_size=0.05, path=None):
        self.lower = np.array(lower)
        self.upper = np.array(upper)
        self.cell_size = cell_size
        self.path = path
        self.compensation = -(np.array(self.lower[:2] / self.cell_size).astype(np.int32))
        with open(path, "rb") as f:
            self.map = np.load(f, allow_pickle=True)

    def get_obs(self, positon, angle, size):
        """

        :param positon: center point in world coordinates for cropping
        :param angle: rotation angle
        :param size_: observation range size
        :return: cropped elevation map
        """

        size_ = np.array(size)

        pos = np.floor(np.array(positon[:2], dtype=float) / self.cell_size + self.compensation)
        side_length = int(np.sqrt(size_[0] ** 2 + size_[1] ** 2) // self.cell_size)

        low = (pos[:2] - side_length // 2).astype(int)
        up = (pos[:2] + side_length // 2).astype(int)

        map_h, map_w = self.map.shape

        # Clamp slice indices to valid map bounds
        low_c = np.clip(low, 0, [map_h, map_w])
        up_c = np.clip(up + 1, 0, [map_h, map_w])

        if low_c[0] >= up_c[0] or low_c[1] >= up_c[1]:
            # Robot entirely outside the map — return flat zero patch
            full_side = side_length + 1
            return np.zeros((full_side, full_side), dtype=self.map.dtype)

        raw = self.map[low_c[0]: up_c[0], low_c[1]: up_c[1]]

        # Pad with edge values when the robot is near the map boundary
        pad_top    = int(low_c[0] - low[0])
        pad_bottom = int((up[0] + 1) - up_c[0])
        pad_left   = int(low_c[1] - low[1])
        pad_right  = int((up[1] + 1) - up_c[1])

        if pad_top or pad_bottom or pad_left or pad_right:
            local_map = np.pad(raw, ((pad_top, pad_bottom), (pad_left, pad_right)), mode="edge")
        else:
            local_map = raw

        h, w = local_map.shape
        center = (w // 2, h // 2)

        M = cv2.getRotationMatrix2D(center, -angle, 1.0)
        rotated = cv2.warpAffine(local_map, M, (w, h))

        low_clip = np.array(center) - size_ / 2 // self.cell_size
        up_clip = np.array(center) + size_ / 2 // self.cell_size

        return rotated[
               int(low_clip[0]): int(up_clip[0]) + 1,
               int(low_clip[1]): int(up_clip[1]) + 1,
               ]

    def get_range_map(self, low, up):
        x1 = floor(low[0] / self.cell_size + self.compensation[0])
        y1 = floor(low[1] / self.cell_size + self.compensation[1])

        x2 = floor(up[0] / self.cell_size + self.compensation[0])
        y2 = floor(up[1] / self.cell_size + self.compensation[1])

        return self.map[x1: x2 + 1, y1: y2 + 1]


class Terrain:

    def __init__(self, name, prim_path="/World/terrain"):
        self.prim_path = prim_path
        terrain_dir = Path(__file__).parent
        with open(terrain_dir / "config" / f"{name}.yaml") as f:
            self.config = yaml.safe_load(f)
        self.obstacles = dict()
        if "obstacles" in self.config:
            for subname, obstacle in self.config["obstacles"].items():
                assert (terrain_dir / obstacle["path"]).exists()
                self.obstacles[subname] = obstacle.copy()
                self.obstacles[subname]["path"] = str(terrain_dir / obstacle["path"])

        else:
            default_usd = terrain_dir / "usd" / f"{name}.usd"
            assert default_usd.exists()
            self.obstacles["terrain"] = {"path": str(default_usd)}

        if "task_info" in self.config:
            self.birth = [self.config["task_info"]]
        else:
            with open(terrain_dir / "birth" / f"{name}.json", 'r') as f:
                self.birth = json.load(f)

        self.map = MapHelper(
            path=terrain_dir / "map" / f"{name}.map",
            **self.config["map"],
        )

    def apply(self, stage, ):
        from ftr_envs.utils.prim import add_usd
        # apply obstacles
        for name, obst in self.obstacles.items():
            add_usd_fun = partial(add_usd, usd_file=obst["path"], prim_path=f"{self.prim_path}/{name}")
            if "position" in obst:
                add_usd_fun = partial(add_usd_fun, pos=obst["position"])
            if "orient" in obst:
                add_usd_fun = partial(add_usd_fun, orient=obst["orient"])
            if "scale" in obst:
                add_usd_fun = partial(add_usd_fun, scale=obst["scale"])
            add_usd_fun()

        # apply config
        if "prim_config" in self.config:
            for cfg in self.config["prim_config"]["set_attrs"]:
                _prim_path = self.prim_path + "/" + cfg["prim_path"]
                attr_name = cfg["attr_name"]
                value = cfg["value"]

                if isinstance(value, list):
                    value = tuple(value)

                if attr_name == "xformOp:orient":
                    from pxr.Gf import Quatd
                    value = Quatd(*value)
                elif attr_name == "xformOp:translate":
                    from pxr.Gf import Vec3d
                    value = Vec3d(*value)
                elif attr_name == "xformOp:scale":
                    from pxr.Gf import Vec3f
                    value = Vec3f(*value)

                try:
                    stage.GetPrimAtPath(_prim_path).GetAttribute(attr_name).Set(value)

                except RuntimeError as e:
                    import carb
                    carb.log_error(f"Failed to set {_prim_path} {attr_name} to {value}: {e}")
                    raise e

    @classmethod
    def list_all(cls):
        terrain_dir = Path(__file__).parent
        config_files = set(os.listdir(terrain_dir / "config"))
        return [name.removesuffix(".yaml") for name in config_files]

    @classmethod
    def check(cls):
        """
        Check assets and config files for existence and correctness.
        Command: python3 -c "from ftr_envs.assets.terrain.terrain import Terrain; Terrain.check()"
        :return:
        """
        import io
        import textwrap
        import traceback

        terrain_dir = Path(__file__).parent
        config_files = set(os.listdir(terrain_dir / "config"))

        error_msg = dict()
        for config_file in config_files:
            name = config_file.removesuffix(".yaml")
            try:
                cls(name)
            except Exception as e:
                buf = io.StringIO()
                traceback.print_exception(e, file=buf)
                error_msg[name] = buf.getvalue()

        if len(error_msg) == 0:
            print("All config files are valid.")
        else:
            print("The following config files have errors:")
            for name, msg in error_msg.items():
                print(f"{name}:")
                print(textwrap.indent(msg, " " * 4))

if __name__=="__main__":
    # regenerate map in scale
    import numpy as np                                                                          
    scale = 2         
    cells = int(440 * scale)  # 880 for 2×                                                      
    m = np.zeros((cells, cells), dtype=np.float32)
    with open("src/FTR-benchmark/ftr_envs/assets/terrain/map/ground.map", "wb") as f:           
        np.save(f, m) 
        print("done")
