# Copyright (c) MONAI Consortium
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#     http://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
import logging
import os
from typing import Any, Callable, Dict, Optional, Sequence

import numpy as np
from lib.transforms import BufferContoursd, LoadImagePatchd, PostFilterLabeld
from monai.transforms import FromMetaTensord, LoadImaged, SaveImaged, SqueezeDimd

from monailabel.interfaces.tasks.infer_v2 import InferType
from monailabel.tasks.infer.bundle import BundleInferTask
from monailabel.transform.post import FindContoursd, RenameKeyd
from monailabel.transform.writer import PolygonWriter

logger = logging.getLogger(__name__)


class HovernetNuclei(BundleInferTask):
    """
    This provides Inference Engine for pre-trained Hovernet segmentation + Classification model.
    """

    def __init__(self, path: str, conf: Dict[str, str], preset_checkpoint: Optional[str] = None,
                 training_metadata: Optional[Dict] = None, **kwargs):
        super().__init__(
            path,
            conf,
            type=InferType.SEGMENTATION,
            add_post_restore=False,
            pre_filter=[LoadImaged],
            post_filter=[FromMetaTensord, SaveImaged],
            load_strict=True,
            **kwargs,
        )

        # Override Labels
        self.labels = {
            "Other": 1,
            "Inflammatory": 2,
            "Epithelial": 3,
            "Spindle-Shaped": 4,
        }
        self.label_colors = {
            "Other": (255, 0, 0),
            "Inflammatory": (255, 255, 0),
            "Epithelial": (0, 0, 255),
            "Spindle-Shaped": (0, 255, 0),
        }

        # BundleInferTask.__init__ may return early (missing config/model)
        # without calling super().__init__(), leaving _config unset.
        if not hasattr(self, "_config"):
            return

        self._config["label_colors"] = self.label_colors
        self.training_metadata = training_metadata or {}

        # When a specific checkpoint is pre-selected, lock this task to that file.
        # Removes model_filename from _config so _get_network() bypasses the
        # user-selection branch and uses self.path directly.
        if preset_checkpoint:
            self._config.pop("model_filename", None)
            checkpoint_path = os.path.join(path, "models", preset_checkpoint)
            self.path = [checkpoint_path]
            logger.info(f"HovernetNuclei preset checkpoint: {checkpoint_path}")

    def pre_transforms(self, data=None) -> Sequence[Callable]:
        t = [LoadImagePatchd(keys="image", mode="RGB", dtype=np.uint8, padding=False)]
        t.extend([x for x in super().pre_transforms(data)])
        return t

    def post_transforms(self, data=None) -> Sequence[Callable]:
        d = data or {}
        min_size = int(d.get("min_size", 64))
        min_hole = int(d.get("min_hole", 64))
        max_poly_area = int(d.get("max_poly_area", 128 * 128))
        buffer_distance = float(d.get("buffer_distance", 0.5))

        t = [x for x in super().post_transforms(data)]
        t.extend(
            [
                RenameKeyd(source_key="type_map", target_key="pred"),
                SqueezeDimd(keys="pred", dim=0),
                PostFilterLabeld(keys="pred", min_size=min_size, min_hole=min_hole),
                FindContoursd(keys="pred", labels=self.labels, max_poly_area=max_poly_area),
                BufferContoursd(keys="pred", distance=buffer_distance),
            ]
        )
        return t

    def info(self) -> Dict[str, Any]:
        d = super().info()
        d["pathology"] = True
        d["description"] = "HoVerNet Nuclei Segmentation (3x3 Fast Mode)"
        d["configurable_thresholds"] = {
            "min_size": {"default": 64, "description": "최소 객체 크기 (px)"},
            "min_hole": {"default": 64, "description": "최소 hole 크기 (px)"},
            "min_poly_area": {"default": 30, "description": "최소 polygon 면적 (px²)"},
            "max_poly_area": {"default": 16384, "description": "최대 polygon 면적 (px²)"},
            "buffer_distance": {"default": 0.5, "description": "contour 확장 거리 (px)"},
            "marker_threshold": {"default": 0.4, "description": "HoVerNet marker 임계값"},
            "sobel_kernel_size": {"default": 21, "description": "Sobel 커널 크기"},
            "marker_radius": {"default": 2, "description": "marker dilation 반경"},
        }
        if self.training_metadata:
            d["training_metadata"] = self.training_metadata
        return d

    def writer(self, data, extension=None, dtype=None):
        writer = PolygonWriter(label=self.output_label_key, json=self.output_json_key)
        return writer(data)
