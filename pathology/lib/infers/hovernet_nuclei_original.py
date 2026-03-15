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
import sys
from typing import Any, Callable, Dict, Optional, Sequence

import numpy as np
import torch
import torch.nn.functional as F
from lib.inferers import ReflectPadHoVerNetInferer
from lib.original_hovernet import OriginalHoVerNetWrapper
from lib.transforms import BufferContoursd, FindContoursFromInstanceMapd, LoadImagePatchd
from monai.config import KeysCollection
from monai.data import MetaTensor
from monai.inferers import Inferer
from monai.transforms import FromMetaTensord, LoadImaged, MapTransform, SaveImaged, SqueezeDimd
from monai.utils import convert_to_numpy

from monailabel.interfaces.tasks.infer_v2 import InferType
from monailabel.tasks.infer.bundle import BundleConstants, BundleInferTask
from monailabel.transform.post import FindContoursd, RenameKeyd
from monailabel.transform.writer import PolygonWriter

logger = logging.getLogger(__name__)


class UndoLoadImagePatchTransposed(MapTransform):
    """Undo the np.moveaxis(0,1) that LoadImagePatchd applies.

    LoadImagePatchd transposes images from (H,W,C) to (W,H,C). This causes the
    model to see a transposed image, producing different (degraded) results.
    This transform undoes that transpose so the model sees the correct orientation.
    """

    def __call__(self, data):
        d = dict(data)
        for key in self.keys:
            img = d[key]
            if isinstance(img, MetaTensor):
                t = img.cpu() if img.is_cuda else img
                arr = t.numpy()
                arr = np.moveaxis(arr, 0, 1)  # undo: (W, H, C) → (H, W, C)
                meta = img.meta.copy() if hasattr(img, 'meta') else {}
                meta["spatial_shape"] = np.asarray(arr.shape[:-1])
                d[key] = MetaTensor(arr, meta=meta, device=img.device)
            elif isinstance(img, torch.Tensor):
                d[key] = img.transpose(0, 1).contiguous()
            else:
                d[key] = np.moveaxis(img, 0, 1)
        return d


class ApplyMoveAxisForFindContoursd(MapTransform):
    """Apply np.moveaxis(0,1) to convert (H,W) → (W,H) before FindContoursd.

    FindContoursd expects data in (W,H) format (as produced by LoadImagePatchd)
    and internally does moveaxis(0,1) to get (H,W) for cv2.findContours.
    When we've undone LoadImagePatchd's transpose, our data is (H,W).
    This transform converts it to (W,H) so FindContoursd works correctly.
    """

    def __call__(self, data):
        d = dict(data)
        for key in self.keys:
            p = d[key]
            if isinstance(p, torch.Tensor):
                d[key] = p.T
            else:
                d[key] = np.moveaxis(p, 0, 1)
        return d


class OriginalHoVerNetPostProcessd(MapTransform):
    """Replace MONAI's HoVerNet post-processing with the original hover_net post_proc.

    Converts the raw model output dict (nucleus_prediction, horizontal_vertical,
    type_prediction) into instance_map and type_map using the EXACT same
    post-processing as the original HoverNet codebase (/app/hover_net).

    This ensures identical results to the original pipeline (cv2.Sobel, cv2.GaussianBlur,
    cv2.morphologyEx with MORPH_ELLIPSE kernel, etc.).

    Args:
        undo_spatial_transpose: If True, undo the np.moveaxis(0,1) from LoadImagePatchd
            before processing and re-apply it after. This is needed because LoadImagePatchd
            transposes (H,W) → (W,H) but the original post_proc expects (H,W).
            Set True for QuPath pipeline, False for direct use (e.g. verify scripts).
    """

    def __init__(self, keys="pred", nr_types=5, undo_spatial_transpose=False,
                 sobel_kernel_size=21, marker_threshold=0.4):
        super().__init__(keys)
        self.nr_types = nr_types
        self.undo_spatial_transpose = undo_spatial_transpose
        self.sobel_kernel_size = sobel_kernel_size
        self.marker_threshold = marker_threshold

        # Import original post-processing
        if "/app/hover_net" not in sys.path:
            sys.path.insert(0, "/app/hover_net")
        from models.hovernet.post_proc import process as _orig_process

        self._orig_process = _orig_process

    def __call__(self, data):
        d = dict(data)

        # Get raw model outputs from the sliding window inferer
        # These are stored under "pred" as a dict by BundleInferTask
        pred = d.get("pred", d)

        np_raw = convert_to_numpy(pred["nucleus_prediction"]).squeeze()   # [2, S0, S1]
        hv_raw = convert_to_numpy(pred["horizontal_vertical"]).squeeze()  # [2, S0, S1]
        tp_raw = convert_to_numpy(pred["type_prediction"]).squeeze()      # [5, S0, S1]

        if self.undo_spatial_transpose:
            # LoadImagePatchd does np.moveaxis(0,1): (H,W,C) → (W,H,C)
            # So model outputs are in (C, W, H) space. Transpose back to (C, H, W).
            np_raw = np.swapaxes(np_raw, -2, -1)   # [2, H, W]
            hv_raw = np.swapaxes(hv_raw, -2, -1)   # [2, H, W]
            tp_raw = np.swapaxes(tp_raw, -2, -1)    # [5, H, W]
            # Swap HV channels: model saw transposed image, so its h_dir = original v_dir
            hv_raw = hv_raw[::-1].copy()

        # Apply softmax to NP → fg probability (same as infer_step in run_desc.py)
        np_prob = F.softmax(torch.from_numpy(np_raw), dim=0)[1].numpy()  # [H, W]

        # Apply softmax + argmax to TP → class IDs
        tp_class = torch.argmax(
            F.softmax(torch.from_numpy(tp_raw), dim=0), dim=0
        ).numpy().astype(np.float32)  # [H, W]

        # HV: [2, H, W] → [H, W, 2]
        hv_hwc = np.transpose(hv_raw, (1, 2, 0))

        # Stack in original format: [tp(1), np(1), hv(2)] = 4 channels
        pred_map = np.concatenate([
            tp_class[..., None],
            np_prob[..., None],
            hv_hwc,
        ], axis=-1)  # [H, W, 4]

        # Run original post-processing (watershed + type assignment)
        pred_inst, inst_info = self._orig_process(
            pred_map, nr_types=self.nr_types, return_centroids=True,
            sobel_kernel_size=self.sobel_kernel_size,
            marker_threshold=self.marker_threshold,
        )

        # Build type_map from instance info
        type_map = np.zeros_like(pred_inst, dtype=np.int32)
        if inst_info:
            for inst_id, info in inst_info.items():
                if info.get("type") is not None:
                    type_map[pred_inst == inst_id] = info["type"]

        if self.undo_spatial_transpose:
            # Transpose results back to (W, H) for downstream transforms
            # (FindContoursd expects (W, H) and does moveaxis to get (H, W))
            type_map = type_map.T.copy()
            pred_inst = pred_inst.T.copy()

        d["type_map"] = type_map[None]          # [1, S0, S1]
        d["instance_map"] = pred_inst[None]     # [1, S0, S1]
        d["instance_info"] = inst_info

        # Clean up the intermediate keys
        for k in ["nucleus_prediction", "horizontal_vertical", "type_prediction"]:
            d.pop(k, None)
        d.pop("pred", None)

        return d


class OriginalModeBundleConstants(BundleConstants):
    """Override to use inference_original.json (original mode: patch_size=270, out_size=80)."""

    def configs(self):
        return ["inference_original.json"]


class HovernetNucleiOriginal(BundleInferTask):
    """
    Inference engine for pre-trained HoVerNet in 'original' mode
    (5x5 decoder convolutions, patch_size=270, out_size=80).

    Uses the ORIGINAL HoverNet model architecture (from hover_net repo)
    instead of MONAI's reimplementation, because the two architectures
    have structural differences (upsampling mode, padding, residual blocks,
    decoder branches) that produce different outputs with the same weights.
    """

    def __init__(
        self,
        path: str,
        conf: Dict[str, str],
        tf2pt_checkpoint: Optional[str] = None,
        preset_checkpoint: Optional[str] = None,
        training_metadata: Optional[Dict] = None,
        **kwargs,
    ):
        super().__init__(
            path,
            conf,
            const=OriginalModeBundleConstants(),
            type=InferType.SEGMENTATION,
            add_post_restore=False,
            pre_filter=[LoadImaged],
            post_filter=[FromMetaTensord, SaveImaged],
            load_strict=True,
            **kwargs,
        )

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

        # Store the TF2PT checkpoint path for loading the original model
        self.tf2pt_checkpoint = tf2pt_checkpoint

        if preset_checkpoint:
            self._config.pop("model_filename", None)
            checkpoint_path = os.path.join(path, "models", preset_checkpoint)
            self.path = [checkpoint_path]
            logger.info(f"HovernetNucleiOriginal preset checkpoint: {checkpoint_path}")

    def _get_network(self, device, data):
        """
        Override to use the original HoverNet model instead of MONAI's HoVerNet.

        When tf2pt_checkpoint is set, loads from the TF2PT checkpoint (original
        model architecture). Otherwise (preset_checkpoint / MONAI-trained),
        falls back to the base class which loads MONAI's HoVerNet.
        """
        if not self.tf2pt_checkpoint:
            return super()._get_network(device, data)

        cached = self._networks.get(device)
        checkpoint_path = self.tf2pt_checkpoint

        if checkpoint_path:
            statbuf = os.stat(checkpoint_path) if os.path.exists(checkpoint_path) else None
        else:
            statbuf = None

        if cached:
            if statbuf and statbuf.st_mtime == cached[1]:
                return cached[0]
            elif not statbuf:
                return cached[0]

        # Create original HoverNet wrapper
        wrapper = OriginalHoVerNetWrapper(nr_types=5, mode="original")

        if checkpoint_path and os.path.exists(checkpoint_path):
            logger.info(f"Loading original HoverNet from TF2PT checkpoint: {checkpoint_path}")
            checkpoint = torch.load(checkpoint_path, map_location=torch.device(device), weights_only=False)

            # TF2PT format: {"desc": state_dict, ...}
            state_dict = checkpoint.get("desc", checkpoint)
            wrapper.model.load_state_dict(state_dict, strict=True)
            logger.info(f"Loaded {len(state_dict)} keys into original HoverNet model")
        else:
            logger.warning(f"TF2PT checkpoint not found: {checkpoint_path}. Using random weights!")

        wrapper.eval()
        wrapper.to(torch.device(device))
        mtime = statbuf.st_mtime if statbuf else 0
        self._networks[device] = (wrapper, mtime)

        return wrapper

    def pre_transforms(self, data=None) -> Sequence[Callable]:
        t = [
            LoadImagePatchd(keys="image", mode="RGB", dtype=np.uint8, padding=False),
            UndoLoadImagePatchTransposed(keys="image"),
        ]
        t.extend([x for x in super().pre_transforms(data)])
        return t

    def post_transforms(self, data=None) -> Sequence[Callable]:
        # Use instance_map (unique ID per nucleus) for contour extraction instead
        # of type_map (which merges adjacent same-type nuclei into one polygon).
        d = data or {}
        min_size = int(d.get("min_size", 64))
        min_hole = int(d.get("min_hole", 64))
        min_poly_area = int(d.get("min_poly_area", 30))
        max_poly_area = int(d.get("max_poly_area", 128 * 128))
        buffer_distance = float(d.get("buffer_distance", 0.5))
        marker_threshold = float(d.get("marker_threshold", 0.4))
        sobel_kernel_size = int(d.get("sobel_kernel_size", 21))

        return [
            OriginalHoVerNetPostProcessd(
                keys="pred", nr_types=5,
                sobel_kernel_size=sobel_kernel_size,
                marker_threshold=marker_threshold,
            ),
            FindContoursFromInstanceMapd(
                keys="pred",
                labels=self.labels,
                label_colors=self.label_colors,
                min_poly_area=min_poly_area,
                max_poly_area=max_poly_area,
                min_size=min_size,
                min_hole=min_hole,
            ),
            BufferContoursd(keys="pred", distance=buffer_distance),
        ]

    def inferer(self, data=None) -> Inferer:
        # Get the default SlidingWindowHoVerNetInferer from bundle config
        base = super().inferer(data)
        # Replace with our reflect-padding variant (same params)
        return ReflectPadHoVerNetInferer(
            roi_size=base.roi_size,
            sw_batch_size=base.sw_batch_size,
            overlap=base.overlap,
            mode=base.mode,
            sigma_scale=base.sigma_scale,
            padding_mode=base.padding_mode,
            cval=base.cval,
            sw_device=base.sw_device,
            device=base.device,
            progress=base.progress,
            extra_input_padding=base.extra_input_padding,
        )

    def info(self) -> Dict[str, Any]:
        d = super().info()
        d["pathology"] = True
        d["description"] = "HoVerNet Nuclei Segmentation (5x5 Original Mode)"
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
