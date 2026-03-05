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
import pathlib

import numpy as np
import openslide
import torch
from monai.config import KeysCollection
from monai.data import MetaTensor
from monai.transforms import MapTransform
from monai.utils import PostFix, convert_to_numpy, ensure_tuple
from PIL import Image
from scipy.ndimage import binary_fill_holes
from skimage.morphology import remove_small_holes, remove_small_objects

logger = logging.getLogger(__name__)


class LoadImagePatchd(MapTransform):
    def __init__(self, keys: KeysCollection, mode="RGB", dtype=np.uint8, padding=True):
        super().__init__(keys)
        self.mode = mode
        self.dtype = dtype
        self.padding = padding

    def __call__(self, data):
        d = dict(data)
        for key in self.keys:
            size = None
            tile_size = None

            if not isinstance(d[key], str):
                image_np = d[key]
            else:
                name = d[key]
                ext = pathlib.Path(name).suffix
                if ext == ".npy":
                    image_np = np.load(d[key])
                else:
                    location = d.get("location", (0, 0))
                    level = d.get("level", 0)
                    size = d.get("size", None)

                    # Model input size
                    tile_size = d.get("tile_size", size)
                    if not ext or ext in (
                        ".bif",
                        ".mrxs",
                        ".ndpi",
                        ".scn",
                        ".svs",
                        ".svslide",
                        ".tif",
                        ".tiff",
                        ".vms",
                        ".vmu",
                    ):
                        slide = openslide.OpenSlide(name)
                        size = size if size else slide.dimensions
                        img = slide.read_region(location, level, size)
                    else:
                        img = Image.open(d[key])
                        d["location"] = [0, 0]
                        d["size"] = [0, 0]

                    img = img.convert(self.mode) if self.mode else img
                    image_np = np.array(img, dtype=self.dtype)

            image_np = np.moveaxis(image_np, 0, 1)
            meta_dict_key = f"{key}_{PostFix.meta()}"
            meta_dict = d.get(meta_dict_key)
            if meta_dict is None:
                d[meta_dict_key] = dict()
                meta_dict = d.get(meta_dict_key)

            meta_dict["spatial_shape"] = np.asarray(image_np.shape[:-1])
            meta_dict["original_channel_dim"] = -1
            meta_dict["original_affine"] = None  # type: ignore
            logger.debug(f"Image shape: {image_np.shape} vs size: {size} vs tile_size: {tile_size}")

            if self.padding and tile_size and (image_np.shape[0] != tile_size[0] or image_np.shape[1] != tile_size[1]):
                image_np = self.pad_to_shape(image_np, tile_size)
            d[key] = MetaTensor(image_np, meta=meta_dict, device=d.get("device"))
        return d

    @staticmethod
    def pad_to_shape(img, shape):
        img_shape = img.shape[:-1]
        s_diff = np.array(shape) - np.array(img_shape)
        diff = [(0, s_diff[0]), (0, s_diff[1]), (0, 0)]
        return np.pad(
            img,
            diff,
            mode="constant",
            constant_values=0,
        )


class PostFilterLabeld(MapTransform):
    def __init__(self, keys: KeysCollection, min_size=64, min_hole=64):
        super().__init__(keys)
        self.min_size = min_size
        self.min_hole = min_hole

    def __call__(self, data):
        d = dict(data)
        for key in self.keys:
            label = convert_to_numpy(d[key]) if isinstance(d[key], torch.Tensor) else d[key]
            label = label.astype(np.uint8)
            if self.min_hole:
                label = remove_small_holes(label, area_threshold=self.min_hole)
            label = binary_fill_holes(label).astype(np.uint8)
            if self.min_size:
                label = remove_small_objects(label, min_size=self.min_size)

            d[key] = np.where(label > 0, d[key], 0)
        return d


class BufferContoursd(MapTransform):
    """Expand polygon contours by a given distance to compensate for
    the half-pixel inset produced by cv2.findContours().

    Operates on data["result"]["annotation"]["elements"][*]["contours"],
    the structure produced by FindContoursd.
    """

    def __init__(self, keys: KeysCollection, distance: float = 0.5, result="result", result_output_key="annotation"):
        super().__init__(keys)
        self.distance = distance
        self.result = result
        self.result_output_key = result_output_key

    def __call__(self, data):
        d = dict(data)
        annotation = (d.get(self.result) or {}).get(self.result_output_key)
        if not annotation:
            return d
        for element in annotation.get("elements", []):
            expanded = []
            for contour in element.get("contours", []):
                expanded.extend(self._buffer_contour(contour))
            element["contours"] = expanded
        return d

    def _buffer_contour(self, contour):
        from shapely.geometry import MultiPolygon, Polygon

        if len(contour) < 3:
            return [contour]
        try:
            poly = Polygon(contour)
            if not poly.is_valid:
                poly = poly.buffer(0)
            if poly.is_empty:
                return [contour]
            buffered = poly.buffer(self.distance, join_style=2)  # mitre
            if buffered.is_empty:
                return [contour]
            geoms = buffered.geoms if isinstance(buffered, MultiPolygon) else [buffered]
            return [[[c[0], c[1]] for c in g.exterior.coords[:-1]] for g in geoms]
        except Exception:
            return [contour]


class FindContoursFromInstanceMapd(MapTransform):
    """Extract per-nucleus contours from instance_map instead of type_map.

    Unlike FindContoursd (which creates binary masks per type and merges adjacent
    same-type nuclei into one polygon), this extracts a separate contour for each
    nucleus instance, preserving boundaries between touching cells.

    Reads:
        d["instance_map"]  — [1, H, W] integer array, unique ID per nucleus
        d["instance_info"] — {inst_id: {"type": int, ...}} from post-processing
        d["location"]      — [x, y] WSI offset
        d["size"]          — [w, h] patch size

    Writes:
        d[result][result_output_key] — same format as FindContoursd output
    """

    def __init__(
        self,
        keys: KeysCollection,
        labels: dict,
        label_colors: dict = None,
        min_poly_area: int = 30,
        max_poly_area: int = 0,
        result: str = "result",
        result_output_key: str = "annotation",
    ):
        super().__init__(keys)
        self.labels = labels
        self.label_colors = label_colors or {}
        self.min_poly_area = min_poly_area
        self.max_poly_area = max_poly_area
        self.result = result
        self.result_output_key = result_output_key
        self._type_to_label = {v: k for k, v in labels.items()}

    def __call__(self, data):
        import cv2

        d = dict(data)
        inst_map = d.get("instance_map")
        inst_info = d.get("instance_info", {})
        location = d.get("location", [0, 0])
        size = d.get("size", [0, 0])

        if inst_map is None:
            return d

        if isinstance(inst_map, (torch.Tensor, MetaTensor)):
            inst_map = convert_to_numpy(inst_map)
        inst_map = np.squeeze(inst_map)

        label_contours = {}

        for inst_id in np.unique(inst_map):
            if inst_id == 0:
                continue

            info = inst_info.get(inst_id, {})
            type_id = info.get("type", 0)
            if type_id == 0:
                continue

            label_name = self._type_to_label.get(type_id)
            if label_name is None:
                continue

            mask = (inst_map == inst_id).astype(np.uint8)
            contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

            for contour in contours:
                contour = np.squeeze(contour, axis=1)
                if contour.ndim != 2 or len(contour) < 3:
                    continue

                area = cv2.contourArea(contour)
                if area < self.min_poly_area:
                    continue
                if self.max_poly_area > 0 and area > self.max_poly_area:
                    continue

                contour[:, 0] += location[0]
                contour[:, 1] += location[1]

                label_contours.setdefault(label_name, []).append(
                    contour.astype(int).tolist()
                )

        elements = []
        labels_out = {}
        for label_name in self.labels:
            if label_name not in label_contours:
                continue
            elements.append({"label": label_name, "contours": label_contours[label_name]})
            if label_name in self.label_colors:
                labels_out[label_name] = self.label_colors[label_name]

        annotation = {
            "location": location,
            "size": size,
            "elements": elements,
            "labels": labels_out,
        }

        r = d.get(self.result, {})
        r[self.result_output_key] = annotation
        d[self.result] = r

        return d


class ConvertInteractiveClickSignals(MapTransform):
    """
    ConvertInteractiveClickSignals converts interactive annotation information (e.g. from DSA) into a format expected
    by NuClick. Typically, it will take point annotations from data["annotations"][<source_annotation_key>], convert
    it to 2d points, and place it in data[<target_data_key>].
    """

    def __init__(
        self, source_annotation_keys: KeysCollection, target_data_keys: KeysCollection, allow_missing_keys: bool = False
    ):
        super().__init__(target_data_keys, allow_missing_keys)
        self.source_annotation_keys = ensure_tuple(source_annotation_keys)
        self.target_data_keys = ensure_tuple(target_data_keys)

    def __call__(self, data):
        data = dict(data)
        annotations = data.get("annotations", {})
        annotations = {} if annotations is None else annotations

        logger.info(f"Annotations: {annotations.keys()}")
        for source_annotation_key, target_data_key in zip(self.source_annotation_keys, self.target_data_keys):
            if source_annotation_key in annotations:
                points = annotations.get(source_annotation_key)["points"]
                logger.info(f"Nuclick points={points}")
                points = [coords[0:2] for coords in points]
                data[target_data_key] = points
            elif not self.allow_missing_keys:
                raise KeyError(
                    f"source_annotation_key={source_annotation_key} not found in annotation keys={annotations.keys()}"
                )
        return data
