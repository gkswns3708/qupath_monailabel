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
import copy
import logging
import os
import pathlib
from typing import Dict, Optional

import numpy as np
from lib.hovernet import PatchExtractor
from lib.utils import split_dataset
from monai.utils import optional_import
from PIL import Image
from scipy.ndimage import label
from tqdm import tqdm

from monailabel.interfaces.datastore import Datastore
from monailabel.tasks.train.bundle import BundleConstants, BundleTrainTask
from monailabel.utils.others.generic import remove_file

logger = logging.getLogger(__name__)


class HovernetNuclei(BundleTrainTask):
    def __init__(self, path: str, conf: Dict[str, str], const: Optional[BundleConstants] = None):
        super().__init__(path, conf, const, enable_tracking=True)
        self.tile_size = (1024, 1024)
        self.patch_size = (540, 540)
        self.step_size = (164, 164)
        self.extract_type = "mirror"

    def remove_file(path):
        if os.path.exists(path):
            os.remove(path)

    def _fetch_datalist(self, request, datastore):
        cache_dir = os.path.join(self.bundle_path, "cache", "train_ds")
        remove_file(cache_dir)

        source = request.get("dataset_source")
        max_region = request.get("dataset_max_region", (10240, 10240))
        max_region = (max_region, max_region) if isinstance(max_region, int) else max_region[:2]

        # HoVer-Net 4-class mapping (ASAP XML PartOfGroup → class ID)
        groups = {
            "other": 1,
            "inflammatory": 2,
            "epithelial": 3,
            "spindle-shaped": 4,
        }

        ds = split_dataset(
            datastore=datastore,
            cache_dir=cache_dir,
            source=source,
            groups=groups,
            tile_size=self.tile_size,
            max_region=max_region,
            limit=request.get("dataset_limit", 0),
            randomize=request.get("dataset_randomize", True),
        )
        logger.info(f"Split data (len: {len(ds)}) based on each nuclei")

        limit = request.get("dataset_limit", 0)
        ds_new: list = []
        xtractor = PatchExtractor(self.patch_size, self.step_size)
        out_dir = os.path.join(cache_dir, "nuclei_hovernet")
        os.makedirs(out_dir, exist_ok=True)

        for d in tqdm(ds):
            if 0 < limit < len(ds_new):
                ds_new = ds_new[:limit]
                break

            base_name = pathlib.Path(d["image"]).stem
            img = np.array(Image.open(d["image"]).convert("RGB"))
            ann_type = np.array(Image.open(d["label"]))

            cv2, has_cv2 = optional_import("cv2")
            if has_cv2:
                numLabels, ann_inst, _, _ = cv2.connectedComponentsWithStats(ann_type, 4, cv2.CV_32S)
            else:
                ann_inst, numLabels = label(ann_type)

            ann = np.dstack([ann_inst, ann_type])

            img = np.concatenate([img, ann], axis=-1)
            sub_patches = xtractor.extract(img, self.extract_type)

            pbar_format = "Extracting: |{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}]"
            pbar = tqdm(total=len(sub_patches), leave=False, bar_format=pbar_format, ascii=True, position=1)

            for idx, patch in enumerate(sub_patches):
                image_patch = patch[..., :3]
                inst_map_patch = patch[..., 3:4]
                type_map_patch = patch[..., 4:5]

                i = f"{out_dir}/{base_name}_{idx:03d}_image.npy"
                j = f"{out_dir}/{base_name}_{idx:03d}_inst_map.npy"
                k = f"{out_dir}/{base_name}_{idx:03d}_type_map.npy"

                np.save(i, image_patch)
                np.save(j, inst_map_patch)
                np.save(k, type_map_patch)
                ds_new.append({"image": i, "label_inst": j, "label_type": k})
                pbar.update()
            pbar.close()

            if 0 < limit < len(ds_new):
                ds_new = ds_new[:limit]
                break

        logger.info(f"Final Records with hovernet patches: {len(ds_new)}")
        return ds_new

    def _load_checkpoint(self, output_dir, pretrained, train_handlers):
        pass

    def _resolve_pretrained(self, request):
        """사용자가 선택한 model_filename을 stage0 pretrained 경로로 변환."""
        if not request.get("pretrained", True):
            logger.info("pretrained=False: skipping all checkpoints, using random initialization.")
            return None

        model_filename = request.get("model_filename")
        if model_filename:
            if isinstance(model_filename, list):
                model_filename = model_filename[0]
            candidate = os.path.join(self.bundle_path, "models", model_filename)
            if os.path.exists(candidate):
                logger.info(f"Using user-selected checkpoint for stage0: {candidate}")
                return candidate
        # 기본값: stage0 전용 checkpoint
        fallback = os.path.join(self.bundle_path, "models", "stage0", "model.pt")
        return fallback if os.path.exists(fallback) else None

    def _final_filename(self, request):
        """Determine the final checkpoint filename with _3x3 suffix."""
        model_name = request.get("model_name")
        if model_name:
            if not model_name.endswith(".pt"):
                model_name = f"{model_name}.pt"
            # Ensure _3x3 suffix for fast mode
            if not model_name.endswith("_3x3.pt"):
                model_name = model_name.replace(".pt", "_3x3.pt")
            return model_name
        return "HoverNet_3x3.pt"

    def run_single_gpu(self, request, overrides):
        logger.info("+++++++++++ Running STAGE 0.........................")
        overrides["stage"] = 0
        overrides["network_def#freeze_encoder"] = True
        pretrained = self._resolve_pretrained(request)
        if pretrained:
            overrides["network_def#pretrained_url"] = pathlib.Path(pretrained).as_uri()
        super().run_single_gpu(request, overrides)

        logger.info("+++++++++++ Running STAGE 1.........................")
        overrides["stage"] = 1
        overrides["network_def#freeze_encoder"] = False
        overrides["network_def#pretrained_url"] = None
        filename = self._final_filename(request)
        overrides["train#handlers[2]#final_filename"] = filename
        logger.info(f"Stage 1 final model will be saved as: {filename}")
        super().run_single_gpu(request, overrides)

    def run_multi_gpu(self, request, cmd, env):
        logger.info("+++++++++++ Running STAGE 0.........................")
        cmd1 = copy.deepcopy(cmd)
        cmd1.extend(["--stage", "0", "--network_def#freeze_encoder", "true"])
        pretrained = self._resolve_pretrained(request)
        if pretrained:
            cmd1.extend(["--network_def#pretrained_url", pathlib.Path(pretrained).as_uri()])
        super().run_multi_gpu(request, cmd1, env)

        logger.info("+++++++++++ Running STAGE 1.........................")
        cmd2 = copy.deepcopy(cmd)
        cmd2.extend(["--stage", "1", "--network_def#freeze_encoder", "false"])
        cmd2.extend(["--network_def#pretrained_url", "None"])
        filename = self._final_filename(request)
        cmd2.extend(["--train#handlers[2]#final_filename", filename])
        logger.info(f"Stage 1 final model will be saved as: {filename}")
        super().run_multi_gpu(request, cmd2, env)

    def __call__(self, request, datastore: Datastore):
        request["force_multi_gpu"] = True
        return super().__call__(request, datastore)
