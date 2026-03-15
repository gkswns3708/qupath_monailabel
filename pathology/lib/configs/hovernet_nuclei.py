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

import json as _json
import logging
import os
from typing import Any, Dict, Optional, Union

import lib.infers
import lib.trainers
from monai.bundle import download

from monailabel.config import settings
from monailabel.interfaces.config import TaskConfig
from monailabel.interfaces.tasks.infer_v2 import InferTask
from monailabel.interfaces.tasks.train import TrainTask

logger = logging.getLogger(__name__)


def _is_final_checkpoint(filepath: str) -> bool:
    """Return True only for final trained checkpoints, excluding intermediates."""
    base = os.path.basename(filepath)
    if base == "model.pt":
        return False
    if "epoch=" in base:
        return False
    if "key_metric=" in base:
        return False
    return True


class HovernetNuclei(TaskConfig):
    def init(self, name: str, model_dir: str, conf: Dict[str, str], planner: Any, **kwargs):
        super().init(name, model_dir, conf, planner, **kwargs)

        bundle_name = "pathology_nuclei_segmentation_classification"
        zoo_source = conf.get("zoo_source", settings.MONAI_ZOO_SOURCE)
        version = conf.get("hovernet_nuclei", "0.2.6")

        self.bundle_path = os.path.join(self.model_dir, bundle_name)
        if not os.path.exists(self.bundle_path):
            download(name=bundle_name, version=version, bundle_dir=self.model_dir, source=zoo_source)

    def infer(self) -> Union[InferTask, Dict[str, InferTask]]:
        import glob as _glob

        models_dir = os.path.join(self.bundle_path, "models")

        # BundleInferTask requires model.pt to exist for initialization.
        # Create symlink if missing (actual weights come from preset_checkpoint).
        model_pt = os.path.join(models_dir, "model.pt")
        if not os.path.exists(model_pt):
            all_pts = sorted(_glob.glob(os.path.join(models_dir, "*.pt")))
            if all_pts:
                os.symlink(all_pts[0], model_pt)
                logger.info(f"Created model.pt symlink → {os.path.basename(all_pts[0])}")

        # Filter: exclude intermediates (epoch/key_metric checkpoints), model.pt symlink, 5x5
        checkpoints = sorted(
            [
                p for p in _glob.glob(os.path.join(models_dir, "*.pt"))
                if _is_final_checkpoint(p)
                and "5x5" not in os.path.basename(p)
            ],
            key=os.path.getmtime,
        )

        if not checkpoints:
            return {"hovernet_nuclei_3x3": lib.infers.HovernetNuclei(self.bundle_path, self.conf)}

        tasks: Dict[str, InferTask] = {}
        for cp_path in checkpoints:
            cp_name = os.path.basename(cp_path)
            cp_base = cp_name[:-3]  # remove ".pt"
            key = cp_base  # show checkpoint name directly in QuPath UI

            # Load training metadata if available
            meta_path = cp_path.replace(".pt", ".meta.json")
            training_metadata = {}
            if os.path.exists(meta_path):
                try:
                    with open(meta_path) as f:
                        training_metadata = _json.load(f)
                except Exception:
                    pass

            tasks[key] = lib.infers.HovernetNuclei(
                self.bundle_path, self.conf, preset_checkpoint=cp_name,
                training_metadata=training_metadata,
            )
            logger.info(f"Registered infer task '{key}' → {cp_name}")

        return tasks

    def trainer(self) -> Optional[TrainTask]:
        task: TrainTask = lib.trainers.HovernetNuclei(self.bundle_path, self.conf)
        return task
