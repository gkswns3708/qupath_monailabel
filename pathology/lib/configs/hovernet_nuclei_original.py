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
from typing import Any, Dict, Optional, Union

import lib.infers
import lib.trainers
from monai.bundle import download

from monailabel.config import settings
from monailabel.interfaces.config import TaskConfig
from monailabel.interfaces.tasks.infer_v2 import InferTask
from monailabel.interfaces.tasks.train import TrainTask

logger = logging.getLogger(__name__)


class HovernetNucleiOriginal(TaskConfig):
    # TF2PT checkpoint: original HoverNet weights in the format matching
    # the original model architecture (not MONAI's reimplementation).
    TF2PT_CHECKPOINT = "/app/true_hovernet_checkpoint/hovernet_original_consep_type_tf2pytorch"

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

        # BundleInferTask requires model.pt to exist for initialization.
        # Create symlink if missing (actual weights come from tf2pt_checkpoint).
        models_dir = os.path.join(self.bundle_path, "models")
        model_pt = os.path.join(models_dir, "model.pt")
        if not os.path.exists(model_pt):
            all_pts = sorted(_glob.glob(os.path.join(models_dir, "*.pt")))
            if all_pts:
                os.symlink(all_pts[0], model_pt)
                logger.info(f"Created model.pt symlink → {os.path.basename(all_pts[0])}")

        tasks: Dict[str, InferTask] = {}

        # TF2PT checkpoint (original HoverNet architecture)
        tf2pt_path = self.TF2PT_CHECKPOINT
        if os.path.exists(tf2pt_path):
            tasks["hovernet_nuclei_5x5"] = lib.infers.HovernetNucleiOriginal(
                self.bundle_path, self.conf, tf2pt_checkpoint=tf2pt_path
            )
            logger.info(f"Registered 5x5 (original-mode) infer task -> {tf2pt_path}")
        else:
            logger.warning(f"TF2PT checkpoint not found: {tf2pt_path}")

        # Trained 5x5 checkpoints (MONAI HoVerNet architecture)
        for cp_path in sorted(
            [
                p for p in _glob.glob(os.path.join(models_dir, "*.pt"))
                if os.path.basename(p) != "model.pt"
                and "5x5" in os.path.basename(p)
                and "3x3" not in os.path.basename(p)
            ],
            key=os.path.getmtime,
        ):
            cp_name = os.path.basename(cp_path)
            cp_base = cp_name[:-3]  # remove ".pt"
            key = cp_base  # show checkpoint name directly in QuPath UI
            tasks[key] = lib.infers.HovernetNucleiOriginal(
                self.bundle_path, self.conf, preset_checkpoint=cp_name
            )
            logger.info(f"Registered infer task '{key}' → {cp_name}")

        if not tasks:
            return {"hovernet_nuclei_5x5": lib.infers.HovernetNucleiOriginal(
                self.bundle_path, self.conf
            )}

        return tasks

    def trainer(self) -> Optional[TrainTask]:
        task: TrainTask = lib.trainers.HovernetNucleiOriginal(self.bundle_path, self.conf)
        return task
