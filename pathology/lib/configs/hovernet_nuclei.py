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

        # Original-mode checkpoints (*_5x5.pt) are incompatible with the
        # fast-mode network.  They are handled by hovernet_nuclei_original.
        checkpoints = sorted(
            [
                p for p in _glob.glob(os.path.join(models_dir, "*.pt"))
                if not os.path.basename(p).endswith("_5x5.pt")
            ],
            key=os.path.getmtime,
        )

        if not checkpoints:
            return lib.infers.HovernetNuclei(self.bundle_path, self.conf)

        tasks: Dict[str, InferTask] = {}
        for cp_path in checkpoints:
            cp_name = os.path.basename(cp_path)   # e.g. "model_epoch=10.pt"
            cp_base = cp_name[:-3]                 # remove ".pt"

            # Keep the legacy "hovernet_nuclei" key for model.pt so existing
            # QuPath sessions / training configs are not broken.
            key = "hovernet_nuclei" if cp_name == "model.pt" else f"hovernet_nuclei__{cp_base}"

            tasks[key] = lib.infers.HovernetNuclei(
                self.bundle_path, self.conf, preset_checkpoint=cp_name
            )
            logger.info(f"Registered infer task '{key}' → {cp_name}")

        return tasks

    def trainer(self) -> Optional[TrainTask]:
        task: TrainTask = lib.trainers.HovernetNuclei(self.bundle_path, self.conf)
        return task
