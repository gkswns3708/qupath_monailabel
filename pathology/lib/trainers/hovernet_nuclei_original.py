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

from lib.trainers.hovernet_nuclei import HovernetNuclei

from monailabel.tasks.train.bundle import BundleConstants

logger = logging.getLogger(__name__)


class HovernetNucleiOriginal(HovernetNuclei):
    """
    HoverNet trainer for 'original' mode (5x5 decoder convolutions,
    patch_size=270, out_size=80).
    """

    def __init__(self, path: str, conf: Dict[str, str], const: Optional[BundleConstants] = None):
        super().__init__(path, conf, const)
        # PatchExtractor step_size matches out_size for original mode
        self.step_size = (80, 80)

    def config(self):
        # Skip HovernetNuclei.config() filtering (which removes 5x5),
        # go directly to BundleTrainTask.config() and apply our own filter.
        from monailabel.tasks.train.bundle import BundleTrainTask

        c = BundleTrainTask.config(self)
        c["learning_rate"] = 0.0001
        c["output_filename"] = ""
        if "model_filename" in c and isinstance(c["model_filename"], list):
            c["model_filename"] = [
                f for f in c["model_filename"]
                if f != "model.pt"
                and "3x3" not in f
            ]
        return c

    def _final_filename(self, request):
        """Determine the final checkpoint filename as {base}_{tag}.pt."""
        tag = request.get("output_filename", "").strip().replace(".pt", "") or "trained"

        model_filename = request.get("model_filename")
        if isinstance(model_filename, list):
            model_filename = model_filename[0] if model_filename else None

        if model_filename:
            name = model_filename.replace(".pt", "")
            base = self._extract_base(name, "5x5")
            return f"{base}_{tag}.pt"

        return f"5x5_HoverNet_{tag}.pt"

    def _resolve_pretrained(self, request):
        """Resolve pretrained checkpoint, defaulting to OFFICIAL original-mode weights."""
        if not request.get("pretrained", True):
            logger.info("pretrained=False: using random initialization.")
            return None

        model_filename = request.get("model_filename")
        if model_filename:
            if isinstance(model_filename, list):
                model_filename = model_filename[0]
            candidate = os.path.join(self.bundle_path, "models", model_filename)
            if os.path.exists(candidate):
                logger.info(f"Using user-selected checkpoint for stage0: {candidate}")
                return candidate

        # Default: pretrained original-mode checkpoint (CoNSeP_HoverNet_5x5.pt)
        official = os.path.join(self.bundle_path, "models", "CoNSeP_HoverNet_5x5.pt")
        if os.path.exists(official):
            logger.info(f"Using pretrained original-mode checkpoint: {official}")
            return official

        # Fallback: original_stage0 directory
        fallback = os.path.join(self.bundle_path, "models", "original_stage0", "model.pt")
        return fallback if os.path.exists(fallback) else None

    def run_single_gpu(self, request, overrides):
        lr = request.get("learning_rate")
        if lr is not None:
            overrides["learning_rate"] = float(lr)

        # Set original mode parameters
        overrides["hovernet_mode"] = "original"
        overrides["patch_size"] = 270
        overrides["out_size"] = 80
        # Stage0 uses separate dir; stage1 saves to models/ with 5x5_ prefix
        overrides["ckpt_dir_stage0"] = os.path.join(self.bundle_path, "models", "original_stage0")

        logger.info("+++++++++++ Running STAGE 0 (original mode).........................")
        overrides["stage"] = 0
        overrides["network_def#freeze_encoder"] = True
        pretrained = self._resolve_pretrained(request)
        if pretrained:
            overrides["network_def#pretrained_url"] = pathlib.Path(pretrained).as_uri()
        # Call BundleTrainTask.run_single_gpu (skip HovernetNuclei's override)
        super(HovernetNuclei, self).run_single_gpu(request, overrides)

        logger.info("+++++++++++ Running STAGE 1 (original mode).........................")
        overrides["stage"] = 1
        overrides["network_def#freeze_encoder"] = False
        overrides["network_def#pretrained_url"] = None
        filename = self._final_filename(request)
        overrides["ckpt_final_filename"] = filename
        logger.info(f"Stage 1 final model will be saved as: {filename}")
        super(HovernetNuclei, self).run_single_gpu(request, overrides)

    def run_multi_gpu(self, request, cmd, env):
        lr = request.get("learning_rate")

        logger.info("+++++++++++ Running STAGE 0 (original mode).........................")
        cmd1 = copy.deepcopy(cmd)
        cmd1.extend([
            "--hovernet_mode", "original",
            "--patch_size", "270",
            "--out_size", "80",
            "--ckpt_dir_stage0", os.path.join(self.bundle_path, "models", "original_stage0"),
            "--stage", "0",
            "--network_def#freeze_encoder", "true",
        ])
        if lr is not None:
            cmd1.extend(["--learning_rate", str(float(lr))])
        pretrained = self._resolve_pretrained(request)
        if pretrained:
            cmd1.extend(["--network_def#pretrained_url", pathlib.Path(pretrained).as_uri()])
        super(HovernetNuclei, self).run_multi_gpu(request, cmd1, env)

        logger.info("+++++++++++ Running STAGE 1 (original mode).........................")
        cmd2 = copy.deepcopy(cmd)
        cmd2.extend([
            "--hovernet_mode", "original",
            "--patch_size", "270",
            "--out_size", "80",
            "--ckpt_dir_stage0", os.path.join(self.bundle_path, "models", "original_stage0"),
            "--stage", "1",
            "--network_def#freeze_encoder", "false",
            "--network_def#pretrained_url", "None",
        ])
        if lr is not None:
            cmd2.extend(["--learning_rate", str(float(lr))])
        filename = self._final_filename(request)
        cmd2.extend(["--ckpt_final_filename", filename])
        logger.info(f"Stage 1 final model will be saved as: {filename}")
        super(HovernetNuclei, self).run_multi_gpu(request, cmd2, env)
