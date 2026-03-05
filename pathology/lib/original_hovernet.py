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

"""
Wrapper around the original HoverNet model (from hover_net repo) that
provides a MONAI-compatible interface: accepts [0,1]-normalized input and
returns a dict with MONAI key names.

This is needed because MONAI's HoVerNet reimplementation has architectural
differences (bilinear vs nearest upsampling, different padding, etc.) that
produce significantly different outputs even with identical weights.
"""

import sys

import torch
import torch.nn as nn

# Mapping from original HoverNet output keys to MONAI key names
_KEY_MAP = {
    "np": "nucleus_prediction",
    "hv": "horizontal_vertical",
    "tp": "type_prediction",
}


class OriginalHoVerNetWrapper(nn.Module):
    """
    Wraps the original HoverNet model to match MONAI's expected interface.

    - Input:  [B, 3, H, W] tensor in [0, 1] range (MONAI preprocessing)
    - Output: dict with keys "nucleus_prediction", "horizontal_vertical",
              "type_prediction" (matching MONAI's HoVerNet output format)

    Internally multiplies input by 255 because the original model's forward()
    divides by 255.
    """

    def __init__(self, nr_types=5, mode="original"):
        super().__init__()

        # Import original HoverNet from the hover_net repo
        if "/app/hover_net" not in sys.path:
            sys.path.insert(0, "/app/hover_net")
        from models.hovernet.net_desc import HoVerNet as OrigHoVerNet

        self.model = OrigHoVerNet(nr_types=nr_types, mode=mode)

    def forward(self, imgs):
        # MONAI preprocessing normalizes to [0, 1].
        # Original model does `imgs = imgs / 255.0` internally,
        # so we multiply by 255 to compensate.
        out = self.model(imgs * 255.0)

        # Map original keys ("np", "hv", "tp") to MONAI keys
        return {_KEY_MAP[k]: v for k, v in out.items()}
