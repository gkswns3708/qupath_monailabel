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

from __future__ import annotations

from collections.abc import Sequence
from typing import Any, Callable

import numpy as np
import torch
import torch.nn.functional as F

from monai.apps.pathology.inferers import SlidingWindowHoVerNetInferer


class ReflectPadHoVerNetInferer(SlidingWindowHoVerNetInferer):
    """
    SlidingWindowHoVerNetInferer that uses reflect padding for extra_input_padding
    instead of constant (zero) padding.

    The parent class uses self.padding_mode for both:
      1. extra_input_padding (image boundary padding) - reflect is better here
      2. process_output (padding 80x80 output to 270x270 window) - must be constant
         because padding_size (95) > input_size (80), and reflect requires padding < input

    This subclass applies reflect padding for (1) and keeps constant for (2).
    """

    def __call__(
        self,
        inputs: torch.Tensor,
        network: Callable[..., torch.Tensor | Sequence[torch.Tensor] | dict[Any, torch.Tensor]],
        *args: Any,
        **kwargs: Any,
    ) -> torch.Tensor | tuple[torch.Tensor, ...] | dict[Any, torch.Tensor]:
        if self.extra_input_padding:
            image_size_original = inputs.shape[2:]
            num_spatial_dims = len(image_size_original)

            # Apply reflect padding for image boundaries (matches original HoverNet)
            inputs = F.pad(inputs, pad=tuple(self.extra_input_padding), mode="reflect")

            # Temporarily disable extra_input_padding so parent doesn't pad again
            saved_padding = self.extra_input_padding
            self.extra_input_padding = None

            results = super().__call__(inputs, network, *args, **kwargs)

            self.extra_input_padding = saved_padding

            # Crop results back to original size (same logic as parent)
            extra_slicing: list[slice] = []
            num_padded_dims = len(saved_padding) // 2
            for sp in range(num_padded_dims):
                slice_dim = slice(
                    saved_padding[sp * 2],
                    image_size_original[num_spatial_dims - sp - 1] + saved_padding[sp * 2],
                )
                extra_slicing.insert(0, slice_dim)
            for _ in range(len(inputs.shape) - num_padded_dims):
                extra_slicing.insert(0, slice(None))

            if isinstance(results, dict):
                for k, v in results.items():
                    results[k] = v[extra_slicing]
            elif isinstance(results, (list, tuple)):
                results = type(results)([res[extra_slicing] for res in results])
            elif isinstance(results, (torch.Tensor, np.ndarray)):
                results = results[extra_slicing]

            return results

        return super().__call__(inputs, network, *args, **kwargs)
