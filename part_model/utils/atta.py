"""Module for handling transforms and saving/loading perturbation."""

from typing import Any

import torch
import torchvision
import torchvision.transforms.functional as T
from torchvision.transforms.functional import InterpolationMode


class ATTA:
    """Implementation of ATTA.

    NOTE: this implementation is slightly different from the original ATTA.
    We only keep perturbation from one epoch earlier and not all the previous
    epochs. This allows us to store perturbation in the dimension after
    transformed instead of original size. This should help reduce the memory
    requirement by about half.

    Reference:
        Zheng et al., "Efficient Adversarial Training with Transferable
        Adversarial Examples," CVPR 2020. https://github.com/hzzheng93/ATTA
    """

    def __init__(
        self,
        input_dim: tuple[int, int, int] = (3, 224, 224),
        num_samples: int = 1000,
        cache_dim: tuple[int, int, int] | None = None,
    ) -> None:
        # Transform order: crop, resize, flip
        # TODO: Assume params: [index, orig_size, crop, flip]
        self._input_dim: tuple[int, int, int] = input_dim
        self._cache_dim: tuple[int, int, int] = (
            cache_dim if cache_dim is not None else self._input_dim
        )
        self._num_samples: int = num_samples
        self._resize_interp = InterpolationMode.BILINEAR
        self._resize_to_cache = torchvision.transforms.Resize(
            self._cache_dim[1:], interpolation=self._resize_interp
        )
        # Perturbation cache
        self._cache = torch.zeros((self._num_samples,) + self._cache_dim)
        # Transform param cache
        self._params = [None] * num_samples

    def update(
        self, perturbation: torch.Tensor, params: list[torch.Tensor]
    ) -> None:
        """Inverse-transform perturbation and update cache.

        Args:
            perturbation: New perturbation to cache.
            params: New transform params to cache.
        """
        assert all([len(perturbation) == len(p) for p in params]), (
            "Inputs and corresponding transform params have mismatched length "
            f" ({len(perturbation)} vs {[len(p) for p in params]})!"
        )
        indices = params[0]
        if perturbation.shape[-3:] != self._cache_dim:
            perturbation = self._resize_to_cache(perturbation)
        self._cache[indices] = perturbation.cpu()

    def _invert_perturbation(self, index: int) -> torch.Tensor:
        """Invert cached perturbation at index.

        Args:
            index: Index of cached perturbation to load.

        Returns:
            Inverted cached perturbation.
        """
        pert: torch.Tensor = self._cache[index]
        param: list[Any] = self._params[index]

        # Flip
        if param[-1]:
            pert = T.hflip(pert)

        # Resize
        crop_top, crop_left, crop_height, crop_width = param[-2]
        pert = T.resize(
            pert,
            (crop_height, crop_width),
            interpolation=self._resize_interp,
        )

        # Pad to original size (left, top, right and bottom)
        orig_height, orig_width = param[-3]
        crop_right = orig_width - crop_width - crop_left
        crop_bot = orig_height - crop_height - crop_top
        pert = T.pad(pert, (crop_left, crop_top, crop_right, crop_bot))
        return pert

    def _transform_perturbation(
        self, perturbation: torch.Tensor, params: list[torch.Tensor]
    ) -> torch.Tensor:
        """Apply new transforms on original-sized perturbation.

        Args:
            perturbation: Perturbation in the original size.
            params: Transform params to apply to perturbation.

        Returns:
            Transformed perturbation.
        """
        perturbation = T.crop(perturbation, *params[-2])
        perturbation = T.resize(
            perturbation, self._input_dim[-2:], antialias=True
        )
        if params[-1]:
            perturbation = T.hflip(perturbation)
        return perturbation

    def apply(
        self, inpt: torch.Tensor, params: list[torch.Tensor]
    ) -> torch.Tensor:
        """Apply cached perturbation to inpt.

        Args:
            inpt: _description_
            params:

        Returns:
            _description_
        """
        assert all([len(inpt) == len(p) for p in params]), (
            "Inputs and corresponding transform params have mismatched length "
            f" ({len(inpt)} vs {[len(p) for p in params]})!"
        )
        for i, _ in enumerate(inpt):
            # First entry of params is sample index
            idx = params[0][i]
            new_params = [p[i] for p in params[1:]]
            if self._params[idx] is None:
                self._params[idx] = new_params
                continue

            # Retrieve cached perturbation and invert it to original size
            pert = self._invert_perturbation(idx)

            # Apply new transform
            pert = self._transform_perturbation(pert, new_params)

            # Apply perturbation to image
            inpt[i] += pert

            # Cache new transform params for next epoch. This must come after
            # self._invert_perturbation(idx) since it relies on cached params.
            self._params[idx] = new_params

        inpt.clamp_(0, 1)
        return inpt
