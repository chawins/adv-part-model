"""Define commonly used types."""

import torch
from jaxtyping import Float

BatchImages = Float[torch.Tensor, "batch channels height width"]
Logits = Float[torch.Tensor, "batch classes"]
