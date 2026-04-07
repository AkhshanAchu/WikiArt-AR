import torch
from typing import List


def uncertainty_loss(losses: List[torch.Tensor],
                     log_vars: torch.Tensor) -> torch.Tensor:
    # Unbind into separate scalar tensors to avoid in-place indexing
    # on the parameter during backward (was causing autograd version errors)
    log_vars_split = log_vars.unbind()
    total = sum(
        torch.exp(-log_vars_split[i]) * l + log_vars_split[i]
        for i, l in enumerate(losses)
    )
    return total
