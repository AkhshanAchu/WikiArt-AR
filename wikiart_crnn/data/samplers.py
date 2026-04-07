import torch
from torch.utils.data import WeightedRandomSampler


def make_weighted_sampler(labels, class_weights: torch.Tensor) -> WeightedRandomSampler:
    sample_weights = [class_weights[lbl].item() for lbl in labels]
    return WeightedRandomSampler(sample_weights, len(sample_weights))
