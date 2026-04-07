import torch
import geoopt


@torch.no_grad()
def hyperbolic_retrieve(
    query_emb: torch.Tensor,
    gallery_embs: torch.Tensor,
    ball: geoopt.PoincareBall,
    top_k: int = 5,
) -> torch.Tensor:
    dists = ball.dist(query_emb.unsqueeze(0), gallery_embs)
    return torch.topk(-dists, k=top_k).indices
