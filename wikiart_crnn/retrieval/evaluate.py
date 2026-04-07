import torch
import geoopt
from .hyperbolic_retrieval import hyperbolic_retrieve


@torch.no_grad()
def compute_map(model, loader, ball: geoopt.PoincareBall,
                device, top_k: int = 5) -> float:
    model.eval()
    all_embs, all_labels = [], []

    for imgs, _, _, a_lbl in loader:
        emb = model.encode(imgs.to(device))
        all_embs.append(emb.cpu())
        all_labels.extend(a_lbl.tolist())

    all_embs   = torch.cat(all_embs, dim=0).to(device)
    all_labels = torch.tensor(all_labels, device=device)
    N          = all_embs.shape[0]

    ap_sum = 0.0
    for i in range(N):
        top_idx = hyperbolic_retrieve(all_embs[i], all_embs, ball, top_k=top_k + 1)
        top_idx = top_idx[top_idx != i][:top_k]
        correct = (all_labels[top_idx] == all_labels[i]).float()
        prec_at = torch.cumsum(correct, dim=0) / (torch.arange(top_k, device=device) + 1)
        ap_sum += (prec_at * correct).sum().item() / correct.sum().clamp(min=1).item()

    return ap_sum / N
