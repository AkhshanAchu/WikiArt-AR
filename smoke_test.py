import torch
import torch.nn as nn

from wikiart_crnn.config import DEVICE, NUM_ARTISTS
from wikiart_crnn.models import WikiArtCRNNMamba
from wikiart_crnn.losses import SupConLoss, uncertainty_loss


def smoke_test():
    print("Running smoke test (no data required)...")
    model = WikiArtCRNNMamba(pretrained=False).to(DEVICE)
    model.eval()

    dummy = torch.randn(4, 3, 224, 224).to(DEVICE)
    with torch.no_grad():
        out = model(dummy)

    print(f"  embedding      : {out['embedding'].shape}")
    print(f"  hyp_embedding  : {out['hyp_embedding'].shape}")
    print(f"  style logits   : {out['style'].shape}")
    print(f"  genre logits   : {out['genre'].shape}")
    print(f"  artist logits  : {out['artist'].shape}")
    print(f"  adversarial    : {out['adversarial'].shape}")

    supcon = SupConLoss()
    labels = torch.randint(0, NUM_ARTISTS, (4,)).to(DEVICE)
    l = supcon(out["embedding"], labels)
    print(f"  SupCon loss    : {l.item():.4f}")

    losses  = [torch.tensor(0.5), torch.tensor(0.3), torch.tensor(0.8)]
    log_var = nn.Parameter(torch.zeros(3))
    lw = uncertainty_loss(losses, log_var)
    print(f"  Uncertainty-weighted loss: {lw.item():.4f}")

    print("  Smoke test PASSED ✓")


if __name__ == "__main__":
    smoke_test()
