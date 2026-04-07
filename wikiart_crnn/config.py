import torch

try:
    from mamba_ssm import Mamba as _Mamba
    USE_MAMBA = True
    Mamba = _Mamba
except ImportError:
    USE_MAMBA = False
    Mamba = None
    print("[WARN] mamba-ssm not found — falling back to multi-head attention")

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

NUM_STYLES  = 27
NUM_GENRES  = 9
NUM_ARTISTS = 1114
EMBED_DIM   = 256
