import ast
import torch
from pathlib import Path
from PIL import Image
from torch.utils.data import Dataset
import pandas as pd

from .transforms import get_transform

import ast
import pandas as pd
from pathlib import Path
from PIL import Image
import torch
from torch.utils.data import Dataset

import ast
import pandas as pd
from pathlib import Path
from PIL import Image

import torch
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
import torchvision.transforms as T

from torchvision.io import read_image




# ─────────────────────────────────────────────
# Meta-genre clustering (27 styles → 9 groups)
# ─────────────────────────────────────────────
STYLE_TO_METAGENRE = {
    "Early Renaissance":          "Renaissance",
    "High Renaissance":           "Renaissance",
    "Northern Renaissance":       "Renaissance",
    "Mannerism Late Renaissance": "Renaissance",
    "Baroque":                    "Baroque_Rococo",
    "Rococo":                     "Baroque_Rococo",
    "Romanticism":                "Romantic_Realist",
    "Realism":                    "Romantic_Realist",
    "Contemporary Realism":       "Romantic_Realist",
    "Impressionism":              "Impressionist",
    "Post Impressionism":         "Impressionist",
    "Pointillism":                "Impressionist",
    "Expressionism":              "Expressionist",
    "Abstract Expressionism":     "Expressionist",
    "Action painting":            "Expressionist",
    "Color Field Painting":       "Expressionist",
    "Cubism":                     "Cubist",
    "Analytical Cubism":          "Cubist",
    "Synthetic Cubism":           "Cubist",
    "Art Nouveau Modern":         "Modern_AvantGarde",
    "Fauvism":                    "Modern_AvantGarde",
    "Symbolism":                  "Modern_AvantGarde",
    "Naive Art Primitivism":      "Modern_AvantGarde",
    "Minimalism":                 "Contemporary",
    "New Realism":                "Contemporary",
    "Pop Art":                    "Contemporary",
    "Ukiyo e":                    "NonWestern",
}

METAGENRE_TO_IDX = {
    m: i for i, m in enumerate(sorted(set(STYLE_TO_METAGENRE.values())))
}
# {'Baroque_Rococo':0, 'Contemporary':1, 'Cubist':2, 'Expressionist':3,
#  'Impressionist':4, 'Modern_AvantGarde':5, 'NonWestern':6,
#  'Renaissance':7, 'Romantic_Realist':8}

# ─────────────────────────────────────────────
# Dataset
# ─────────────────────────────────────────────
class WikiArtDataset(Dataset):
    def __init__(
        self,
        root: str,
        csv_path: str,
        split: str = "train",
        style_to_idx: dict = None,
        artist_to_idx: dict = None,
    ):
        self.root  = Path(root)
        self.split = split

        df = pd.read_csv(csv_path)
        df = df[df["subset"] == split].reset_index(drop=True)

        # Style from folder prefix
        df["style"] = df["filename"].apply(lambda x: x.split("/")[0])

        # Normalise style name: "Abstract_Expressionism" → "Abstract Expressionism"
        df["style"] = df["style"].str.replace("_", " ")

        # Map style → metagenre
        df["metagenre"] = df["style"].map(STYLE_TO_METAGENRE)

        # Drop rows whose style isn't in our mapping (shouldn't happen, but safe)
        before = len(df)
        df = df.dropna(subset=["metagenre"]).reset_index(drop=True)
        if len(df) < before:
            print(f"[{split}] Dropped {before - len(df)} rows with unknown style")

        # Build label maps — reuse train maps for val/test
        self.style_to_idx = style_to_idx or {
            s: i for i, s in enumerate(sorted(df["style"].unique()))
        }
        self.artist_to_idx = artist_to_idx or {
            a: i for i, a in enumerate(sorted(df["artist"].unique()))
        }
        self.metagenre_to_idx = METAGENRE_TO_IDX   # fixed global map

        # Build sample list, skip missing files or unseen labels
        self.samples = []
        skipped = 0
        for _, row in df.iterrows():
            path   = self.root / row["filename"]
            style  = row["style"]
            artist = row["artist"]
            mg     = row["metagenre"]

            if not path.exists():
                skipped += 1
                continue
            if style not in self.style_to_idx or artist not in self.artist_to_idx:
                skipped += 1
                continue

            self.samples.append((
                path,
                self.style_to_idx[style],           # label 0: style  (27 classes)
                self.metagenre_to_idx[mg],          # label 1: metagenre (9 classes)
                self.artist_to_idx[artist],         # label 2: artist
            ))

        if skipped:
            print(f"[{split}] Skipped {skipped} samples (missing file / unknown label)")
        print(f"[{split}] {len(self.samples)} samples | "
              f"{self.num_styles} styles | "
              f"{self.num_metagenres} metagenres | "
              f"{self.num_artists} artists")

        self.style_labels    = [s[1] for s in self.samples]
        self.metagenre_labels= [s[2] for s in self.samples]
        self.artist_labels   = [s[3] for s in self.samples]

        self.transform = get_transform(split)

    # ------------------------------------------------------------------
    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        path, style_idx, mg_idx, artist_idx = self.samples[idx]
        img = Image.open(path).convert("RGB")

        return self.transform(img), style_idx, mg_idx, artist_idx

    # ------------------------------------------------------------------
    def class_weights(self, labels: list, num_classes: int) -> torch.Tensor:
        """Inverse-frequency weights normalised to sum → num_classes."""
        num_classes = max(num_classes, max(labels) + 1)
        counts = torch.zeros(num_classes)
        for lbl in labels:
            counts[lbl] += 1
        counts  = counts.clamp(min=1)
        weights = 1.0 / counts
        return weights / weights.sum() * num_classes

    def sample_weights(self, labels: list, num_classes: int) -> torch.Tensor:
        """Per-sample weight for WeightedRandomSampler (based on style by default)."""
        cls_w = self.class_weights(labels, num_classes)
        return torch.tensor([cls_w[lbl].item() for lbl in labels])

    # ------------------------------------------------------------------
    @property
    def num_styles(self):     return len(self.style_to_idx)
    @property
    def num_metagenres(self): return len(self.metagenre_to_idx)
    @property
    def num_artists(self):    return len(self.artist_to_idx)


# ─────────────────────────────────────────────
# DataLoader factory
# ─────────────────────────────────────────────
def get_dataloaders(
    root: str,
    csv_path: str,
    batch_size: int = 32,
    num_workers: int = 4,
    weighted_sampling: bool = True,   # balance style classes during training
):
    # Train dataset — builds the label maps
    train_ds = WikiArtDataset(root, csv_path, split="train")

    # Val/test share train's label maps so indices are consistent
    val_ds = WikiArtDataset(
        root, csv_path, split="test",
        style_to_idx=train_ds.style_to_idx,
        artist_to_idx=train_ds.artist_to_idx,
    )

    # ── Sampler ──────────────────────────────
    if weighted_sampling:
        sample_w = train_ds.sample_weights(train_ds.style_labels, train_ds.num_styles)
        sampler  = WeightedRandomSampler(
            weights     = sample_w,
            num_samples = len(sample_w),
            replacement = True,
        )
        train_shuffle = False   # sampler and shuffle are mutually exclusive
    else:
        sampler       = None
        train_shuffle = True

    # ── Loaders ──────────────────────────────
    common = dict(
        batch_size  = batch_size,
        num_workers = num_workers,
        pin_memory  = True,
        persistent_workers = num_workers > 0,
        prefetch_factor=2
    )

    train_loader = DataLoader(
        train_ds,
        sampler = sampler,
        shuffle = train_shuffle,
        drop_last = True,          # keeps batch size stable for BN layers
        **common,
    )

    val_loader = DataLoader(
        val_ds,
        shuffle   = False,
        drop_last = False,
        **common,
    )

    return train_loader, val_loader, train_ds, val_ds
