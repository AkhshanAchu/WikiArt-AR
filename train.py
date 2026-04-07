import csv
import json
import logging
import time
from pathlib import Path

import geoopt
import torch
from tqdm import tqdm

from wikiart_crnn.config import DEVICE, USE_MAMBA, NUM_STYLES, NUM_GENRES, NUM_ARTISTS
from wikiart_crnn.data import WikiArtDataset, get_dataloaders, make_weighted_sampler
from wikiart_crnn.losses import SupConLoss
from wikiart_crnn.models import WikiArtCRNNMamba
from wikiart_crnn.retrieval import compute_map
from wikiart_crnn.training import train_one_epoch, validate

# ── Paths ──────────────────────────────────────────────────────────────────────
DATA_ROOT  = r"C:\Users\akhsh\Downloads\archive (5)"
CSV_PATH   = r"C:\Users\akhsh\Downloads\archive (5)\classes.csv"
LOG_DIR    = Path("logs")

# ── Hyper-params ───────────────────────────────────────────────────────────────
EPOCHS     = 30
BATCH_SIZE = 32
LR         = 3e-4
MAP_K      = 5          # top-K for mAP; computed every epoch
CKPT_PATH  = "best_model.pt"


# ── Logging setup ──────────────────────────────────────────────────────────────
def setup_logging(log_dir: Path) -> logging.Logger:
    log_dir.mkdir(parents=True, exist_ok=True)
    logger = logging.getLogger("wikiart")
    logger.setLevel(logging.DEBUG)

    fmt = logging.Formatter("%(asctime)s  %(levelname)-7s  %(message)s",
                            datefmt="%Y-%m-%d %H:%M:%S")

    ch = logging.StreamHandler()
    ch.setLevel(logging.INFO)
    ch.setFormatter(fmt)

    fh = logging.FileHandler(log_dir / "train.log", mode="a", encoding="utf-8")
    fh.setLevel(logging.DEBUG)
    fh.setFormatter(fmt)

    logger.addHandler(ch)
    logger.addHandler(fh)
    return logger


def save_csv(history: list, path: Path) -> None:
    if not history:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=history[0].keys())
        writer.writeheader()
        writer.writerows(history)


def save_json(history: list, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(history, f, indent=2)


# ── mAP with two progress bars ─────────────────────────────────────────────────
@torch.no_grad()
def _compute_map_with_progress(model, loader, ball, device, top_k: int) -> float:
    from wikiart_crnn.retrieval.hyperbolic_retrieval import hyperbolic_retrieve

    model.eval()
    all_embs, all_labels = [], []

    for imgs, _, _, a_lbl in tqdm(loader,
                                   desc=f"           [mAP@{top_k} embed]",
                                   leave=False,
                                   dynamic_ncols=True,
                                   unit="batch"):
        emb = model.encode(imgs.to(device))
        all_embs.append(emb.cpu())
        all_labels.extend(a_lbl.tolist())

    all_embs   = torch.cat(all_embs, dim=0).to(device)
    all_labels = torch.tensor(all_labels, device=device)
    N          = all_embs.shape[0]

    ap_sum = 0.0
    for i in tqdm(range(N),
                  desc=f"           [mAP@{top_k} score]",
                  leave=False,
                  dynamic_ncols=True,
                  unit="q"):
        top_idx = hyperbolic_retrieve(all_embs[i], all_embs, ball, top_k=top_k + 1)
        top_idx = top_idx[top_idx != i][:top_k]
        correct = (all_labels[top_idx] == all_labels[i]).float()
        prec_at = torch.cumsum(correct, dim=0) / (
            torch.arange(top_k, device=device) + 1
        )
        ap_sum += (prec_at * correct).sum().item() / correct.sum().clamp(min=1).item()

    return ap_sum / N


# ── Main ───────────────────────────────────────────────────────────────────────
def main():
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    logger = setup_logging(LOG_DIR)

    train_loader, val_loader, train_ds, val_ds = get_dataloaders(
        root        = DATA_ROOT,
        csv_path    = CSV_PATH,
        batch_size  = BATCH_SIZE,
        num_workers = 2
    )

    style_w  = train_ds.class_weights(train_ds.style_labels,    train_ds.num_styles)
    genre_w  = train_ds.class_weights(train_ds.metagenre_labels, train_ds.num_metagenres)
    artist_w = train_ds.class_weights(train_ds.artist_labels,   train_ds.num_artists)

    imgs, styles, metagenres, artists = next(iter(train_loader))
    logger.info(
        f"Batch  → imgs:{imgs.shape}  styles:{styles.shape}  "
        f"metagenres:{metagenres.shape}  artists:{artists.shape}"
    )
    logger.info(
        f"Classes → styles:{train_ds.num_styles}  "
        f"metagenres:{train_ds.num_metagenres}  "
        f"artists:{train_ds.num_artists}"
    )

    model  = WikiArtCRNNMamba(pretrained=True).to(DEVICE)
    supcon = SupConLoss(temperature=0.07)
    ball   = geoopt.PoincareBall(c=1.0)

    backbone_params = list(model.backbone.parameters())
    other_params    = [p for n, p in model.named_parameters()
                       if not n.startswith("backbone")]
    optimizer = torch.optim.AdamW(
        [
            {"params": backbone_params, "lr": LR * 0.1},
            {"params": other_params,    "lr": LR},
        ],
        weight_decay=1e-4,
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
        optimizer, T_0=10, T_mult=2
    )

    best_map = 0.0
    history  = []

    logger.info("=" * 70)
    logger.info(f"  WikiArt CRNN | device={DEVICE} | mamba={USE_MAMBA}")
    logger.info(f"  Styles={NUM_STYLES}  Genres={NUM_GENRES}  Artists={NUM_ARTISTS}")
    logger.info(f"  mAP@{MAP_K} computed every epoch  |  logs → {LOG_DIR}/")
    logger.info("=" * 70)

    epoch_bar = tqdm(range(1, EPOCHS + 1), desc="Epochs", unit="ep",
                     position=0, leave=True, dynamic_ncols=True)

    for epoch in epoch_bar:
        t0 = time.time()

        train_metrics = train_one_epoch(
            model, train_loader, optimizer,
            style_w, genre_w, artist_w, supcon, DEVICE, epoch,
        )
        scheduler.step()

        val_metrics = validate(
            model, val_loader, style_w, genre_w, artist_w, DEVICE
        )

        map_score = _compute_map_with_progress(
            model, val_loader, ball, DEVICE, top_k=MAP_K
        )

        elapsed = time.time() - t0
        lr_now  = optimizer.param_groups[1]["lr"]

        row = {
            "epoch"             : epoch,
            "elapsed_s"         : round(elapsed, 1),
            "lr"                : round(lr_now, 8),
            "train_loss"        : round(train_metrics["loss"],        6),
            "train_style_acc"   : round(train_metrics["style_acc"],   6),
            "train_genre_acc"   : round(train_metrics["genre_acc"],   6),
            "train_artist_acc"  : round(train_metrics["artist_acc"],  6),
            "val_loss"          : round(val_metrics["val_loss"],      6),
            "val_style_acc"     : round(val_metrics["val_style_acc"], 6),
            "val_genre_acc"     : round(val_metrics["val_genre_acc"], 6),
            "val_artist_acc"    : round(val_metrics["val_artist_acc"],6),
            f"map_at_{MAP_K}"   : round(map_score, 6),
        }
        history.append(row)

        logger.info(
            f"[Epoch {epoch:3d}/{EPOCHS}] {elapsed:.1f}s | lr={lr_now:.2e}"
        )
        logger.info(
            f"  Train  loss={train_metrics['loss']:.4f} | "
            f"style={train_metrics['style_acc']:.3f} | "
            f"genre={train_metrics['genre_acc']:.3f} | "
            f"artist={train_metrics['artist_acc']:.3f}"
        )
        logger.info(
            f"  Val    loss={val_metrics['val_loss']:.4f} | "
            f"style={val_metrics['val_style_acc']:.3f} | "
            f"genre={val_metrics['val_genre_acc']:.3f} | "
            f"artist={val_metrics['val_artist_acc']:.3f}"
        )
        logger.info(f"  mAP@{MAP_K} = {map_score:.4f}")

        epoch_bar.set_postfix(
            val_loss  = f"{val_metrics['val_loss']:.4f}",
            val_style = f"{val_metrics['val_style_acc']:.3f}",
            map       = f"{map_score:.4f}",
        )

        if map_score > best_map:
            best_map = map_score
            torch.save(
                {
                    "epoch"         : epoch,
                    "model_state"   : model.state_dict(),
                    "optimizer"     : optimizer.state_dict(),
                    "best_map"      : best_map,
                    "style_to_idx"  : train_ds.style_to_idx,
                    "artist_to_idx" : train_ds.artist_to_idx,
                    "genre_to_idx"  : train_ds.metagenre_to_idx,
                },
                CKPT_PATH,
            )
            logger.info(f"  ✓ Checkpoint saved  best_map@{MAP_K}={best_map:.4f}")

        # Persist logs after every epoch so nothing is lost on crash
        save_csv(history,  LOG_DIR / "history.csv")
        save_json(history, LOG_DIR / "history.json")

    logger.info("=" * 70)
    logger.info(f"Training complete.  Best mAP@{MAP_K} = {best_map:.4f}")
    logger.info(f"Logs saved to  {LOG_DIR}/")
    logger.info("=" * 70)

    return model, history


if __name__ == "__main__":
    main()
