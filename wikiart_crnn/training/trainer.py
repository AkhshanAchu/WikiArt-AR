import torch
import torch.nn.functional as F
from tqdm import tqdm
from ..losses import bias_aware_ce, uncertainty_loss


def train_one_epoch(model, loader, optimizer,
                    style_w, genre_w, artist_w,
                    supcon, device, epoch: int) -> dict:
    model.train()
    style_w  = style_w.to(device)
    genre_w  = genre_w.to(device)
    artist_w = artist_w.to(device)
    total_loss = style_acc = genre_acc = artist_acc = 0.0
    n = 0

    pbar = tqdm(loader, desc=f"Epoch {epoch:3d} [Train]", leave=True,
                dynamic_ncols=True, unit="batch")

    for imgs, s_lbl, g_lbl, a_lbl in pbar:
        imgs, s_lbl, g_lbl, a_lbl = (
            imgs.to(device), s_lbl.to(device),
            g_lbl.to(device), a_lbl.to(device),
        )

        out = model(imgs)

        l_style  = bias_aware_ce(out["style"],  s_lbl, style_w)
        l_genre  = bias_aware_ce(out["genre"],  g_lbl, genre_w)
        l_artist = bias_aware_ce(out["artist"], a_lbl, artist_w)
        l_con    = supcon(out["embedding"], a_lbl)
        l_adv    = F.cross_entropy(out["adversarial"], s_lbl)

        l_cls = uncertainty_loss([l_style, l_genre, l_artist], model.log_var)
        loss  = l_cls + 0.1 * l_con + 0.05 * l_adv

        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()

        bs          = imgs.size(0)
        total_loss += loss.item() * bs
        style_acc  += (out["style"].argmax(1)  == s_lbl).sum().item()
        genre_acc  += (out["genre"].argmax(1)  == g_lbl).sum().item()
        artist_acc += (out["artist"].argmax(1) == a_lbl).sum().item()
        n          += bs

        pbar.set_postfix(
            loss=f"{total_loss/n:.4f}",
            style=f"{style_acc/n:.3f}",
            genre=f"{genre_acc/n:.3f}",
            artist=f"{artist_acc/n:.3f}",
        )

    return {
        "loss"       : total_loss / n,
        "style_acc"  : style_acc  / n,
        "genre_acc"  : genre_acc  / n,
        "artist_acc" : artist_acc / n,
    }


@torch.no_grad()
def validate(model, loader, style_w, genre_w, artist_w, device) -> dict:
    model.eval()
    style_w  = style_w.to(device)
    genre_w  = genre_w.to(device)
    artist_w = artist_w.to(device)
    total_loss = style_acc = genre_acc = artist_acc = 0.0
    n = 0

    pbar = tqdm(loader, desc="           [Val]  ", leave=True,
                dynamic_ncols=True, unit="batch")

    for imgs, s_lbl, g_lbl, a_lbl in pbar:
        imgs, s_lbl, g_lbl, a_lbl = (
            imgs.to(device), s_lbl.to(device),
            g_lbl.to(device), a_lbl.to(device),
        )

        out = model(imgs)
        loss = (
            bias_aware_ce(out["style"],  s_lbl, style_w) +
            bias_aware_ce(out["genre"],  g_lbl, genre_w) +
            bias_aware_ce(out["artist"], a_lbl, artist_w)
        )

        bs          = imgs.size(0)
        total_loss += loss.item() * bs
        style_acc  += (out["style"].argmax(1)  == s_lbl).sum().item()
        genre_acc  += (out["genre"].argmax(1)  == g_lbl).sum().item()
        artist_acc += (out["artist"].argmax(1) == a_lbl).sum().item()
        n          += bs

        pbar.set_postfix(
            loss=f"{total_loss/n:.4f}",
            style=f"{style_acc/n:.3f}",
            genre=f"{genre_acc/n:.3f}",
            artist=f"{artist_acc/n:.3f}",
        )

    return {
        "val_loss"       : total_loss / n,
        "val_style_acc"  : style_acc  / n,
        "val_genre_acc"  : genre_acc  / n,
        "val_artist_acc" : artist_acc / n,
    }
