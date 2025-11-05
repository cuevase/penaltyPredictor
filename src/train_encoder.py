import argparse
import json
import os
import random
from typing import Tuple

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, random_split

from data.penalty_dataset import PenaltyPoseDataset, penalty_collate_fn, LABEL_TO_ID
from models.transformers import EncoderConfig, PoseTransformerEncoder


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def compute_class_weights(labels_json_path: str) -> torch.Tensor:
    with open(labels_json_path, "r") as f:
        data = json.load(f)
    counts = {k: 0 for k in LABEL_TO_ID.keys()}
    for meta in data.values():
        lbl = meta.get("label")
        if lbl in counts:
            counts[lbl] += 1
    totals = np.array([counts["left"], counts["center"], counts["right"]], dtype=np.float32)
    totals = np.clip(totals, 1.0, None)
    inv = 1.0 / totals
    weights = inv / inv.sum() * 3.0
    return torch.tensor(weights, dtype=torch.float32)


def accuracy(pred: torch.Tensor, target: torch.Tensor) -> float:
    return (pred.argmax(dim=1) == target).float().mean().item()


def macro_f1(pred: torch.Tensor, target: torch.Tensor, num_classes: int) -> float:
    preds = pred.argmax(dim=1).cpu().numpy()
    t = target.cpu().numpy()
    f1s = []
    for c in range(num_classes):
        tp = np.sum((preds == c) & (t == c))
        fp = np.sum((preds == c) & (t != c))
        fn = np.sum((preds != c) & (t == c))
        prec = tp / (tp + fp + 1e-8)
        rec = tp / (tp + fn + 1e-8)
        f1 = 2 * prec * rec / (prec + rec + 1e-8)
        f1s.append(f1)
    return float(np.mean(f1s))


def train_one_epoch(model, loader, opt, device, scaler, class_weights):
    model.train()
    total_loss = total_acc = total = 0
    for batch in loader:
        batch = {k: v.to(device) if torch.is_tensor(v) else v for k, v in batch.items()}
        with torch.autocast(device_type=device.split(":")[0] if device != "cpu" else "cpu", enabled=(device != "cpu")):
            logits = model(batch)
            loss = F.cross_entropy(logits, batch["labels"], weight=class_weights.to(device) if class_weights is not None else None)
        opt.zero_grad()
        if scaler is not None and device != "cpu":
            scaler.scale(loss).backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(opt)
            scaler.update()
        else:
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()

        bs = batch["labels"].size(0)
        total += bs
        total_loss += loss.item() * bs
        total_acc += accuracy(logits.detach(), batch["labels"]) * bs
    return total_loss / total, total_acc / total


@torch.no_grad()
def evaluate(model, loader, device) -> Tuple[float, float, float]:
    model.eval()
    total_loss = total_acc = total = 0
    all_logits = []
    all_labels = []
    for batch in loader:
        batch = {k: v.to(device) if torch.is_tensor(v) else v for k, v in batch.items()}
        logits = model(batch)
        loss = F.cross_entropy(logits, batch["labels"])  # report only
        bs = batch["labels"].size(0)
        total += bs
        total_loss += loss.item() * bs
        total_acc += accuracy(logits, batch["labels"]) * bs
        all_logits.append(logits.cpu())
        all_labels.append(batch["labels"].cpu())
    logits = torch.cat(all_logits)
    labels = torch.cat(all_labels)
    f1 = macro_f1(logits, labels, num_classes=3)
    return total_loss / total, total_acc / total, f1


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--labels_json", type=str, default="data/labels.json")
    p.add_argument("--batch_size", type=int, default=16)
    p.add_argument("--epochs", type=int, default=30)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--dropout", type=float, default=0.1)
    p.add_argument("--d_model", type=int, default=128)
    p.add_argument("--n_heads", type=int, default=4)
    p.add_argument("--n_layers", type=int, default=4)
    p.add_argument("--dim_ff", type=int, default=256)
    p.add_argument("--use_cls_token", action="store_true")
    p.add_argument("--positional_encoding", type=str, default="sinusoidal", choices=["sinusoidal", "learned"])
    p.add_argument("--horizon_percent", type=float, default=None)
    p.add_argument("--normalize_per_sample", action="store_true")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--val_split", type=float, default=0.2)
    p.add_argument("--use_class_weights", action="store_true")
    p.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    args = p.parse_args()

    set_seed(args.seed)

    ds = PenaltyPoseDataset(
        labels_json_path=args.labels_json,
        horizon_percent=args.horizon_percent,
        normalize_per_sample=args.normalize_per_sample,
    )

    n_val = max(1, int(len(ds) * args.val_split))
    n_train = len(ds) - n_val
    train_ds, val_ds = random_split(ds, [n_train, n_val])

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, collate_fn=penalty_collate_fn)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, collate_fn=penalty_collate_fn)

    cfg = EncoderConfig(
        pose_dim=66,
        d_model=args.d_model,
        n_heads=args.n_heads,
        n_layers=args.n_layers,
        dim_feedforward=args.dim_ff,
        dropout=args.dropout,
        num_classes=3,
        use_cls_token=args.use_cls_token,
        positional_encoding=args.positional_encoding,
    )
    model = PoseTransformerEncoder(cfg).to(args.device)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr)
    scaler = torch.cuda.amp.GradScaler(enabled=(args.device != "cpu"))
    class_weights = compute_class_weights(args.labels_json) if args.use_class_weights else None

    best_f1 = -1.0
    os.makedirs("models", exist_ok=True)

    for epoch in range(1, args.epochs + 1):
        tr_loss, tr_acc = train_one_epoch(model, train_loader, opt, args.device, scaler, class_weights)
        va_loss, va_acc, va_f1 = evaluate(model, val_loader, args.device)
        print(f"Epoch {epoch:03d} | train loss {tr_loss:.4f} acc {tr_acc:.3f} | val loss {va_loss:.4f} acc {va_acc:.3f} f1 {va_f1:.3f}")
        if va_f1 > best_f1:
            best_f1 = va_f1
            torch.save({
                "model_state": model.state_dict(),
                "config": cfg.__dict__,
                "val_f1": va_f1,
            }, os.path.join("models", "encoder_transformer.pt"))
            print(f"Saved best encoder to models/encoder_transformer.pt (F1={va_f1:.3f})")


if __name__ == "__main__":
    main()


