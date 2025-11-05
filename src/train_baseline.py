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
from models.baselines import MeanPoolMLP


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


def train_one_epoch(model, loader, opt, device, class_weights):
    model.train()
    total_loss = total_acc = total = 0
    for batch in loader:
        batch = {k: v.to(device) if torch.is_tensor(v) else v for k, v in batch.items()}
        logits = model(batch)
        loss = F.cross_entropy(logits, batch["labels"], weight=class_weights.to(device) if class_weights is not None else None)
        opt.zero_grad()
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
        loss = F.cross_entropy(logits, batch["labels"])  # for reporting
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
    p.add_argument("--epochs", type=int, default=20)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--dropout", type=float, default=0.1)
    p.add_argument("--hidden_dim", type=int, default=128)
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

    model = MeanPoolMLP(input_dim=66, hidden_dim=args.hidden_dim, num_classes=3, dropout=args.dropout).to(args.device)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr)
    class_weights = compute_class_weights(args.labels_json) if args.use_class_weights else None

    best_f1 = -1.0
    os.makedirs("models", exist_ok=True)

    for epoch in range(1, args.epochs + 1):
        tr_loss, tr_acc = train_one_epoch(model, train_loader, opt, args.device, class_weights)
        va_loss, va_acc, va_f1 = evaluate(model, val_loader, args.device)
        print(f"Epoch {epoch:03d} | train loss {tr_loss:.4f} acc {tr_acc:.3f} | val loss {va_loss:.4f} acc {va_acc:.3f} f1 {va_f1:.3f}")
        if va_f1 > best_f1:
            best_f1 = va_f1
            torch.save({
                "model_state": model.state_dict(),
                "config": {
                    "input_dim": 66,
                    "hidden_dim": args.hidden_dim,
                    "dropout": args.dropout,
                },
                "val_f1": va_f1,
            }, os.path.join("models", "baseline_meanpool_mlp.pt"))
            print(f"Saved best baseline to models/baseline_meanpool_mlp.pt (F1={va_f1:.3f})")


if __name__ == "__main__":
    main()


