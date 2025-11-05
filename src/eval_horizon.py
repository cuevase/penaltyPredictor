import argparse
import os
import random
from typing import List, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, random_split
import matplotlib.pyplot as plt

from data.penalty_dataset import PenaltyPoseDataset, penalty_collate_fn
from models.baselines import MeanPoolMLP
from models.transformers import EncoderConfig, PoseTransformerEncoder


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


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


def train_eval_once(model, train_loader, val_loader, device, epochs: int, lr: float) -> Tuple[float, float]:
    opt = torch.optim.AdamW(model.parameters(), lr=lr)
    for _ in range(epochs):
        model.train()
        for batch in train_loader:
            batch = {k: v.to(device) if torch.is_tensor(v) else v for k, v in batch.items()}
            logits = model(batch)
            loss = F.cross_entropy(logits, batch["labels"]) 
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()

    # Evaluate
    model.eval()
    all_logits, all_labels = [], []
    with torch.no_grad():
        for batch in val_loader:
            batch = {k: v.to(device) if torch.is_tensor(v) else v for k, v in batch.items()}
            logits = model(batch)
            all_logits.append(logits.cpu())
            all_labels.append(batch["labels"].cpu())
    logits = torch.cat(all_logits)
    labels = torch.cat(all_labels)
    acc = accuracy(logits, labels)
    f1 = macro_f1(logits, labels, num_classes=3)
    return acc, f1


def sweep(
    model_type: str,
    horizons: List[float],
    labels_json: str,
    batch_size: int,
    epochs: int,
    lr: float,
    device: str,
    use_cls_token: bool,
    d_model: int,
    n_heads: int,
    n_layers: int,
    dim_ff: int,
    dropout: float,
    normalize_per_sample: bool,
    seed: int,
):
    set_seed(seed)
    results = []
    for hp in horizons:
        ds = PenaltyPoseDataset(labels_json_path=labels_json, horizon_percent=hp, normalize_per_sample=normalize_per_sample)
        n_val = max(1, int(len(ds) * 0.2))
        n_train = len(ds) - n_val
        train_ds, val_ds = random_split(ds, [n_train, n_val])
        train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, collate_fn=penalty_collate_fn)
        val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, collate_fn=penalty_collate_fn)

        if model_type == "baseline":
            model = MeanPoolMLP(input_dim=66, hidden_dim=d_model, num_classes=3, dropout=dropout).to(device)
        else:
            cfg = EncoderConfig(
                pose_dim=66,
                d_model=d_model,
                n_heads=n_heads,
                n_layers=n_layers,
                dim_feedforward=dim_ff,
                dropout=dropout,
                num_classes=3,
                use_cls_token=use_cls_token,
                positional_encoding="sinusoidal",
            )
            model = PoseTransformerEncoder(cfg).to(device)

        acc, f1 = train_eval_once(model, train_loader, val_loader, device, epochs=epochs, lr=lr)
        print(f"horizon={hp:.2f} -> acc={acc:.3f}, f1={f1:.3f}")
        results.append((hp, acc, f1))

    # Plot
    os.makedirs("plots", exist_ok=True)
    xs = [r[0] * 100 for r in results]
    accs = [r[1] for r in results]
    f1s = [r[2] for r in results]
    plt.figure(figsize=(7,4))
    plt.plot(xs, accs, marker='o', label='Accuracy')
    plt.plot(xs, f1s, marker='s', label='Macro-F1')
    plt.xlabel('Horizon (% of frames)')
    plt.ylabel('Score')
    plt.title(f'Horizon sweep ({model_type})')
    plt.grid(True, alpha=0.3)
    plt.legend()
    out_path = os.path.join('plots', f'horizon_sweep_{model_type}.png')
    plt.tight_layout()
    plt.savefig(out_path)
    print(f"Saved plot to {out_path}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--model', choices=['baseline', 'encoder'], default='encoder')
    ap.add_argument('--labels_json', type=str, default='data/labels.json')
    ap.add_argument('--horizons', type=str, default='0.2,0.3,0.4,0.6')
    ap.add_argument('--batch_size', type=int, default=16)
    ap.add_argument('--epochs', type=int, default=8)
    ap.add_argument('--lr', type=float, default=3e-4)
    ap.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu')
    ap.add_argument('--use_cls_token', action='store_true')
    ap.add_argument('--d_model', type=int, default=128)
    ap.add_argument('--n_heads', type=int, default=4)
    ap.add_argument('--n_layers', type=int, default=4)
    ap.add_argument('--dim_ff', type=int, default=256)
    ap.add_argument('--dropout', type=float, default=0.1)
    ap.add_argument('--normalize_per_sample', action='store_true')
    ap.add_argument('--seed', type=int, default=42)
    args = ap.parse_args()

    horizons = [float(x.strip()) for x in args.horizons.split(',') if x.strip()]
    sweep(
        model_type=args.model,
        horizons=horizons,
        labels_json=args.labels_json,
        batch_size=args.batch_size,
        epochs=args.epochs,
        lr=args.lr,
        device=args.device,
        use_cls_token=args.use_cls_token,
        d_model=args.d_model,
        n_heads=args.n_heads,
        n_layers=args.n_layers,
        dim_ff=args.dim_ff,
        dropout=args.dropout,
        normalize_per_sample=args.normalize_per_sample,
        seed=args.seed,
    )


if __name__ == '__main__':
    main()


