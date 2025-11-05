<!-- 398ebde6-5123-43dd-a176-b83f1c64b428 ed9d17ea-ba8a-4855-84fc-d91ce510fabb -->
# Step-by-step plan: Transformer encoder for penalty direction

## Clarifications needed

1) Target horizon: predict using only the first X% of frames (e.g., 40%), or first T seconds (e.g., 0.4s)?
2) Model form: encoder-only sequence classifier (CLS pooling) vs simple mean pooling baseline first?

## Milestone 0 — Data wiring (read-only)

- Inspect `data/labels.json` mapping of pose `.npy` → {label, video} and a few `.npy` shapes to confirm feature dims and FPS. ✅ Done

## Milestone 1 — Dataset + collate

- Create `src/data/penalty_dataset.py` with a `PenaltyPoseDataset` that loads `.npy` pose sequences given `labels.json` and applies:
  - Optional sub-sequence truncation by horizon (percent or seconds)
  - Optional normalization (per-keypoint mean/std or min-max)
  - Label mapping: {left, center, right} → {0,1,2}
  - Implement `collate_fn` to pad variable length sequences and build `key_padding_mask`.
  ✅ Done (see `src/data/penalty_dataset.py`)

## Milestone 2 — Baseline models

- Implement `src/models/baselines.py`:
  - Temporal mean pooling + MLP classifier
  - 1D temporal CNN (optional)
  - Training script `src/train_baseline.py` using the dataset, saving metrics and a confusion matrix.
  ✅ Done (`MeanPoolMLP`, `train_baseline.py` with accuracy/macro-F1 + checkpoint)

## Milestone 3 — Transformer encoder (minimal)

- Module structure:
  - `src/models/transformers/__init__.py`
  - `src/models/transformers/config.py` (dataclass: d_model, n_heads, n_layers, dropout, horizon, etc.)
  - `src/models/transformers/embeddings.py` (linear projection from pose_dim→d_model; sinusoidal or learnable positional enc)
  - `src/models/transformers/encoder.py` (stack of `nn.TransformerEncoderLayer` with `batch_first=True`)
  - `src/models/transformers/heads.py` (CLS token or masked mean pooling → linear classifier)
  - `src/models/transformers/masks.py` (builders for `key_padding_mask`, future `attn_mask` if needed)
  - Training script `src/train_encoder.py` with AdamW, grad clipping, mixed precision optional, accuracy/macro-F1, checkpointing
  ✅ Done

## Milestone 4 — Early-time evaluation

- Add CLI flags `--horizon_percent` or `--horizon_seconds` to dataset.
- Evaluate performance vs horizon values (e.g., 20%, 30%, 40%, 60%).
- Plot accuracy/F1 vs horizon (`plots/encoder_horizon.png`).
Status: Pending

## Milestone 5 — Inference + visualization

- `src/infer_clip.py` loads a pose `.npy` or a video → pose pipeline → encoder → predicted class with confidence.
- Optional: overlay prediction timeline on `data/visualized_poses/` frames or video.
Status: Pending

## Milestone 6 — Repro + docs

- `MODELS.md` section describing encoder config and training commands.
- Save best config JSON and random seeds for reproducibility.
Status: Partially Done (README updated; configs saved in checkpoints)

## Implementation notes

- Input formatting: shapes `[B, T, pose_dim]` with `batch_first=True` end-to-end.
- Use `key_padding_mask` for padded timesteps; no `attn_mask` (not autoregressive).
- Start with sinusoidal positional encodings; swap to learned if beneficial.
- Class imbalance: compute class weights from labels for CE loss if needed.
- Small data guardrails: dropout, weight decay, early stopping on val macro-F1.

### To-dos

- [x] Verify .npy shapes, pose_dim, and sequence lengths
- [x] Implement dataset and collate with padding and masks
- [x] Add mean-pooling MLP baseline and trainer
- [x] Implement encoder modules, embeddings, and classification head
- [x] Training loop with schedulers, clipping, mixed precision optional
- [ ] Add horizon truncation and run early-time evaluation sweep
- [ ] Create inference script for a single clip or video
- [x] Document configs and training commands; save checkpoints and seeds

### Quick commands

Baseline (mean-pool MLP):
```bash
python src/train_baseline.py --labels_json data/labels.json --batch_size 16 --epochs 20 --use_class_weights --normalize_per_sample --horizon_percent 0.4
```

Transformer encoder:
```bash
python src/train_encoder.py --labels_json data/labels.json --batch_size 8 --epochs 30 --use_class_weights --use_cls_token --normalize_per_sample --horizon_percent 0.4
```

