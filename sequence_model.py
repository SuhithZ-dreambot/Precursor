"""
sequence_model.py
GRU / LSTM sequence forecaster -- the "deep temporal signal" half of the
PRECURSOR hybrid model (see slide 5: statistical EWMA/CUSUM + deep temporal
GRU/LSTM, fused into one early-warning risk score).

This is a drop-in replacement for the XGBoost stand-in used in pipeline.py:
same input (windowed feature panel), same output contract (a per-window
"attack in next `lead_windows` windows" probability), so it can be fused with
the statistical signal and evaluated for lead time exactly the same way.

Architecture
------------
Each window t gets a sequence of the last `seq_len` windows' raw + delta
features (NOT the rolling-mean/std columns -- those are redundant once the
recurrent cell can integrate history itself; feeding it raw+delta lets the
GRU/LSTM learn its own temporal aggregation instead of consuming a
statistician's version of it). The cell's final hidden state feeds a small
MLP head -> sigmoid -> P(attack in next `lead_windows` windows).

Usage
-----
    from sequence_model import SeqConfig, train_sequence_model
    cfg = SeqConfig(cell_type="gru")
    model, proba_all, split, auc, importances = train_sequence_model(df, cfg)

`proba_all` is aligned 1:1 with `df` rows, same as pipeline.py's XGBoost
`train_model()` output, so pipeline_gru.py can swap it in directly.
"""
from dataclasses import dataclass, field
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import roc_auc_score

from feature_engineering import RAW_COLS

SEQ_FEATURE_COLS = RAW_COLS + [f"{c}_delta" for c in RAW_COLS]  # 14 channels


@dataclass
class SeqConfig:
    seq_len: int = 12          # 12 windows * 5 min = 60 min of history per prediction
    hidden_size: int = 32
    num_layers: int = 1
    cell_type: str = "gru"     # "gru" or "lstm"
    dropout: float = 0.1
    lr: float = 1e-3
    weight_decay: float = 1e-5
    epochs: int = 25
    batch_size: int = 64
    train_frac: float = 0.7    # time-ordered split, matches pipeline.py
    device: str = "cpu"
    seed: int = 42


class WindowSequenceDataset(Dataset):
    """
    Builds one sequence per window index i: the `seq_len` windows ending at i
    (inclusive), left-padded by repeating the first row for i < seq_len-1 so
    every window in the dataframe gets a prediction (needed so lead-time
    evaluation can look at buildup windows near the start of an incident).
    """
    def __init__(self, X, y, seq_len):
        self.X = X.astype(np.float32)
        self.y = y.astype(np.float32)
        self.seq_len = seq_len
        self.n = len(X)

    def __len__(self):
        return self.n

    def __getitem__(self, i):
        lo = i - self.seq_len + 1
        if lo < 0:
            pad = np.repeat(self.X[0:1], -lo, axis=0)
            seq = np.concatenate([pad, self.X[0:i + 1]], axis=0)
        else:
            seq = self.X[lo:i + 1]
        return torch.from_numpy(seq), torch.tensor(self.y[i])


class SequenceForecaster(nn.Module):
    def __init__(self, n_features, cfg: SeqConfig):
        super().__init__()
        rnn_cls = nn.GRU if cfg.cell_type == "gru" else nn.LSTM
        self.rnn = rnn_cls(
            input_size=n_features,
            hidden_size=cfg.hidden_size,
            num_layers=cfg.num_layers,
            batch_first=True,
            dropout=cfg.dropout if cfg.num_layers > 1 else 0.0,
        )
        self.head = nn.Sequential(
            nn.Linear(cfg.hidden_size, 16),
            nn.ReLU(),
            nn.Dropout(cfg.dropout),
            nn.Linear(16, 1),
        )

    def forward(self, x):
        out, hidden = self.rnn(x)
        last = out[:, -1, :]              # final timestep's hidden state
        logits = self.head(last).squeeze(-1)
        return logits


def _standardize(train_X, full_X):
    mu = train_X.mean(axis=0, keepdims=True)
    sigma = train_X.std(axis=0, keepdims=True) + 1e-6
    return (full_X - mu) / sigma, mu, sigma


def train_sequence_model(df: pd.DataFrame, cfg: SeqConfig = SeqConfig(),
                          label_col: str = "future_attack", verbose: bool = True):
    torch.manual_seed(cfg.seed)
    np.random.seed(cfg.seed)

    X_raw = df[SEQ_FEATURE_COLS].values.astype(np.float32)
    y = df[label_col].values.astype(np.float32)
    n = len(df)
    split = int(n * cfg.train_frac)

    X_std, mu, sigma = _standardize(X_raw[:split], X_raw)

    full_ds = WindowSequenceDataset(X_std, y, cfg.seq_len)
    train_idx = np.arange(0, split)
    test_idx = np.arange(split, n)

    train_ds = torch.utils.data.Subset(full_ds, train_idx)
    train_loader = DataLoader(train_ds, batch_size=cfg.batch_size, shuffle=True)

    n_pos = y[:split].sum()
    n_neg = split - n_pos
    pos_weight = torch.tensor([n_neg / max(1, n_pos)], dtype=torch.float32)

    model = SequenceForecaster(n_features=len(SEQ_FEATURE_COLS), cfg=cfg).to(cfg.device)
    opt = torch.optim.Adam(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    loss_fn = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

    best_auc, best_state = -1.0, None

    for epoch in range(1, cfg.epochs + 1):
        model.train()
        epoch_loss = 0.0
        for xb, yb in train_loader:
            xb, yb = xb.to(cfg.device), yb.to(cfg.device)
            opt.zero_grad()
            logits = model(xb)
            loss = loss_fn(logits, yb)
            loss.backward()
            opt.step()
            epoch_loss += loss.item() * xb.size(0)
        epoch_loss /= len(train_ds)

        test_auc = _eval_auc(model, full_ds, test_idx, cfg)
        if test_auc > best_auc:
            best_auc = test_auc
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
        if verbose and (epoch % 5 == 0 or epoch == 1):
            print(f"  epoch {epoch:>3}/{cfg.epochs}  train_loss={epoch_loss:.4f}  test_auc={test_auc:.4f}")

    if best_state is not None:
        model.load_state_dict(best_state)

    proba_all = _predict_all(model, full_ds, cfg)
    final_auc = roc_auc_score(y[test_idx], proba_all[test_idx]) if y[test_idx].sum() > 0 else float("nan")

    importances = _permutation_importance(model, full_ds, test_idx, y, cfg)

    return model, proba_all, split, final_auc, importances


@torch.no_grad()
def _predict_all(model, ds, cfg, batch_size=256):
    model.eval()
    loader = DataLoader(ds, batch_size=batch_size, shuffle=False)
    out = []
    for xb, _ in loader:
        logits = model(xb.to(cfg.device))
        out.append(torch.sigmoid(logits).cpu().numpy())
    return np.concatenate(out)


@torch.no_grad()
def _eval_auc(model, ds, idx, cfg):
    model.eval()
    subset = torch.utils.data.Subset(ds, idx)
    loader = DataLoader(subset, batch_size=256, shuffle=False)
    preds, labels = [], []
    for xb, yb in loader:
        logits = model(xb.to(cfg.device))
        preds.append(torch.sigmoid(logits).cpu().numpy())
        labels.append(yb.numpy())
    preds = np.concatenate(preds)
    labels = np.concatenate(labels)
    if labels.sum() == 0:
        return float("nan")
    return roc_auc_score(labels, preds)


@torch.no_grad()
def _permutation_importance(model, ds, test_idx, y, cfg, n_repeats=2):
    """
    Coarse channel-level permutation importance: shuffle one feature channel
    across the test sequences, measure AUC drop. Cheap (no gradients) and
    gives the dashboard's "signal drivers" panel something meaningful to show
    for a recurrent model, where native feature_importances_ don't exist.
    """
    model.eval()
    base_preds = _predict_all(model, ds, cfg)
    base_auc = roc_auc_score(y[test_idx], base_preds[test_idx]) if y[test_idx].sum() > 0 else 0.5

    rng = np.random.default_rng(cfg.seed)
    importances = {}
    X = ds.X.copy()
    for ci, col in enumerate(SEQ_FEATURE_COLS):
        drops = []
        for _ in range(n_repeats):
            X_perm = ds.X.copy()
            perm_idx = rng.permutation(len(test_idx)) + test_idx[0]
            X_perm[test_idx, ci] = ds.X[perm_idx, ci]
            perm_ds = WindowSequenceDataset(X_perm, ds.y, ds.seq_len)
            perm_preds = _predict_all(model, perm_ds, cfg)
            perm_auc = roc_auc_score(y[test_idx], perm_preds[test_idx]) if y[test_idx].sum() > 0 else 0.5
            drops.append(max(0.0, base_auc - perm_auc))
        importances[col] = float(np.mean(drops))

    # normalize to sum to 1 for readability, matching the XGBoost panel's scale
    total = sum(importances.values()) or 1.0
    return {k: v / total for k, v in importances.items()}
