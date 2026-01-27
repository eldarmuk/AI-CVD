import os
import argparse
from typing import Optional

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, SubsetRandomSampler, Sampler
from torch.optim import AdamW
import matplotlib.pyplot as plt
import random
try:
    from sklearn.metrics import average_precision_score, f1_score
    _HAS_SKLEARN = True
except Exception:
    _HAS_SKLEARN = False
from .generate_sequences import SEQ_LEN


# Configurations
FEATURE_COUNT = 33 #! UPDATE THIS IF FEATURES CHANGE
BATCH_SIZE = 64
LR = 1e-3
WEIGHT_DECAY = 1e-4
EPOCHS = 50
PATIENCE = 10
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
PATHS = {
    "X_train": "data/processed/sequences/X_train.npy",
    "y_train": "data/processed/sequences/y_train.npy",
    "X_val": "data/processed/sequences/X_val.npy",
    "y_val": "data/processed/sequences/y_val.npy",
    "X_test": "data/processed/sequences/X_test.npy",
}
SAVE_PATH = "models/bi_gru_best.pt"
SEED = 42

# Multi-class weights (capped for stability)
NUM_CLASSES = 4
CLASS_WEIGHTS = torch.tensor([1.0, 10.0, 15.0, 25.0]).to(DEVICE)

def _worker_init_fn(worker_id: int) -> None:
    """Top-level worker init fn so it is picklable on Windows spawn.

    Seeds numpy, random and torch for each worker deterministically.
    """
    worker_seed = SEED + worker_id
    np.random.seed(worker_seed)
    random.seed(worker_seed)
    try:
        torch.manual_seed(worker_seed)
    except Exception:
        pass


class NPYDataset(Dataset):
    """Dataset that memory-maps a .npy file and returns samples by index.

    Expects the numpy array to have shape (N, T, C) or (N, T, C-like).
    """

    def __init__(self, npy_path: str, labels_path: Optional[str] = None):
        self.npy_path = npy_path
        self.labels_path = labels_path
        self.data = None
        self.labels = None
        
        # We need these constants to calculate shape from file size
        self.seq_len = SEQ_LEN
        self.n_features = FEATURE_COUNT
        self.bytes_per_sample = self.seq_len * self.n_features * 4
        self.dtype_x = np.float32
        self.dtype_y = np.int8

        file_size = os.path.getsize(npy_path)

        if file_size % self.bytes_per_sample == 0:
            self.offset = 0
            self.n_samples = file_size // self.bytes_per_sample
        else:
            # Assume 128-byte header (standard numpy save)
            remaining = file_size - 128
            if remaining > 0 and remaining % self.bytes_per_sample == 0:
                self.offset = 128
                self.n_samples = remaining // self.bytes_per_sample
            else:
                # Fallback: Just floor division and warn
                self.offset = 0
                self.n_samples = file_size // self.bytes_per_sample
                print(f"Warning: File {npy_path} alignment is unclear. Using N={self.n_samples}")
        
        # Calculate N (number of samples) immediately to support __len__
        if not os.path.exists(npy_path):
             raise FileNotFoundError(f"Feature file not found: {npy_path}")
        
    def _ensure_open(self):
        """Lazy loads the memmap only when needed."""
        if self.data is None:
            # Map X (Features)
            self.data = np.memmap(
                self.npy_path, 
                dtype=self.dtype_x, 
                mode='r', 
                shape=(self.n_samples, self.seq_len, self.n_features),
                offset=self.offset
            )
            
            # Map Y (Labels) if present
            if self.labels_path and os.path.exists(self.labels_path):
                l_size = os.path.getsize(self.labels_path)
                l_offset = 128 if (l_size % 1 != 0) else 0 # Simple heuristic
                # Better: Check alignment with X
                if l_size == self.n_samples: # Raw int8
                    l_offset = 0
                elif l_size == self.n_samples + 128: # Header int8
                    l_offset = 128
                else:
                    l_offset = 0 # Hope for best

                self.data_y = np.memmap(
                    self.labels_path,
                    dtype=self.dtype_y, # adjust if you saved as float
                    mode='r',
                    shape=(self.n_samples,),
                    offset=l_offset
                )
            else:
                self.data_y = None

    def __len__(self):
        self._ensure_open()
        return self.n_samples

    def __getitem__(self, idx):
        self._ensure_open()
        x = np.array(self.data[idx], dtype=np.float32, copy=True)
        if self.data_y is None:
            return torch.from_numpy(x)
        
        y = self.data_y[idx]
        return torch.from_numpy(x), torch.tensor(y).float()


class DotProductAttention(nn.Module):
    def __init__(self, hidden_dim: int):
        super().__init__()
        # We'll project GRU outputs to a query and compute attention weights via dot-product
        self.scale = 1.0 / (hidden_dim ** 0.5)

    def forward(self, h: torch.Tensor):
        # h: (batch, seq_len, hidden_dim)
        # For dot-product attention collapse, use a learnable context vector q
        # Here we'll use the last timestep as query: h[:, -1, :]
        q = h[:, -1, :].unsqueeze(1)  # (batch, 1, hidden_dim)
        attn_scores = torch.bmm(q, h.transpose(1, 2)).squeeze(1)  # (batch, seq_len)
        attn_scores = attn_scores * self.scale
        attn_weights = torch.softmax(attn_scores, dim=-1)  # (batch, seq_len)
        attn_weights = attn_weights.unsqueeze(1)  # (batch,1,seq_len)
        context = torch.bmm(attn_weights, h).squeeze(1)  # (batch, hidden_dim)
        return context, attn_weights.squeeze(1)


class BiGRUAttention(nn.Module):
    def __init__(self, input_dim=33, hidden_dim=64, num_layers=2, dropout=0.3, num_classes=4):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.gru = nn.GRU(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
            bidirectional=True,
        )
        # after bidirectional GRU, hidden size doubles
        self.attention = DotProductAttention(hidden_dim * 2)
        self.fc = nn.Sequential(
            nn.Linear(hidden_dim * 2, 64),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(64, num_classes),
        )

    def forward(self, x):
        # x: (batch, seq_len, input_dim)
        h, _ = self.gru(x)  # h: (batch, seq_len, hidden_dim*2)
        context, attn = self.attention(h)
        out = self.fc(context)  # (batch, num_classes)
        return out, attn

class StratifiedValidationSampler(Sampler):
    """scans y_val to find ALL positives and mixes them with random negatives."""

    def __init__(self, labels_path, n_negatives=5000, dataset_len=None):
        self.labels_path = labels_path
        self.n_negatives = n_negatives
        self.dataset_len = dataset_len
        self.indices = self._build_indices()

    def _build_indices(self):
        print("Scanning Validation Labels for Stratified Sampling...")
        l_size = os.path.getsize(self.labels_path)
        offset = 128 if l_size % 1 != 0 else 0
        n_samples = l_size - offset
        y = np.memmap(self.labels_path, dtype='int8', mode='r', offset=offset, shape=(n_samples,))

        # Limit to dataset_len if provided
        max_len = self.dataset_len if self.dataset_len is not None else n_samples
        y = y[:max_len]

        pos_indices = np.where(y > 0)[0]
        neg_indices = np.where(y == 0)[0]

        print(f"  Found {len(pos_indices)} Positives and {len(neg_indices)} Negatives in Val.")

        if len(neg_indices) > self.n_negatives:
            selected_neg = np.random.choice(neg_indices, size=self.n_negatives, replace=False)
        else:
            selected_neg = neg_indices

        indices = np.concatenate([pos_indices, selected_neg])
        np.random.shuffle(indices)

        print(f"  Created Golden Validation Set: {len(indices)} samples ({len(pos_indices)} Pos / {len(selected_neg)} Neg)")
        return indices.tolist()

    def __iter__(self):
        return iter(self.indices)

    def __len__(self):
        return len(self.indices)


def train_loop(
    model,
    train_loader,
    val_loader,
    device,
    epochs: int = EPOCHS,
    patience: int = PATIENCE,
    save_path: str = SAVE_PATH,
    lr: float = LR,
    weight_decay: float = WEIGHT_DECAY,
):
    criterion = nn.CrossEntropyLoss(weight=CLASS_WEIGHTS)
    opt = AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)

    best_val_loss = float("inf")
    epochs_no_improve = 0
    history = {"train_loss": [], "val_loss": []}

    for epoch in range(1, epochs + 1):
        model.train()
        train_loss = 0.0
        n = 0
        for batch in train_loader:
            if isinstance(batch, (list, tuple)):
                x, y = batch
            else:
                raise RuntimeError("Train loader must return (x, y)")
            x = x.to(device)
            y = y.to(device).long()

            opt.zero_grad()
            logits, _ = model(x)
            loss = criterion(logits, y)
            loss.backward()

            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

            opt.step()

            batch_size = x.size(0)
            train_loss += loss.item() * batch_size
            n += batch_size

        train_loss /= n
        history["train_loss"].append(train_loss)

        # Validation
        model.eval()
        val_loss = 0.0
        vn = 0
        y_trues = []
        y_preds = []
        with torch.no_grad():
            for batch in val_loader:
                x, y = batch
                x = x.to(device)
                y = y.to(device).long()
                logits, _ = model(x)
                loss = criterion(logits, y)
                val_loss += loss.item() * x.size(0)
                vn += x.size(0)
                # collect for metrics
                preds = torch.argmax(logits, dim=1).cpu().numpy()
                y_np = y.cpu().numpy()
                y_preds.append(preds)
                y_trues.append(y_np)
        val_loss = val_loss / max(vn, 1)
        history["val_loss"].append(val_loss)

        # Metrics (optional: f1 macro)
        f1_macro = 0.0
        if _HAS_SKLEARN and len(y_trues) > 0:
            try:
                y_trues_flat = np.concatenate(y_trues)
                y_preds_flat = np.concatenate(y_preds)
                f1_macro = f1_score(y_trues_flat, y_preds_flat, average="macro")
            except: pass

        print(f"Epoch {epoch:02d}: Train Loss={train_loss:.4f} | Val Loss={val_loss:.4f} | Val F1_macro={f1_macro:.4f}")

        # Early stopping
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            epochs_no_improve = 0
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            torch.save(model.state_dict(), save_path)
        else:
            epochs_no_improve += 1
            if epochs_no_improve >= patience:
                print("Early stopping triggered.")
                break

    # plot
    plt.figure()
    plt.plot(history["train_loss"], label="train_loss")
    plt.plot(history["val_loss"], label="val_loss")
    plt.xlabel("epoch")
    plt.ylabel("loss")
    plt.legend()
    plt.tight_layout()
    plt.savefig("training_curve.png")
    plt.close()

    return history


def main(args):
    device = DEVICE

    # Set seeds for reproducibility
    seed = args.seed
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    train_dataset = NPYDataset(args.x_train, args.y_train)
    val_dataset = NPYDataset(args.x_val, args.y_val)

    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=0,
        pin_memory=True,
        worker_init_fn=_worker_init_fn,
    )


    # Pass dataset length to sampler to avoid out-of-bounds indices
    stratified_sampler = StratifiedValidationSampler(args.y_val, n_negatives=5000, dataset_len=len(val_dataset))
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        sampler=stratified_sampler,
        num_workers=0,
        pin_memory=True,
        worker_init_fn=_worker_init_fn,
    )

    model = BiGRUAttention(input_dim=args.input_dim, hidden_dim=64, num_layers=2, dropout=0.3, num_classes=NUM_CLASSES)
    model.to(device)

    train_loop(
        model,
        train_loader,
        val_loader,
        device,
        epochs=args.epochs,
        patience=args.patience,
        save_path=args.save_path,
        lr=args.lr,
        weight_decay=args.weight_decay,
    )

    print("Training complete. Best model saved to:", args.save_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--x_train", default=PATHS["X_train"])
    parser.add_argument("--y_train", default=PATHS["y_train"])
    parser.add_argument("--x_val", default=PATHS["X_val"])
    parser.add_argument("--y_val", default=PATHS["y_val"])
    parser.add_argument("--batch_size", type=int, default=BATCH_SIZE)
    parser.add_argument("--epochs", type=int, default=EPOCHS)
    parser.add_argument("--patience", type=int, default=PATIENCE)
    parser.add_argument("--save_path", default=SAVE_PATH)
    parser.add_argument("--input_dim", type=int, default=33)
    parser.add_argument("--lr", type=float, default=LR)
    parser.add_argument("--weight_decay", type=float, default=WEIGHT_DECAY)
    parser.add_argument("--seed", type=int, default=SEED)

    args = parser.parse_args()
    main(args)
