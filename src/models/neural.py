"""
src/models/neural.py
---------------------
MLP with learned champion embeddings for draft recommendation.

Architecture:
  1. Champion embedding table (shared for picks & bans)
  2. Aggregate team embeddings (mean-pool over present champions)
  3. Concatenate with role and team side encodings
  4. Feed through stacked linear → BatchNorm → ReLU → Dropout blocks
  5. Output: logits over champion pool (softmax at inference)

Usage:
    from src.models.neural import DraftMLP, train_epoch, evaluate
"""

from __future__ import annotations

import pathlib
from typing import Iterator

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
import concurrent.futures
import functools

from src.utils.config import get
from src.utils.logger import get_logger

logger = get_logger(__name__)

MODEL_DIR = pathlib.Path(get("training.model_save_dir", "models"))

NUM_CHAMPIONS: int = get("data.num_champions", 165)
EMB_DIM: int = get("features.embedding_dim", 64)
HIDDEN_DIMS: list[int] = get("model.neural.hidden_dims", [256, 128, 64])
DROPOUT: float = get("model.neural.dropout", 0.3)
NUM_ROLES: int = len(get("data.roles", ["TOP", "JUNGLE", "MID", "ADC", "SUPPORT"]))


class DraftMLP(nn.Module):
    """Multi-layer perceptron draft recommendation model with champion embeddings.

    Args:
        num_champions:  Vocabulary size (= number of unique champions).
        embedding_dim:  Dimension of champion embedding vectors.
        hidden_dims:    Sizes of the hidden layers.
        dropout:        Dropout probability applied after each hidden layer.
    """

    def __init__(
        self,
        num_champions: int = NUM_CHAMPIONS,
        embedding_dim: int = EMB_DIM,
        hidden_dims: list[int] | None = None,
        dropout: float = DROPOUT,
    ) -> None:
        super().__init__()
        self.num_champions = num_champions
        self.embedding_dim = embedding_dim
        hidden_dims = hidden_dims or HIDDEN_DIMS

        # Shared champion embedding table (index 0 = padding/unknown)
        self.champion_emb = nn.Embedding(
            num_embeddings=num_champions + 1,
            embedding_dim=embedding_dim,
            padding_idx=0,
        )

        # Input size:
        #   4 aggregated champion vectors (blue_picks, red_picks, blue_bans, red_bans)
        #   + role one-hot + team one-hot
        input_dim = 4 * embedding_dim + NUM_ROLES + 2

        layers: list[nn.Module] = []
        prev = input_dim
        for h in hidden_dims:
            layers += [
                nn.Linear(prev, h),
                nn.BatchNorm1d(h),
                nn.ReLU(inplace=True),
                nn.Dropout(dropout),
            ]
            prev = h
        layers.append(nn.Linear(prev, num_champions))

        self.net = nn.Sequential(*layers)

    # ------------------------------------------------------------------
    # Forward pass helpers
    # ------------------------------------------------------------------

    def _aggregate(self, indices: torch.Tensor) -> torch.Tensor:
        """Mean-pool embeddings for a batch of champion index sequences.

        Args:
            indices: LongTensor of shape ``(batch, seq_len)`` where 0 = padding.

        Returns:
            FloatTensor of shape ``(batch, embedding_dim)``.
        """
        embs = self.champion_emb(indices)          # (B, S, D)
        mask = (indices != 0).unsqueeze(-1).float()
        summed = (embs * mask).sum(dim=1)
        count = mask.sum(dim=1).clamp(min=1)
        return summed / count

    def forward(
        self,
        blue_picks: torch.Tensor,
        red_picks: torch.Tensor,
        blue_bans: torch.Tensor,
        red_bans: torch.Tensor,
        role: torch.Tensor,
        team: torch.Tensor,
    ) -> torch.Tensor:
        """Compute logits over the champion pool.

        Args:
            blue_picks: (B, 5) champion indices, 0-padded.
            red_picks:  (B, 5) champion indices, 0-padded.
            blue_bans:  (B, 5) champion indices, 0-padded.
            red_bans:   (B, 5) champion indices, 0-padded.
            role:       (B, NUM_ROLES) one-hot.
            team:       (B, 2) one-hot — blue=``[1,0]``, red=``[0,1]``.

        Returns:
            (B, num_champions) logits.
        """
        bp = self._aggregate(blue_picks)
        rp = self._aggregate(red_picks)
        bb = self._aggregate(blue_bans)
        rb = self._aggregate(red_bans)
        x = torch.cat([bp, rp, bb, rb, role.float(), team.float()], dim=-1)
        return self.net(x)

    # ------------------------------------------------------------------
    # Convenience predict
    # ------------------------------------------------------------------

    @torch.no_grad()
    def predict_proba(
        self,
        blue_picks: torch.Tensor,
        red_picks: torch.Tensor,
        blue_bans: torch.Tensor,
        red_bans: torch.Tensor,
        role: torch.Tensor,
        team: torch.Tensor,
    ) -> np.ndarray:
        """Return softmax probabilities as a numpy array."""
        self.eval()
        logits = self(blue_picks, red_picks, blue_bans, red_bans, role, team)
        return torch.softmax(logits, dim=-1).cpu().numpy()


# ---------------------------------------------------------------------------
# Training utilities
# ---------------------------------------------------------------------------

def train_epoch(
    model: DraftMLP,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
) -> float:
    """Run one training epoch.

    Args:
        model:     The MLP model.
        loader:    DataLoader yielding ``(X, y)`` batches.  ``X`` is the flat
                   feature vector produced by :class:`~src.features.champion_encoder.DraftStateEncoder`.
        optimizer: Torch optimiser.
        device:    Compute device.

    Returns:
        Mean cross-entropy loss over the epoch.
    """
    model.train()
    criterion = nn.CrossEntropyLoss()
    total_loss = 0.0

    for batch in loader:
        *inputs, y_batch = batch
        inputs = [tensor.to(device) for tensor in inputs]
        y_batch = y_batch.to(device)
        optimizer.zero_grad(set_to_none=True)

        if len(inputs) == 1:
            # Flat-vector path used by the baseline encoder.
            logits = _flat_forward(model, inputs[0].float())
        else:
            blue_picks, red_picks, blue_bans, red_bans, role, team = inputs
            logits = model(
                blue_picks.long(),
                red_picks.long(),
                blue_bans.long(),
                red_bans.long(),
                role.float(),
                team.float(),
            )
        loss = criterion(logits, y_batch)
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        total_loss += loss.item() * len(y_batch)

    return total_loss / len(loader.dataset)


def evaluate(
    model: DraftMLP,
    loader: DataLoader,
    device: torch.device,
) -> dict[str, float]:
    """Evaluate model on a DataLoader.

    Returns:
        Dict with ``"loss"`` and ``"top1_acc"`` keys.
    """
    model.eval()
    criterion = nn.CrossEntropyLoss()
    total_loss = 0.0
    correct = 0

    with torch.no_grad():
        for batch in loader:
            *inputs, y_batch = batch
            inputs = [tensor.to(device) for tensor in inputs]
            y_batch = y_batch.to(device)
            if len(inputs) == 1:
                logits = _flat_forward(model, inputs[0].float())
            else:
                blue_picks, red_picks, blue_bans, red_bans, role, team = inputs
                logits = model(
                    blue_picks.long(),
                    red_picks.long(),
                    blue_bans.long(),
                    red_bans.long(),
                    role.float(),
                    team.float(),
                )
            total_loss += criterion(logits, y_batch).item() * len(y_batch)
            correct += (logits.argmax(dim=1) == y_batch).sum().item()

    n = len(loader.dataset)
    return {"loss": total_loss / n, "top1_acc": correct / n}


def _flat_forward(model: DraftMLP, X: torch.Tensor) -> torch.Tensor:
    """Route a flat feature vector through the MLP's ``net`` directly.

    When using :class:`~src.features.champion_encoder.DraftStateEncoder` the
    input is already a multi-hot concatenation; we bypass the embedding lookup
    and aggregate layers and feed directly into the linear stack.

    This keeps ``DraftMLP`` compatible with both the embedding-based forward
    and the flat feature-vector path used during training with the baseline
    encoder.
    """
    # Feature vector: 4*N + 7 → project down with first Linear layer
    return model.net(X)


def build_dataloader(
    X: np.ndarray,
    y: np.ndarray,
    batch_size: int = 512,
    shuffle: bool = True,
) -> DataLoader:
    """Wrap numpy arrays in a DataLoader."""
    dataset = TensorDataset(
        torch.tensor(X, dtype=torch.float32),
        torch.tensor(y, dtype=torch.long),
    )
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, num_workers=0)


def save_model(model: DraftMLP, path: pathlib.Path | None = None) -> pathlib.Path:
    """Save model state dict."""
    path = path or MODEL_DIR / "mlp_recommender.pt"
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "state_dict": model.state_dict(),
            "num_champions": model.num_champions,
            "input_dim": model.net[0].in_features,
        },
        path,
    )
    logger.info("Saved DraftMLP to %s", path)
    return path


def save_checkpoint(
    model: DraftMLP,
    path: pathlib.Path | None = None,
    optimizer_state: dict | None = None,
    scheduler_state: dict | None = None,
    epoch: int | None = None,
    best_acc: float | None = None,
) -> pathlib.Path:
    """Save a resumable training checkpoint."""
    path = path or MODEL_DIR / "mlp_recommender.pt"
    path.parent.mkdir(parents=True, exist_ok=True)
    ckpt = {
        "state_dict": model.state_dict(),
        "num_champions": model.num_champions,
        "input_dim": model.net[0].in_features,
    }
    if optimizer_state is not None:
        ckpt["optimizer_state"] = optimizer_state
    if scheduler_state is not None:
        ckpt["scheduler_state"] = scheduler_state
    if epoch is not None:
        ckpt["epoch"] = epoch
    if best_acc is not None:
        ckpt["best_acc"] = best_acc
    torch.save(ckpt, path)
    logger.info("Saved DraftMLP checkpoint to %s", path)
    return path


def load_model(path: pathlib.Path, device: torch.device | None = None) -> DraftMLP:
    """Load a DraftMLP from a checkpoint file using a guarded torch.load.

    Uses a short threaded timeout so the UI can recover if file I/O blocks.
    """
    device = device or torch.device("cpu")

    def _torch_load(p):
        return torch.load(p, map_location=device)

    try:
        with concurrent.futures.ThreadPoolExecutor(max_workers=1) as ex:
            future = ex.submit(functools.partial(_torch_load, path))
            ckpt = future.result(timeout=8)
    except concurrent.futures.TimeoutError as e:
        logger.warning("torch.load timeout when loading %s: %s", path, e)
        raise TimeoutError("Model load timed out") from e

    input_dim = ckpt.get("input_dim")
    if input_dim is None:
        input_dim = ckpt["state_dict"]["net.0.weight"].shape[1]
    model = DraftMLP(num_champions=ckpt["num_champions"])
    model.net[0] = nn.Linear(input_dim, model.net[0].out_features)
    model.load_state_dict(ckpt["state_dict"])
    model.to(device)
    logger.info("Loaded DraftMLP from %s", path)
    return model


def load_checkpoint(
    path: pathlib.Path,
    device: torch.device | None = None,
) -> tuple[DraftMLP, dict | None, dict | None, int, float]:
    """Load a resumable checkpoint using direct pickle to avoid torch.load overhead.

    Returns:
        (model, optimizer_state, scheduler_state, epoch, best_acc)
    """
    device = device or torch.device("cpu")

    def _torch_load(p):
        return torch.load(p, map_location=device)

    try:
        with concurrent.futures.ThreadPoolExecutor(max_workers=1) as ex:
            future = ex.submit(functools.partial(_torch_load, path))
            ckpt = future.result(timeout=8)
    except concurrent.futures.TimeoutError as e:
        logger.warning("torch.load timeout when loading %s: %s", path, e)
        raise TimeoutError("Checkpoint load timed out") from e

    model = DraftMLP(num_champions=ckpt["num_champions"])
    input_dim = ckpt.get("input_dim")
    if input_dim is None:
        input_dim = ckpt["state_dict"]["net.0.weight"].shape[1]
    model.net[0] = nn.Linear(input_dim, model.net[0].out_features)
    model.load_state_dict(ckpt["state_dict"])
    model.to(device)

    optimizer_state = ckpt.get("optimizer_state")
    scheduler_state = ckpt.get("scheduler_state")
    epoch = ckpt.get("epoch", 0)
    best_acc = ckpt.get("best_acc", 0.0)

    logger.info("Loaded DraftMLP checkpoint from %s (epoch %d)", path, epoch)
    return model, optimizer_state, scheduler_state, epoch, best_acc
