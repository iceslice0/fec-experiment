import argparse
import csv
import math
from pathlib import Path
from typing import Tuple, Optional

import numpy as np
import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader

# Use either pytorch_lightning or lightning depending on install
try:
    import pytorch_lightning as pl
except ImportError:
    import lightning as pl  # type: ignore


# --------------------------------------------------------------------
# Utilities: seeding, codebook, Hamming distance
# --------------------------------------------------------------------

def set_seed(seed: int = 42):
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def hamming_distance(a: np.ndarray, b: np.ndarray) -> int:
    return int(np.sum(a != b))


def sample_codebook(
    total_codewords: int,
    n: int,
    min_dist_frac: float = 0.2,
    max_tries: int = 100000,
    verbose: bool = True,
) -> np.ndarray:
    """
    Sample a random binary codebook in {0,1}^{total_codewords x n}
    with minimum pairwise Hamming distance >= min_dist_frac * n.
    """
    min_dist = int(math.ceil(min_dist_frac * n))
    codebook = []
    tries = 0
    while len(codebook) < total_codewords and tries < max_tries:
        tries += 1
        candidate = np.random.randint(0, 2, size=(n,), dtype=np.int8)
        ok = True
        for c in codebook:
            if hamming_distance(candidate, c) < min_dist:
                ok = False
                break
        if ok:
            codebook.append(candidate)
            if verbose:
                print(f"Accepted codeword {len(codebook)}/{total_codewords} after {tries} tries")
            tries = 0
    if len(codebook) < total_codewords:
        raise RuntimeError("Failed to sample codebook with required minimum distance")
    codebook_arr = np.stack(codebook, axis=0)  # [total_codewords, n]
    # Report actual pairwise distance statistics.
    d_min = math.inf
    d_sum = 0
    d_count = 0
    for i in range(total_codewords):
        for j in range(i + 1, total_codewords):
            d = hamming_distance(codebook_arr[i], codebook_arr[j])
            d_min = min(d_min, d)
            d_sum += d
            d_count += 1
    d_avg = d_sum / d_count if d_count else 0.0
    if verbose:
        print(f"Sampled codebook with d_min = {d_min} ({d_min/n:.3f} * n)")
        print(f"Sampled codebook with d_avg = {d_avg:.6f} ({d_avg/n:.3f} * n)")
    return codebook_arr


def cluster_codebook(
    codebook: np.ndarray,
    num_classes: int,
    codewords_per_class: int,
) -> np.ndarray:
    """
    Greedily assign codewords to classes by always adding the closest
    unassigned codeword to whichever class has the smallest Hamming
    distance to it.
    """
    total_codewords, n = codebook.shape
    if num_classes * codewords_per_class > total_codewords:
        raise ValueError("Not enough sampled codewords for requested clustering")

    permutation = np.random.permutation(total_codewords)
    clusters = [[int(permutation[i])] for i in range(num_classes)]
    unassigned = list(permutation[num_classes:])

    while any(len(cluster) < codewords_per_class for cluster in clusters):
        best_pair = None
        best_dist = math.inf
        for idx in unassigned:
            for class_idx, cluster in enumerate(clusters):
                if len(cluster) >= codewords_per_class:
                    continue
                dist = min(
                    hamming_distance(codebook[idx], codebook[member_idx])
                    for member_idx in cluster
                )
                if dist < best_dist:
                    best_dist = dist
                    best_pair = (idx, class_idx)
        if best_pair is None:
            break
        idx, class_idx = best_pair
        clusters[class_idx].append(idx)
        unassigned.remove(idx)

    for class_idx, members in enumerate(clusters):
        if len(members) < codewords_per_class:
            raise RuntimeError(
                f"Unable to fill class {class_idx} with {codewords_per_class} members"
            )

    grouped = np.stack(
        [codebook[np.array(cluster, dtype=np.int64)] for cluster in clusters], axis=0
    )  # [num_classes, codewords_per_class, n]
    return grouped


def save_codebook(path: str | Path, codebook: np.ndarray, args) -> None:
    """Save the clustered codebook as one CSV row per class codeword."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    bit_fields = [f"bit_{j:04d}" for j in range(args.n)]
    fieldnames = [
        "C",
        "m",
        "n",
        "q",
        "min_dist_frac",
        "seed",
        "class_id",
        "codeword_id",
        *bit_fields,
    ]

    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for class_id in range(args.C):
            for codeword_id in range(args.m):
                bits = codebook[class_id, codeword_id].astype(np.int8, copy=False)
                row = {
                    "C": args.C,
                    "m": args.m,
                    "n": args.n,
                    "q": args.q,
                    "min_dist_frac": args.min_dist_frac,
                    "seed": args.seed,
                    "class_id": class_id,
                    "codeword_id": codeword_id,
                }
                row.update({field: int(bit) for field, bit in zip(bit_fields, bits)})
                writer.writerow(row)


# --------------------------------------------------------------------
# Dataset and DataModule
# --------------------------------------------------------------------

class BernoulliCodeDataset(Dataset):
    """
    Synthetic dataset: X is a noisy version of a codeword sampled from a
    randomly chosen class, and labels correspond to the class index.
    """
    def __init__(
        self,
        codebook: np.ndarray,
        q: float,
        size: int,
        device: Optional[torch.device] = None,
    ):
        super().__init__()
        self.codebook = codebook  # [C, m, n]
        self.q = float(q)
        self.size = int(size)
        self.C, self.m, self.n = self.codebook.shape
        self.total_codewords = self.C * self.m

        flat_codebook = codebook.reshape(self.total_codewords, self.n)

        # Pre-generate samples for reproducibility
        class_indices = np.random.randint(0, self.C, size=self.size, dtype=np.int64)
        offsets = np.random.randint(0, self.m, size=self.size, dtype=np.int64)
        codeword_indices = class_indices * self.m + offsets
        C_y = flat_codebook[codeword_indices]  # [size, n]
        noise = np.random.rand(self.size, self.n) < self.q   # E_j ~ Bern(q)
        X = np.logical_xor(C_y, noise).astype(np.int8)       # X = c ⊕ E

        self.X = torch.from_numpy(X.astype(np.float32))      # {0,1} as float
        self.y = torch.from_numpy(class_indices.astype(np.int64))

        if device is not None:
            self.X = self.X.to(device)
            self.y = self.y.to(device)

    def __len__(self):
        return self.size

    def __getitem__(self, idx: int):
        return self.X[idx], self.y[idx]


class BernoulliCodeDataModule(pl.LightningDataModule):
    def __init__(
        self,
        codebook: np.ndarray,
        q: float,
        train_size: int = 50000,
        val_size: int = 10000,
        test_size: int = 100000,
        batch_size: int = 512,
        num_workers: int = 4,
        device: Optional[torch.device] = None,
    ):
        super().__init__()
        self.codebook = codebook
        self.q = q
        self.train_size = train_size
        self.val_size = val_size
        self.test_size = test_size
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.device = device

    def setup(self, stage=None):
        self.train_ds = BernoulliCodeDataset(
            self.codebook, self.q, self.train_size, device=None
        )
        self.val_ds = BernoulliCodeDataset(
            self.codebook, self.q, self.val_size, device=None
        )
        self.test_ds = BernoulliCodeDataset(
            self.codebook, self.q, self.test_size, device=None
        )

    def train_dataloader(self):
        return DataLoader(
            self.train_ds,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            pin_memory=True,
        )

    def val_dataloader(self):
        return DataLoader(
            self.val_ds,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=True,
        )

    def test_dataloader(self):
        return DataLoader(
            self.test_ds,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=True,
        )


# --------------------------------------------------------------------
# Linear softmax classifier (single FC layer)
# --------------------------------------------------------------------

class LinearSoftmaxClassifier(pl.LightningModule):
    def __init__(
        self,
        n: int,
        num_classes: int,
        lr: float = 1e-2,
        weight_decay: float = 0.0,
        scheduler_T_max: int = 20,
    ):
        super().__init__()
        self.save_hyperparameters()
        self.fc = nn.Linear(n, num_classes, bias=True)
        self.criterion = nn.CrossEntropyLoss()

    def forward(self, x):
        # x: [B, n], entries in {0,1}
        return self.fc(x)

    def training_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)
        loss = self.criterion(logits, y)
        preds = logits.argmax(dim=1)
        acc = (preds == y).float().mean()
        self.log("train_loss", loss, on_step=False, on_epoch=True)
        self.log("train_acc", acc, on_step=False, on_epoch=True)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)
        loss = self.criterion(logits, y)
        preds = logits.argmax(dim=1)
        acc = (preds == y).float().mean()
        self.log("val_loss", loss, on_step=False, on_epoch=True)
        self.log("val_acc", acc, on_step=False, on_epoch=True)

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(
            self.parameters(),
            lr=self.hparams.lr,
            weight_decay=self.hparams.weight_decay,
        )
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=self.hparams.scheduler_T_max
        )
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "interval": "epoch",
                "frequency": 1,
            },
        }


def _class_min_distances(
    xb: torch.Tensor,
    codebook_flat: torch.Tensor,
    num_classes: int,
    codewords_per_class: int,
) -> torch.Tensor:
    """
    Compute the minimum Hamming distance from each sample to each class by
    taking the minimum over the codewords that belong to each class.
    """
    B = xb.size(0)
    diffs = xb.unsqueeze(1) != codebook_flat.unsqueeze(0)  # [B, M, n]
    dists = diffs.sum(dim=2).float()                       # [B, M]
    class_dists = (
        dists.view(B, num_classes, codewords_per_class).min(dim=2).values
    )  # [B, num_classes]
    return class_dists


def _class_sum_distances(
    xb: torch.Tensor,
    codebook_flat: torch.Tensor,
    num_classes: int,
    codewords_per_class: int,
) -> torch.Tensor:
    """
    Compute the total Hamming distance from each sample to each class by
    summing the distances to each codeword in the class.
    """
    B = xb.size(0)
    diffs = xb.unsqueeze(1) != codebook_flat.unsqueeze(0)  # [B, M, n]
    dists = diffs.sum(dim=2).float()                       # [B, M]
    class_dists = dists.view(B, num_classes, codewords_per_class).sum(
        dim=2
    )  # [B, num_classes]
    return class_dists


# --------------------------------------------------------------------
# Measurement (1): ML Hamming decoder vs learned softmax
# --------------------------------------------------------------------

def evaluate_ml_decoder(
    model: nn.Module,
    codebook: np.ndarray,
    dataloader: DataLoader,
    device: torch.device,
):
    """
    Returns:
      coincidence_prob = P[ argmax z_i(X) == argmin d_H(X, class_i) ]
      A_soft, A_ml = accuracies of softmax and ML decoder over classes.
    """
    model.eval()
    C, m, n = codebook.shape
    codebook_flat = torch.from_numpy(codebook.reshape(C * m, n).astype(np.int64)).to(device)

    total = 0
    correct_soft = 0
    correct_ml = 0
    coincide = 0

    with torch.no_grad():
        for x, y in dataloader:
            x = x.to(device)
            y = y.to(device)
            B = x.size(0)

            logits = model(x)
            soft_pred = logits.argmax(dim=1)

            xb = (x > 0.5).long()
            class_dists = _class_min_distances(xb, codebook_flat, C, m)  # [B, C]
            ml_pred = torch.argmin(class_dists, dim=1)

            total += B
            correct_soft += (soft_pred == y).sum().item()
            correct_ml += (ml_pred == y).sum().item()
            coincide += (soft_pred == ml_pred).sum().item()

    A_soft = correct_soft / total
    A_ml = correct_ml / total
    coincidence_prob = coincide / total
    return coincidence_prob, A_soft, A_ml


# --------------------------------------------------------------------
# Simple 1D linear regression helper
# --------------------------------------------------------------------

def linear_regression(x: np.ndarray, y: np.ndarray) -> Tuple[float, float, float]:
    """
    Univariate linear regression: y ≈ intercept + slope * x.
    Returns (slope, intercept, R^2).
    """
    x = x.astype(np.float64)
    y = y.astype(np.float64)
    x_mean = x.mean()
    y_mean = y.mean()
    vx = x - x_mean
    vy = y - y_mean
    var_x = np.mean(vx * vx)
    if var_x == 0.0:
        return float("nan"), float("nan"), float("nan")
    cov_xy = np.mean(vx * vy)
    slope = cov_xy / var_x
    intercept = y_mean - slope * x_mean
    y_pred = intercept + slope * x
    ss_tot = np.sum((y - y_mean) ** 2)
    ss_res = np.sum((y - y_pred) ** 2)
    R2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else float("nan")
    return slope, intercept, R2


# --------------------------------------------------------------------
# Measurement (2): logits as affine transforms of Hamming distance
# --------------------------------------------------------------------

def measure_logits_vs_hamming(
    model: nn.Module,
    codebook: np.ndarray,
    dataloader: DataLoader,
    device: torch.device,
):
    """
    Returns (alpha, beta, R^2) for global fit:
      Δz_{i,j}(x) ≈ alpha + beta * Δd_{i,j}(x)
    aggregated over all i<j and all test samples x.
    """
    model.eval()
    C, m, n = codebook.shape
    codebook_flat = torch.from_numpy(codebook.reshape(C * m, n).astype(np.int64)).to(device)

    all_logits = []
    all_dists = []

    with torch.no_grad():
        for x, y in dataloader:
            x = x.to(device)
            logits = model(x)  # [B,C]
            #xb = (x > 0.5).long()
            class_dists = _class_sum_distances(x, codebook_flat, C, m)  # [B,C]
            all_logits.append(logits.cpu())
            all_dists.append(class_dists.cpu())

    Z = torch.cat(all_logits, dim=0).numpy()  # [B,C]
    D = torch.cat(all_dists, dim=0).numpy()   # [B,C]
    B = Z.shape[0]

    delta_z_list = []
    delta_d_list = []
    for i in range(C):
        for j in range(i + 1, C):
            dz = Z[:, j] - Z[:, i]
            dd = D[:, j] - D[:, i]
            delta_z_list.append(dz)
            delta_d_list.append(dd)

    Delta_z = np.concatenate(delta_z_list, axis=0)
    Delta_d = np.concatenate(delta_d_list, axis=0)

    slope, intercept, R2 = linear_regression(Delta_d, Delta_z)
    return slope, intercept, R2


# --------------------------------------------------------------------
# Measurement (3): cross-entropy vs Hamming margin
# --------------------------------------------------------------------

def measure_ce_vs_hamming_margin(
    model: nn.Module,
    codebook: np.ndarray,
    dataloader: DataLoader,
    device: torch.device,
    num_bins: int = 20,
):
    """
    For correctly classified samples, computes gamma_t(x) and CE loss.
    Bins by gamma, computes mean loss in each bin, and fits:

      log \bar{L}(gamma) ≈ intercept + slope * gamma

    Returns:
      bin_centers, mean_losses, slope, intercept, R^2
    """
    model.eval()
    C, m, n = codebook.shape
    codebook_flat = torch.from_numpy(codebook.reshape(C * m, n).astype(np.int64)).to(device)

    gammas = []
    losses = []

    ce_loss = nn.CrossEntropyLoss(reduction="none")

    with torch.no_grad():
        for x, y in dataloader:
            x = x.to(device)
            y = y.to(device)
            logits = model(x)  # [B,C]
            xb = (x > 0.5).long()
            class_dists = _class_min_distances(xb, codebook_flat, C, m)  # [B,C]
            B = x.size(0)

            # Softmax predictions
            soft_pred = logits.argmax(dim=1)
            correct_mask = soft_pred.eq(y)

            if correct_mask.sum() == 0:
                continue

            d_t = class_dists[torch.arange(B, device=device), y]  # [B]
            diff = class_dists - d_t.unsqueeze(1)                 # [B,C]
            diff[torch.arange(B, device=device), y] = float("inf")
            gamma = diff.min(dim=1).values                        # [B]

            loss_vec = ce_loss(logits, y)                   # [B]

            gammas.append(gamma[correct_mask].cpu())
            losses.append(loss_vec[correct_mask].cpu())

    if not gammas:
        raise RuntimeError("No correctly classified samples to measure margin statistics")

    gamma_all = torch.cat(gammas).numpy()
    loss_all = torch.cat(losses).numpy()

    # Quantile binning in gamma
    quantiles = np.linspace(0.0, 1.0, num_bins + 1)
    bin_edges = np.quantile(gamma_all, quantiles)
    bin_edges = np.unique(bin_edges)
    if bin_edges.size <= 2:
        raise RuntimeError("Not enough distinct gamma values for binning")

    num_bins_eff = bin_edges.size - 1
    bin_centers = []
    mean_losses = []

    for k in range(num_bins_eff):
        left = bin_edges[k]
        right = bin_edges[k + 1]
        mask = (gamma_all >= left) & (
            gamma_all <= right if k == num_bins_eff - 1 else gamma_all < right
        )
        if np.sum(mask) == 0:
            continue
        bin_centers.append(gamma_all[mask].mean())
        mean_losses.append(loss_all[mask].mean())

    bin_centers = np.array(bin_centers)
    mean_losses = np.array(mean_losses)

    log_losses = np.log(mean_losses)
    slope, intercept, R2 = linear_regression(bin_centers, log_losses)
    # log L ≈ intercept + slope * gamma
    return bin_centers, mean_losses, slope, intercept, R2


# --------------------------------------------------------------------
# Measurement (4): Jeffreys divergence vs Hamming distance
# (analytic for the Bernoulli channel)
# --------------------------------------------------------------------

def jeffreys_bernoulli(p: float, q: float) -> float:
    """Jeffreys divergence between Bernoulli(p) and Bernoulli(q)."""
    eps = 1e-12

    def kl(a, b):
        a = np.clip(a, eps, 1 - eps)
        b = np.clip(b, eps, 1 - eps)
        return a * np.log(a / b) + (1 - a) * np.log((1 - a) / (1 - b))

    return float(kl(p, q) + kl(q, p))


def measure_jeffreys_vs_hamming(
    codebook: np.ndarray,
    q: float,
):
    """
    For the BSC(q) with product Bernoulli structure:
      J(p(.|i), p(.|j)) = d_H(c_i, c_j) * J_bit,
    where J_bit = J(Ber(q), Ber(1-q)).
    Returns (pairwise_dists, Jeffreys_values, J_bit) computed using the
    minimum Hamming distance between codewords belonging to each class.
    """
    C, m, n = codebook.shape
    J_bit = jeffreys_bernoulli(q, 1.0 - q)
    dists = []
    Js = []
    for i in range(C):
        for j in range(i + 1, C):
            min_d = math.inf
            for cw_i in codebook[i]:
                for cw_j in codebook[j]:
                    min_d = min(min_d, hamming_distance(cw_i, cw_j))
            dists.append(int(min_d))
            Js.append(min_d * J_bit)
    return np.array(dists), np.array(Js), J_bit


# --------------------------------------------------------------------
# Main script
# --------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Bernoulli codeword experiment")
    parser.add_argument("--C", type=int, default=32, help="number of classes")
    parser.add_argument("--m", type=int, default=1, help="number of codewords per class")
    parser.add_argument("--n", type=int, default=256, help="code length")
    parser.add_argument("--q", type=float, default=0.4, help="BSC crossover probability")
    parser.add_argument("--min_dist_frac", type=float, default=0.3,
                        help="minimum relative Hamming distance for codebook")
    parser.add_argument("--train_size", type=int, default=50000)
    parser.add_argument("--val_size", type=int, default=10000)
    parser.add_argument("--test_size", type=int, default=100000)
    parser.add_argument("--batch_size", type=int, default=512)
    parser.add_argument("--max_epochs", type=int, default=20)
    parser.add_argument("--lr", type=float, default=1e-2)
    parser.add_argument("--weight_decay", type=float, default=0.0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--gpus", type=int, default=1,
                        help="number of GPUs to use (0 for CPU)")
    parser.add_argument(
        "--codebook-out",
        default=None,
        help="Optional CSV path for saving the sampled and clustered codebook.",
    )
    args = parser.parse_args()

    set_seed(args.seed)

    total_codewords = args.C * args.m
    print("Sampling codebook...")
    flat_codebook = sample_codebook(
        total_codewords=total_codewords,
        n=args.n,
        min_dist_frac=args.min_dist_frac,
        verbose=True,
    )
    print(
        f"Clustering {total_codewords} sampled codewords into "
        f"{args.C} classes with {args.m} codewords each"
    )
    codebook = cluster_codebook(flat_codebook, args.C, args.m)
    if args.codebook_out:
        save_codebook(args.codebook_out, codebook, args)
        print(f"Saved codebook: {args.codebook_out}")

    device = torch.device(
        "cuda" if args.gpus > 0 and torch.cuda.is_available() else "cpu"
    )
    print(f"Using device: {device}")

    dm = BernoulliCodeDataModule(
        codebook=codebook,
        q=args.q,
        train_size=args.train_size,
        val_size=args.val_size,
        test_size=args.test_size,
        batch_size=args.batch_size,
        num_workers=4,
        device=None,
    )
    dm.setup()

    model = LinearSoftmaxClassifier(
        n=args.n,
        num_classes=args.C,
        lr=args.lr,
        weight_decay=args.weight_decay,
        scheduler_T_max=args.max_epochs,
    )

    trainer = pl.Trainer(
        max_epochs=args.max_epochs,
        accelerator="gpu" if device.type == "cuda" else "cpu",
        devices=args.gpus if device.type == "cuda" else 1,
        log_every_n_steps=50,
    )
    trainer.fit(model, dm)

    # Move model to device for evaluation
    model.to(device)

    # --------------------------------------------------------------
    # (1) Recovery of ML Hamming decoder
    # --------------------------------------------------------------
    print("\n=== Measurement (1): ML decoder recovery ===")
    coincidence, A_soft, A_ml = evaluate_ml_decoder(
        model, codebook, dm.test_dataloader(), device
    )
    print(f"Coincidence probability P(soft == ML): {coincidence:.6f}")
    print(f"Softmax accuracy A_soft: {A_soft:.6f}")
    print(f"ML Hamming decoder accuracy A_ml: {A_ml:.6f}")

    # --------------------------------------------------------------
    # (2) Logits vs Hamming distance
    # --------------------------------------------------------------
    print("\n=== Measurement (2): logits vs Hamming distance ===")
    slope, intercept, R2 = measure_logits_vs_hamming(
        model, codebook, dm.test_dataloader(), device
    )
    print(f"Δz ≈ intercept + slope * Δd with intercept={intercept:.6f}, slope={slope:.6f}, R^2={R2:.6f}")

    # # --------------------------------------------------------------
    # # (3) Cross-entropy vs Hamming margin
    # # --------------------------------------------------------------
    # print("\n=== Measurement (3): cross-entropy vs Hamming margin ===")
    # bin_centers, mean_losses, slope, intercept, R2_margin = \
    #     measure_ce_vs_hamming_margin(
    #         model, codebook, dm.test_dataloader(), device, num_bins=20
    #     )
    # print("Bin centers (gamma):", bin_centers)
    # print("Mean CE losses:", mean_losses)
    # print(
    #     f"log L ≈ intercept + slope * gamma with "
    #     f"intercept={intercept:.6f}, slope={slope:.6f}, R^2={R2_margin:.6f}"
    # )

    # # --------------------------------------------------------------
    # # (4) Jeffreys divergence vs Hamming distance
    # # --------------------------------------------------------------
    # print("\n=== Measurement (4): Jeffreys divergence vs Hamming distance ===")
    # dists, Js, J_bit = measure_jeffreys_vs_hamming(codebook, args.q)
    # print(f"Per-bit Jeffreys divergence J_bit = {J_bit:.6f}")
    # print("Example pairs (d_H, J):")
    # for k in range(min(5, len(dists))):
    #     print(f"  d_H = {dists[k]}, J = {Js[k]:.6f}")

    print("\nDone.")


if __name__ == "__main__":
    main()
