"""Model definition and training utilities for the Osteosarcoma project."""

from __future__ import annotations

import math
import random
from pathlib import Path
from typing import Iterable, Sequence

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

from sklearn.metrics import (
    accuracy_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
    precision_recall_fscore_support,
    roc_curve,
)

from data import (
    CSV_PATH,
    IMG_ROOTS,
    RAD_CACHE_DIR,
    DEVICE,
    IMG_SIZE,
    TumorDataset,
    build_or_load_radiomics_cache,
    load_df_from_csv,
)

BACKBONE = "inception_v3"
MAX_EPOCHS = 150
MIN_EPOCHS = 60
PATIENCE = 50
LR = 1e-4
NUM_WORKERS = 4
PIN_MEMORY = DEVICE.type == "cuda"

W_MIN, W_MAX = 0.2, 3.0
LS_MIN, LS_MAX = -math.log(W_MAX), -math.log(W_MIN)
REG_SCALE = 0.2

RAD_MAX_IMAGES: int | None = None
RAD_DEBUG = False

VAL_FRACTION_FROM_VALPAT = 0.13
TRAIN_PATIENTS = ["Case-3", "P9"]
VAL_SPLIT_PATIENTS = ["Case-48"]
TEST_PATIENT = "Case-4"

CKPT_DIR = Path("./checkpoints")
CKPT_DIR.mkdir(parents=True, exist_ok=True)

MC_EVAL_SEEDS = [1, 2, 3, 4, 5]


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():  # pragma: no cover - GPU specific
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


class MultiModalHierNet(nn.Module):
    """CNN + radiomics fusion network with learnable task weights."""

    def __init__(
        self,
        backbone: str,
        rad_in_dim: int,
        *,
        rad_hidden: int = 256,
        fusion_dim: int = 256,
        att_hidden: int = 128,
        p_drop: float = 0.3,
    ) -> None:
        super().__init__()

        import timm

        self.cnn = timm.create_model(
            backbone, pretrained=True, num_classes=0, global_pool="avg"
        )
        with torch.no_grad():
            dummy = torch.zeros(1, 3, IMG_SIZE, IMG_SIZE)
            cnn_dim = self.cnn(dummy).shape[1]

        self.cnn_proj = nn.Sequential(
            nn.Linear(cnn_dim, fusion_dim),
            nn.BatchNorm1d(fusion_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(p_drop),
        )

        self.rad_net = nn.Sequential(
            nn.BatchNorm1d(rad_in_dim),
            nn.Linear(rad_in_dim, rad_hidden),
            nn.ReLU(inplace=True),
            nn.Dropout(p_drop),
            nn.Linear(rad_hidden, fusion_dim),
            nn.ReLU(inplace=True),
        )

        self.att_gate = nn.Sequential(
            nn.Linear(2 * fusion_dim, att_hidden),
            nn.ReLU(inplace=True),
            nn.Dropout(p_drop),
            nn.Linear(att_hidden, 2),
            nn.Softmax(dim=1),
        )

        self.head_coarse = nn.Linear(fusion_dim, 2)
        self.head_fine = nn.Linear(fusion_dim, 2)

        init_wA = 0.5
        init_wB = 1.5
        self.log_sigma_a = nn.Parameter(
            torch.tensor([-math.log(init_wA)], dtype=torch.float32)
        )
        self.log_sigma_b = nn.Parameter(
            torch.tensor([-math.log(init_wB)], dtype=torch.float32)
        )

    def forward(self, x_img: torch.Tensor, x_rad: torch.Tensor):
        f_img = self.cnn_proj(self.cnn(x_img))
        f_rad = self.rad_net(x_rad)
        h = torch.cat([f_img, f_rad], dim=1)
        w = self.att_gate(h)
        w_img, w_rad = w[:, :1], w[:, 1:]
        f_fused = w_img * f_img + w_rad * f_rad
        logits_a = self.head_coarse(f_fused)
        logits_b = self.head_fine(f_fused)
        return logits_a, logits_b


def hierarchical_loss_uncertainty(
    model: MultiModalHierNet,
    logits_a: torch.Tensor,
    logits_b: torch.Tensor,
    y3: torch.Tensor,
    *,
    W_A: torch.Tensor | None = None,
    W_B: torch.Tensor | None = None,
    reg_scale: float = REG_SCALE,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    y_a = (y3 != 0).long()
    y_b = torch.where(y3 == 1, 0, torch.where(y3 == 2, 1, -1)).long()
    mask_b = y_b >= 0

    loss_a = nn.functional.cross_entropy(logits_a, y_a, weight=W_A)
    if mask_b.any():
        loss_b = nn.functional.cross_entropy(
            logits_b[mask_b], y_b[mask_b], weight=W_B
        )
    else:
        loss_b = logits_a.new_tensor(0.0)

    inv_var_a = torch.exp(-model.log_sigma_a)
    inv_var_b = torch.exp(-model.log_sigma_b)
    loss = inv_var_a * loss_a + inv_var_b * loss_b + reg_scale * (
        model.log_sigma_a + model.log_sigma_b
    )
    return loss, loss_a.detach(), loss_b.detach()


def fuse_probs_to_three_classes(
    logits_a: torch.Tensor, logits_b: torch.Tensor
) -> torch.Tensor:
    pa = torch.softmax(logits_a, dim=1)
    pb = torch.softmax(logits_b, dim=1)
    p_non_tumor = pa[:, 0:1]
    p_tumor = pa[:, 1:2]
    p_nvt = pb[:, 0:1]
    p_vi = pb[:, 1:2]
    p_cls0 = p_non_tumor
    p_cls1 = p_tumor * p_nvt
    p_cls2 = p_tumor * p_vi
    return torch.cat([p_cls0, p_cls1, p_cls2], dim=1)


def set_mc_dropout(model: nn.Module, enable: bool = True) -> None:
    if not enable:
        model.eval()
        return
    model.eval()
    for module in model.modules():
        if isinstance(module, (nn.Dropout, nn.Dropout1d, nn.Dropout2d, nn.AlphaDropout)):
            module.train()


def _sens_spec_at_targets(
    y_true_bin: np.ndarray,
    y_score: np.ndarray,
    *,
    spec_target: float = 0.9,
    sens_target: float = 0.9,
) -> tuple[float, float]:
    if len(np.unique(y_true_bin)) < 2:
        return float("nan"), float("nan")

    fpr, tpr, _ = roc_curve(y_true_bin, y_score)
    spec = 1.0 - fpr

    spec_rev = spec[::-1]
    tpr_rev = tpr[::-1]

    if spec_target < spec_rev[0] or spec_target > spec_rev[-1]:
        sens_at_spec = float("nan")
    else:
        sens_at_spec = float(np.interp(spec_target, spec_rev, tpr_rev))

    if sens_target < tpr[0] or sens_target > tpr[-1]:
        spec_at_sens = float("nan")
    else:
        spec_at_sens = float(np.interp(sens_target, tpr, spec))

    return sens_at_spec, spec_at_sens


def compute_overall_and_perclass(
    y_true: Sequence[int],
    y_pred: Sequence[int],
    y_prob: np.ndarray,
    classes: Sequence[str],
) -> tuple[dict[str, float], list[dict[str, float]]]:
    y_true_arr = np.array(y_true)
    y_pred_arr = np.array(y_pred)

    overall = {
        "accuracy": accuracy_score(y_true_arr, y_pred_arr),
        "f1_weighted": f1_score(y_true_arr, y_pred_arr, average="weighted"),
        "f1_macro": f1_score(y_true_arr, y_pred_arr, average="macro"),
    }

    y_true_bin = np.eye(len(classes))[y_true_arr]
    try:
        overall["roc_auc_macro"] = roc_auc_score(y_true_bin, y_prob, multi_class="ovr")
    except ValueError:
        overall["roc_auc_macro"] = float("nan")

    cm = confusion_matrix(y_true_arr, y_pred_arr, labels=list(range(len(classes))))
    per_class = []
    for idx, cls in enumerate(classes):
        tp = cm[idx, idx]
        fn = cm[idx, :].sum() - tp
        fp = cm[:, idx].sum() - tp
        tn = cm.sum() - tp - fn - fp
        specificity = tn / (tn + fp) if (tn + fp) > 0 else float("nan")
        precision, recall, f1, support = precision_recall_fscore_support(
            (y_true_arr == idx).astype(int),
            (y_pred_arr == idx).astype(int),
            average="binary",
            zero_division=0,
        )
        sens_at_spec90, spec_at_sens90 = _sens_spec_at_targets(
            y_true_bin[:, idx], y_prob[:, idx]
        )
        try:
            auc = roc_auc_score(y_true_bin[:, idx], y_prob[:, idx])
        except ValueError:
            auc = float("nan")
        per_class.append(
            {
                "cls": cls,
                "support": int(support),
                "precision": float(precision),
                "recall": float(recall),
                "specificity": float(specificity),
                "sens_at_spec90": float(sens_at_spec90),
                "spec_at_sens90": float(spec_at_sens90),
                "f1": float(f1),
                "auc": float(auc),
            }
        )

    return overall, per_class


def print_metrics_block(
    title: str,
    overall: dict[str, float],
    per_class: Sequence[dict[str, float]],
) -> None:
    print("\n" + title)
    print("=" * len(title))
    for k, v in overall.items():
        print(f"{k:>16}: {v:.4f}")
    hdr = (
        f"{'Class':<22} {'Support':>7}  "
        f"{'Prec':>6}  {'Recall':>6}  {'Spec':>6}  "
        f"{'Sens@Spec90':>12}  {'Spec@Sens90':>12}  "
        f"{'F1':>6}  {'AUC':>6}"
    )
    print("-" * len(hdr))
    print(hdr)
    for r in per_class:
        print(
            f"{r['cls']:<22} {r['support']:>7d}  "
            f"{r['precision']:>6.3f}  {r['recall']:>6.3f}  {r['specificity']:>6.3f}  "
            f"{r['sens_at_spec90']:>12.3f}  {r['spec_at_sens90']:>12.3f}  "
            f"{r['f1']:>6.3f}  {r['auc']:>6.3f}"
        )


def stratified_val_from_patient(
    df_patient: pd.DataFrame, val_fraction: float, seed: int
) -> tuple[pd.DataFrame, pd.DataFrame]:
    val_idx: list[pd.Index] = []
    for _, group in df_patient.groupby("y", sort=False):
        n = len(group)
        if n <= 1:
            n_val = 0
        else:
            n_val = int(round(val_fraction * n))
            n_val = max(1, n_val)
            n_val = min(n_val, n - 1)
        if n_val > 0:
            val_idx.append(group.sample(n=n_val, random_state=seed).index)
    if val_idx:
        val_idx_combined = pd.Index(np.concatenate(val_idx))
    else:
        val_idx_combined = pd.Index([])
    val_df = df_patient.loc[val_idx_combined].copy()
    remainder_df = df_patient.drop(val_idx_combined).copy()
    return val_df, remainder_df


def _build_val_from_patients(
    train_pool_df: pd.DataFrame,
    patients: Iterable[str],
    frac: float,
    seed_: int,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    patients = [p for p in patients if p in train_pool_df["patient"].unique()]
    if not patients:
        raise ValueError(
            "Validation patients do not match any patient in the training pool."
        )

    val_parts: list[pd.DataFrame] = []
    remainder_parts: list[pd.DataFrame] = []
    for patient in patients:
        subset = train_pool_df[train_pool_df["patient"] == patient].copy()
        val_p, rem_p = stratified_val_from_patient(subset, frac, seed_)
        val_parts.append(val_p)
        remainder_parts.append(rem_p)

    val_df_local = (
        pd.concat(val_parts, axis=0).sort_index() if len(val_parts) else train_pool_df.iloc[0:0]
    )
    remainder_df_local = (
        pd.concat(remainder_parts, axis=0).sort_index()
        if len(remainder_parts)
        else train_pool_df.copy()
    )
    return val_df_local, remainder_df_local


def train_model_once(seed: int = 2025):
    set_seed(seed)
    df, classes = load_df_from_csv(CSV_PATH, IMG_ROOTS)

    rad_df, feat_names, _ = build_or_load_radiomics_cache(
        df,
        RAD_CACHE_DIR,
        pixelSpacing=(1.0, 1.0, 1.0),
        voxelArrayShift=0.0,
        binWidth=5.0,
        device=DEVICE,
        max_images=RAD_MAX_IMAGES,
        debug=RAD_DEBUG,
    )

    df = df[df["path"].isin(rad_df["path"])].reset_index(drop=True)
    rad_df = rad_df[rad_df["path"].isin(df["path"])].reset_index(drop=True)

    train_pool = df[df["patient"].isin(TRAIN_PATIENTS)].copy()
    test_df = df[df["patient"] == TEST_PATIENT].copy()

    print("\n[Sanity] Patients in TRAIN pool:", sorted(train_pool["patient"].unique()))
    print("[Sanity] Patients in TEST set:", sorted(test_df["patient"].unique()))
    assert TEST_PATIENT not in set(train_pool["patient"])
    assert set(test_df["patient"]) == {TEST_PATIENT}

    val_df, _ = _build_val_from_patients(
        train_pool, VAL_SPLIT_PATIENTS, VAL_FRACTION_FROM_VALPAT, seed
    )

    train_df = pd.concat(
        [train_pool[~train_pool.index.isin(val_df.index)]], axis=0
    ).sort_index()

    print("\n[Training split sizes — Patient-level]")
    print(f"  TRAIN images (pool minus VAL-slices): {len(train_df)}")
    print(
        f"  VAL images (from {VAL_SPLIT_PATIENTS}): {len(val_df)}"
    )
    print(f"  TEST images (all {TEST_PATIENT}):     {len(test_df)}")
    print("[Sanity] Per-class counts (TRAIN):", train_df["label"].value_counts().to_dict())
    print("[Sanity] Per-class counts (VAL):  ", val_df["label"].value_counts().to_dict())
    print("[Sanity] Per-class counts (TEST): ", test_df["label"].value_counts().to_dict())

    train_paths = set(train_df["path"])
    rad_train = (
        rad_df[rad_df["path"].isin(train_paths)][feat_names].to_numpy(dtype=np.float32)
    )
    rad_mean = rad_train.mean(axis=0)
    rad_std = rad_train.std(axis=0, ddof=0)
    rad_stats = {"mean": rad_mean, "std": rad_std}

    loaders = {
        "train": DataLoader(
            TumorDataset(train_df, rad_df, feat_names, rad_stats, train=True),
            batch_size=16,
            shuffle=True,
            num_workers=NUM_WORKERS,
            pin_memory=PIN_MEMORY,
        ),
        "val": DataLoader(
            TumorDataset(val_df, rad_df, feat_names, rad_stats, train=False),
            batch_size=16,
            shuffle=False,
            num_workers=NUM_WORKERS,
            pin_memory=PIN_MEMORY,
        ),
    }

    model = MultiModalHierNet(BACKBONE, rad_in_dim=len(feat_names)).to(DEVICE)

    y_a_train = (train_df["y"].values != 0).astype(int)
    cnt_a = np.bincount(y_a_train, minlength=2).astype(float)
    w_a = 1.0 / (cnt_a / cnt_a.sum() + 1e-9)
    w_a = w_a * (2.0 / w_a.sum())
    W_A = torch.tensor(w_a, dtype=torch.float32, device=DEVICE)

    tumor_rows = train_df[train_df["y"] != 0]
    y_b_train = np.where(tumor_rows["y"].values == 1, 0, 1)
    cnt_b = np.bincount(y_b_train, minlength=2).astype(float)
    w_b = 1.0 / (cnt_b / cnt_b.sum() + 1e-9)
    w_b = w_b * (2.0 / w_b.sum())
    W_B = torch.tensor(w_b, dtype=torch.float32, device=DEVICE)

    cnt_3 = train_df["y"].value_counts().sort_index().astype(float)
    w_3 = 1.0 / (cnt_3 / cnt_3.sum() + 1e-9)
    w_3 = w_3 * (3.0 / w_3.sum())
    W_3 = w_3

    optimizer = optim.Adam(model.parameters(), lr=LR)

    best_f1 = -float("inf")
    best_state: dict[str, torch.Tensor] | None = None
    wait = 0
    ckpt_path = CKPT_DIR / "best_patient_split.pt"

    for epoch in range(1, MAX_EPOCHS + 1):
        model.train()
        train_loss = 0.0
        seen_train = 0
        for x_img, x_rad, y in loaders["train"]:
            x_img = x_img.to(DEVICE, non_blocking=True)
            x_rad = x_rad.to(DEVICE, non_blocking=True)
            y = y.to(DEVICE, non_blocking=True)

            optimizer.zero_grad()
            logits_a, logits_b = model(x_img, x_rad)
            loss, loss_a, loss_b = hierarchical_loss_uncertainty(
                model, logits_a, logits_b, y, W_A=W_A, W_B=W_B, reg_scale=REG_SCALE
            )
            loss.backward()
            optimizer.step()

            with torch.no_grad():
                model.log_sigma_a.clamp_(LS_MIN, LS_MAX)
                model.log_sigma_b.clamp_(LS_MIN, LS_MAX)

            train_loss += float(loss.detach().cpu())
            seen_train += y.size(0)

        train_loss /= max(1, len(loaders["train"]))
        print(
            f"[Epoch {epoch:03d}] train images: {seen_train}/{len(train_df)} | Train Loss: {train_loss:.3f}"
        )

        if epoch % 5 == 0:
            wA = float(torch.exp(-model.log_sigma_a).detach().cpu())
            wB = float(torch.exp(-model.log_sigma_b).detach().cpu())
            print(f"    Learned task weights ~ wA={wA:.3f}, wB={wB:.3f}")

        model.eval()
        T: list[int] = []
        P: list[int] = []
        Q: list[list[float]] = []
        seen_val = 0
        with torch.no_grad():
            for x_img, x_rad, y in loaders["val"]:
                x_img = x_img.to(DEVICE, non_blocking=True)
                x_rad = x_rad.to(DEVICE, non_blocking=True)
                y = y.to(DEVICE, non_blocking=True)

                logits_a, logits_b = model(x_img, x_rad)
                prob3 = fuse_probs_to_three_classes(logits_a, logits_b)
                T += y.cpu().tolist()
                P += prob3.argmax(1).cpu().tolist()
                Q += prob3.cpu().tolist()
                seen_val += y.size(0)

        print(f"[Epoch {epoch:03d}] val   images: {seen_val}/{len(val_df)}")
        f1w = f1_score(T, P, average="weighted")
        print(f"Epoch {epoch:03d} | Val F1w {f1w:.4f}")

        if epoch % 10 == 0:
            overall, perclass = compute_overall_and_perclass(
                T, P, np.array(Q), classes
            )
            print_metrics_block(f"VAL METRICS @ Epoch {epoch}", overall, perclass)

        if f1w > best_f1:
            best_f1 = f1w
            best_state = {k: v.cpu() for k, v in model.state_dict().items()}
            torch.save(
                {
                    "state_dict": best_state,
                    "backbone": BACKBONE,
                    "img_size": IMG_SIZE,
                    "classes": classes,
                    "rad_mean": rad_stats["mean"],
                    "rad_std": rad_stats["std"],
                    "feat_names": feat_names,
                    "val_f1_weighted": best_f1,
                    "epoch": epoch,
                    "W_3": w_3.to_dict(),
                },
                ckpt_path,
            )
            print(f"[Checkpoint] New best F1w={best_f1:.4f} saved → {ckpt_path}")
            wait = 0
        else:
            wait += 1

        if epoch >= MIN_EPOCHS and wait >= PATIENCE:
            print(f"Early stopping at epoch {epoch}")
            break

    if ckpt_path.exists():
        payload = torch.load(ckpt_path, map_location="cpu")
        model.load_state_dict({k: v.to(DEVICE) for k, v in payload["state_dict"].items()})
        print(
            f"[Checkpoint] Loaded best model (Val F1w={payload.get('val_f1_weighted', float('nan')):.4f})"
        )
    elif best_state:
        model.load_state_dict({k: v.to(DEVICE) for k, v in best_state.items()})

    wA_final = float(torch.exp(-model.log_sigma_a).detach().cpu())
    wB_final = float(torch.exp(-model.log_sigma_b).detach().cpu())
    print(f"Final learned task weights ~ wA={wA_final:.3f}, wB={wB_final:.3f}")

    artifacts = dict(
        rad_df=rad_df,
        feat_names=feat_names,
        rad_stats=rad_stats,
        df=df,
        classes=classes,
        test_df=test_df,
        best_ckpt=str(ckpt_path),
        W_3=W_3,
    )
    return model, artifacts


def train_one_fold_loocv(
    df_all: pd.DataFrame,
    rad_df: pd.DataFrame,
    feat_names: Sequence[str],
    classes: Sequence[str],
    *,
    test_patient: str,
    seed: int = 42,
):
    set_seed(seed)

    df = df_all

    train_pool = df[df["patient"] != test_patient].copy()
    test_df = df[df["patient"] == test_patient].copy()

    print("\n" + "=" * 90)
    print(f"[FOLD] Test patient: {test_patient}")
    print("=" * 90)
    print("[Sanity] Patients in TRAIN pool:", sorted(train_pool["patient"].unique()))
    print("[Sanity] Patients in TEST set:", sorted(test_df["patient"].unique()))
    assert test_patient not in set(train_pool["patient"])
    assert set(test_df["patient"]) == {test_patient}

    val_df, _ = _build_val_from_patients(
        train_pool, VAL_SPLIT_PATIENTS, VAL_FRACTION_FROM_VALPAT, seed
    )
    train_df = train_pool[~train_pool.index.isin(val_df.index)].copy()

    print("\n[Training split sizes — Patient-level]")
    print(f"  TRAIN images (pool minus VAL-slices): {len(train_df)}")
    print(
        f"  VAL images (from {VAL_SPLIT_PATIENTS} ∩ train_pool): {len(val_df)}"
    )
    print(f"  TEST images (all {test_patient}):                 {len(test_df)}")
    print("[Sanity] Per-class counts (TRAIN):", train_df["label"].value_counts().to_dict())
    print("[Sanity] Per-class counts (VAL):  ", val_df["label"].value_counts().to_dict())
    print("[Sanity] Per-class counts (TEST): ", test_df["label"].value_counts().to_dict())

    train_paths = set(train_df["path"])
    rad_train = (
        rad_df[rad_df["path"].isin(train_paths)][feat_names].to_numpy(dtype=np.float32)
    )
    rad_mean = rad_train.mean(axis=0)
    rad_std = rad_train.std(axis=0, ddof=0)
    rad_stats = {"mean": rad_mean, "std": rad_std}

    loaders = {
        "train": DataLoader(
            TumorDataset(train_df, rad_df, feat_names, rad_stats, train=True),
            batch_size=16,
            shuffle=True,
            num_workers=NUM_WORKERS,
            pin_memory=PIN_MEMORY,
        ),
        "val": DataLoader(
            TumorDataset(val_df, rad_df, feat_names, rad_stats, train=False),
            batch_size=16,
            shuffle=False,
            num_workers=NUM_WORKERS,
            pin_memory=PIN_MEMORY,
        ),
    }

    model = MultiModalHierNet(BACKBONE, rad_in_dim=len(feat_names)).to(DEVICE)

    y_a_train = (train_df["y"].values != 0).astype(int)
    cnt_a = np.bincount(y_a_train, minlength=2).astype(float)
    w_a = 1.0 / (cnt_a / cnt_a.sum() + 1e-9)
    w_a = w_a * (2.0 / w_a.sum())
    W_A = torch.tensor(w_a, dtype=torch.float32, device=DEVICE)

    tumor_rows = train_df[train_df["y"] != 0]
    y_b_train = np.where(tumor_rows["y"].values == 1, 0, 1)
    cnt_b = np.bincount(y_b_train, minlength=2).astype(float)
    w_b = 1.0 / (cnt_b / cnt_b.sum() + 1e-9)
    w_b = w_b * (2.0 / w_b.sum())
    W_B = torch.tensor(w_b, dtype=torch.float32, device=DEVICE)

    cnt_3 = train_df["y"].value_counts().sort_index().astype(float)
    w_3 = 1.0 / (cnt_3 / cnt_3.sum() + 1e-9)
    w_3 = w_3 * (3.0 / w_3.sum())
    W_3 = w_3

    optimizer = optim.Adam(model.parameters(), lr=LR)

    best_f1 = -float("inf")
    best_state: dict[str, torch.Tensor] | None = None
    wait = 0

    for epoch in range(1, MAX_EPOCHS + 1):
        model.train()
        train_loss = 0.0
        seen_train = 0
        for x_img, x_rad, y in loaders["train"]:
            x_img = x_img.to(DEVICE, non_blocking=True)
            x_rad = x_rad.to(DEVICE, non_blocking=True)
            y = y.to(DEVICE, non_blocking=True)

            optimizer.zero_grad()
            logits_a, logits_b = model(x_img, x_rad)
            loss, _, _ = hierarchical_loss_uncertainty(
                model, logits_a, logits_b, y, W_A=W_A, W_B=W_B, reg_scale=REG_SCALE
            )
            loss.backward()
            optimizer.step()

            with torch.no_grad():
                model.log_sigma_a.clamp_(LS_MIN, LS_MAX)
                model.log_sigma_b.clamp_(LS_MIN, LS_MAX)

            train_loss += float(loss.detach().cpu())
            seen_train += y.size(0)

        train_loss /= max(1, len(loaders["train"]))
        print(
            f"[Epoch {epoch:03d}] train images: {seen_train}/{len(train_df)} | Train Loss: {train_loss:.3f}"
        )

        if epoch % 5 == 0:
            wA = float(torch.exp(-model.log_sigma_a).detach().cpu())
            wB = float(torch.exp(-model.log_sigma_b).detach().cpu())
            print(f"    Learned task weights ~ wA={wA:.3f}, wB={wB:.3f}")

        model.eval()
        T: list[int] = []
        P: list[int] = []
        Q: list[list[float]] = []
        seen_val = 0
        with torch.no_grad():
            for x_img, x_rad, y in loaders["val"]:
                x_img = x_img.to(DEVICE, non_blocking=True)
                x_rad = x_rad.to(DEVICE, non_blocking=True)
                y = y.to(DEVICE, non_blocking=True)

                logits_a, logits_b = model(x_img, x_rad)
                prob3 = fuse_probs_to_three_classes(logits_a, logits_b)
                T += y.cpu().tolist()
                P += prob3.argmax(1).cpu().tolist()
                Q += prob3.cpu().tolist()
                seen_val += y.size(0)

        print(f"[Epoch {epoch:03d}] val   images: {seen_val}/{len(val_df)}")
        f1w = f1_score(T, P, average="weighted")
        print(f"Epoch {epoch:03d} | Val F1w {f1w:.4f}")

        if epoch % 10 == 0:
            overall, perclass = compute_overall_and_perclass(
                T, P, np.array(Q), classes
            )
            print_metrics_block(
                f"VAL METRICS @ Epoch {epoch} (fold test={test_patient})",
                overall,
                perclass,
            )

        if f1w > best_f1:
            best_f1 = f1w
            best_state = {k: v.cpu() for k, v in model.state_dict().items()}
            wait = 0
        else:
            wait += 1

        if epoch >= MIN_EPOCHS and wait >= PATIENCE:
            print(f"Early stopping at epoch {epoch}")
            break

    if best_state:
        model.load_state_dict({k: v.to(DEVICE) for k, v in best_state.items()})

    wA_final = float(torch.exp(-model.log_sigma_a).detach().cpu())
    wB_final = float(torch.exp(-model.log_sigma_b).detach().cpu())
    print(f"Final learned task weights ~ wA={wA_final:.3f}, wB={wB_final:.3f}")

    artifacts = dict(
        rad_df=rad_df,
        feat_names=feat_names,
        rad_stats=rad_stats,
        df=df,
        classes=classes,
        test_df=test_df,
        W_3=W_3,
    )
    return model, artifacts


def run_patient_loocv(seed: int = 42):
    set_seed(seed)
    df_all, classes = load_df_from_csv(CSV_PATH, IMG_ROOTS)

    rad_df_full, feat_names, _ = build_or_load_radiomics_cache(
        df_all,
        RAD_CACHE_DIR,
        pixelSpacing=(1.0, 1.0, 1.0),
        voxelArrayShift=0.0,
        binWidth=5.0,
        device=DEVICE,
        max_images=RAD_MAX_IMAGES,
        debug=RAD_DEBUG,
    )
    df_all = df_all[df_all["path"].isin(rad_df_full["path"])].reset_index(drop=True)
    rad_df_full = rad_df_full[rad_df_full["path"].isin(df_all["path"])].reset_index(
        drop=True
    )

    candidate_patients = ["Case-3", "P9", "Case-48", "Case-4"]
    patients = [p for p in candidate_patients if p in set(df_all["patient"].unique())]
    if len(patients) < 2:
        raise RuntimeError("Need at least 2 patients for LOPO-CV")

    print("\n[LOPO-CV] Patients participating:", patients)

    fold_overall_means = []
    fold_names = []

    from evaluation import evaluate_model_5x  # local import to avoid circular deps

    for test_patient in patients:
        model, artifacts_fold = train_one_fold_loocv(
            df_all=df_all,
            rad_df=rad_df_full,
            feat_names=feat_names,
            classes=classes,
            test_patient=test_patient,
            seed=seed,
        )

        dfres, _ = evaluate_model_5x(
            model,
            artifacts_fold,
            seeds_eval=MC_EVAL_SEEDS,
            patient_name=test_patient,
        )

        fold_overall_means.append(dfres.mean(axis=0).to_dict())
        fold_names.append(test_patient)

    overall_df = pd.DataFrame(fold_overall_means, index=fold_names)
    print("\n" + "=" * 90)
    print("=== LOPO-CV Overall Metrics (Mean ± Std across folds) ===")
    for col in overall_df.columns:
        mu = overall_df[col].mean()
        sd = overall_df[col].std(ddof=1) if len(overall_df[col]) > 1 else 0.0
        print(f"{col:>16}: {mu:.4f} ± {sd:.4f}")

    return overall_df


if __name__ == "__main__":
    model, artifacts = train_model_once(seed=2025)
    from evaluation import evaluate_model_5x

    evaluate_model_5x(model, artifacts, seeds_eval=MC_EVAL_SEEDS)
