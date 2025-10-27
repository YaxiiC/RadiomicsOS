# efficientnet_radiomics.py
# === EfficientNet-B0 3-Class Classifier + Radiomics Fusion (FIRST-ORDER radiomics only) ===

import os, re, time, json, math, random, pathlib
from pathlib import Path
import logging
import warnings

import numpy as np
import pandas as pd
from PIL import Image

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader

from torchvision import transforms
import timm

from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score, f1_score, roc_auc_score, classification_report,
    confusion_matrix, precision_recall_fscore_support, roc_curve
)
from sklearn.preprocessing import label_binarize

# ---------------- Radiomics deps ----------------
import SimpleITK as sitk
from torchradiomics import (
    TorchRadiomicsFirstOrder,  # <-- using only first-order extractor
    inject_torch_radiomics
)

# Optional/robust import of shape extractors across versions
try:
    from torchradiomics import TorchRadiomicsShape2D as _Shape2DClass
    _SHAPE_MODE = "shape2d"
except Exception:
    try:
        from torchradiomics import TorchRadiomicsShape as _Shape2DClass
        _SHAPE_MODE = "shape_fallback"
    except Exception:
        _Shape2DClass = None
        _SHAPE_MODE = None
# tqdm progress bar (safe fallback if unavailable)
try:
    from tqdm.auto import tqdm
except Exception:
    tqdm = None


# ---- Cache I/O helpers: parquet if available, else CSV ----
def _parquet_available():
    try:
        import pyarrow  # noqa: F401
        return True
    except Exception:
        try:
            import fastparquet  # noqa: F401
            return True
        except Exception:
            return False

_PARQUET_OK = _parquet_available()

def _read_cache(df_path_base: Path):
    pqt = df_path_base.with_suffix(".parquet")
    csv = df_path_base.with_suffix(".csv")
    if pqt.exists() and _PARQUET_OK:
        return pd.read_parquet(pqt)
    if csv.exists():
        return pd.read_csv(csv)
    return None  # no cache yet

def _write_cache(df: pd.DataFrame, df_path_base: Path, autosave: bool = False, i: int | None = None):
    """
    Writes df to parquet if engine available; else CSV.
    If autosave=True and i is provided, suffix with _partial_{i}.
    """
    stem = df_path_base.stem
    parent = df_path_base.parent
    if autosave and i is not None:
        stem = f"{stem}_partial_{i}"
    if _PARQUET_OK:
        out = parent / f"{stem}.parquet"
        df.to_parquet(out, index=False)
    else:
        out = parent / f"{stem}.csv"
        df.to_csv(out, index=False)
    return out

def _read_feat_names(path_base: Path):
    # feature names shared across parquet/csv
    names_json = path_base.with_suffix(".json")
    return json.loads(names_json.read_text()) if names_json.exists() else None

def _write_feat_names(names: list[str], path_base: Path):
    names_json = path_base.with_suffix(".json")
    names_json.write_text(json.dumps(names))
    return names_json

logging.getLogger("torchradiomics").setLevel(logging.ERROR)
warnings.filterwarnings("ignore", category=UserWarning)

# ----------------------- Paths & Config ------------------------
CSV_PATH  = pathlib.Path("/home/yaxi/Osteosarcoma-UT/ML_Features_1144.csv")
IMG_ROOTS = [
    pathlib.Path("/home/yaxi/Osteosarcoma-UT/Training-Set-1"),
    pathlib.Path("/home/yaxi/Osteosarcoma-UT/Training-Set-2")
]
RAD_CACHE_DIR = pathlib.Path("./rad_cache")

BACKBONE   = "inception_v3"   # was: "efficientnet_b2"
IMG_SIZE   = 299              # InceptionV3 expects 299x299
DEVICE     = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#IMG_SIZE   = 224
MAX_EPOCHS = 100
MIN_EPOCHS = 50
PATIENCE   = 30
LR = 1e-4
NUM_WORKERS = 4
PIN_MEMORY  = DEVICE.type == "cuda"

# --- Radiomics cache debug knobs ---
RAD_MAX_IMAGES = None   # e.g., 20 for a fast sanity check; None for all images
RAD_DEBUG      = False  # True to print per-image details (path, shape, min/max, time)

# ----------------------- NEW: Patient-level split config -----------------------
# (Same idea as script 2)
TRAIN_PATIENTS = ["Case-3", "P9", "Case-48"]
VAL_FROM_PATIENT = ["Case-3", "P9", "Case-48"]   # subset (or all) of TRAIN patients to carve a small VAL slice from
TEST_PATIENT = "Case-4"
VAL_FRACTION = 0.13  # 10% from the specified VAL_FROM_PATIENT patients will be moved from TRAIN → VAL

# ----------------------- Reproducibility -----------------------
def set_seed(seed: int):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

# ---------------------- Transforms -----------------------------
transform_train = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.RandomResizedCrop(IMG_SIZE, scale=(0.8, 1.0)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomVerticalFlip(),
    transforms.RandomRotation(15),
    transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.15, hue=0.02),
    transforms.RandomAffine(degrees=0, translate=(0.05, 0.05), shear=5),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485,0.456,0.406], std=[0.229,0.224,0.225])
])


transform_eval = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

# =======================================================================
# Radiomics extractor (FIRST-ORDER ONLY)
# =======================================================================
def extract_all_radiomics(
    x: torch.Tensor,
    voxelArrayShift: float = 0.0,
    pixelSpacing = [1.0, 1.0, 1.0],
    binWidth: float | None = None
):
    """
    FIRST-ORDER + SHAPE2D (texture families disabled for speed).
    Works for 2D tiles or 3D volumes.
      - For 2D: spacing uses (x, y) in SimpleITK (we map from [z,y,x] style input).
      - For 3D: spacing uses (x, y, z).
    Returns (features_dict, feature_names_list).
    """
    # ---- 1) Prepare SITK images from torch tensors
    img_np  = x.to(dtype=torch.float64, non_blocking=False).cpu().numpy()
    mask_np = (x > 0).to(dtype=torch.uint8,  non_blocking=False).cpu().numpy()

    sitk_img  = sitk.GetImageFromArray(img_np)
    sitk_mask = sitk.GetImageFromArray(mask_np)

    # Dim-aware spacing
    dim = sitk_img.GetDimension()
    if dim == 2:
        # SimpleITK expects (x, y)
        sx = float(pixelSpacing[1]) if len(pixelSpacing) >= 2 else 1.0
        sy = float(pixelSpacing[0]) if len(pixelSpacing) >= 1 else 1.0
        sitk_img.SetSpacing((sx, sy))
        sitk_mask.SetSpacing((sx, sy))
    elif dim == 3:
        if len(pixelSpacing) != 3:
            raise ValueError("For 3D, pixelSpacing must be [z, y, x] length 3.")
        # SimpleITK expects (x, y, z)
        sitk_img.SetSpacing((pixelSpacing[2], pixelSpacing[1], pixelSpacing[0]))
        sitk_mask.SetSpacing((pixelSpacing[2], pixelSpacing[1], pixelSpacing[0]))
    else:
        raise ValueError(f"Unsupported image dimension: {dim}")

    # ---- 2) Inject & prepare extractor kwargs
    inject_torch_radiomics()

    base_compute = dict(
        voxelBased=False,
        padDistance=1,
        kernelRadius=1,
        maskedKernel=False,
        voxelBatch=512,
        dtype=torch.float64,
        device=x.device
    )
    base_settings = dict(voxelArrayShift=voxelArrayShift)
    if binWidth is not None:
        base_settings["binWidth"] = float(binWidth)

    # ---- 3) First-Order + Shape2D
    extractors = [
        TorchRadiomicsFirstOrder(sitk_img, sitk_mask, **base_settings, **base_compute)
    ]

    if _Shape2DClass is not None:
        if _SHAPE_MODE == "shape2d":
            extractors.append(_Shape2DClass(sitk_img, sitk_mask, **base_settings, **base_compute))
        elif _SHAPE_MODE == "shape_fallback":
            extractors.append(_Shape2DClass(
                sitk_img, sitk_mask,
                force2D=True, force2DDimension=0,
                **base_settings, **base_compute
            ))

    # ---- 4) Execute and collect (skip feature maps/ITK images)
    features, names = {}, []
    for ext in extractors:
        out = ext.execute()
        for k, v in out.items():
            if isinstance(v, sitk.Image):
                continue
            tv = v if isinstance(v, torch.Tensor) else torch.as_tensor(v, dtype=torch.float64, device=x.device)
            if torch.isfinite(tv).all():
                features[k] = tv
                names.append(k)
    return features, names

# =======================================================================
# Radiomics cache builder (progress bar + debug + autosave)
# =======================================================================
def build_or_load_radiomics_cache(
    df: pd.DataFrame,
    cache_dir: Path,
    pixelSpacing=[1.0,1.0,1.0],
    voxelArrayShift=0.0,
    binWidth=5.0,
    device=DEVICE,
    max_images: int | None = RAD_MAX_IMAGES,
    debug: bool = RAD_DEBUG
):
    """
    Builds/loads radiomics cache (First-Order + Shape2D).
    Uses Parquet if available; otherwise falls back to CSV automatically.
    Autosaves every 100 images using the same strategy.
    """
    cache_dir.mkdir(parents=True, exist_ok=True)
    base = cache_dir / "radiomics_cache"  # we'll add .parquet or .csv dynamically

    cached = _read_cache(base)
    feat_names = _read_feat_names(cache_dir / "radiomics_feat_names")
    if cached is not None and feat_names is not None:
        kind = "Parquet" if _PARQUET_OK and (cache_dir / "radiomics_cache.parquet").exists() else "CSV"
        print(f"[Radiomics] Using existing {kind} cache: {cache_dir}")
        return cached, feat_names, {"table": cached, "names": cache_dir / "radiomics_feat_names.json"}

    rows = []
    feat_names = None
    paths = df["path"].astype(str).unique().tolist()
    if max_images is not None:
        paths = paths[:max_images]
    total = len(paths)
    store_kind = "Parquet" if _PARQUET_OK else "CSV"
    print(f"[Radiomics] Computing FO+Shape2D for {total} images... (store={store_kind})")

    pbar = tqdm(total=total, desc="Radiomics(FO+S2D)", unit="img") if tqdm else None
    t0 = time.time()

    try:
        for i, p in enumerate(paths, 1):
            t_img0 = time.time()
            try:
                pil = Image.open(p).convert("L")  # grayscale
                arr = np.array(pil, dtype=np.float32)
                x = torch.from_numpy(arr).to(device=device, dtype=torch.float32)

                if debug:
                    print(f"[{i:04d}/{total}] {p} | shape={tuple(arr.shape)} | min={float(arr.min()):.1f} max={float(arr.max()):.1f}")

                feats_dict, feat_names_i = extract_all_radiomics(
                    x, voxelArrayShift=voxelArrayShift, pixelSpacing=pixelSpacing, binWidth=binWidth
                )
                if feat_names is None:
                    feat_names = feat_names_i
                    if debug:
                        print(f"[Feature names] {len(feat_names)} FO+Shape2D features")

                row = {"path": p}
                for k in feat_names:
                    v = feats_dict[k]
                    if isinstance(v, torch.Tensor):
                        v = v.detach().to("cpu").item()
                    row[k] = float(v)
                rows.append(row)

            except Exception as e:
                msg = f"[Radiomics][WARN] {p}: {e}"
                if pbar: pbar.write(msg)
                else: print(msg)

            # progress + ETA
            if pbar:
                pbar.update(1)
                if i % 25 == 0 or i == total:
                    elapsed = time.time() - t0
                    ips = i / max(elapsed, 1e-9)
                    eta_s = (total - i) / max(ips, 1e-9)
                    pbar.set_postfix_str(f"{ips:.2f} img/s | ETA {eta_s/60:.1f} min")
            else:
                if i % 50 == 0 or i == total:
                    elapsed = time.time() - t0
                    print(f"[Radiomics] {i}/{total} done ({elapsed:.1f}s)")

            # autosave every 100 images
            if i % 100 == 0:
                tmp_df = pd.DataFrame(rows)
                out = _write_cache(tmp_df, cache_dir / "radiomics_cache", autosave=True, i=i)
                if pbar: pbar.write(f"[Autosave] {out}")
                else: print(f"[Autosave] {out}")

            # per-image timing
            t_img = time.time() - t_img0
            if debug and t_img > 5.0:
                print(f"[SLOW] {p} took {t_img:.1f}s")

    finally:
        if pbar: pbar.close()

    if not rows:
        raise RuntimeError("No radiomics rows computed — check inputs.")

    rad_df = pd.DataFrame(rows)
    out_main = _write_cache(rad_df, cache_dir / "radiomics_cache", autosave=False)
    names_path = _write_feat_names(feat_names, cache_dir / "radiomics_feat_names")
    print(f"[Radiomics] Saved cache → {out_main}")
    return rad_df, feat_names, {"table": out_main, "names": names_path}

# =======================================================================
# Data loading utilities
# =======================================================================
def load_df_from_csv(csv_path: pathlib.Path, roots):
    df = pd.read_csv(csv_path)
    df["image.name"] = df["image.name"].astype(str)

    def clean_label(s: str) -> str:
        s_low = str(s).strip().lower().replace("_", "-").replace(" ", "-")
        if "non" in s_low and "viable" in s_low: return "Non-Viable-Tumor"
        if "non-tumor" in s_low or "nontumor" in s_low: return "Non-Tumor"
        return "Viable"

    def canonical_key(s: str):
        stem = pathlib.Path(str(s)).stem.lower()
        nums = re.findall(r'\d+', stem)
        if len(nums) < 4:
            return None
        return f"case{nums[0]}a{nums[1]}{nums[2]}{nums[3]}"

    exts = {".jpg", ".jpeg", ".png", ".tif", ".tiff", ".bmp"}
    by_key, dups = {}, set()
    for root in roots:
        for p in root.rglob("*"):
            if p.is_file() and p.suffix.lower() in exts:
                key = canonical_key(p.name)
                if key is None: continue
                if key in by_key: dups.add(key)
                else: by_key[key] = p

    df["canon"] = df["image.name"].map(canonical_key)
    before = len(df)
    df = df.dropna(subset=["canon"]).copy()
    has_file_mask = df["canon"].isin(by_key)
    resolved = int(has_file_mask.sum())
    df = df[has_file_mask].copy()
    df["path"] = df["canon"].map(lambda k: str(by_key[k]))
    df["label"] = df["classification"].apply(clean_label)
    classes = ["Non-Tumor", "Non-Viable-Tumor", "Viable"]
    df = df[df["label"].isin(classes)].copy()
    after_label = len(df)
    df["y"] = df["label"].map({c: i for i, c in enumerate(classes)})

    # ---- NEW: derive patient_id from path or filename (supports "Case-XX" or "PXX")
    def extract_patient_id(s: str):
        s = str(s)
        m = re.search(r'(Case-\d+|P\d+)', s, flags=re.IGNORECASE)
        return m.group(1).title() if m else None

    df["patient_id"] = df["path"].map(extract_patient_id)
    # Fallback: try from original image name if path failed
    mask_missing = df["patient_id"].isna()
    if mask_missing.any():
        df.loc[mask_missing, "patient_id"] = df.loc[mask_missing, "image.name"].map(extract_patient_id)

    print("\n[Data loading summary]")
    print(f"  CSV rows:                     {before}")
    print(f"  Rows with valid key:          {before - int(df['canon'].isna().sum())}")
    print(f"  Rows with resolved image path:{resolved}")
    print(f"  Usable rows after label clean:{after_label}")
    print(f"  TOTAL IMAGES LOADED:          {len(df)}")
    print(f"  Per-class counts:             {df['label'].value_counts().to_dict()}")
    if dups:
        print(f"  Note: {len(dups)} duplicate basenames detected on disk (kept first occurrence).")
    print(f"  Distinct patients detected:   {df['patient_id'].nunique()} (some may be None)")
    return df, classes

# =======================================================================
# Dataset w/ radiomics vector
# =======================================================================
class TumorDataset(Dataset):
    def __init__(self, df, rad_df, feat_names, rad_stats, train=False):
        """
        df: DataFrame containing ['path','y']
        rad_df: DataFrame with ['path', <feats...>]
        feat_names: list[str] order
        rad_stats: dict {'mean': np.ndarray, 'std': np.ndarray} aligned to feat_names
        """
        self.df = df.reset_index(drop=True)
        self.tf = transform_train if train else transform_eval
        self.feat_names = feat_names
        self.mean = rad_stats["mean"].astype(np.float32)
        self.std  = rad_stats["std"].astype(np.float32)
        self.rad_lookup = rad_df.set_index("path")[feat_names]

    def __len__(self): 
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        p = row["path"]
        if not os.path.exists(p):
            raise FileNotFoundError(f"Image path does not exist: {p}")
        img = Image.open(p).convert("RGB")
        x_img = self.tf(img)

        r = self.rad_lookup.loc[p].to_numpy(dtype=np.float32)
        r = r - self.mean
        denom = np.where(self.std > 1e-8, self.std, 1.0)
        r = r / denom
        x_rad = torch.from_numpy(r)

        return x_img, x_rad, int(row["y"])

# =======================================================================
# Metrics helpers
# =======================================================================
def _sens_spec_at_targets(y_true_bin, y_score, spec_target=0.9, sens_target=0.9):
    """Return (sensitivity@spec_target, specificity@sens_target)."""
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


def compute_overall_and_perclass(y_true, y_pred, y_prob, classes):
    y_true_arr = np.asarray(y_true)
    y_prob_arr = np.asarray(y_prob)
    labels_idx = list(range(len(classes)))
    acc = accuracy_score(y_true_arr, y_pred)
    f1_mi = f1_score(y_true_arr, y_pred, average="micro")
    f1_ma = f1_score(y_true_arr, y_pred, average="macro")
    f1_w = f1_score(y_true_arr, y_pred, average="weighted")
    try: auc_ovr = roc_auc_score(y_true_arr, y_prob_arr, multi_class="ovr", average="macro")
    except: auc_ovr = float("nan")
    try: auc_ovo = roc_auc_score(y_true_arr, y_prob_arr, multi_class="ovo", average="macro")
    except: auc_ovo = float("nan")

    prec, rec, f1c, support = precision_recall_fscore_support(
        y_true_arr, y_pred, labels=labels_idx, average=None, zero_division=0
    )
    cm = confusion_matrix(y_true_arr, y_pred, labels=labels_idx)
    specs = []
    for i in labels_idx:
        TP = cm[i, i]; FP = cm[:, i].sum() - TP; FN = cm[i, :].sum() - TP; TN = cm.sum() - TP - FP - FN
        specs.append(TN / (TN + FP) if (TN + FP) > 0 else float("nan"))

    try:
        Y_bin = label_binarize(y_true_arr, classes=labels_idx)
        auc_class = [roc_auc_score(Y_bin[:, i], y_prob_arr[:, i]) if Y_bin[:, i].min() != Y_bin[:, i].max() else float("nan") for i in labels_idx]
    except:
        auc_class = [float("nan")] * len(labels_idx)

    per_class = []
    sens_spec_pairs = []
    for i, c in enumerate(classes):
        sens_at_spec, spec_at_sens = _sens_spec_at_targets(
            (y_true_arr == i).astype(int), y_prob_arr[:, i]
        )
        sens_spec_pairs.append((sens_at_spec, spec_at_sens))
        per_class.append(
            dict(
                cls=c,
                support=int(support[i]),
                precision=prec[i],
                recall=rec[i],
                specificity=specs[i],
                sens_at_spec90=sens_at_spec,
                spec_at_sens90=spec_at_sens,
                f1=f1c[i],
                auc=auc_class[i],
            )
        )
    overall = dict(
        acc=acc,
        f1_micro=f1_mi,
        f1_macro=f1_ma,
        f1_weighted=f1_w,
        auc_macro_ovr=auc_ovr,
        auc_macro_ovo=auc_ovo,
        sens_at_spec90_macro=float(np.nanmean([p[0] for p in sens_spec_pairs])) if sens_spec_pairs else float("nan"),
        spec_at_sens90_macro=float(np.nanmean([p[1] for p in sens_spec_pairs])) if sens_spec_pairs else float("nan"),
    )
    return overall, per_class

def print_metrics_block(title, overall, per_class):
    print(f"\n=== {title} ===")
    print(f"  Accuracy: {overall['acc']:.4f}")
    print(f"  F1 (micro/macro/w): {overall['f1_micro']:.4f} / {overall['f1_macro']:.4f} / {overall['f1_weighted']:.4f}")
    print(f"  Macro AUC (OvR/OvO): {overall['auc_macro_ovr']:.4f} / {overall['auc_macro_ovo']:.4f}")
    print(f"  Sens@Spec90 (macro): {overall['sens_at_spec90_macro']:.4f}")
    print(f"  Spec@Sens90 (macro): {overall['spec_at_sens90_macro']:.4f}")
    hdr = f"{'Class':<22} {'Support':>7}  {'Prec':>6}  {'Rec':>6}  {'Spec':>6}  {'Sens@Spec90':>12}  {'Spec@Sens90':>12}  {'F1':>6}  {'AUC':>6}"
    print("\nPer-class metrics:")
    print(hdr)
    print("-" * len(hdr))
    for r in per_class:
        print(
            f"{r['cls']:<22} {r['support']:>7d}  "
            f"{r['precision']:>6.3f}  {r['recall']:>6.3f}  {r['specificity']:>6.3f}  "
            f"{r['sens_at_spec90']:>12.3f}  {r['spec_at_sens90']:>12.3f}  "
            f"{r['f1']:>6.3f}  {r['auc']:>6.3f}"
        )

# =======================================================================
# Multimodal model (ATTENTION-BASED fusion)
# =======================================================================
class MultiModalNet(nn.Module):
    """
    Attention-based fusion:
      - Project CNN and radiomics branches to the same fusion_dim.
      - Concatenate both projections and predict a 2-way attention vector (softmax).
      - Fuse by weighted sum of the two branch embeddings.
      - Classify from the fused embedding.
    """
    def __init__(self, backbone: str, num_classes: int, rad_in_dim: int,
                 rad_hidden=256, fusion_dim=256, att_hidden=128, p_drop=0.3):
        super().__init__()

        # CNN encoder (no classifier head)
        self.cnn = timm.create_model(backbone, pretrained=True, num_classes=0, global_pool="avg")
        with torch.no_grad():
            dummy = torch.zeros(1, 3, IMG_SIZE, IMG_SIZE)
            cnn_dim = self.cnn(dummy).shape[1]

        # Project CNN features -> fusion_dim
        self.cnn_proj = nn.Sequential(
            nn.Linear(cnn_dim, fusion_dim),
            nn.BatchNorm1d(fusion_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(p_drop),
        )

        # Radiomics MLP -> fusion_dim
        self.rad_net = nn.Sequential(
            nn.BatchNorm1d(rad_in_dim),
            nn.Linear(rad_in_dim, rad_hidden),
            nn.ReLU(inplace=True),
            nn.Dropout(p_drop),
            nn.Linear(rad_hidden, fusion_dim),
            nn.ReLU(inplace=True),
        )

        # Attention gate
        self.att_gate = nn.Sequential(
            nn.Linear(2 * fusion_dim, att_hidden),
            nn.ReLU(inplace=True),
            nn.Dropout(p_drop),
            nn.Linear(att_hidden, 2),
            nn.Softmax(dim=1)  # outputs [w_img, w_rad]
        )

        # Classifier on fused embedding
        self.classifier = nn.Linear(fusion_dim, num_classes)

    def forward(self, x_img, x_rad):
        f_img = self.cnn_proj(self.cnn(x_img))  # [B, fusion_dim]
        f_rad = self.rad_net(x_rad)             # [B, fusion_dim]
        h = torch.cat([f_img, f_rad], dim=1)    # [B, 2*fusion_dim]
        w = self.att_gate(h)                    # [B, 2]
        w_img, w_rad = w[:, :1], w[:, 1:]
        f_fused = w_img * f_img + w_rad * f_rad
        return self.classifier(f_fused)

# =======================================================================
# Train once (build cache, normalize rads, train)
# =======================================================================
def train_model_once(seed=42):
    set_seed(seed)
    df, classes = load_df_from_csv(CSV_PATH, IMG_ROOTS)
    num_classes = len(classes)

    # Build/load radiomics cache (FO + Shape2D)
    rad_df, feat_names, _paths = build_or_load_radiomics_cache(
        df, RAD_CACHE_DIR,
        pixelSpacing=[1.0,1.0,1.0], voxelArrayShift=0.0, binWidth=5.0, device=DEVICE,
        max_images=RAD_MAX_IMAGES, debug=RAD_DEBUG
    )

    # Keep only rows in both tables
    df = df[df["path"].isin(rad_df["path"])].reset_index(drop=True)
    rad_df = rad_df[rad_df["path"].isin(df["path"])].reset_index(drop=True)

    # ----------------------- NEW: Patient-level split -----------------------
    if "patient_id" not in df.columns:
        raise RuntimeError("patient_id column missing; check load_df_from_csv() extraction.")

    # TEST set: single held-out patient
    test_df = df[df["patient_id"].str.title() == str(TEST_PATIENT).title()].copy()

    # TRAIN pool: specified training patients
    train_pool = df[df["patient_id"].str.title().isin([p.title() for p in TRAIN_PATIENTS])].copy()

    # From a subset of TRAIN patients, carve out a small VAL slice
    rng = np.random.default_rng(seed)
    val_mask = pd.Series(False, index=train_pool.index)
    for pid in VAL_FROM_PATIENT:
        pid_norm = str(pid).title()
        idx = train_pool.index[train_pool["patient_id"].str.title() == pid_norm].to_numpy()
        if len(idx) == 0:
            continue
        n_val = max(1, int(len(idx) * VAL_FRACTION))
        chosen = rng.choice(idx, size=min(n_val, len(idx)), replace=False)
        val_mask.loc[chosen] = True

    val_df = train_pool[val_mask].reset_index(drop=True)
    train_df = train_pool[~val_mask].reset_index(drop=True)

    print("\n[Patient-level split]")
    def _per_patient_counts(x):
        return x.groupby("patient_id")["y"].size().to_dict()
    print(f"  TRAIN images: {len(train_df)} | per-patient: {_per_patient_counts(train_df)}")
    print(f"  VAL images:   {len(val_df)}   | per-patient: {_per_patient_counts(val_df)}")
    print(f"  TEST images:  {len(test_df)}  | patient: {TEST_PATIENT}")

    # Radiomics normalization: fit on TRAIN ONLY
    train_paths = set(train_df["path"])
    rad_train = rad_df[rad_df["path"].isin(train_paths)][feat_names].to_numpy(dtype=np.float32)
    rad_mean = rad_train.mean(axis=0)
    rad_std  = rad_train.std(axis=0, ddof=0)
    rad_stats = {"mean": rad_mean, "std": rad_std}

    loaders = {
        "train": DataLoader(
            TumorDataset(train_df, rad_df, feat_names, rad_stats, train=True),
            batch_size=16, shuffle=True,  num_workers=NUM_WORKERS, pin_memory=PIN_MEMORY
        ),
        "val": DataLoader(
            TumorDataset(val_df, rad_df, feat_names, rad_stats, train=False),
            batch_size=16, shuffle=False, num_workers=NUM_WORKERS, pin_memory=PIN_MEMORY
        )
    }

    model = MultiModalNet(BACKBONE, num_classes, rad_in_dim=len(feat_names)).to(DEVICE)

    # class weights by inverse frequency
    counts = np.bincount(train_df["y"], minlength=num_classes).astype(float)
    weights = (1.0 / (counts / counts.sum() + 1e-9))
    weights = weights * (num_classes / weights.sum())
    class_weights = torch.tensor(weights, dtype=torch.float32).to(DEVICE)

    criterion = nn.CrossEntropyLoss(weight=class_weights)
    optimizer = optim.AdamW(model.parameters(), lr=LR)

    best_state, best_f1 = None, -1
    wait = 0
    for epoch in range(1, MAX_EPOCHS + 1):
        # ---- train ----
        model.train(); train_loss = 0.0; seen_train = 0
        for x_img, x_rad, y in loaders["train"]:
            x_img = x_img.to(DEVICE, non_blocking=True)
            x_rad = x_rad.to(DEVICE, non_blocking=True)
            y     = y.to(DEVICE, non_blocking=True)

            optimizer.zero_grad()
            out = model(x_img, x_rad)
            loss = criterion(out, y)
            loss.backward(); optimizer.step()
            train_loss += loss.item()
            seen_train += y.size(0)

        train_loss /= max(1, len(loaders["train"]))
        print(f"[Epoch {epoch:03d}] train batches covered images: {seen_train}/{len(train_df)}")

        # ---- val ----
        model.eval(); T, P, Q = [], [], []; seen_val = 0
        with torch.no_grad():
            for x_img, x_rad, y in loaders["val"]:
                x_img = x_img.to(DEVICE, non_blocking=True)
                x_rad = x_rad.to(DEVICE, non_blocking=True)
                y     = y.to(DEVICE, non_blocking=True)

                out = model(x_img, x_rad); prob = torch.softmax(out, 1)
                T += y.cpu().tolist(); P += prob.argmax(1).cpu().tolist(); Q += prob.cpu().tolist()
                seen_val += y.size(0)
        print(f"[Epoch {epoch:03d}] val   batches covered images: {seen_val}/{len(val_df)}")

        f1w = f1_score(T, P, average="weighted")
        print(f"Epoch {epoch:03d} | Train Loss {train_loss:.3f} | Val F1w {f1w:.4f}")

        if epoch % 10 == 0:
            overall, perclass = compute_overall_and_perclass(T, P, np.array(Q), classes)
            print_metrics_block(f"VAL METRICS @ Epoch {epoch}", overall, perclass)

        if f1w > best_f1:
            best_f1 = f1w
            best_state = {k: v.cpu() for k, v in model.state_dict().items()}
            wait = 0
        else:
            wait += 1
        if epoch >= MIN_EPOCHS and wait >= PATIENCE:
            print(f"Early stopping at epoch {epoch}")
            break

    model.load_state_dict({k: v.to(DEVICE) for k, v in best_state.items()})
    artifacts = dict(rad_df=rad_df, feat_names=feat_names, rad_stats=rad_stats)
    return model, df, classes, artifacts

# =======================================================================
# Evaluate on patient-held-out test set (re-using train radiomics normalization)
# =======================================================================
def evaluate_model(model, df, classes, artifacts, seeds_eval=[1,2,3,4,5]):
    rad_df      = artifacts["rad_df"]
    feat_names  = artifacts["feat_names"]
    rad_stats   = artifacts["rad_stats"]

    # Fixed TEST set = held-out patient (same across seeds; seeds only affect any RNG in evaluation if present)
    test_df = df[df["patient_id"].str.title() == str(TEST_PATIENT).title()].copy()
    print(f"\n[Evaluation] TEST images: {len(test_df)} | patient: {TEST_PATIENT}")

    results = []
    perclass_all = []

    for s in seeds_eval:
        # (kept the loop to mimic prior interface; test set remains the same)
        loader = DataLoader(
            TumorDataset(test_df, rad_df, feat_names, rad_stats, train=False),
            batch_size=16, shuffle=False, num_workers=NUM_WORKERS, pin_memory=PIN_MEMORY
        )

        y_true, y_pred, y_prob = [], [], []
        model.eval(); seen_test = 0
        with torch.no_grad():
            for x_img, x_rad, y in loader:
                x_img = x_img.to(DEVICE, non_blocking=True)
                x_rad = x_rad.to(DEVICE, non_blocking=True)
                y     = y.to(DEVICE, non_blocking=True)

                out = model(x_img, x_rad); prob = torch.softmax(out, 1)
                y_true += y.cpu().tolist()
                y_pred += prob.argmax(1).cpu().tolist()
                y_prob += prob.cpu().tolist()
                seen_test += y.size(0)

        print(f"[Evaluation seed={s}] test batches covered images: {seen_test}/{len(test_df)}")

        y_prob = np.array(y_prob)
        overall, perclass = compute_overall_and_perclass(y_true, y_pred, y_prob, classes)
        print_metrics_block(f"TEST EVALUATION (seed={s})", overall, perclass)

        results.append(overall)
        perclass_all.append(perclass)

    dfres = pd.DataFrame(results, index=[f"eval_seed_{s}" for s in seeds_eval])
    print("\n=== Overall Summary (Mean ± Std) ===")
    for col in dfres.columns:
        print(f"{col:>14}: {dfres[col].mean():.4f} ± {dfres[col].std(ddof=1):.4f}")

    print("\n=== Per-Class Summary (Mean ± Std) ===")
    metrics = [
        "precision",
        "recall",
        "specificity",
        "sens_at_spec90",
        "spec_at_sens90",
        "f1",
        "auc",
    ]
    rows = []
    for i, cls in enumerate(classes):
        values = {m: [perclass_all[s][i][m] for s in range(len(seeds_eval))] for m in metrics}
        rows.append({
            "Class": cls,
            "Support(mean)": np.mean([perclass_all[s][i]["support"] for s in range(len(seeds_eval))]),
            **{f"{m}_mean": np.mean(values[m]) for m in metrics},
            **{f"{m}_std": np.std(values[m], ddof=1) for m in metrics},
        })

    dfpc = pd.DataFrame(rows)
    hdr = (
        f"{'Class':<22} "
        f"{'Prec':>10} {'±':>3} "
        f"{'Rec':>10} {'±':>3} "
        f"{'Spec':>10} {'±':>3} "
        f"{'Sens@Spec90':>12} {'±':>3} "
        f"{'Spec@Sens90':>12} {'±':>3} "
        f"{'F1':>10} {'±':>3} "
        f"{'AUC':>10} {'±':>3}"
    )
    print(hdr)
    print("-" * len(hdr))
    for _, r in dfpc.iterrows():
        print(
            f"{r['Class']:<22} "
            f"{r['precision_mean']:>10.3f} ±{r['precision_std']:<5.3f} "
            f"{r['recall_mean']:>10.3f} ±{r['recall_std']:<5.3f} "
            f"{r['specificity_mean']:>10.3f} ±{r['specificity_std']:<5.3f} "
            f"{r['sens_at_spec90_mean']:>12.3f} ±{r['sens_at_spec90_std']:<5.3f} "
            f"{r['spec_at_sens90_mean']:>12.3f} ±{r['spec_at_sens90_std']:<5.3f} "
            f"{r['f1_mean']:>10.3f} ±{r['f1_std']:<5.3f} "
            f"{r['auc_mean']:>10.3f} ±{r['auc_std']:<5.3f}"
        )
    return dfres, dfpc

# =======================================================================
# Main
# =======================================================================
if __name__ == "__main__":
    model, df, classes, artifacts = train_model_once(seed=42)
    evaluate_model(model, df, classes, artifacts)