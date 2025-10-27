# multimodal_hierarchical_radiomics.py
# === Multimodal (CNN + Radiomics) + Hierarchical Loss (Per-head class weights + Learnable task weights) ===

import os, re, time, json, math, random, pathlib
from pathlib import Path
import logging
import warnings

import numpy as np
import pandas as pd
from PIL import Image

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader

from torchvision import transforms
import timm

from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score, f1_score, roc_auc_score,
    confusion_matrix, precision_recall_fscore_support, roc_curve
)
from sklearn.preprocessing import label_binarize

# ---------------- Radiomics deps ----------------
import SimpleITK as sitk
from torchradiomics import (
    TorchRadiomicsFirstOrder,
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

# ====================== Cache I/O helpers: parquet if available, else CSV ======================
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
    names_json = path_base.with_suffix(".json")
    return json.loads(names_json.read_text()) if names_json.exists() else None

def _write_feat_names(names: list[str], path_base: Path):
    names_json = path_base.with_suffix(".json")
    names_json.write_text(json.dumps(names))
    return names_json

logging.getLogger("torchradiomics").setLevel(logging.ERROR)
warnings.filterwarnings("ignore", category=UserWarning)

# ====================== Paths & Config ======================
CSV_PATH  = pathlib.Path("/home/yaxi/Osteosarcoma-UT/ML_Features_1144.csv")
IMG_ROOTS = [
    pathlib.Path("/home/yaxi/Osteosarcoma-UT/Training-Set-1"),
    pathlib.Path("/home/yaxi/Osteosarcoma-UT/Training-Set-2")
]
RAD_CACHE_DIR = pathlib.Path("./rad_cache")

BACKBONE   = "inception_v3"       # Inception v3 backbone
DEVICE     = torch.device("cuda" if torch.cuda.is_available() else "cpu")
IMG_SIZE   = 299                   # Inception v3 uses 299x299 inputs
MAX_EPOCHS = 150
MIN_EPOCHS = 60
PATIENCE   = 50
LR = 1e-4
NUM_WORKERS = 4
PIN_MEMORY  = DEVICE.type == "cuda"

# --- Hierarchical uncertainty-weighting guardrails ---
W_MIN, W_MAX  = 0.2, 3.0
LS_MIN, LS_MAX = -math.log(W_MAX), -math.log(W_MIN)
REG_SCALE = 0.2

# --- Radiomics cache debug knobs ---
RAD_MAX_IMAGES = None   # e.g., 50 for a fast sanity check; None for all images
RAD_DEBUG      = False  # True to print per-image details

# --- Patient-level split config (used for baseline & for CV val slicing) ---
VAL_FRACTION_FROM_VALPAT = 0.13          # small validation slice fraction from selected patients
TRAIN_PATIENTS = ["Case-3", "P9"]        # train only
VAL_PATIENTS   = ["Case-48"]             # all of Case-48 in val
TEST_PATIENT   = "Case-4" # kept for backward compatibility (single-run)

# --- Checkpointing ---
CKPT_DIR = pathlib.Path("./checkpoints")
CKPT_DIR.mkdir(parents=True, exist_ok=True)
def _best_ckpt_path():
    return CKPT_DIR / "best_patient_split.pt"

# --- MC-Dropout evaluation seeds (5 runs) ---
MC_EVAL_SEEDS = [1, 2, 3, 4, 5]

# ====================== Reproducibility ======================
def set_seed(seed: int):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

# ====================== Transforms ======================
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

# ====================== Radiomics extractor (First-Order + optional Shape2D) ======================
def extract_all_radiomics(
    x: torch.Tensor,
    voxelArrayShift: float = 0.0,
    pixelSpacing = [1.0, 1.0, 1.0],
    binWidth: float | None = None
):
    img_np  = x.to(dtype=torch.float64, non_blocking=False).cpu().numpy()
    mask_np = (x > 0).to(dtype=torch.uint8,  non_blocking=False).cpu().numpy()

    sitk_img  = sitk.GetImageFromArray(img_np)
    sitk_mask = sitk.GetImageFromArray(mask_np)

    dim = sitk_img.GetDimension()
    if dim == 2:
        sx = float(pixelSpacing[1]) if len(pixelSpacing) >= 2 else 1.0
        sy = float(pixelSpacing[0]) if len(pixelSpacing) >= 1 else 1.0
        sitk_img.SetSpacing((sx, sy))
        sitk_mask.SetSpacing((sx, sy))
    elif dim == 3:
        if len(pixelSpacing) != 3:
            raise ValueError("For 3D, pixelSpacing must be [z, y, x] length 3.")
        sitk_img.SetSpacing((pixelSpacing[2], pixelSpacing[1], pixelSpacing[0]))
        sitk_mask.SetSpacing((pixelSpacing[2], pixelSpacing[1], pixelSpacing[0]))
    else:
        raise ValueError(f"Unsupported image dimension: {dim}")

    inject_torch_radiomics()

    base_compute = dict(
        voxelBased=False, padDistance=1, kernelRadius=1, maskedKernel=False,
        voxelBatch=512, dtype=torch.float64, device=x.device
    )
    base_settings = dict(voxelArrayShift=voxelArrayShift)
    if binWidth is not None: base_settings["binWidth"] = float(binWidth)

    extractors = [TorchRadiomicsFirstOrder(sitk_img, sitk_mask, **base_settings, **base_compute)]
    if _Shape2DClass is not None:
        if _SHAPE_MODE == "shape2d":
            extractors.append(_Shape2DClass(sitk_img, sitk_mask, **base_settings, **base_compute))
        elif _SHAPE_MODE == "shape_fallback":
            extractors.append(_Shape2DClass(
                sitk_img, sitk_mask, force2D=True, force2DDimension=0,
                **base_settings, **base_compute
            ))

    features, names = {}, []
    for ext in extractors:
        out = ext.execute()
        for k, v in out.items():
            if isinstance(v, sitk.Image): continue
            tv = v if isinstance(v, torch.Tensor) else torch.as_tensor(v, dtype=torch.float64, device=x.device)
            if torch.isfinite(tv).all():
                features[k] = tv
                names.append(k)
    return features, names

# ====================== Radiomics cache builder ======================
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
    cache_dir.mkdir(parents=True, exist_ok=True)
    base = cache_dir / "radiomics_cache"

    cached = _read_cache(base)
    feat_names = _read_feat_names(cache_dir / "radiomics_feat_names")
    if cached is not None and feat_names is not None:
        kind = "Parquet" if _PARQUET_OK and (cache_dir / "radiomics_cache.parquet").exists() else "CSV"
        print(f"[Radiomics] Using existing {kind} cache: {cache_dir}")
        return cached, feat_names, {"table": cached, "names": cache_dir / "radiomics_feat_names.json"}

    rows = []
    feat_names = None
    paths = df["path"].astype(str).unique().tolist()
    if max_images is not None: paths = paths[:max_images]
    total = len(paths)
    store_kind = "Parquet" if _PARQUET_OK else "CSV"
    print(f"[Radiomics] Computing FO+Shape2D for {total} images... (store={store_kind})")

    pbar = tqdm(total=total, desc="Radiomics(FO+S2D)", unit="img") if tqdm else None
    t0 = time.time()

    try:
        for i, p in enumerate(paths, 1):
            try:
                pil = Image.open(p).convert("L")
                arr = np.array(pil, dtype=np.float32)
                x = torch.from_numpy(arr).to(device=device, dtype=torch.float32)

                if debug:
                    print(f"[{i:04d}/{total}] {p} | shape={tuple(arr.shape)} | min={float(arr.min()):.1f} max={float(arr.max()):.1f}")

                feats_dict, feat_names_i = extract_all_radiomics(
                    x, voxelArrayShift=voxelArrayShift, pixelSpacing=pixelSpacing, binWidth=binWidth
                )
                if feat_names is None:
                    feat_names = feat_names_i
                    if debug: print(f"[Feature names] {len(feat_names)} FO+Shape2D features")

                row = {"path": p}
                for k in feat_names:
                    v = feats_dict[k]
                    if isinstance(v, torch.Tensor): v = v.detach().to("cpu").item()
                    row[k] = float(v)
                rows.append(row)

            except Exception as e:
                msg = f"[Radiomics][WARN] {p}: {e}"
                if pbar: pbar.write(msg)
                else: print(msg)

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

            if i % 100 == 0:
                tmp_df = pd.DataFrame(rows)
                out = _write_cache(tmp_df, cache_dir / "radiomics_cache", autosave=True, i=i)
                if pbar: pbar.write(f"[Autosave] {out}")
                else: print(f"[Autosave] {out}")

    finally:
        if pbar: pbar.close()

    if not rows:
        raise RuntimeError("No radiomics rows computed — check inputs.")

    rad_df = pd.DataFrame(rows)
    out_main = _write_cache(rad_df, cache_dir / "radiomics_cache", autosave=False)
    names_path = _write_feat_names(feat_names, cache_dir / "radiomics_feat_names")
    print(f"[Radiomics] Saved cache → {out_main}")
    return rad_df, feat_names, {"table": out_main, "names": names_path}

# ====================== Data loading ======================
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
        if len(nums) < 4: return None
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

    # --- add patient id ---
    def _extract_patient_id_from_string(s: str) -> str:
        s = str(s).lower().replace("_", "-").replace(" ", "-")
        if re.search(r"\bcase-?3\b", s):   return "Case-3"
        if re.search(r"\bcase-?4\b", s):   return "Case-4"
        if re.search(r"\bcase-?48\b", s):  return "Case-48"
        if re.search(r"\bp-?9\b", s):      return "P9"
        return "Unknown"

    def assign_patient_id(row) -> str:
        for src in (row.get("image.name", ""), row.get("path", "")):
            pid = _extract_patient_id_from_string(src)
            if pid != "Unknown":
                return pid
        return "Unknown"

    df["patient"] = df.apply(assign_patient_id, axis=1)

    print("\n[Data loading summary]")
    print(f"  CSV rows:                     {before}")
    print(f"  Rows with valid key:          {before - int(df['canon'].isna().sum())}")
    print(f"  Rows with resolved image path:{resolved}")
    print(f"  Usable rows after label clean:{after_label}")
    print(f"  TOTAL IMAGES LOADED:          {len(df)}")
    print(f"  Per-class counts:             {df['label'].value_counts().to_dict()}")
    print(f"  Per-patient counts:           {df['patient'].value_counts().to_dict()}")
    if dups:
        print(f"  Note: {len(dups)} duplicate basenames detected on disk (kept first occurrence).")
    expected = {"Case-3", "Case-4", "Case-48", "P9"}
    seen = set(df["patient"].unique())
    print("\n[Dataset patients] seen:", sorted(seen))
    missing = expected - seen
    if missing:
        print("[Warn] Expected patients missing from dataset:", sorted(missing))
    return df, classes

# ====================== Dataset (returns image, radiomics, label) ======================
class TumorDataset(Dataset):
    def __init__(self, df, rad_df, feat_names, rad_stats, train=False):
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

# ====================== Metrics helpers ======================
def _sens_spec_at_targets(y_true_bin, y_score, spec_target=0.9, sens_target=0.9):
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
        sens_at_spec, spec_at_sens = _sens_spec_at_targets((y_true_arr == i).astype(int), y_prob_arr[:, i])
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

# ====================== Multimodal Hierarchical Model ======================
class MultiModalHierNet(nn.Module):
    """
    CNN + Radiomics → attention fusion → two heads:
      - head_coarse: Non-Tumor vs Tumor (2-way)
      - head_fine:   Non-Viable vs Viable (2-way)
    Learnable uncertainty weights: log_sigma_a, log_sigma_b
    """
    def __init__(self, backbone: str, rad_in_dim: int,
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

        # Attention gate for fusion
        self.att_gate = nn.Sequential(
            nn.Linear(2 * fusion_dim, att_hidden),
            nn.ReLU(inplace=True),
            nn.Dropout(p_drop),
            nn.Linear(att_hidden, 2),
            nn.Softmax(dim=1)  # [w_img, w_rad]
        )

        # Two hierarchical heads on fused embedding
        self.head_coarse = nn.Linear(fusion_dim, 2)  # Non-Tumor vs Tumor
        self.head_fine   = nn.Linear(fusion_dim, 2)  # Non-Viable vs Viable

        # --- Learnable task weights (explicit init so exp(-log_sigma) = 1.0 and 1.5) ---
        init_wA = 0.5
        init_wB = 1.5
        self.log_sigma_a = nn.Parameter(torch.tensor([-math.log(init_wA)], dtype=torch.float32))
        self.log_sigma_b = nn.Parameter(torch.tensor([-math.log(init_wB)], dtype=torch.float32))
        
    def forward(self, x_img, x_rad):
        f_img = self.cnn_proj(self.cnn(x_img))  # [B, fusion_dim]
        f_rad = self.rad_net(x_rad)             # [B, fusion_dim]
        h = torch.cat([f_img, f_rad], dim=1)    # [B, 2*fusion_dim]
        w = self.att_gate(h)                    # [B, 2]
        w_img, w_rad = w[:, :1], w[:, 1:]
        f_fused = w_img * f_img + w_rad * f_rad
        logits_a = self.head_coarse(f_fused)
        logits_b = self.head_fine(f_fused)
        return logits_a, logits_b

# ====================== Hierarchical Loss ======================
def hierarchical_loss_uncertainty(model, logits_a, logits_b, y3, W_A=None, W_B=None, reg_scale=REG_SCALE):
    """
    y3 in {0: Non-Tumor, 1: Non-Viable-Tumor, 2: Viable}
    L = exp(-λA)*CE_A + exp(-λB)*CE_B + reg_scale*(λA + λB)
    """
    y_a = (y3 != 0).long()  # 0: Non-Tumor, 1: Tumor
    y_b = torch.where(y3 == 1, 0, torch.where(y3 == 2, 1, -1)).long()  # 0=NVT, 1=V, -1=untouched
    mask_b = (y_b >= 0)

    loss_a = nn.functional.cross_entropy(logits_a, y_a, weight=W_A)
    if mask_b.any():
        loss_b = nn.functional.cross_entropy(logits_b[mask_b], y_b[mask_b], weight=W_B)
        #focal_b = FocalCE(gamma=2.0, alpha=torch.tensor([0.25, 0.75], device=logits_b.device))
        #loss_b = focal_b(logits_b[mask_b], y_b[mask_b])
    else:
        loss_b = logits_a.new_tensor(0.0)

    inv_var_a = torch.exp(-model.log_sigma_a)
    inv_var_b = torch.exp(-model.log_sigma_b)
    loss = inv_var_a * loss_a + inv_var_b * loss_b + reg_scale * (model.log_sigma_a + model.log_sigma_b)
    return loss, loss_a.detach(), loss_b.detach()

# ====================== Prob fusion (2->3) ======================
def fuse_probs_to_three_classes(logits_a, logits_b):
    pa = torch.softmax(logits_a, dim=1)  # [B,2]
    pb = torch.softmax(logits_b, dim=1)  # [B,2]
    p_non_tumor = pa[:, 0:1]
    p_tumor     = pa[:, 1:2]
    p_nvt = pb[:, 0:1]
    p_vi  = pb[:, 1:2]
    p_cls0 = p_non_tumor
    p_cls1 = p_tumor * p_nvt
    p_cls2 = p_tumor * p_vi
    return torch.cat([p_cls0, p_cls1, p_cls2], dim=1)

# ====================== MC Dropout toggler ======================
def set_mc_dropout(model, enable=True):
    """
    If enable=True: set only Dropout modules to train(); keep everything else in eval().
    If enable=False: set full model back to eval().
    """
    if not enable:
        model.eval()
        return
    model.eval()
    for m in model.modules():
        if isinstance(m, (nn.Dropout, nn.Dropout1d, nn.Dropout2d, nn.AlphaDropout)):
            m.train()

# ====================== Patient-level VAL helper ======================
def stratified_val_from_patient(df_patient: pd.DataFrame, val_fraction: float, seed: int):
    """
    Build a validation subset from a single patient's data with per-class representation.
    Ensures (when class count >=2) at least 1 in val and at least 1 remains outside val.
    Returns: val_df, remainder_df
    """
    val_idx = []
    for y, g in df_patient.groupby("y", sort=False):
        n = len(g)
        if n <= 1:
            n_val = 0
        else:
            n_val = int(round(val_fraction * n))
            n_val = max(1, n_val)
            n_val = min(n_val, n - 1)
        if n_val > 0:
            val_idx.append(g.sample(n=n_val, random_state=seed).index)
    val_idx = pd.Index(np.concatenate(val_idx)) if len(val_idx) else pd.Index([])
    val_df = df_patient.loc[val_idx].copy()
    remainder_df = df_patient.drop(val_idx).copy()
    return val_df, remainder_df

# ===== helper to build validation from one or multiple training patients (filters by presence) =====
def _build_val_from_patients(train_pool_df, patients, frac, seed_):
    if isinstance(patients, str):
        patients = [patients]
    patients = [p for p in patients if p in train_pool_df["patient"].unique()]
    if not patients:
        raise ValueError("VAL_FROM_PATIENT does not match any patient in the training pool.")

    val_parts, remainder_parts = [], []
    for p in patients:
        subset = train_pool_df[train_pool_df["patient"] == p].copy()
        val_p, rem_p = stratified_val_from_patient(subset, frac, seed_)
        val_parts.append(val_p); remainder_parts.append(rem_p)

    val_df_local = pd.concat(val_parts, axis=0).sort_index() if len(val_parts) else train_pool_df.iloc[0:0].copy()
    remainder_df_local = pd.concat(remainder_parts, axis=0).sort_index() if len(remainder_parts) else train_pool_df.copy()
    return val_df_local, remainder_df_local

# ====================== Train Once (baseline, kept for compatibility) ======================
def train_model_once(seed=2025):
    set_seed(seed)
    df, classes = load_df_from_csv(CSV_PATH, IMG_ROOTS)

    # Build/load radiomics cache (FO + Shape2D)
    rad_df, feat_names, _paths = build_or_load_radiomics_cache(
        df, RAD_CACHE_DIR,
        pixelSpacing=[1.0,1.0,1.0], voxelArrayShift=0.0, binWidth=5.0, device=DEVICE,
        max_images=RAD_MAX_IMAGES, debug=RAD_DEBUG
    )

    # restrict to images that have radiomics
    df = df[df["path"].isin(rad_df["path"])].reset_index(drop=True)
    rad_df = rad_df[rad_df["path"].isin(df["path"])].reset_index(drop=True)

    # ====== PATIENT-LEVEL SPLIT ======
    train_pool = df[df["patient"].isin(TRAIN_PATIENTS)].copy()
    test_df    = df[df["patient"] == TEST_PATIENT].copy()

    print("\n[Sanity] Patients in TRAIN pool:", sorted(train_pool["patient"].unique().tolist()))
    print("[Sanity] Patients in TEST set:", sorted(test_df["patient"].unique().tolist()))
    assert TEST_PATIENT not in set(train_pool["patient"]), "Leak: test patient appears in TRAIN pool!"
    assert set(test_df["patient"]) == {TEST_PATIENT}, "TEST set contains non-test patients!"

    # Build validation from the specified patient(s)
    val_df, _train_pool_after_val = _build_val_from_patients(train_pool, VAL_FROM_PATIENT, VAL_FRACTION_FROM_VALPAT, seed)

    train_df = pd.concat([
        train_pool[~train_pool.index.isin(val_df.index)],
    ], axis=0).sort_index()

    print("\n[Training split sizes — Patient-level]")
    print(f"  TRAIN images (pool minus VAL-slices): {len(train_df)}")
    print(f"  VAL images (from {VAL_FROM_PATIENT}): {len(val_df)}")
    print(f"  TEST images (all {TEST_PATIENT}):     {len(test_df)}")
    print("[Sanity] Per-class counts (TRAIN):", train_df["label"].value_counts().to_dict())
    print("[Sanity] Per-class counts (VAL):  ", val_df["label"].value_counts().to_dict())
    print("[Sanity] Per-class counts (TEST): ", test_df["label"].value_counts().to_dict())

    # radiomics normalization: fit on TRAIN only
    train_paths = set(train_df["path"])
    rad_train = rad_df[rad_df["path"].isin(train_paths)][feat_names].to_numpy(dtype=np.float32)
    rad_mean = rad_train.mean(axis=0)
    rad_std  = rad_train.std(axis=0, ddof=0)
    rad_stats = {"mean": rad_mean, "std": rad_std}

    loaders = {
        "train": DataLoader(
            TumorDataset(train_df, rad_df, feat_names, rad_stats, train=True),
            batch_size=16, shuffle=True, num_workers=NUM_WORKERS, pin_memory=PIN_MEMORY
        ),
        "val": DataLoader(
            TumorDataset(val_df, rad_df, feat_names, rad_stats, train=False),
            batch_size=16, shuffle=False, num_workers=NUM_WORKERS, pin_memory=PIN_MEMORY
        )
    }

    # model
    model = MultiModalHierNet(BACKBONE, rad_in_dim=len(feat_names)).to(DEVICE)

    # ---- per-head + per-label class weights ----
    # Head A (Non-Tumor vs Tumor)
    y_a_train = (train_df["y"].values != 0).astype(int)
    cnt_a = np.bincount(y_a_train, minlength=2).astype(float)
    w_a = 1.0 / (cnt_a / cnt_a.sum() + 1e-9)
    w_a = w_a * (2.0 / w_a.sum())
    W_A = torch.tensor(w_a, dtype=torch.float32, device=DEVICE)

    # Head B (Non-Viable vs Viable), tumor-only rows
    tumor_rows = train_df[train_df["y"] != 0]
    y_b_train = np.where(tumor_rows["y"].values == 1, 0, 1)
    cnt_b = np.bincount(y_b_train, minlength=2).astype(float)
    w_b = 1.0 / (cnt_b / cnt_b.sum() + 1e-9)
    w_b = w_b * (2.0 / w_b.sum())
    W_B = torch.tensor(w_b, dtype=torch.float32, device=DEVICE)

    # 3-class label weight (for reporting/debugging convenience)
    cnt_3 = train_df["y"].value_counts().sort_index().astype(float)
    w_3 = 1.0 / (cnt_3 / cnt_3.sum() + 1e-9)
    w_3 = w_3 * (3.0 / w_3.sum())  # normalize so mean weight = 1
    W_3 = torch.tensor(w_3.values, dtype=torch.float32, device=DEVICE)

    print(f"[Per-head class weights] W_A={w_a.tolist()}  W_B={w_b.tolist()}")
    print(f"[Per-label (3-way)] counts={cnt_3.to_dict()} weights={w_3.round(4).to_dict()}")

    # optimizer: boost lr for log_sigma params
    optimizer = optim.AdamW([
        {"params": model.cnn.parameters(),        "lr": LR},
        {"params": model.cnn_proj.parameters(),   "lr": LR},
        {"params": model.rad_net.parameters(),    "lr": LR},
        {"params": model.att_gate.parameters(),   "lr": LR},
        {"params": model.head_coarse.parameters(),"lr": LR},
        {"params": model.head_fine.parameters(),  "lr": LR},
        {"params": [model.log_sigma_a, model.log_sigma_b], "lr": LR*8, "weight_decay": 0.0},
    ], lr=LR)

    # initial weights
    wA0 = float(torch.exp(-model.log_sigma_a).detach().cpu())
    wB0 = float(torch.exp(-model.log_sigma_b).detach().cpu())
    print(f"Init learned task weights ~ wA={wA0:.3f}, wB={wB0:.3f}")

    best_state, best_f1 = None, -1
    wait = 0
    ckpt_path = _best_ckpt_path()

    for epoch in range(1, MAX_EPOCHS + 1):
        # -------- Train --------
        model.train(); train_loss = 0.0; seen_train = 0
        for x_img, x_rad, y in loaders["train"]:
            x_img = x_img.to(DEVICE, non_blocking=True)
            x_rad = x_rad.to(DEVICE, non_blocking=True)
            y     = y.to(DEVICE, non_blocking=True)

            optimizer.zero_grad()
            logits_a, logits_b = model(x_img, x_rad)
            loss, la, lb = hierarchical_loss_uncertainty(
                model, logits_a, logits_b, y, W_A=W_A, W_B=W_B, reg_scale=REG_SCALE
            )
            loss.backward(); optimizer.step()

            with torch.no_grad():
                model.log_sigma_a.clamp_(LS_MIN, LS_MAX)
                model.log_sigma_b.clamp_(LS_MIN, LS_MAX)

            train_loss += float(loss.detach().cpu())
            seen_train += y.size(0)

        train_loss /= max(1, len(loaders["train"]))
        print(f"[Epoch {epoch:03d}] train images: {seen_train}/{len(train_df)} | Train Loss: {train_loss:.3f}")

        if epoch % 5 == 0:
            wA = float(torch.exp(-model.log_sigma_a).detach().cpu())
            wB = float(torch.exp(-model.log_sigma_b).detach().cpu())
            print(f"    Learned task weights ~ wA={wA:.3f}, wB={wB:.3f}")

        # -------- Val --------
        model.eval(); T, P, Q = [], [], []; seen_val = 0
        with torch.no_grad():
            for x_img, x_rad, y in loaders["val"]:
                x_img = x_img.to(DEVICE, non_blocking=True)
                x_rad = x_rad.to(DEVICE, non_blocking=True)
                y     = y.to(DEVICE, non_blocking=True)

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
            overall, perclass = compute_overall_and_perclass(T, P, np.array(Q), classes)
            print_metrics_block(f"VAL METRICS @ Epoch {epoch}", overall, perclass)

        # Save best
        if f1w > best_f1:
            best_f1 = f1w
            best_state = {k: v.cpu() for k, v in model.state_dict().items()}
            torch.save({
                "state_dict": best_state,
                "backbone": BACKBONE,
                "img_size": IMG_SIZE,
                "classes": classes,
                "rad_mean": rad_stats["mean"],
                "rad_std":  rad_stats["std"],
                "feat_names": feat_names,
                "val_f1_weighted": best_f1,
                "epoch": epoch,
                "W_3": w_3.to_dict()
            }, ckpt_path)
            print(f"[Checkpoint] New best F1w={best_f1:.4f} saved → {ckpt_path}")
            wait = 0
        else:
            wait += 1

        if epoch >= MIN_EPOCHS and wait >= PATIENCE:
            print(f"Early stopping at epoch {epoch}")
            break

    # reload best
    if ckpt_path.exists():
        payload = torch.load(ckpt_path, map_location="cpu")
        model.load_state_dict({k: v.to(DEVICE) for k, v in payload["state_dict"].items()})
        print(f"[Checkpoint] Loaded best model (Val F1w={payload.get('val_f1_weighted', float('nan')):.4f})")
    elif best_state:
        model.load_state_dict({k: v.to(DEVICE) for k, v in best_state.items()})

    wA = float(torch.exp(-model.log_sigma_a).detach().cpu())
    wB = float(torch.exp(-model.log_sigma_b).detach().cpu())
    print(f"Final learned task weights ~ wA={wA:.3f}, wB={wB:.3f}")

    artifacts = dict(
        rad_df=rad_df, feat_names=feat_names, rad_stats=rad_stats,
        df=df, classes=classes, test_df=test_df,
        best_ckpt=str(ckpt_path),
        W_3=W_3
    )
    return model, artifacts

# ====================== Evaluate once on fixed TEST patient ======================
def evaluate_model(model, artifacts):
    df = artifacts["df"]; classes = artifacts["classes"]
    rad_df = artifacts["rad_df"]; feat_names = artifacts["feat_names"]; rad_stats = artifacts["rad_stats"]
    test_df = artifacts["test_df"]

    if "best_ckpt" in artifacts:
        print(f"[Eval] Using best checkpoint: {artifacts['best_ckpt']}")

    print("\n[Sanity] Test patients:", sorted(test_df["patient"].unique().tolist()))
    assert set(test_df["patient"]) == {TEST_PATIENT}, "Leak: TEST contains non-test patients!"
    print("[Sanity] Per-class counts (TEST):", test_df["label"].value_counts().to_dict())
    print(f"[Evaluation] TEST images (all {TEST_PATIENT}): {len(test_df)}")

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

            logits_a, logits_b = model(x_img, x_rad)
            prob = fuse_probs_to_three_classes(logits_a, logits_b)
            y_true += y.cpu().tolist()
            y_pred += prob.argmax(1).cpu().tolist()
            y_prob += prob.cpu().tolist()
            seen_test += y.size(0)

    print(f"[Evaluation] test images processed: {seen_test}/{len(test_df)}")

    y_prob = np.array(y_prob)
    overall, perclass = compute_overall_and_perclass(y_true, y_pred, y_prob, classes)
    print_metrics_block(f"TEST EVALUATION ({TEST_PATIENT})", overall, perclass)

    return overall, perclass

# ====================== Evaluate 5x with MC Dropout ======================
def evaluate_model_5x(model, artifacts, seeds_eval=MC_EVAL_SEEDS, patient_name=None):
    """
    Slightly extended: patient_name allows using this function for LOPO folds.
    If None, falls back to global TEST_PATIENT.
    """
    classes = artifacts["classes"]
    rad_df = artifacts["rad_df"]; feat_names = artifacts["feat_names"]; rad_stats = artifacts["rad_stats"]
    test_df = artifacts["test_df"]

    pname = patient_name if patient_name is not None else TEST_PATIENT

    print("\n[Sanity] Test patients:", sorted(test_df["patient"].unique().tolist()))
    assert set(test_df["patient"]) == {pname}, "Leak: TEST contains non-test patients!"
    print("[Sanity] Per-class counts (TEST):", test_df["label"].value_counts().to_dict())
    print(f"[Evaluation] TEST images (all {pname}): {len(test_df)}")

    base_loader = DataLoader(
        TumorDataset(test_df, rad_df, feat_names, rad_stats, train=False),
        batch_size=16, shuffle=False, num_workers=NUM_WORKERS, pin_memory=PIN_MEMORY
    )

    results = []
    perclass_all = []

    for s in seeds_eval:
        set_seed(s)
        set_mc_dropout(model, enable=True)  # stochastic forward pass

        y_true, y_pred, y_prob = [], [], []
        seen_test = 0
        with torch.no_grad():
            for x_img, x_rad, y in base_loader:
                x_img = x_img.to(DEVICE, non_blocking=True)
                x_rad = x_rad.to(DEVICE, non_blocking=True)
                y     = y.to(DEVICE, non_blocking=True)

                logits_a, logits_b = model(x_img, x_rad)  # dropout active
                prob = fuse_probs_to_three_classes(logits_a, logits_b)
                y_true += y.cpu().tolist()
                y_pred += prob.argmax(1).cpu().tolist()
                y_prob += prob.cpu().tolist()
                seen_test += y.size(0)

        print(f"[Evaluation seed={s} (MC dropout)] test images: {seen_test}/{len(test_df)}")

        y_prob = np.array(y_prob)
        overall, perclass = compute_overall_and_perclass(y_true, y_pred, y_prob, classes)
        print_metrics_block(f"TEST EVALUATION (seed={s}, MC-Dropout, patient={pname})", overall, perclass)

        results.append(overall)
        perclass_all.append(perclass)

    # restore normal eval mode
    set_mc_dropout(model, enable=False)

    # === Overall mean ± std summary ===
    dfres = pd.DataFrame(results, index=[f"eval_seed_{s}" for s in seeds_eval])
    print("\n=== Overall Summary (Mean ± Std over 5 MC-Dropout passes) ===")
    for col in dfres.columns:
        print(f"{col:>16}: {dfres[col].mean():.4f} ± {dfres[col].std(ddof=1):.4f}")

    # === Per-class mean ± std summary ===
    print("\n=== Per-Class Summary (Mean ± Std over 5 MC-Dropout passes) ===")
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
        agg = {m: [perclass_all[k][i][m] for k in range(len(seeds_eval))] for m in metrics}
        rows.append({
            "Class": cls,
            **{f"{m}_mean": float(np.mean(agg[m])) for m in metrics},
            **{f"{m}_std":  float(np.std(agg[m], ddof=1)) for m in metrics},
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

# ====================== LOPO-CV: train a single fold (minimal change; reuses your logic) ======================
def train_one_fold_loocv(df_all, rad_df, feat_names, classes, test_patient: str, seed=42):
    """
    Mirrors 'train_model_once' but:
    - builds train/val using all patients except the test patient
    - keeps all your weight computations/prints intact
    - returns model and per-fold artifacts (with test_df set to this fold's patient)
    """
    set_seed(seed)

    # restrict to images that have radiomics (df_all already filtered in driver)
    df = df_all

    # ====== PATIENT-LEVEL SPLIT for this fold ======
    train_pool = df[df["patient"] != test_patient].copy()
    test_df    = df[df["patient"] == test_patient].copy()

    print("\n" + "="*90)
    print(f"[FOLD] Test patient: {test_patient}")
    print("="*90)
    print("[Sanity] Patients in TRAIN pool:", sorted(train_pool["patient"].unique().tolist()))
    print("[Sanity] Patients in TEST set:", sorted(test_df["patient"].unique().tolist()))
    assert test_patient not in set(train_pool["patient"]), "Leak: test patient appears in TRAIN pool!"
    assert set(test_df["patient"]) == {test_patient}, "TEST set contains non-test patients!"

    # Build validation from specified training patients (filtered internally)
    val_df, _ = _build_val_from_patients(train_pool, VAL_FROM_PATIENT, VAL_FRACTION_FROM_VALPAT, seed)
    train_df = train_pool[~train_pool.index.isin(val_df.index)].copy()

    print("\n[Training split sizes — Patient-level]")
    print(f"  TRAIN images (pool minus VAL-slices): {len(train_df)}")
    print(f"  VAL images (from {VAL_FROM_PATIENT} ∩ train_pool): {len(val_df)}")
    print(f"  TEST images (all {test_patient}):                 {len(test_df)}")
    print("[Sanity] Per-class counts (TRAIN):", train_df["label"].value_counts().to_dict())
    print("[Sanity] Per-class counts (VAL):  ", val_df["label"].value_counts().to_dict())
    print("[Sanity] Per-class counts (TEST): ", test_df["label"].value_counts().to_dict())

    # radiomics normalization: fit on TRAIN only
    train_paths = set(train_df["path"])
    rad_train = rad_df[rad_df["path"].isin(train_paths)][feat_names].to_numpy(dtype=np.float32)
    rad_mean = rad_train.mean(axis=0)
    rad_std  = rad_train.std(axis=0, ddof=0)
    rad_stats = {"mean": rad_mean, "std": rad_std}

    loaders = {
        "train": DataLoader(
            TumorDataset(train_df, rad_df, feat_names, rad_stats, train=True),
            batch_size=16, shuffle=True, num_workers=NUM_WORKERS, pin_memory=PIN_MEMORY
        ),
        "val": DataLoader(
            TumorDataset(val_df, rad_df, feat_names, rad_stats, train=False),
            batch_size=16, shuffle=False, num_workers=NUM_WORKERS, pin_memory=PIN_MEMORY
        )
    }

    # model
    model = MultiModalHierNet(BACKBONE, rad_in_dim=len(feat_names)).to(DEVICE)

    # ---- per-head + per-label class weights (unchanged logic) ----
    y_a_train = (train_df["y"].values != 0).astype(int)
    cnt_a = np.bincount(y_a_train, minlength=2).astype(float)
    w_a = 1.0 / (cnt_a / cnt_a.sum() + 1e-9); w_a = w_a * (2.0 / w_a.sum())
    W_A = torch.tensor(w_a, dtype=torch.float32, device=DEVICE)

    tumor_rows = train_df[train_df["y"] != 0]
    y_b_train = np.where(tumor_rows["y"].values == 1, 0, 1)
    cnt_b = np.bincount(y_b_train, minlength=2).astype(float)
    w_b = 1.0 / (cnt_b / cnt_b.sum() + 1e-9); w_b = w_b * (2.0 / w_b.sum())
    W_B = torch.tensor(w_b, dtype=torch.float32, device=DEVICE)

    cnt_3 = train_df["y"].value_counts().sort_index().astype(float)
    w_3 = 1.0 / (cnt_3 / cnt_3.sum() + 1e-9); w_3 = w_3 * (3.0 / w_3.sum())
    W_3 = torch.tensor(w_3.values, dtype=torch.float32, device=DEVICE)

    print(f"[Per-head class weights] W_A={w_a.tolist()}  W_B={w_b.tolist()}")
    print(f"[Per-label (3-way)] counts={cnt_3.to_dict()} weights={w_3.round(4).to_dict()}")

    # optimizer (same)
    optimizer = optim.AdamW([
        {"params": model.cnn.parameters(),        "lr": LR},
        {"params": model.cnn_proj.parameters(),   "lr": LR},
        {"params": model.rad_net.parameters(),    "lr": LR},
        {"params": model.att_gate.parameters(),   "lr": LR},
        {"params": model.head_coarse.parameters(),"lr": LR},
        {"params": model.head_fine.parameters(),  "lr": LR},
        {"params": [model.log_sigma_a, model.log_sigma_b], "lr": LR*8, "weight_decay": 0.0},
    ], lr=LR)

    # initial learned task weights print
    wA0 = float(torch.exp(-model.log_sigma_a).detach().cpu())
    wB0 = float(torch.exp(-model.log_sigma_b).detach().cpu())
    print(f"Init learned task weights ~ wA={wA0:.3f}, wB={wB0:.3f}")

    best_state, best_f1 = None, -1
    wait = 0

    for epoch in range(1, MAX_EPOCHS + 1):
        model.train(); train_loss = 0.0; seen_train = 0
        for x_img, x_rad, y in loaders["train"]:
            x_img = x_img.to(DEVICE, non_blocking=True)
            x_rad = x_rad.to(DEVICE, non_blocking=True)
            y     = y.to(DEVICE, non_blocking=True)

            optimizer.zero_grad()
            logits_a, logits_b = model(x_img, x_rad)
            loss, la, lb = hierarchical_loss_uncertainty(
                model, logits_a, logits_b, y, W_A=W_A, W_B=W_B, reg_scale=REG_SCALE
            )
            loss.backward(); optimizer.step()

            with torch.no_grad():
                model.log_sigma_a.clamp_(LS_MIN, LS_MAX)
                model.log_sigma_b.clamp_(LS_MIN, LS_MAX)

            train_loss += float(loss.detach().cpu())
            seen_train += y.size(0)

        train_loss /= max(1, len(loaders["train"]))
        print(f"[Epoch {epoch:03d}] train images: {seen_train}/{len(train_df)} | Train Loss: {train_loss:.3f}")

        if epoch % 5 == 0:
            wA = float(torch.exp(-model.log_sigma_a).detach().cpu())
            wB = float(torch.exp(-model.log_sigma_b).detach().cpu())
            print(f"    Learned task weights ~ wA={wA:.3f}, wB={wB:.3f}")

        # -------- Val --------
        model.eval(); T, P, Q = [], [], []; seen_val = 0
        with torch.no_grad():
            for x_img, x_rad, y in loaders["val"]:
                x_img = x_img.to(DEVICE, non_blocking=True)
                x_rad = x_rad.to(DEVICE, non_blocking=True)
                y     = y.to(DEVICE, non_blocking=True)

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
            overall, perclass = compute_overall_and_perclass(T, P, np.array(Q), classes)
            print_metrics_block(f"VAL METRICS @ Epoch {epoch} (fold test={test_patient})", overall, perclass)

        if f1w > best_f1:
            best_f1 = f1w
            best_state = {k: v.cpu() for k, v in model.state_dict().items()}
            wait = 0
        else:
            wait += 1

        if epoch >= MIN_EPOCHS and wait >= PATIENCE:
            print(f"Early stopping at epoch {epoch}")
            break

    # restore best (no disk write per fold to keep light)
    if best_state:
        model.load_state_dict({k: v.to(DEVICE) for k, v in best_state.items()})

    wA = float(torch.exp(-model.log_sigma_a).detach().cpu())
    wB = float(torch.exp(-model.log_sigma_b).detach().cpu())
    print(f"Final learned task weights ~ wA={wA:.3f}, wB={wB:.3f}")

    artifacts = dict(
        rad_df=rad_df, feat_names=feat_names, rad_stats=rad_stats,
        df=df, classes=classes, test_df=test_df,
        W_3=W_3
    )
    return model, artifacts

# ====================== LOPO-CV driver ======================
def run_patient_loocv(seed=42):
    set_seed(seed)
    df_all, classes = load_df_from_csv(CSV_PATH, IMG_ROOTS)

    # Build radiomics ONCE across all images we have
    rad_df_full, feat_names, _paths = build_or_load_radiomics_cache(
        df_all, RAD_CACHE_DIR,
        pixelSpacing=[1.0,1.0,1.0], voxelArrayShift=0.0, binWidth=5.0, device=DEVICE,
        max_images=RAD_MAX_IMAGES, debug=RAD_DEBUG
    )
    # keep overlap only
    df_all = df_all[df_all["path"].isin(rad_df_full["path"])].reset_index(drop=True)
    rad_df_full = rad_df_full[rad_df_full["path"].isin(df_all["path"])].reset_index(drop=True)

    # Candidate patients (in stable order); use only those present
    candidate_patients = ["Case-3", "P9", "Case-48", "Case-4"]
    patients = [p for p in candidate_patients if p in set(df_all["patient"].unique())]
    if len(patients) < 2:
        raise RuntimeError(f"Need at least 2 patients for LOPO-CV; found {patients}")

    print("\n[LOPO-CV] Patients participating:", patients)

    fold_overall_means = []
    fold_names = []

    for test_patient in patients:
        model, artifacts_fold = train_one_fold_loocv(
            df_all=df_all, rad_df=rad_df_full, feat_names=feat_names, classes=classes,
            test_patient=test_patient, seed=seed
        )

        # Per-fold evaluation (5x MC)
        dfres, dfpc = evaluate_model_5x(model, artifacts_fold, seeds_eval=MC_EVAL_SEEDS, patient_name=test_patient)

        # Aggregate overall metrics for this fold by averaging over MC seeds
        fold_overall_means.append(dfres.mean(axis=0).to_dict())
        fold_names.append(test_patient)

    # ===== Aggregate across folds (macro over test patients) =====
    overall_df = pd.DataFrame(fold_overall_means, index=fold_names)
    print("\n" + "="*90)
    print("=== LOPO-CV Overall Metrics (Mean ± Std across folds) ===")
    for col in overall_df.columns:
        mu = overall_df[col].mean()
        sd = overall_df[col].std(ddof=1) if len(overall_df[col]) > 1 else 0.0
        print(f"{col:>16}: {mu:.4f} ± {sd:.4f}")

    return overall_df

# ====================== Main ======================
if __name__ == "__main__":
    
    model, artifacts = train_model_once(seed=2025)
    evaluate_model_5x(model, artifacts, seeds_eval=MC_EVAL_SEEDS)

