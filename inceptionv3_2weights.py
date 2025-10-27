# === Inception v3 3-Class Classifier: Patient-Level Splits ===
import os, math, random, pathlib, re
import numpy as np, pandas as pd
import torch, torch.nn as nn, torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image

from sklearn.metrics import (
    accuracy_score, f1_score, roc_auc_score, classification_report,
    confusion_matrix, precision_recall_fscore_support
)
from sklearn.preprocessing import label_binarize

import timm

# ----------------------- Paths & Config ------------------------
CSV_PATH  = pathlib.Path("/home/yaxi/Osteosarcoma-UT/ML_Features_1144.csv")
IMG_ROOTS = [
    pathlib.Path("/home/yaxi/Osteosarcoma-UT/Training-Set-1"),
    pathlib.Path("/home/yaxi/Osteosarcoma-UT/Training-Set-2")
]

BACKBONE   = "inception_v3"              # <<< CHANGED: Inception v3 (timm)
DEVICE     = torch.device("cuda" if torch.cuda.is_available() else "cpu")
IMG_SIZE   = 299                          # <<< CHANGED: Inception v3 default input size
MAX_EPOCHS = 100
MIN_EPOCHS = 20
PATIENCE   = 40
WARMUP_EPOCHS = 10
LR = 1e-4
NUM_WORKERS = 4
PIN_MEMORY  = DEVICE.type == "cuda"

# ---------------- Patient-level split config -------------------
VAL_FRACTION_FROM_VALPAT = 0.13          # small val slice taken from selected patients
TRAIN_PATIENTS = ["Case-3", "P9", "Case-48"]
VAL_FROM_PATIENT = ["Case-3", "P9", "Case-48"]   # subset of train patients to draw val from
TEST_PATIENT = "Case-4"                              # exclusive held-out test patient
MIN_PER_CLASS_VAL = 1                                # ensure at least 1 per class when possible

# ---- Weight control for uncertainty loss (keeps learning stable) ----
W_MIN, W_MAX = 0.2, 3.0                                       # keep effective weights in [0.2, 3.0]
LS_MIN, LS_MAX = -math.log(W_MAX), -math.log(W_MIN)           # clamp range for log_sigma
REG_SCALE = 0.2                                               # smaller than 0.5 so weights don't explode

# ----------------------- Reproducibility -----------------------
def set_seed(seed: int):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

# ---------------------- Transforms -----------------------------
# Note: Inception v3 was trained on ImageNet mean/std; keep same normalization.
transform_train = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(15),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])
transform_eval = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

class TumorDataset(Dataset):
    def __init__(self, df, train=False):
        self.df = df.reset_index(drop=True)
        self.tf = transform_train if train else transform_eval
    def __len__(self): return len(self.df)
    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        p = row["path"]
        if not os.path.exists(p):
            raise FileNotFoundError(f"Image path does not exist: {p}")
        img = Image.open(p).convert("RGB")
        return self.tf(img), int(row["y"])

# ---------------------- Load CSV + Patient IDs -----------------
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

# ---------------------- Patient-level split helper -------------
def make_patient_splits(
    df: pd.DataFrame,
    train_patients: list,
    val_from_patients: list,
    test_patient: str,
    val_frac: float,
    classes: list,
    rng_seed: int = 42
):
    """
    Returns: train_df, val_df, test_df
    - test_df = all images from TEST_PATIENT
    - val_df  = stratified sample (per-class) from VAL_FROM_PATIENT ∩ TRAIN_PATIENTS
    - train_df= remaining images from TRAIN_PATIENTS after removing val_df
    """
    assert "patient" in df.columns, "load_df_from_csv must add a 'patient' column"

    # --- test set: exclusive patient ---
    test_df = df[df["patient"] == test_patient].copy()
    if test_df.empty:
        print(f"[Warn] TEST_PATIENT '{test_patient}' not found; test will be empty.")

    # --- candidate pool for train (and val slice is carved from within) ---
    train_pool = df[df["patient"].isin(train_patients)].copy()
    if train_pool.empty:
        raise ValueError("No rows matched TRAIN_PATIENTS. Check names in df['patient'].")

    # --- where the val slice may be drawn from (must be subset of train patients) ---
    val_cands = train_pool[train_pool["patient"].isin(val_from_patients)].copy()

    # --- build a small stratified val set per-class ---
    rng = np.random.RandomState(rng_seed)
    val_idx = []
    for cls in range(len(classes)):
        cls_rows = val_cands[val_cands["y"] == cls]
        if cls_rows.empty:
            continue
        n = int(round(val_frac * len(cls_rows)))
        if len(cls_rows) >= MIN_PER_CLASS_VAL:
            n = max(n, MIN_PER_CLASS_VAL)
        n = min(n, len(cls_rows))
        choose_idx = rng.choice(cls_rows.index.values, size=n, replace=False)
        val_idx.extend(choose_idx.tolist())

    val_df = val_cands.loc[sorted(set(val_idx))].copy()

    # --- final train = train_pool minus validation indices ---
    train_df = train_pool.drop(index=val_df.index).copy()

    # --- sanity checks: no patient leakage with test ---
    tr_p = set(train_df["patient"].unique())
    va_p = set(val_df["patient"].unique())
    te_p = set(test_df["patient"].unique())
    if tr_p & te_p or va_p & te_p:
        raise AssertionError("Patient leakage detected between (train/val) and test!")
    # val & train may share patient ids; their images are disjoint by construction.

    print("\n[Patient-level split]")
    print("  Train patients:", sorted(tr_p))
    print("  Val patients:  ", sorted(va_p))
    print("  Test patient:  ", sorted(te_p))
    print(f"  Counts -> train: {len(train_df)} | val: {len(val_df)} | test: {len(test_df)}")
    print("  Train per-class:", train_df["y"].value_counts().sort_index().to_dict())
    print("  Val   per-class:", val_df["y"].value_counts().sort_index().to_dict())
    print("  Test  per-class:", test_df["y"].value_counts().sort_index().to_dict())

    return train_df, val_df, test_df

# ---------------------- Metrics Helpers ------------------------
def compute_overall_and_perclass(y_true, y_pred, y_prob, classes):
    labels_idx = list(range(len(classes)))
    acc = accuracy_score(y_true, y_pred)
    f1_mi = f1_score(y_true, y_pred, average="micro")
    f1_ma = f1_score(y_true, y_pred, average="macro")
    f1_w = f1_score(y_true, y_pred, average="weighted")

    try: auc_ovr = roc_auc_score(y_true, y_prob, multi_class="ovr", average="macro")
    except: auc_ovr = float("nan")
    try: auc_ovo = roc_auc_score(y_true, y_prob, multi_class="ovo", average="macro")
    except: auc_ovo = float("nan")

    prec, rec, f1c, support = precision_recall_fscore_support(
        y_true, y_pred, labels=labels_idx, average=None, zero_division=0
    )
    cm = confusion_matrix(y_true, y_pred, labels=labels_idx)
    specs = []
    for i in labels_idx:
        TP = cm[i, i]; FP = cm[:, i].sum() - TP; FN = cm[i, :].sum() - TP; TN = cm.sum() - TP - FP - FN
        specs.append(TN / (TN + FP) if (TN + FP) > 0 else float("nan"))

    try:
        Y_bin = label_binarize(y_true, classes=labels_idx)
        auc_class = [roc_auc_score(Y_bin[:, i], y_prob[:, i]) if Y_bin[:, i].min() != Y_bin[:, i].max() else float("nan") for i in labels_idx]
    except:
        auc_class = [float("nan")] * len(labels_idx)

    per_class = []
    for i, c in enumerate(classes):
        per_class.append(dict(cls=c, support=int(support[i]), precision=prec[i], recall=rec[i],
                              specificity=specs[i], f1=f1c[i], auc=auc_class[i]))
    overall = dict(acc=acc, f1_micro=f1_mi, f1_macro=f1_ma, f1_weighted=f1_w,
                   auc_macro_ovr=auc_ovr, auc_macro_ovo=auc_ovo)
    return overall, per_class

def print_metrics_block(title, overall, per_class):
    print(f"\n=== {title} ===")
    print(f"  Accuracy: {overall['acc']:.4f}")
    print(f"  F1 (micro/macro/w): {overall['f1_micro']:.4f} / {overall['f1_macro']:.4f} / {overall['f1_weighted']:.4f}")
    print(f"  Macro AUC (OvR/OvO): {overall['auc_macro_ovr']:.4f} / {overall['auc_macro_ovo']:.4f}")
    hdr = f"{'Class':<22} {'Support':>7}  {'Prec':>6}  {'Rec':>6}  {'Spec':>6}  {'F1':>6}  {'AUC':>6}"
    print("\nPer-class metrics:")
    print(hdr)
    print("-" * len(hdr))
    for r in per_class:
        print(f"{r['cls']:<22} {r['support']:>7d}  {r['precision']:>6.3f}  {r['recall']:>6.3f}  {r['specificity']:>6.3f}  {r['f1']:>6.3f}  {r['auc']:>6.3f}")

# ---------------------- Hierarchical Model ---------------------
class HierCNN(nn.Module):
    """
    Two-head hierarchical model:
      - head_coarse: Non-Tumor vs Tumor  (2-way)
      - head_fine:   Non-Viable vs Viable (2-way), trained only on Tumor samples
    Learnable uncertainty weights (Kendall et al.) are log variances:
      log_sigma_a, log_sigma_b
    """
    def __init__(self, backbone: str):
        super().__init__()
        # Inception v3 feature extractor with AVG pooling
        self.body = timm.create_model(backbone, pretrained=True, num_classes=0, global_pool="avg")
        feat = self.body.num_features
        self.head_coarse = nn.Linear(feat, 2)  # Non-Tumor vs Tumor
        self.head_fine   = nn.Linear(feat, 2)  # Non-Viable vs Viable

        # Initial effective task weights
        wA0 = 1.5   # for Non-Tumor vs Tumor
        wB0 = 1.0   # for Non-Viable vs Viable
        self.log_sigma_a = nn.Parameter(torch.tensor([-math.log(wA0)]))
        self.log_sigma_b = nn.Parameter(torch.tensor([-math.log(wB0)]))

    def forward(self, x):
        f = self.body(x)                  # [B, feat]
        logits_a = self.head_coarse(f)    # [B, 2]
        logits_b = self.head_fine(f)      # [B, 2]
        return logits_a, logits_b

# ---------------------- Hierarchical Loss ----------------------
def hierarchical_loss_uncertainty(model, logits_a, logits_b, y3, W_A=None, W_B=None, reg_scale=REG_SCALE):
    """
    y3: 3-class label in {0: Non-Tumor, 1: Non-Viable-Tumor, 2: Viable}
    L = exp(-λA)*CE_A + exp(-λB)*CE_B + reg_scale*(λA + λB), with learnable λA, λB.
    W_A, W_B: optional per-head class weights (len=2 tensors) for imbalance correction.
    """
    # Head A target: 0 if Non-Tumor else 1 (Tumor)
    y_a = (y3 != 0).long()

    # Head B target (tumor-only): 0=Non-Viable, 1=Viable, -1 undefined otherwise
    y_b = torch.where(y3 == 1, 0, torch.where(y3 == 2, 1, -1)).long()
    mask_b = (y_b >= 0)

    loss_a = nn.functional.cross_entropy(logits_a, y_a, weight=W_A)
    if mask_b.any():
        loss_b = nn.functional.cross_entropy(logits_b[mask_b], y_b[mask_b], weight=W_B)
    else:
        loss_b = logits_a.new_tensor(0.0)

    inv_var_a = torch.exp(-model.log_sigma_a)   # exp(-λA)
    inv_var_b = torch.exp(-model.log_sigma_b)   # exp(-λB)

    loss = inv_var_a * loss_a + inv_var_b * loss_b + reg_scale * (model.log_sigma_a + model.log_sigma_b)
    return loss, loss_a.detach(), loss_b.detach()

# ---------------------- Prob fusion (2->3) ---------------------
def fuse_probs_to_three_classes(logits_a, logits_b):
    """
    logits_a: [B,2] Non-Tumor vs Tumor
    logits_b: [B,2] Non-Viable vs Viable (conditional on Tumor)
    returns: probs3 [B,3] in order [Non-Tumor, Non-Viable-Tumor, Viable]
    """
    pa = torch.softmax(logits_a, dim=1)  # [B,2]
    pb = torch.softmax(logits_b, dim=1)  # [B,2]

    p_non_tumor = pa[:, 0:1]
    p_tumor     = pa[:, 1:2]
    p_nvt = pb[:, 0:1]    # Non-Viable-Tumor
    p_vi  = pb[:, 1:2]    # Viable

    p_cls0 = p_non_tumor
    p_cls1 = p_tumor * p_nvt
    p_cls2 = p_tumor * p_vi
    probs3 = torch.cat([p_cls0, p_cls1, p_cls2], dim=1)  # [B,3]
    return probs3

# ---------------------- Train Once (Patient-level splits) -----
def train_model_once(seed=42):
    set_seed(seed)
    df, classes = load_df_from_csv(CSV_PATH, IMG_ROOTS)

    # ----- PATIENT-LEVEL SPLIT (no leakage) -----
    train_df, val_df, test_df_fixed = make_patient_splits(
        df,
        train_patients=TRAIN_PATIENTS,
        val_from_patients=VAL_FROM_PATIENT,
        test_patient=TEST_PATIENT,
        val_frac=VAL_FRACTION_FROM_VALPAT,
        classes=classes,
        rng_seed=seed,
    )

    print("\n[Training split sizes]")
    print(f"  TRAIN images: {len(train_df)}")
    print(f"  VAL images:   {len(val_df)}")
    print(f"  TEST images:  {len(test_df_fixed)} (fixed, held-out patient: {TEST_PATIENT})")

    loaders = {
        "train": DataLoader(TumorDataset(train_df, train=True), batch_size=16, shuffle=True,  num_workers=NUM_WORKERS, pin_memory=PIN_MEMORY),
        "val":   DataLoader(TumorDataset(val_df,   train=False), batch_size=16, shuffle=False, num_workers=NUM_WORKERS, pin_memory=PIN_MEMORY)
    }

    model = HierCNN(BACKBONE).to(DEVICE)

    # ---- per-head class weights from TRAIN SET (imbalance-aware) ----
    # Head A: Non-Tumor (0) vs Tumor (1)
    y_a_train = (train_df["y"].values != 0).astype(int)
    cnt_a = np.bincount(y_a_train, minlength=2).astype(float)
    w_a = 1.0 / (cnt_a / cnt_a.sum() + 1e-9)
    w_a = w_a * (2.0 / w_a.sum())   # normalize to mean ~1
    W_A = torch.tensor(w_a, dtype=torch.float32, device=DEVICE)

    # Head B: only tumor rows -> 0=Non-Viable, 1=Viable
    tumor_rows = train_df[train_df["y"] != 0]
    if len(tumor_rows) == 0:
        raise ValueError("Train set has no tumor rows; cannot train fine head.")
    y_b_train = np.where(tumor_rows["y"].values == 1, 0, 1)  # map 1→0 (Non-Viable), 2→1 (Viable)
    cnt_b = np.bincount(y_b_train, minlength=2).astype(float)
    w_b = 1.0 / (cnt_b / cnt_b.sum() + 1e-9)
    w_b = w_b * (2.0 / w_b.sum())
    W_B = torch.tensor(w_b, dtype=torch.float32, device=DEVICE)

    print(f"[Per-head class weights] W_A={w_a.tolist()}  W_B={w_b.tolist()}")

    # Param groups: moderate LR boost for log-σ (×8), no weight decay change
    base_lr = LR
    optimizer = optim.AdamW([
        {"params": model.body.parameters(),        "lr": base_lr},
        {"params": model.head_coarse.parameters(), "lr": base_lr},
        {"params": model.head_fine.parameters(),   "lr": base_lr},
        {"params": [model.log_sigma_a, model.log_sigma_b], "lr": base_lr * 8, "weight_decay": 0.0},
    ], lr=base_lr)

    # Print initial task weights
    wA0 = float(torch.exp(-model.log_sigma_a).detach().cpu())
    wB0 = float(torch.exp(-model.log_sigma_b).detach().cpu())
    print(f"Init learned task weights ~ wA={wA0:.3f}, wB={wB0:.3f}")

    best_state, best_f1 = None, -1
    wait = 0
    for epoch in range(1, MAX_EPOCHS + 1):
        # ---------------- Train ----------------
        model.train(); train_loss = 0.0; seen_train = 0
        for x, y in loaders["train"]:
            x, y = x.to(DEVICE), y.to(DEVICE)
            optimizer.zero_grad()
            logits_a, logits_b = model(x)
            loss, la, lb = hierarchical_loss_uncertainty(
                model, logits_a, logits_b, y,
                W_A=W_A, W_B=W_B, reg_scale=REG_SCALE
            )
            loss.backward(); optimizer.step()

            # Clamp log-sigmas to keep task weights in [W_MIN, W_MAX]
            with torch.no_grad():
                model.log_sigma_a.clamp_(LS_MIN, LS_MAX)
                model.log_sigma_b.clamp_(LS_MIN, LS_MAX)

            train_loss += float(loss.detach().cpu())
            seen_train += y.size(0)
        train_loss /= max(1, len(loaders["train"]))
        print(f"[Epoch {epoch:03d}] train images seen: {seen_train}/{len(train_df)}  | Train Loss: {train_loss:.3f}")

        # Peek at learned task weights every few epochs
        if epoch % 5 == 0:
            wA = float(torch.exp(-model.log_sigma_a).detach().cpu())
            wB = float(torch.exp(-model.log_sigma_b).detach().cpu())
            print(f"    Learned task weights ~ wA={wA:.3f}, wB={wB:.3f}")

        # ---------------- Val ----------------
        model.eval(); T, P, Q = [], [], []; seen_val = 0
        with torch.no_grad():
            for x, y in loaders["val"]:
                x, y = x.to(DEVICE), y.to(DEVICE)
                logits_a, logits_b = model(x)
                prob3 = fuse_probs_to_three_classes(logits_a, logits_b)
                T += y.cpu().tolist()
                P += prob3.argmax(1).cpu().tolist()
                Q += prob3.cpu().tolist()
                seen_val += y.size(0)
        print(f"[Epoch {epoch:03d}] val   images seen: {seen_val}/{len(val_df)}")

        f1w = f1_score(T, P, average="weighted")
        print(f"Epoch {epoch:03d} | Val F1w {f1w:.4f}")

        if epoch % 10 == 0 and len(Q) > 0:
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

    if best_state is None:
        best_state = {k: v.cpu() for k, v in model.state_dict().items()}
    model.load_state_dict({k: v.to(DEVICE) for k, v in best_state.items()})

    # Final learned task weights
    wA = float(torch.exp(-model.log_sigma_a).detach().cpu())
    wB = float(torch.exp(-model.log_sigma_b).detach().cpu())
    print(f"Final learned task weights ~ wA={wA:.3f}, wB={wB:.3f}")

    return model, df, classes, test_df_fixed

# ---------------------- Evaluate on fixed test -----------------
def evaluate_on_fixed_test(model, test_df: pd.DataFrame, classes: list):
    if test_df is None or len(test_df) == 0:
        print("\n[Evaluation] No test samples available for the held-out patient.")
        return None, None

    print(f"\n[Evaluation] TEST images: {len(test_df)} | patients: {sorted(test_df['patient'].unique())}")
    loader = DataLoader(
        TumorDataset(test_df, train=False),
        batch_size=16, shuffle=False,
        num_workers=NUM_WORKERS, pin_memory=PIN_MEMORY
    )

    y_true, y_pred, y_prob = [], [], []
    model.eval(); seen_test = 0
    with torch.no_grad():
        for x, y in loader:
            x, y = x.to(DEVICE), y.to(DEVICE)
            logits_a, logits_b = model(x)
            prob = fuse_probs_to_three_classes(logits_a, logits_b)
            y_true += y.cpu().tolist()
            y_pred += prob.argmax(1).cpu().tolist()
            y_prob += prob.cpu().tolist()
            seen_test += y.size(0)

    print(f"[Evaluation] test images seen: {seen_test}/{len(test_df)}")
    y_prob = np.array(y_prob)
    overall, perclass = compute_overall_and_perclass(y_true, y_pred, y_prob, classes)
    print_metrics_block("TEST (Held-out Patient)", overall, perclass)
    return overall, perclass

# ---------------------- Main ----------------------------------
if __name__ == "__main__":
    model, df, classes, test_df_fixed = train_model_once(seed=42)
    evaluate_on_fixed_test(model, test_df_fixed, classes)