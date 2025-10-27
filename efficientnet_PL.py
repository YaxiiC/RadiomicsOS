# === EfficientNet-B0 3-Class Classifier: Patient-Level Split (Train: Case-3/P9/Case-48, Val: Case-48 slice, Test: Case-4) ===
import os, math, random, pathlib, re
import numpy as np, pandas as pd
import torch, torch.nn as nn, torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image

from sklearn.metrics import (
    accuracy_score, f1_score, roc_auc_score,
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

BACKBONE   = "efficientnet_b0"   # <<< changed backbone
DEVICE     = torch.device("cuda" if torch.cuda.is_available() else "cpu")
IMG_SIZE   = 224                  # <<< native EfficientNet-B0 input size
MAX_EPOCHS = 80
MIN_EPOCHS = 20
PATIENCE   = 40
WARMUP_EPOCHS = 10
LR = 1e-4
NUM_WORKERS = 4
PIN_MEMORY  = DEVICE.type == "cuda"

# --- Patient split config ---
VAL_FRACTION_FROM_VALPAT = 0.12                 # small validation slice fraction
TRAIN_PATIENTS = ["Case-4", "P9", "Case-48"]    # Train set patients
VAL_FROM_PATIENT = "Case-48"                    # Take val slice from Case-48
TEST_PATIENT = "Case-3"                         # Hold out all of Case-4 for test

# --- Regularization / Dropout config --------------------------
DROP_RATE = 0.30            # standard dropout inside timm head
DROP_PATH_RATE = 0.10       # stochastic depth (supported by EfficientNet-B0 in timm)
MC_DROPOUT_EVAL_SEEDS = [1,2,3,4,5]  # 5 stochastic evaluation passes

# ----------------------- Reproducibility -----------------------
def set_seed(seed: int):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

# ---------------------- Transforms -----------------------------
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

# ---------------------- Load CSV -------------------------------
def load_df_from_csv(csv_path: pathlib.Path, roots):
    df = pd.read_csv(csv_path)
    df["image.name"] = df["image.name"].astype(str)

    def clean_label(s: str) -> str:
        s_low = str(s).strip().lower().replace("_", "-").replace(" ", "-")
        if "non" in s_low and "viable" in s_low: return "Non-Viable-Tumor"
        if "non-tumor" in s_low or "nontumor" in s_low: return "Non-Tumor"
        return "Viable"

    # --- canonical key that tolerates separators/extension mismatches ---
    def canonical_key(s: str):
        stem = pathlib.Path(str(s)).stem.lower()
        nums = re.findall(r'\d+', stem)
        if len(nums) < 4:
            return None
        return f"case{nums[0]}a{nums[1]}{nums[2]}{nums[3]}"

    # --- index files on disk by canonical key ---
    exts = {".jpg", ".jpeg", ".png", ".tif", ".tiff", ".bmp"}
    by_key = {}
    dups = set()

    for root in roots:
        for p in root.rglob("*"):
            if p.is_file() and p.suffix.lower() in exts:
                key = canonical_key(p.name)
                if key is None:
                    continue
                if key in by_key:
                    dups.add(key)
                else:
                    by_key[key] = p

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
    if dups:
        print(f"  Note: {len(dups)} duplicate basenames detected on disk (kept first occurrence).")
    print(f"  Per-patient counts:           {df['patient'].value_counts().to_dict()}")

    expected = {"Case-3", "Case-4", "Case-48", "P9"}
    seen = set(df["patient"].unique())
    print("\n[Dataset patients] seen:", sorted(seen))
    missing = expected - seen
    if missing:
        print("[Warn] Expected patients missing from dataset:", sorted(missing))

    return df, classes

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
        y_true, y_pred, labels=list(range(len(classes))), average=None, zero_division=0
    )
    cm = confusion_matrix(y_true, y_pred, labels=list(range(len(classes))))
    specs = []
    for i in range(len(classes)):
        TP = cm[i, i]; FP = cm[:, i].sum() - TP; FN = cm[i, :].sum() - TP; TN = cm.sum() - TP - FP - FN
        specs.append(TN / (TN + FP) if (TN + FP) > 0 else float("nan"))

    try:
        Y_bin = label_binarize(y_true, classes=list(range(len(classes))))
        auc_class = [roc_auc_score(Y_bin[:, i], y_prob[:, i]) if Y_bin[:, i].min() != Y_bin[:, i].max() else float("nan") for i in range(len(classes))]
    except:
        auc_class = [float("nan")] * len(classes)

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

# ---------------------- Model ---------------------------------
class CNNClassifier(nn.Module):
    def __init__(self, backbone: str, num_classes: int):
        super().__init__()
        kwargs = dict(pretrained=True, num_classes=num_classes, global_pool="avg")
        # add EfficientNet-friendly regularization
        if "inception" not in backbone.lower():
            kwargs.update(dict(drop_path_rate=DROP_PATH_RATE))
        kwargs.update(dict(drop_rate=DROP_RATE))
        self.model = timm.create_model(backbone, **kwargs)
    def forward(self, x): return self.model(x)

# --- Utility: enable ONLY dropout layers during eval (MC Dropout)
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
        if isinstance(m, (nn.Dropout, nn.Dropout2d, nn.Dropout1d, nn.AlphaDropout)):
            m.train()

# ---------------------- Stratified VAL helper ------------------
def stratified_val_from_patient(df_patient: pd.DataFrame, val_fraction: float, seed: int):
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

# ---------------------- Train Once -----------------------------
def train_model_once(seed=42):
    set_seed(seed)
    df, classes = load_df_from_csv(CSV_PATH, IMG_ROOTS)
    num_classes = len(classes)

    # ----- PATIENT-LEVEL SPLIT -----
    train_pool = df[df["patient"].isin(TRAIN_PATIENTS)].copy()
    test_df   = df[df["patient"] == TEST_PATIENT].copy()

    print("\n[Sanity] Patients in TRAIN pool:", sorted(train_pool["patient"].unique().tolist()))
    print("[Sanity] Patients in TEST set:", sorted(test_df["patient"].unique().tolist()))
    assert TEST_PATIENT not in set(train_pool["patient"]), "Leak: test patient appears in TRAIN pool!"
    assert set(test_df["patient"]) == {TEST_PATIENT}, "TEST set contains non-test patients!"

    # VAL slice from Case-48
    val_source = train_pool[train_pool["patient"] == VAL_FROM_PATIENT].copy()
    val_df, val_source_remainder = stratified_val_from_patient(val_source, VAL_FRACTION_FROM_VALPAT, seed)

    train_df = pd.concat([
        train_pool[train_pool["patient"] != VAL_FROM_PATIENT],
        val_source_remainder
    ], axis=0).sort_index()

    print("\n[Training split sizes — Patient-level]")
    print(f"  TRAIN images (Case-3/P9/Case-48 minus Case-48-val): {len(train_df)}")
    print(f"  VAL images (from {VAL_FROM_PATIENT}):               {len(val_df)}")
    print(f"  TEST images (all {TEST_PATIENT}):                  {len(test_df)}")
    print("[Sanity] Per-class counts (TRAIN):", train_df["label"].value_counts().to_dict())
    print("[Sanity] Per-class counts (VAL):  ", val_df["label"].value_counts().to_dict())
    print("[Sanity] Per-class counts (TEST): ", test_df["label"].value_counts().to_dict())
    print("[Peek] TRAIN sample paths:", train_df["path"].head(3).tolist())
    print("[Peek] VAL sample paths:", val_df["path"].head(3).tolist())
    print("[Peek] TEST sample paths:", test_df["path"].head(3).tolist())

    loaders = {
        "train": DataLoader(TumorDataset(train_df, train=True), batch_size=16, shuffle=True,  num_workers=NUM_WORKERS, pin_memory=PIN_MEMORY),
        "val":   DataLoader(TumorDataset(val_df,   train=False), batch_size=16, shuffle=False, num_workers=NUM_WORKERS, pin_memory=PIN_MEMORY)
    }

    model = CNNClassifier(BACKBONE, num_classes).to(DEVICE)
    counts = np.bincount(train_df["y"], minlength=num_classes).astype(float)
    weights = (1.0 / (counts / counts.sum() + 1e-9))
    weights = weights * (num_classes / weights.sum())
    class_weights = torch.tensor(weights, dtype=torch.float32).to(DEVICE)

    criterion = nn.CrossEntropyLoss(weight=class_weights)
    optimizer = optim.AdamW(model.parameters(), lr=LR)

    best_state, best_f1 = None, -1
    wait = 0
    for epoch in range(1, MAX_EPOCHS + 1):
        model.train(); train_loss = 0.0; seen_train = 0
        for x, y in loaders["train"]:
            x, y = x.to(DEVICE), y.to(DEVICE)
            optimizer.zero_grad()
            out = model(x)
            loss = criterion(out, y)
            loss.backward(); optimizer.step()
            train_loss += loss.item()
            seen_train += y.size(0)
        train_loss /= max(1, len(loaders["train"]))
        print(f"[Epoch {epoch:03d}] train batches covered images: {seen_train}/{len(train_df)}")

        model.eval(); T, P, Q = [], [], []; seen_val = 0
        with torch.no_grad():
            for x, y in loaders["val"]:
                x, y = x.to(DEVICE), y.to(DEVICE)
                out = model(x); prob = torch.softmax(out, 1)
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
    return model, df, classes, test_df

# ---------------------- Evaluate 5x with MC Dropout -----------
def evaluate_model_5x(model, test_df, classes, seeds_eval=MC_DROPOUT_EVAL_SEEDS):
    print("\n[Sanity] Test patients:", sorted(test_df["patient"].unique().tolist()))
    assert set(test_df["patient"]) == {TEST_PATIENT}, "Leak: TEST contains non-test patients!"
    print("[Sanity] Per-class counts (TEST):", test_df["label"].value_counts().to_dict())
    print("[Peek] TEST sample paths:", test_df["path"].head(3).tolist())
    print(f"\n[Evaluation] TEST images (all {TEST_PATIENT}): {len(test_df)}")

    results = []
    perclass_all = []

    # Fixed test loader
    base_loader = DataLoader(
        TumorDataset(test_df, train=False),
        batch_size=16, shuffle=False,
        num_workers=NUM_WORKERS, pin_memory=PIN_MEMORY
    )

    for s in seeds_eval:
        set_seed(s)
        set_mc_dropout(model, enable=True)  # turn on MC dropout

        y_true, y_pred, y_prob = [], [], []
        seen_test = 0
        with torch.no_grad():
            for x, y in base_loader:
                x, y = x.to(DEVICE), y.to(DEVICE)
                out = model(x)  # dropout active
                prob = torch.softmax(out, 1)
                y_true += y.cpu().tolist()
                y_pred += prob.argmax(1).cpu().tolist()
                y_prob += prob.cpu().tolist()
                seen_test += y.size(0)

        print(f"[Evaluation seed={s} (MC dropout)] test batches covered images: {seen_test}/{len(test_df)}")

        y_prob = np.array(y_prob)
        overall, perclass = compute_overall_and_perclass(y_true, y_pred, y_prob, classes)
        print_metrics_block(f"TEST EVALUATION (seed={s}, MC-Dropout, patient={TEST_PATIENT})", overall, perclass)

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
    metrics = ["precision", "recall", "specificity", "f1", "auc"]
    rows = []
    for i, cls in enumerate(classes):
        agg = {m: [perclass_all[k][i][m] for k in range(len(seeds_eval))] for m in metrics}
        rows.append({
            "Class": cls,
            **{f"{m}_mean": float(np.mean(agg[m])) for m in metrics},
            **{f"{m}_std":  float(np.std(agg[m], ddof=1)) for m in metrics},
        })

    dfpc = pd.DataFrame(rows)
    hdr = f"{'Class':<22} {'Prec':>10} {'±':>3} {'Rec':>10} {'±':>3} {'Spec':>10} {'±':>3} {'F1':>10} {'±':>3} {'AUC':>10} {'±':>3}"
    print(hdr)
    print("-" * len(hdr))
    for _, r in dfpc.iterrows():
        print(
            f"{r['Class']:<22} "
            f"{r['precision_mean']:>10.3f} ±{r['precision_std']:<5.3f} "
            f"{r['recall_mean']:>10.3f} ±{r['recall_std']:<5.3f} "
            f"{r['specificity_mean']:>10.3f} ±{r['specificity_std']:<5.3f} "
            f"{r['f1_mean']:>10.3f} ±{r['f1_std']:<5.3f} "
            f"{r['auc_mean']:>10.3f} ±{r['auc_std']:<5.3f}"
        )

    return dfres, dfpc

# ---------------------- Main ----------------------------------
if __name__ == "__main__":
    model, df, classes, test_df = train_model_once(seed=42)  # train ONCE
    evaluate_model_5x(model, test_df, classes, seeds_eval=MC_DROPOUT_EVAL_SEEDS)  # 5 stochastic passes