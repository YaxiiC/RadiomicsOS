"""Data loading, preprocessing, and radiomics utilities for the Osteosarcoma project."""

from __future__ import annotations

import json
import os
import re
import time
from pathlib import Path
from typing import Iterable, Sequence

import numpy as np
import pandas as pd
from PIL import Image

import torch
from torch.utils.data import Dataset
from torchvision import transforms

import SimpleITK as sitk
from torchradiomics import TorchRadiomicsFirstOrder, inject_torch_radiomics

try:  # Optional/robust import of shape extractors across versions
    from torchradiomics import TorchRadiomicsShape2D as _Shape2DClass
    _SHAPE_MODE = "shape2d"
except Exception:  # pragma: no cover - fallback path
    try:
        from torchradiomics import TorchRadiomicsShape as _Shape2DClass
        _SHAPE_MODE = "shape_fallback"
    except Exception:  # pragma: no cover - shape extractor unavailable
        _Shape2DClass = None
        _SHAPE_MODE = None


BASE_DATA_DIR = Path("./data")
CSV_PATH = BASE_DATA_DIR / "ML_Features_1144.csv"
IMG_ROOTS: list[Path] = [
    BASE_DATA_DIR / "Training-Set-1",
    BASE_DATA_DIR / "Training-Set-2",
]
RAD_CACHE_DIR = Path("./rad_cache")
RAD_CACHE_DIR.mkdir(parents=True, exist_ok=True)

IMG_SIZE = 299
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def _parquet_available() -> bool:
    try:  # pragma: no cover - optional dependency probe
        import pyarrow  # type: ignore # noqa: F401
        return True
    except Exception:  # pragma: no cover - optional dependency probe
        try:
            import fastparquet  # type: ignore # noqa: F401
            return True
        except Exception:
            return False


_PARQUET_OK = _parquet_available()


def _read_cache(df_path_base: Path) -> pd.DataFrame | None:
    pqt = df_path_base.with_suffix(".parquet")
    csv = df_path_base.with_suffix(".csv")
    if pqt.exists() and _PARQUET_OK:
        return pd.read_parquet(pqt)
    if csv.exists():
        return pd.read_csv(csv)
    return None


def _write_cache(
    df: pd.DataFrame,
    df_path_base: Path,
    *,
    autosave: bool = False,
    i: int | None = None,
) -> Path:
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


def _read_feat_names(path_base: Path) -> list[str] | None:
    names_json = path_base.with_suffix(".json")
    return json.loads(names_json.read_text()) if names_json.exists() else None


def _write_feat_names(names: Sequence[str], path_base: Path) -> Path:
    names_json = path_base.with_suffix(".json")
    names_json.write_text(json.dumps(list(names)))
    return names_json


def extract_all_radiomics(
    x: torch.Tensor,
    *,
    voxelArrayShift: float = 0.0,
    pixelSpacing: Sequence[float] = (1.0, 1.0, 1.0),
    binWidth: float | None = None,
) -> tuple[dict[str, torch.Tensor], list[str]]:
    img_np = x.to(dtype=torch.float64, non_blocking=False).cpu().numpy()
    mask_np = (x > 0).to(dtype=torch.uint8, non_blocking=False).cpu().numpy()

    sitk_img = sitk.GetImageFromArray(img_np)
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
    else:  # pragma: no cover - unsupported dimension guard
        raise ValueError(f"Unsupported image dimension: {dim}")

    inject_torch_radiomics()

    base_compute = dict(
        voxelBased=False,
        padDistance=1,
        kernelRadius=1,
        maskedKernel=False,
        voxelBatch=512,
        dtype=torch.float64,
        device=x.device,
    )
    base_settings = dict(voxelArrayShift=voxelArrayShift)
    if binWidth is not None:
        base_settings["binWidth"] = float(binWidth)

    extractors = [
        TorchRadiomicsFirstOrder(sitk_img, sitk_mask, **base_settings, **base_compute)
    ]
    if _Shape2DClass is not None:
        if _SHAPE_MODE == "shape2d":
            extractors.append(
                _Shape2DClass(sitk_img, sitk_mask, **base_settings, **base_compute)
            )
        elif _SHAPE_MODE == "shape_fallback":
            extractors.append(
                _Shape2DClass(
                    sitk_img,
                    sitk_mask,
                    force2D=True,
                    force2DDimension=0,
                    **base_settings,
                    **base_compute,
                )
            )

    features: dict[str, torch.Tensor] = {}
    names: list[str] = []
    for ext in extractors:
        out = ext.execute()
        for k, v in out.items():
            if isinstance(v, sitk.Image):
                continue
            tv = (
                v
                if isinstance(v, torch.Tensor)
                else torch.as_tensor(v, dtype=torch.float64, device=x.device)
            )
            if torch.isfinite(tv).all():
                features[k] = tv
                names.append(k)
    return features, names


def build_or_load_radiomics_cache(
    df: pd.DataFrame,
    cache_dir: Path,
    *,
    pixelSpacing: Sequence[float] = (1.0, 1.0, 1.0),
    voxelArrayShift: float = 0.0,
    binWidth: float = 5.0,
    device: torch.device = DEVICE,
    max_images: int | None = None,
    debug: bool = False,
) -> tuple[pd.DataFrame, list[str], dict[str, Path | pd.DataFrame]]:
    cache_dir.mkdir(parents=True, exist_ok=True)
    base = cache_dir / "radiomics_cache"

    cached = _read_cache(base)
    feat_names = _read_feat_names(cache_dir / "radiomics_feat_names")
    if cached is not None and feat_names is not None:
        kind = (
            "Parquet"
            if _PARQUET_OK and (cache_dir / "radiomics_cache.parquet").exists()
            else "CSV"
        )
        print(f"[Radiomics] Using existing {kind} cache: {cache_dir}")
        return cached, feat_names, {
            "table": cached,
            "names": cache_dir / "radiomics_feat_names.json",
        }

    rows: list[dict[str, float | str]] = []
    feat_names = None
    paths = df["path"].astype(str).unique().tolist()
    if max_images is not None:
        paths = paths[:max_images]
    total = len(paths)
    store_kind = "Parquet" if _PARQUET_OK else "CSV"
    print(
        f"[Radiomics] Computing FO+Shape2D for {total} images... (store={store_kind})"
    )

    try:
        from tqdm.auto import tqdm  # type: ignore
    except Exception:  # pragma: no cover - tqdm optional
        tqdm = None  # type: ignore

    pbar = tqdm(total=total, desc="Radiomics(FO+S2D)", unit="img") if tqdm else None
    t0 = time.time()

    try:
        for i, p in enumerate(paths, 1):
            try:
                pil_img = Image.open(p).convert("L")
                arr = np.array(pil_img, dtype=np.float32)
                x = torch.from_numpy(arr).to(device=device, dtype=torch.float32)

                if debug:
                    print(
                        f"[{i:04d}/{total}] {p} | shape={tuple(arr.shape)} "
                        f"| min={float(arr.min()):.1f} max={float(arr.max()):.1f}"
                    )

                feats_dict, feat_names_i = extract_all_radiomics(
                    x,
                    voxelArrayShift=voxelArrayShift,
                    pixelSpacing=pixelSpacing,
                    binWidth=binWidth,
                )
                if feat_names is None:
                    feat_names = feat_names_i
                    if debug:
                        print(
                            f"[Feature names] {len(feat_names)} FO+Shape2D features"
                        )

                row: dict[str, float | str] = {"path": p}
                for k in feat_names:
                    v = feats_dict[k]
                    if isinstance(v, torch.Tensor):
                        v = v.detach().to("cpu").item()
                    row[k] = float(v)
                rows.append(row)

            except Exception as exc:  # pragma: no cover - logging path
                msg = f"[Radiomics][WARN] {p}: {exc}"
                if pbar:
                    pbar.write(msg)
                else:
                    print(msg)

            if pbar:
                pbar.update(1)
                if i % 25 == 0 or i == total:
                    elapsed = time.time() - t0
                    ips = i / max(elapsed, 1e-9)
                    eta_s = (total - i) / max(ips, 1e-9)
                    pbar.set_postfix_str(f"{ips:.2f} img/s | ETA {eta_s/60:.1f} min")
            elif i % 50 == 0 or i == total:
                elapsed = time.time() - t0
                print(f"[Radiomics] {i}/{total} done ({elapsed:.1f}s)")

            if i % 100 == 0:
                tmp_df = pd.DataFrame(rows)
                out = _write_cache(
                    tmp_df, cache_dir / "radiomics_cache", autosave=True, i=i
                )
                if pbar:
                    pbar.write(f"[Autosave] {out}")
                else:
                    print(f"[Autosave] {out}")

    finally:
        if pbar:
            pbar.close()

    if not rows:
        raise RuntimeError("No radiomics rows computed — check inputs.")

    rad_df = pd.DataFrame(rows)
    out_main = _write_cache(rad_df, cache_dir / "radiomics_cache", autosave=False)
    names_path = _write_feat_names(feat_names or [], cache_dir / "radiomics_feat_names")
    print(f"[Radiomics] Saved cache → {out_main}")
    return rad_df, feat_names or [], {"table": out_main, "names": names_path}


transform_train = transforms.Compose(
    [
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.RandomResizedCrop(IMG_SIZE, scale=(0.8, 1.0)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomVerticalFlip(),
        transforms.RandomRotation(15),
        transforms.ColorJitter(
            brightness=0.2, contrast=0.2, saturation=0.15, hue=0.02
        ),
        transforms.RandomAffine(degrees=0, translate=(0.05, 0.05), shear=5),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ]
)

transform_eval = transforms.Compose(
    [
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
        ),
    ]
)


def load_df_from_csv(csv_path: Path, roots: Iterable[Path]) -> tuple[pd.DataFrame, list[str]]:
    df = pd.read_csv(csv_path)
    df["image.name"] = df["image.name"].astype(str)

    def clean_label(s: str) -> str:
        s_low = str(s).strip().lower().replace("_", "-").replace(" ", "-")
        if "non" in s_low and "viable" in s_low:
            return "Non-Viable-Tumor"
        if "non-tumor" in s_low or "nontumor" in s_low:
            return "Non-Tumor"
        return "Viable"

    def canonical_key(s: str) -> str | None:
        stem = Path(str(s)).stem.lower()
        nums = re.findall(r"\d+", stem)
        if len(nums) < 4:
            return None
        return f"case{nums[0]}a{nums[1]}{nums[2]}{nums[3]}"

    exts = {".jpg", ".jpeg", ".png", ".tif", ".tiff", ".bmp"}
    by_key: dict[str, Path] = {}
    dups: set[str] = set()
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

    def _extract_patient_id_from_string(s: str) -> str:
        s = str(s).lower().replace("_", "-").replace(" ", "-")
        if re.search(r"\bcase-?3\b", s):
            return "Case-3"
        if re.search(r"\bcase-?4\b", s):
            return "Case-4"
        if re.search(r"\bcase-?48\b", s):
            return "Case-48"
        if re.search(r"\bp-?9\b", s):
            return "P9"
        return "Unknown"

    def assign_patient_id(row: pd.Series) -> str:
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
        print(
            f"  Note: {len(dups)} duplicate basenames detected on disk (kept first occurrence)."
        )
    expected = {"Case-3", "Case-4", "Case-48", "P9"}
    seen = set(df["patient"].unique())
    print("\n[Dataset patients] seen:", sorted(seen))
    missing = expected - seen
    if missing:
        print("[Warn] Expected patients missing from dataset:", sorted(missing))
    return df, classes


class TumorDataset(Dataset):
    def __init__(
        self,
        df: pd.DataFrame,
        rad_df: pd.DataFrame,
        feat_names: Sequence[str],
        rad_stats: dict[str, np.ndarray],
        *,
        train: bool = False,
    ) -> None:
        self.df = df.reset_index(drop=True)
        self.tf = transform_train if train else transform_eval
        self.feat_names = list(feat_names)
        self.mean = rad_stats["mean"].astype(np.float32)
        self.std = rad_stats["std"].astype(np.float32)
        self.rad_lookup = rad_df.set_index("path")[self.feat_names]

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, idx: int):
        row = self.df.iloc[idx]
        path = row["path"]
        if not os.path.exists(path):
            raise FileNotFoundError(f"Image path does not exist: {path}")
        img = Image.open(path).convert("RGB")
        x_img = self.tf(img)

        r = self.rad_lookup.loc[path].to_numpy(dtype=np.float32)
        r = r - self.mean
        denom = np.where(self.std > 1e-8, self.std, 1.0)
        r = r / denom
        x_rad = torch.from_numpy(r)

        return x_img, x_rad, int(row["y"])


__all__ = [
    "BASE_DATA_DIR",
    "CSV_PATH",
    "IMG_ROOTS",
    "RAD_CACHE_DIR",
    "IMG_SIZE",
    "DEVICE",
    "transform_train",
    "transform_eval",
    "build_or_load_radiomics_cache",
    "extract_all_radiomics",
    "load_df_from_csv",
    "TumorDataset",
]
