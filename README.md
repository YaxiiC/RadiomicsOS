# RadiomicsOS

RadiomicsOS is a research prototype for multimodal osteosarcoma histology classification. The project couples convolutional features from an InceptionV3 backbone with handcrafted radiomics descriptors and optimizes them jointly through a hierarchical two-head loss. The training loop enforces patient-level data splits to avoid information leakage when working with the TCIA Osteosarcoma Tumor Assessment cohort.

## Key features

- **Multimodal learning** – Combines CNN image embeddings with radiomics feature vectors extracted through `torchradiomics` and `SimpleITK`.
- **Hierarchical uncertainty-weighted loss** – Learns separate logits for viable versus non-viable tissue while adapting per-head task weights during training.
- **Patient-level evaluation** – Supports leave-one-patient-out (LOPO) cross-validation and Monte Carlo dropout evaluation to better approximate clinical deployment scenarios.
- **Caching utilities** – Automatically caches radiomics features to disk in Parquet or CSV format to accelerate experimentation.

## Repository layout

```
├── README.md                     # Project overview and usage guide
└── inceptionv3_radiomics_2weights.py  # Training, evaluation, and radiomics utilities
```

All of the executable logic lives in `inceptionv3_radiomics_2weights.py`. The script bundles dataset preparation, model definition, training routines, and evaluation helpers.

## Requirements

- Python 3.10+
- CUDA-capable GPU (recommended). The script can run on CPU, albeit significantly slower.
- Python dependencies:
  - `torch` / `torchvision`
  - `timm`
  - `pandas`, `numpy`, `scikit-learn`
  - `Pillow`
  - `SimpleITK`
  - `torchradiomics`
  - Optional: `pyarrow` or `fastparquet` for faster cache serialization, `tqdm` for progress bars

Create and activate a virtual environment, then install dependencies for example via:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -U pip
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121  # adjust for your CUDA
pip install timm pandas numpy scikit-learn pillow SimpleITK torchradiomics pyarrow tqdm
```

> **Note:** Depending on your platform, installing `torchradiomics` might require system-level libraries. Consult the [TorchRadiomics documentation](https://github.com/radiomics-developers/torchradiomics) if you encounter build errors.

## Data preparation

The script expects:

- A CSV file containing tile-level metadata and radiomics descriptors (`ML_Features_1144.csv` in the original study).
- Root directories containing the corresponding histology image tiles (e.g., `Training-Set-1`, `Training-Set-2`).

Update the following constants near the top of `inceptionv3_radiomics_2weights.py` to point to your local copies:

```python
CSV_PATH  = pathlib.Path("/path/to/ML_Features_1144.csv")
IMG_ROOTS = [
    pathlib.Path("/path/to/Training-Set-1"),
    pathlib.Path("/path/to/Training-Set-2")
]
RAD_CACHE_DIR = pathlib.Path("./rad_cache")
```

The cache directory will be created automatically and populated with Parquet/CSV artifacts containing extracted radiomics features. Delete the folder if you need to regenerate them.

## Training the model

The default configuration performs a single training run with a fixed train/validation/test split defined by patient identifiers in the `TRAIN_PATIENTS`, `VAL_PATIENTS`, and `TEST_PATIENT` constants. Launch training with:

```bash
python inceptionv3_radiomics_2weights.py
```

Key training hyperparameters are exposed as module-level constants (`MAX_EPOCHS`, `LR`, augmentation transforms, etc.) for quick experimentation. Checkpoint files are written under `./checkpoints/` (the best patient-split model is saved as `best_patient_split.pt`).

### Leave-one-patient-out cross-validation

To run the LOPO evaluation routine, import the module and call `train_lopo_cv`:

```python
from inceptionv3_radiomics_2weights import train_lopo_cv

results_df = train_lopo_cv(seed=2025)
print(results_df)
```

Each fold trains a fresh model, evaluates it with five Monte Carlo dropout passes, and aggregates the metrics.

## Evaluation

After training, `evaluate_model_5x` performs Monte Carlo dropout evaluation over the held-out patient:

```python
from inceptionv3_radiomics_2weights import train_model_once, evaluate_model_5x, MC_EVAL_SEEDS

model, artifacts = train_model_once(seed=2025)
metrics_df, per_case_df = evaluate_model_5x(model, artifacts, seeds_eval=MC_EVAL_SEEDS)
print(metrics_df.mean())
```

Evaluation metrics include accuracy, macro F1, ROC-AUC, confusion matrices, and precision/recall per class. Outputs are returned as Pandas DataFrames for convenient analysis and visualization.

## Tips for reproducibility

- Set the `seed` argument when calling `train_model_once` or `train_lopo_cv` to make data splits and weight initialization deterministic.
- Adjust the `MC_EVAL_SEEDS` list to control how many stochastic evaluation passes are executed.
- GPU determinism is enforced by disabling cuDNN benchmarking within `set_seed`, but results may still vary slightly across hardware.

## Troubleshooting

- **Missing radiomics features:** ensure the CSV includes the column names referenced inside the dataset loader. The script prints helpful error messages when required columns are absent.
- **Shape extractor availability:** the code attempts to import `TorchRadiomicsShape2D` and falls back to `TorchRadiomicsShape`. If neither is available, shape features are skipped automatically.
- **Slow preprocessing:** radiomics extraction can be time-consuming. Use the `RAD_MAX_IMAGES` debug flag to limit the number of processed tiles during dry runs.

## Citation

If you build upon this repository, please cite the accompanying work or acknowledge the RadiomicsOS project in your publications.
