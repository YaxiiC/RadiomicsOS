# RadiomicsOS

RadiomicsOS is a research prototype for multimodal osteosarcoma histology classification. The project couples convolutional features from an InceptionV3 backbone with handcrafted radiomics descriptors and optimizes them jointly through a learnable hierarchical two-head loss. 

## Key features

- **Multimodal learning** – Combines CNN image embeddings with radiomics feature vectors extracted through `torchradiomics`.
- **Hierarchical uncertainty-weighted loss** – Learns separate logits for viable versus non-viable tissue while adapting per-head task weights during training.


## Repository layout

```
├── README.md                     
└── inceptionv3_radiomics_2weights.py  
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



## Citation

If you build upon this repository, please cite the accompanying work or acknowledge the RadiomicsOS project in your publications.
