# ECE1512 Project A - PyTorch MIL on WSI features

This is the **starter package** for Project A. The project uses Attention-Based Multiple Instance Learning (ABMIL) for whole-slide histopathology classification. After completing Project A, you should understand the standard PyTorch supervised-learning pipeline and be able to construct a basic PyTorch training pipeline on your own.

## 1. Download the pre-extracted features

The pre-extracted histopathology features are hosted on Zenodo:

https://zenodo.org/records/23130543


Download the HDF5 feature files and place them in the designated `feature` folder.

```text
feature/
  C16/patch_feats_pretrain_medical_ssl.h5
  C17/patch_feats_pretrain_medical_ssl.h5
  BRACS/patch_feats_pretrain_medical_ssl.h5
```

If you store the HDF5 feature files elsewhere, edit only `data_dir` in the selected YAML file.

## 2. Coding

Open:

```text
ECE1512_ProjectA_Student.ipynb
```

The notebook is incomplete. Search for **TODO** and complete the requested lines.

## 3. What is provided

- `dataloader.py` - HDF5 reading, dataset-specific splitting, few-shot selection, and padded MIL collation.
- `utils.py` - configuration loading, seeding, feature-dimension checks, and AUROC/F1 metric helpers.
- `config/` - templates for CAMELYON16, CAMELYON17, and BRACS.
- `dataset_csv/` - metadata required by the CAMELYON17 and BRACS split logic.
- `ECE1512_ProjectA_Student.ipynb` - guided PyTorch implementation, baseline experiment, and ablation workspace.

The completed ABMIL model and training loop are deliberately **not** included in the student starter package.

## 4. Expected HDF5 layout

Each top-level HDF5 key is one slide:

```text
slide_id/
  feat    [num_patches, feature_dim]
  coords  [num_patches, 2]   # optional for training
attrs:
  label   integer class label
```

Different slides contain different numbers of patches. `mil_collate_fn` pads a batch and returns a boolean `mask` that marks valid patches.

## 5. Colab/local setup

Install dependencies with:

```bash
pip install -r requirements.txt
```

In Colab, place the project folder in Google Drive or upload it to the runtime, set `PROJECT_DIR` in the notebook, and select `C16`, `C17`, or `BRACS`.

For an initial debugging run, use only a few epochs. Once your pipeline passes the sanity checks, run the full baseline and keep the same split, seed, and evaluation protocol for your ablations.
