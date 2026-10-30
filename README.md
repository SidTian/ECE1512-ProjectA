# ECE1512 Project A

Slide-level classification of whole-slide histopathology features with Attention-Based Multiple Instance Learning (ABMIL). The research extension is TransMIL, compared with two training-only augmentations: random patch subsampling and Gaussian feature noise.

Features and trained weights are not in this repository. Download the HDF5 files before training.

## Setup

```bash
pip install -r requirements.txt
```

Download the pre-extracted features from https://zenodo.org/records/23130543 and place them here:

```text
feature/
  C16/patch_feats_pretrain_medical_ssl.h5
  C17/patch_feats_pretrain_medical_ssl.h5
  BRACS/patch_feats_pretrain_medical_ssl.h5
```

Each slide in the HDF5 file has `feat` with shape `[num_patches, 384]`, optional `coords` with shape `[num_patches, 2]`, and an integer `label` attribute. Slides have different numbers of patches. `mil_collate_fn` pads a batch and returns a mask.

Dataset choices are `C16` (CAMELYON16, 2 classes), `C17` (CAMELYON17, 4 classes), and `BRACS` (7 classes). Splits, learning rate, and epoch count come from `config/`.

## ABMIL baseline

Open `ECE1512_ProjectA_Student.ipynb` and set `DATASET` to `C16`, `C17`, or `BRACS`. The notebook builds the model, trains it, and evaluates the test set after selecting the best validation checkpoint.

## Experiments

Run these from the project directory. Validation and test slides are not augmented.

```bash
python train_patch_subsample.py --dataset BRACS --max-patches 2048
python train_gaussian_noise.py --dataset BRACS --noise-std 0.2
python train_transmil.py --dataset BRACS --max-patches 4096
```

Replace `BRACS` with `C16` or `C17` to run the other datasets.

- `train_patch_subsample.py` keeps the ABMIL model and randomly keeps at most 2048 patches from each training slide.
- `train_gaussian_noise.py` keeps the ABMIL model and adds Gaussian noise with standard deviation 0.2 to training patch features. Feature values have a standard deviation of about 2.2, so this is a mild perturbation.
- `train_transmil.py` trains TransMIL. Slides longer than 4096 patches are subsampled, then sorted by coordinates and arranged on a square grid. The model uses Nyström attention and pyramid position encoding. The reported run uses cross-entropy.

Metrics and curves are written to `results/`. Checkpoint files are gitignored.
