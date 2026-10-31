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

Run these from the project directory. ABMIL noise/subsampling variants perturb only training inputs; TransMIL also caps validation/test bags deterministically at 4096 patches.

```bash
python train_patch_subsample.py --dataset BRACS --max-patches 2048
python train_gaussian_noise.py --dataset BRACS --noise-std 0.2
python train_transmil.py --dataset BRACS --max-patches 4096
```

Replace `BRACS` with `C16` or `C17` to run the other datasets.

- `train_patch_subsample.py` keeps the ABMIL model and randomly keeps at most 2048 patches from each training slide.
- `train_gaussian_noise.py` keeps the ABMIL model and adds Gaussian noise with standard deviation 0.2 to training patch features. Feature values have a standard deviation of about 2.2, so this is a mild perturbation.
- `train_transmil.py` trains TransMIL. Slides longer than 4096 patches are subsampled, then sorted by coordinates and arranged on a square grid. The model uses Nyström attention and pyramid position encoding. The reported run uses cross-entropy.

Training scripts write metric JSON files to results/; the notebook plots saved results. Checkpoint files are gitignored.

## Report and code reproducibility checklist

The report and this repository are separate artifacts. The report describes methods, proposals, comparisons, and limitations; this checklist describes code and saved evidence.

### Data and environment

- [ ] User setup: download features from the source above; features and weights are not tracked. This audit did not download or reload them.
- [x] C16: test IDs contain 'test'; reserve 10% of non-test slides for validation with seed 42 and stratification when feasible. Reported counts: 242 train / 27 validation / 129 test.
- [x] C17: centers 3-4 are test; centers 0-2 are split 90/10 for training/validation with seed 42. Counts: 269 / 30 / 200. This experimental split of available challenge training slides is not the official challenge test partition. Training/validation are split by slide, not grouped by patient.
- [x] BRACS: preserve CSV assignments for available features; counts: 389 / 62 / 86.
- [x] Common settings: frozen 384-dimensional inputs, batch size 1, AdamW, learning rate 1e-4, weight decay 1e-5, 30 epochs, seed 42. Current loops do not implement the YAML warmup field. Baseline classifier dropout is 0; TransMIL attention-output dropout is 0.1.
- [x] Saved baseline notebook environment: PyTorch 2.14.1+cu130; NVIDIA GeForce RTX 4070 Ti SUPER. This is not a verified manifest for every pilot run.
- [ ] Exact dependency versions for all historical runs are not archived; exact numerical reproduction is not guaranteed.

### Execution and evaluation

Run the baseline notebook separately for C16, C17, and BRACS and preserve outputs per dataset. The zero-noise script provides an alternative ABMIL run, but it should not be assumed to reproduce an old notebook checkpoint exactly.

~~~bash
python train_gaussian_noise.py --dataset C16 --noise-std 0 --droprate 0
python train_gaussian_noise.py --dataset C17 --noise-std 0 --droprate 0
python train_gaussian_noise.py --dataset BRACS --noise-std 0 --droprate 0
python train_transmil.py --dataset C16 --max-patches 4096
python train_transmil.py --dataset C17 --max-patches 4096
python train_transmil.py --dataset BRACS --max-patches 4096
python train_transmil.py --dataset BRACS --max-patches 4096 --focal --focal-gamma 2
~~~

- [x] Checkpoints maximize validation macro-AUC + macro-F1, retaining the earlier epoch on ties. Each run evaluates test data after loading that checkpoint.
- [x] utils.py defines macro one-versus-rest AUROC, macro-F1, and accuracy (%). Cross-entropy is averaged over slides. Focal-loss values are not directly comparable with cross-entropy.
- [x] TransMIL randomly samples long training bags and deterministically samples validation/test bags, capped at 4096. ABMIL noise/subsampling variants perturb only training inputs.
Optional future work: multiple seeds, confidence intervals, matched architectural ablations, calibration and class-level error analyses are not included; these are not treated as required missing experiments.

### Report-to-result map

JSON filenames below are under results/.

| Report result | Saved evidence | Best epoch | Test accuracy (%) |
| --- | --- | ---: | ---: |
| C16 ABMIL | c16_noise0.json | 3 | 88.3721 |
| C17 ABMIL | c17_noise0.json | 20 | 80.0000 |
| BRACS ABMIL | BRACS output in ECE1512_ProjectA_Student.ipynb | 10 | 40.6977 |
| C16 TransMIL | c16_transmil4096.json | 3 | 93.7984 |
| C17 TransMIL | c17_transmil4096.json | 15 | 82.0000 |
| BRACS TransMIL | bracs_transmil4096.json | 20 | 48.8372 |
| BRACS Gaussian noise | bracs_noise0.2.json | 6 | 39.5349 |
| BRACS training patch subsampling | bracs_patch2048.json | 4 | 39.5349 |
| BRACS TransMIL focal loss | bracs_transmil4096_focal2.json | 10 | 38.3721 |
| BRACS CLAM-MB pilot | bracs_clam_mb.json (historical entry restored; not rerun) | 10 | 39.5349 |

BRACS baseline: notebook output, AUC 0.7916, F1 0.3439, loss 1.6326. Do not substitute bracs_noise0_drop0.25.json: that separate dropout run has accuracy 38.3721%.

Current TransMIL CE runs append _ce to result filenames; tracked main-comparison files omit it. Current zero-dropout noise runs include _drop0, while tracked C16/C17 zero-noise filenames omit it. Keep fresh outputs separate and check JSON payloads. The notebook uses a fixed C16 checkpoint filename even when another dataset is selected; use dataset-specific names when preserving new runs.

### Figures and artifact limitations

- [x] The final notebook plotting cell reads the three TransMIL JSONs and regenerates the comparison and learning curves; use Matplotlib savefig after each plotting block to save them. Tracked images: results/transmil_vs_abmil.png and results/transmil_curves.png.
- [x] Original Overleaf figure archived as results/abmil_curves_overleaf_original.png. Run python plot_abmil_curves.py to generate results/abmil_curves.png from the C16/C17 zero-noise JSON histories and original BRACS notebook stdout, retaining printed precision. This is not a byte-identical reconstruction.
- [ ] Complete separate C16/C17 baseline notebook executions remain unrecovered; JSON histories and configuration are available.
- [x] Historical train_clam_mb.py recovered unchanged from commit 9da9662 (removed in 0245cd4). Defaults match the saved pilot configuration. See PROVENANCE.md for the command and overwrite precautions.
- [ ] CLAM-MB historical numerical result has not been revalidated; syntax checking is not a training run.
- [x] TransMIL is the selected implemented extension evaluated on all three datasets. Projection width, dropout, patch sampling, ordering, and aggregation change together; this is a complete-pipeline comparison, not a causal attention ablation.

## Source attribution and audit evidence

See [PROVENANCE.md](PROVENANCE.md) for course-starter evidence, historical recovery links, plotting inputs, remaining execution gaps, and specific author-confirmation questions. Existing course statements and MIT copyright are preserved. No third-party code origin is inferred merely from a paper implementation. The original starter archive and authorship/adaptation scope still require confirmation.
