# Evidence and source audit

Audit base: commit `9fa5942521fbc34ccee5244a56c6ddef4f55703e`. No training was run in this audit. Repository history preserves the pre-edit versions.

## Learning curves and baseline records

Run `python plot_abmil_curves.py` from the repository. Inputs are `results/c16_noise0.json`, `results/c17_noise0.json`, and the original 30-epoch BRACS stdout in `ECE1512_ProjectA_Student.ipynb`. BRACS values retain their printed precision (loss: four decimals; AUC/F1: three); they are not reconstructed from pixels or estimated. The generated figure is an evidence-based replacement, not a claim of byte identity with the prior Overleaf figure. No dropout-0.25 or positive-noise experiment is substituted.

C16/C17 JSON histories and test metrics are retained. Full separate baseline notebook executions, historical weights, and exact per-run environments have not been recovered from the tracked history. Existing JSONs are evidence of recorded runs, not newly verified measurements. Baseline instructions and configuration files are in README and config/. Preserve new runs separately.

## Historical CLAM-MB recovery

`train_clam_mb.py` is restored byte-for-byte from [9da9662](https://github.com/SidTian/ECE1512-ProjectA/blob/9da96620a189d786ca3d6a9db3ad421c6aab9917/train_clam_mb.py); it was removed in `0245cd4`. Its default BRACS configuration, k_sample=8, bag_weight=0.7, inverse-frequency weighting, validation AUC+F1 selection and JSON fields agree with the archived pilot. This is a recovered historical entry point, not a newly implemented model or numerical reproduction of the JSON.

In a disposable copy of the repository with features installed, run `python train_clam_mb.py --k-sample 8 --bag-weight 0.7`. It writes `results/bracs_clam_mb.json` and a checkpoint, so preserve the archived JSON before running. Runtime training has not been validated during this audit. TransMIL remains the selected extension evaluated on all three datasets.

## Author-confirmed notebook template origin

The project author confirms that the baseline notebook originates from the school-provided ECE1512 template and describes the supplied local reference as a TODO-completed version. This confirms the course origin of the notebook scaffold, not that the completed code was supplied by the school. The reference is not a blank, uncompleted starter and cannot establish the exact original TODO boundaries by itself.

Reference inspected: `ECE1512_ProjectA_Student.ipynb`, SHA-256 `b2dae0329d66c1bef99bedce54aa869a1036ed80c35e83da7791e2741b0d5a5c`. Both it and the repository notebook contain 37 cells. Direct comparison finds differences including padding-mask support, CSV history export and checkpoint verification in the local reference; the repository adds written answers and a TransMIL result-plotting section. These differences are documented without assigning their authorship.

The local reference preserves a separate BRACS execution: epoch 10, accuracy 40.6977%, AUC 0.7920, F1 0.3439, loss 1.6344; its environment output reports PyTorch 2.11.0+cu128 and RTX 4070 Laptop GPU. It is not the report baseline (AUC 0.7916, loss 1.6326). No values or environment claims in the report are replaced by this separate execution. No training was run for this comparison.

## Code attribution: evidence and unresolved questions

- `utils.py` explicitly identifies itself as utilities provided to students. The notebook identifies `build_HDF5_feat_dataset` as provided and includes course instructions and TODO scaffolding. These are evidence of course starter material; existing statements are preserved. The notebook template origin is now author-confirmed as above. The blank original distribution and its license were not supplied separately, so exact starter-versus-completion boundaries remain unverified.
- Completed notebook model/training cells and the standalone experiment scripts are present in project commit `9da9662`. That establishes repository provenance, not proof of original authorship. No explicit upstream code URL or third-party copyright notice was found in the reviewed Python files. Implementing ABMIL, TransMIL or CLAM does not establish copying from their official repositories.
- The restored CLAM script's source is the project's own historical commit linked above. The new plotting script was added during this artifact audit and parses existing evidence only.
- `dataset_csv/README.md` documents BRACS metadata derived from a local MELD directory and C17 center metadata from a dataset manifest. This establishes metadata provenance, not Python code provenance.
- The existing MIT license and `Copyright (c) 2026 SidTian` are preserved. This does not establish licensing rights over any unconfirmed starter or external portions.

Author confirmation requested: confirm the origins of the separate `dataloader.py`, `utils.py` and `wsi_visualization.py` files; if available, provide the blank notebook template to establish exact completion boundaries. Identify which completed notebook cells and `train_transmil.py`, `train_gaussian_noise.py`, `train_patch_subsample.py`, `train_clam_mb.py` were written by the team. In particular, were `iterative_pinv`, `NystromAttention`, PPEG or CLAM instance-selection routines adapted from an external implementation? If yes, supply repository URL, revision, adapted functions and license. No unsupported upstream attribution has been inserted.

## Split evidence

Dataset papers establish dataset origins. Project-specific partitions are defined by `dataloader.py`, `config/` and `dataset_csv/README.md`. C17 uses centers 3-4 for test, centers 0-2 for a slide-level 90/10 development split (seed 42), not the official challenge test split. Counts remain C16 242/27/129, C17 269/30/200, BRACS 389/62/86. No features were reloaded to independently recount them in this audit.

Repeated seeds, confusion matrices and calibration are optional future analyses, not missing required experiments in this audit.
