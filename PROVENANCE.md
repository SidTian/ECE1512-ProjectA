# Evidence and source audit

Audit base: commit `9fa5942521fbc34ccee5244a56c6ddef4f55703e`. No training was run in this audit. Repository history preserves the pre-edit versions.

## Learning curves and baseline records

Run `python plot_abmil_curves.py` from the repository. Inputs are `results/c16_noise0.json`, `results/c17_noise0.json`, and the original 30-epoch BRACS stdout in `ECE1512_ProjectA_Student.ipynb`. BRACS values retain their printed precision (loss: four decimals; AUC/F1: three); they are not reconstructed from pixels or estimated. The generated figure is an evidence-based replacement, not a claim of byte identity with the prior Overleaf figure. No dropout-0.25 or positive-noise experiment is substituted.

C16/C17 JSON histories and test metrics are retained. Full separate baseline notebook executions, historical weights, and exact per-run environments have not been recovered from the tracked history. Existing JSONs are evidence of recorded runs, not newly verified measurements. Baseline instructions and configuration files are in README and config/. Preserve new runs separately.

## Historical CLAM-MB recovery

`train_clam_mb.py` is restored byte-for-byte from [9da9662](https://github.com/SidTian/ECE1512-ProjectA/blob/9da96620a189d786ca3d6a9db3ad421c6aab9917/train_clam_mb.py); it was removed in `0245cd4`. Its default BRACS configuration, k_sample=8, bag_weight=0.7, inverse-frequency weighting, validation AUC+F1 selection and JSON fields agree with the archived pilot. This is a recovered historical entry point, not a newly implemented model or numerical reproduction of the JSON.

In a disposable copy of the repository with features installed, run `python train_clam_mb.py --k-sample 8 --bag-weight 0.7`. It writes `results/bracs_clam_mb.json` and a checkpoint, so preserve the archived JSON before running. Runtime training has not been validated during this audit. TransMIL remains the selected extension evaluated on all three datasets.

## Author-confirmed notebook template origin

The project author confirms that the baseline notebook originates from the school-provided ECE1512 template and describes the supplied local reference as a TODO-completed version. This confirms the course origin of the notebook scaffold, not that the completed code was supplied by the school. That reference is not a blank starter. The subsequently supplied complete ZIP now establishes the original scaffold and TODO locations; see the verified boundary below.

Reference inspected: `ECE1512_ProjectA_Student.ipynb`, SHA-256 `b2dae0329d66c1bef99bedce54aa869a1036ed80c35e83da7791e2741b0d5a5c`. Both it and the repository notebook contain 37 cells. Direct comparison finds differences including padding-mask support, CSV history export and checkpoint verification in the local reference; the repository adds written answers and a TransMIL result-plotting section. These differences are documented without assigning their authorship.

The local reference preserves a separate BRACS execution: epoch 10, accuracy 40.6977%, AUC 0.7920, F1 0.3439, loss 1.6344; its environment output reports PyTorch 2.11.0+cu128 and RTX 4070 Laptop GPU. It is not the report baseline (AUC 0.7916, loss 1.6326). No values or environment claims in the report are replaced by this separate execution. No training was run for this comparison.

## Verified course starter boundary

The complete school starter ZIP was subsequently supplied and compared directly. Its SHA-256 is `e48c5ec771a12415a3677b50cba1edbd976f5ee18790209ac1850650b4e12da8`. File hashes and normalized-text comparison results are archived in [docs/starter_comparison.json](docs/starter_comparison.json). This supersedes earlier uncertainty about the separate helper files and blank notebook availability.

The following repository files match the course starter text without changes: `dataloader.py`, `utils.py`, `wsi_visualization.py`, `requirements.txt`, all three files under `config/`, both dataset CSVs and `dataset_csv/README.md`. Consequently, their loader logic, split rules, metric helpers, visualization implementation and metadata documentation are course-provided material, not claimed as original project contributions. The BRACS MELD-path and C17 manifest statements in the metadata README are inherited course documentation; the team did not newly establish those provenance claims during this audit.

The original notebook has 37 cells, unfilled TODO calls and no saved execution outputs. It supplies the educational text, model class structure, gated-attention branch definitions, training/evaluation scaffolds, full epoch loop, validation macro-AUC+macro-F1 checkpoint rule and basic plotting/inspection code. Project completions fill DataLoader creation, classifier/projection operations, gated forward operations, attention normalization and aggregation, loss/optimizer setup, training steps and evaluation steps. These completions are distinguished from the supplied surrounding framework. The current notebook has no executable TODO calls left.

Other differences from the blank starter include written question responses, saved BRACS outputs, a checkpoint-save statement and the TransMIL result-reading/plotting section. The repository omits the starter's standalone split/feature inspection cell and contains a repeated DataLoader construction cell. These are documented structural differences, not new experimental evidence. The historical notebook is preserved rather than silently rewriting code underneath saved outputs.

No LICENSE file or explicit license grant was found in the supplied starter package. Existing source statements and the repository MIT license are preserved; its `Copyright (c) 2026 SidTian` does not establish ownership or relicensing authority over course-provided portions. No upstream third-party origin is inferred for the school's code.

## Pending extension provenance

The author is checking TransMIL and CLAM-MB implementation origins with a teammate. Their source/adaptation status remains pending, with no new confirmation required during that discussion. In particular, attribution for `iterative_pinv`, `NystromAttention`, PPEG in `train_transmil.py` and CLAM instance-selection routines in `train_clam_mb.py` is unresolved. If external code was adapted, record its repository, revision, affected functions and license. The separate `train_gaussian_noise.py`, `train_patch_subsample.py` and `verify_features.py` are absent from the starter; absence establishes that they were added later, not proof of original authorship.

The CLAM entry is restored from the project's own historical commit, as documented above. `plot_abmil_curves.py` was added during this artifact audit. No third-party copying claim is made merely because an implementation follows a paper.

## Split evidence

Dataset papers establish dataset origins. Project-specific partitions are defined by `dataloader.py`, `config/` and `dataset_csv/README.md`. C17 uses centers 3-4 for test, centers 0-2 for a slide-level 90/10 development split (seed 42), not the official challenge test split. Counts remain C16 242/27/129, C17 269/30/200, BRACS 389/62/86. No features were reloaded to independently recount them in this audit.

Repeated seeds, confusion matrices and calibration are optional future analyses, not missing required experiments in this audit.
