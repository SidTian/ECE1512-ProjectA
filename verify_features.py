"""Check that the extracted HDF5 features match the project configs."""

import os
from pathlib import Path

import h5py

from dataloader import build_HDF5_feat_dataset, mil_collate_fn
from utils import infer_feature_dim, load_config

ROOT = Path(__file__).resolve().parent
CONFIGS = {
    "C16": ROOT / "config" / "camelyon16_medical_ssl_config.yml",
    "C17": ROOT / "config" / "camelyon17_medical_ssl_config.yml",
    "BRACS": ROOT / "config" / "bracs_medical_ssl_config.yml",
}


def check_h5(path: Path, n_class: int, expected_dim: int) -> None:
    if not path.is_file():
        raise FileNotFoundError(path)
    if path.stat().st_size < 1024:
        raise RuntimeError(f"{path} is too small to be a feature file")

    labels = set()
    with h5py.File(path, "r") as h5:
        slide_ids = list(h5.keys())
        if not slide_ids:
            raise RuntimeError(f"{path} contains no slides")
        for slide_id in slide_ids:
            group = h5[slide_id]
            if "feat" not in group:
                raise KeyError(f"{slide_id} has no feat dataset")
            feat = group["feat"]
            if feat.ndim != 2 or feat.shape[0] == 0 or feat.shape[1] != expected_dim:
                raise ValueError(f"{slide_id}: bad feat shape {feat.shape}")
            if "label" not in group.attrs:
                raise KeyError(f"{slide_id} has no label attribute")
            labels.add(int(group.attrs["label"]))
    print(
        f"  HDF5 slides={len(slide_ids)} raw_labels={sorted(labels)} dim={expected_dim}"
    )
    if max(labels) >= n_class or min(labels) < 0:
        print(f"  warning: raw labels {sorted(labels)} are outside 0..{n_class - 1}")


def main() -> None:
    os.chdir(ROOT)
    failed = False
    for name, config_path in CONFIGS.items():
        print(f"\n[{name}]")
        conf = load_config(str(config_path))
        data_path = (ROOT / conf.data_dir).resolve()
        print(f"  path: {data_path}")
        try:
            check_h5(data_path, int(conf.n_class), int(conf.D_feat))
            train_data, val_data, test_data = build_HDF5_feat_dataset(
                str(data_path), conf
            )
            dim = infer_feature_dim(train_data)
            if dim != int(conf.D_feat):
                raise ValueError(f"feature dim {dim} != config D_feat {conf.D_feat}")
            sample = train_data[0]
            batch = mil_collate_fn([sample])
            print(
                f"  split: train={len(train_data)} val={len(val_data)} test={len(test_data)}"
            )
            print(
                f"  sample: slide={sample['slide_id']} label={sample['label']} "
                f"feat={tuple(sample['input'].shape)} batch={tuple(batch['input'].shape)}"
            )
            print("  PASS")
        except Exception as exc:
            failed = True
            print(f"  FAIL: {type(exc).__name__}: {exc}")

    if failed:
        raise SystemExit(1)
    print("\nAll three feature files passed.")


if __name__ == "__main__":
    main()
