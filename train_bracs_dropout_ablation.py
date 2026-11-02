"""Portable adaptation of the local 2026-10-30 BRACS dropout study.

Reuses this repository's TransMIL; does not introduce a new model implementation.
Historical evidence and adaptation scope: README.md, BRACS dropout ablation.
"""
import argparse
import csv
from datetime import datetime
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time
from importlib.metadata import version, PackageNotFoundError

REPO = Path(__file__).resolve().parent
RATES = [0.0, 0.1, 0.25, 0.5]

def dump(path, value):
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False), encoding='utf-8')

def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()

def load_entry(root):
    source = root / 'source'
    sys.path.insert(0, str(source))
    spec = importlib.util.spec_from_file_location('dropout_model_source', source / 'train_transmil.py')
    entry = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(entry)
    return entry

def set_dropout(model, rate):
    import torch
    layers = {name: layer for name, layer in model.named_modules() if isinstance(layer, torch.nn.Dropout)}
    assert set(layers) == {'layer1.attn.to_out.1', 'layer2.attn.to_out.1'}
    for layer in layers.values():
        layer.p = rate
    return {name: layer.p for name, layer in layers.items()}

def state_hash(model):
    h = hashlib.sha256()
    for name, value in model.state_dict().items():
        h.update(name.encode())
        h.update(value.detach().cpu().contiguous().numpy().tobytes())
    return h.hexdigest()

def prepare(args):
    import torch
    import yaml
    import pandas as pd
    if not torch.cuda.is_available():
        raise RuntimeError('CUDA GPU required by this reproduction protocol.')
    data_path = args.features.resolve()
    if not data_path.is_file():
        raise FileNotFoundError(data_path)
    root = args.output.resolve()
    root.mkdir(parents=True, exist_ok=False)  # Never overwrite another run.
    source = root / 'source'
    source.mkdir()
    for name in ('train_transmil.py', 'dataloader.py', 'utils.py'):
        shutil.copy2(REPO / name, source / name)
    shutil.copytree(REPO / 'dataset_csv', root / 'dataset_csv')
    shutil.copy2(REPO / 'results/bracs_dropout_ablation/splits.csv', root / 'reference_splits.csv')
    shutil.copy2(Path(__file__), root / 'experiment_runner.py')
    config = yaml.safe_load((REPO / 'config/bracs_medical_ssl_config.yml').read_text(encoding='utf-8'))
    config.update(data_dir=str(data_path), dataset_csv_dir=str(root / 'dataset_csv'),
                  seed=42, train_epoch=30, B=1, n_class=7, n_shot=0, lr=1e-4, wd=1e-5,
                  warmup_epoch=0)
    (root / 'config.yml').write_text(yaml.safe_dump(config, sort_keys=False), encoding='utf-8')
    packages = {}
    for name in ('torch', 'torchmetrics', 'numpy', 'pandas', 'h5py', 'scikit-learn', 'pyyaml'):
        try:
            packages[name] = version(name)
        except PackageNotFoundError:
            packages[name] = 'not installed'
    entry = load_entry(root)
    conf = entry.load_config(root / 'config.yml')
    datasets = entry.build_HDF5_feat_dataset(conf.data_dir, conf)
    assert tuple(map(len, datasets)) == (389, 62, 86)
    rows = [{'split': split, 'slide_id': name}
            for split, data in zip(('train', 'val', 'test'), datasets) for name in data.data_names]
    assert rows == pd.read_csv(root / 'reference_splits.csv').to_dict('records')
    assert all(data.feature_dim == 384 for data in datasets)
    hashes = []
    for rate in RATES:
        entry.seed_everything(42)
        model = entry.TransMIL(384, 7).to('cuda')
        set_dropout(model, rate)
        hashes.append(state_hash(model))
    assert len(set(hashes)) == 1
    del model
    dump(root / 'design.json', {'rates': RATES, 'epochs_per_rate': 30, 'seed': 42,
         'max_patches': 4096, 'loss': 'cross_entropy', 'warmup_enabled': False,
         'selection': 'validation macro-AUROC + macro-F1; earliest epoch; lower rate on ties',
         'test_policy': 'only selected configuration after four validation runs',
         'initial_weights_sha256': hashes[0], 'python': sys.version,
         'packages': packages, 'cuda_runtime': torch.version.cuda,
         'gpu': torch.cuda.get_device_name(0),
         'source_sha256': {p.name: digest(p) for p in source.glob('*.py')}})
    print('Preflight PASS: split IDs, feature dimensions, dropout modules and equal initial weights.', flush=True)
    return root


def train(root, rate):
    import torch
    import numpy as np
    import pandas as pd
    entry = load_entry(root)
    directory = root / f'dropout_{rate:g}'
    directory.mkdir(exist_ok=False)
    entry.seed_everything(42)
    conf = entry.load_config(root / 'config.yml')
    train_base, val_base, test_base = entry.build_HDF5_feat_dataset(conf.data_dir, conf)
    assert (len(train_base), len(val_base), len(test_base)) == (389, 62, 86)
    split_rows = [{'split': split, 'slide_id': name} for split, data in [('train', train_base), ('val', val_base), ('test', test_base)] for name in data.data_names]
    pd.DataFrame(split_rows).to_csv(directory / 'splits.csv', index=False)
    reference = pd.read_csv(root / 'reference_splits.csv')
    assert split_rows == reference.to_dict('records')
    train_data = entry.SpatialSquareBag(train_base, 4096, train=True)
    val_data = entry.SpatialSquareBag(val_base, 4096, train=False)
    generator = torch.Generator().manual_seed(42)
    device = torch.device('cuda')
    assert torch.cuda.is_available()
    loader_args = dict(batch_size=1, num_workers=0, pin_memory=True, collate_fn=entry.mil_collate_fn)
    train_loader = entry.DataLoader(train_data, shuffle=True, generator=generator, **loader_args)
    val_loader = entry.DataLoader(val_data, shuffle=False, **loader_args)
    entry.seed_everything(42)
    model = entry.TransMIL(384, 7).to(device)
    actual = set_dropout(model, rate)
    initial_hash = state_hash(model)
    assert initial_hash == json.loads((root / 'design.json').read_text())['initial_weights_sha256']
    dump(directory / 'initialization.json', {'initial_weights_sha256': initial_hash, 'actual_dropout_layers': actual,
        'config': vars(conf), 'gpu': torch.cuda.get_device_name(0), 'parameters': sum(p.numel() for p in model.parameters())})
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=conf.lr, weight_decay=conf.wd)
    best_score, best_epoch, history = float('-inf'), -1, []
    for epoch in range(30):
        started = time.time()
        tr = entry.train_one_epoch(model, criterion, train_loader, optimizer, device)
        va = entry.evaluate(model, criterion, val_loader, device, conf)
        score = va['auc'] + va['f1']
        row = {'epoch': epoch + 1, 'train_loss': tr['loss'], 'train_acc': tr['acc'], 'val_loss': va['loss'],
               'val_acc': va['acc'], 'val_auc': va['auc'], 'val_f1': va['f1'], 'score': score, 'seconds': time.time() - started}
        assert all(np.isfinite(value) for value in row.values())
        if score > best_score:
            best_score, best_epoch = score, epoch + 1
            torch.save({key: value.detach().cpu().clone() for key, value in model.state_dict().items()}, directory / 'best_model.pt')
        history.append(row)
        pd.DataFrame(history).to_csv(directory / 'history.csv', index=False)
        dump(directory / 'progress.json', {'completed_epochs': epoch + 1, 'best_epoch': best_epoch, 'best_score': best_score})
        print(f'dropout={rate:g} Epoch {epoch+1:02d}/30 | train loss={tr["loss"]:.4f} acc={tr["acc"]:.2f} | '
              f'val loss={va["loss"]:.4f} auc={va["auc"]:.4f} f1={va["f1"]:.4f} score={score:.4f} | best={best_epoch} | {row["seconds"]:.1f}s', flush=True)
    dump(directory / 'validation_result.json', {'dropout': rate, 'best_epoch': best_epoch, 'best_score': best_score,
        'best_validation': history[best_epoch - 1], 'epochs': 30, 'initial_weights_sha256': initial_hash,
        'test_evaluated': False, 'actual_dropout_layers': actual})

def test_selected(root):
    import pandas as pd
    import torch
    from sklearn.metrics import classification_report, confusion_matrix
    results = [json.loads((root / f'dropout_{rate:g}' / 'validation_result.json').read_text()) for rate in RATES]
    assert len({r['initial_weights_sha256'] for r in results}) == 1
    winner = max(results, key=lambda result: (result['best_score'], -result['dropout']))
    dump(root / 'selection_before_test.json', {'selected_dropout': winner['dropout'], 'rule': 'max validation AUROC+F1; lower dropout wins ties',
                                             'validation_results': results, 'selected_before_test': True})
    entry = load_entry(root)
    conf = entry.load_config(root / 'config.yml')
    entry.seed_everything(42)
    _, _, test_base = entry.build_HDF5_feat_dataset(conf.data_dir, conf)
    dataset = entry.SpatialSquareBag(test_base, 4096, train=False)
    loader = entry.DataLoader(dataset, batch_size=1, shuffle=False, num_workers=0, collate_fn=entry.mil_collate_fn)
    model = entry.TransMIL(384, 7).to('cuda')
    set_dropout(model, winner['dropout'])
    checkpoint = root / f'dropout_{winner["dropout"]:g}' / 'best_model.pt'
    model.load_state_dict(torch.load(checkpoint, map_location='cpu', weights_only=True))
    model.eval()
    probabilities, labels, names = [], [], []
    total_loss, count = 0., 0
    criterion = torch.nn.CrossEntropyLoss()
    with torch.no_grad():
        for batch in loader:
            target = batch['label'].to('cuda', dtype=torch.long)
            logits = model(batch['input'].to('cuda', dtype=torch.float32))
            assert torch.isfinite(logits).all()
            probabilities.append(torch.softmax(logits, dim=-1).cpu())
            labels.append(target.cpu()); names.extend(batch['slide_id'])
            total_loss += criterion(logits, target).item() * len(target); count += len(target)
    probs, labels = torch.cat(probabilities), torch.cat(labels)
    metrics = entry.classification_metrics(probs, labels, 7)
    metrics['loss'] = total_loss / count
    pred, y = probs.argmax(1).numpy(), labels.numpy()
    frame = pd.DataFrame({'slide_id': names, 'true_label': y, 'predicted_label': pred, 'correct': y == pred})
    for i in range(7):
        frame[f'probability_class_{i}'] = probs[:, i].numpy()
    frame.to_csv(root / 'selected_test_predictions.csv', index=False)
    matrix = confusion_matrix(y, pred, labels=list(range(7)))
    pd.DataFrame(matrix).to_csv(root / 'selected_confusion_matrix.csv', index=False)
    per_class = classification_report(y, pred, labels=list(range(7)), output_dict=True, zero_division=0)
    pd.DataFrame(per_class).transpose().to_csv(root / 'selected_classification_report.csv')
    dump(root / 'selected_test_metrics.json', {'dropout': winner['dropout'], 'epoch': winner['best_epoch'], 'test': metrics,
        'weights_reload': 'PASS', 'predictions': count, 'confusion_matrix': matrix.tolist(), 'classification_report': per_class})
    shutil.copy2(checkpoint, root / 'selected_best_model.pt')
    print('SELECTED_DROPOUT:', winner['dropout'], flush=True)
    print('SELECTED_TEST_METRICS:', json.dumps(metrics), flush=True)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--features', type=Path, default=REPO / 'feature/BRACS/patch_feats_pretrain_medical_ssl.h5')
    parser.add_argument('--output', type=Path, default=REPO / 'runs' / ('bracs_dropout_' + datetime.now().strftime('%Y%m%d_%H%M%S_%f')))
    parser.add_argument('--check-only', action='store_true', help='Check data, split IDs and initialization without training or test evaluation.')
    parser.add_argument('--root', type=Path, help=argparse.SUPPRESS)
    parser.add_argument('--train', type=float, help=argparse.SUPPRESS)
    parser.add_argument('--test', action='store_true', help=argparse.SUPPRESS)
    args = parser.parse_args()
    if args.train is not None:
        assert args.train in RATES
        train(args.root, args.train)
        return
    if args.test:
        test_selected(args.root)
        return
    root = prepare(args)
    if args.check_only:
        return
    print('EXPERIMENT_ROOT:', root, flush=True)
    for rate in RATES:
        env = dict(os.environ, PYTHONUNBUFFERED='1', PYTHONIOENCODING='utf-8')
        with (root / f'train_dropout_{rate:g}.log').open('w', encoding='utf-8') as log:
            process = subprocess.Popen([sys.executable, '-u', str(root / 'experiment_runner.py'), '--root', str(root), '--train', str(rate)],
                stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, encoding='utf-8', errors='replace', env=env)
            for line in process.stdout:
                print(line, end='', flush=True); log.write(line); log.flush()
            code = process.wait()
        if code:
            raise SystemExit(code)
    subprocess.run([sys.executable, '-u', str(root / 'experiment_runner.py'), '--root', str(root), '--test'], check=True)
    print('DROPOUT_EXPERIMENT_COMPLETE:', root, flush=True)

if __name__ == '__main__':
    main()
