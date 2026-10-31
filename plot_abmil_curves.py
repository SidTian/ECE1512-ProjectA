"""Plot archived evidence only; BRACS values retain printed notebook precision."""
import json
import re
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parent

def main():
    notebook = json.loads((ROOT / 'ECE1512_ProjectA_Student.ipynb').read_text(encoding='utf-8'))
    text = ''.join(''.join(o.get('text', [])) for c in notebook['cells'] for o in c.get('outputs', []))
    pattern = r'Epoch (\d+)/30 \| train loss=([\d.]+) acc=([\d.]+) \| val loss=([\d.]+) auc=([\d.]+) f1=([\d.]+)'
    rows = [dict(zip(['epoch','train_loss','train_acc','val_loss','val_auc','val_f1'], map(float, m))) for m in re.findall(pattern, text)]
    assert len(rows) == 30 and [r['epoch'] for r in rows] == list(range(1,31)), 'Expected intact 30-epoch BRACS log'
    assert '40.6977' in text and '0.7916' in text and "DATASET = 'BRACS'" in ''.join(''.join(c.get('source',[])) for c in notebook['cells'])
    runs = [json.loads((ROOT / 'results' / f'{d}_noise0.json').read_text()) for d in ['c16','c17']]
    runs.append(dict(dataset='BRACS', best_epoch=10, history=rows))
    fig, axes = plt.subplots(2, 3, figsize=(10, 4.6), constrained_layout=True)
    for col, run in enumerate(runs):
        h = run['history']; epochs = [r['epoch'] for r in h]
        assert epochs == list(range(1,31))
        for key in ['train_loss','val_loss']:
            axes[0,col].plot(epochs,[r[key] for r in h],label=key.replace('_',' '))
        for key in ['val_auc','val_f1']:
            axes[1,col].plot(epochs,[r[key] for r in h],label=key.replace('_',' '))
        axes[0,col].set_title(run['dataset'])
        for row in range(2):
            ax=axes[row,col]; ax.axvline(run['best_epoch'],ls='--',color='gray'); ax.set_xlabel('Epoch'); ax.legend(fontsize=7); ax.grid(alpha=.2)
    axes[0,0].set_ylabel('Loss'); axes[1,0].set_ylabel('Validation score')
    fig.savefig(ROOT / 'results' / 'abmil_curves.png', dpi=220)
    print('Saved results/abmil_curves.png; BRACS uses rounded original stdout, not estimated values.')

if __name__ == '__main__':
    main()
