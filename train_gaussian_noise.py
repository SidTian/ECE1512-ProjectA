import argparse
import copy
import json
import os
import time
from pathlib import Path

import torch
from torch import nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

from dataloader import build_HDF5_feat_dataset, mil_collate_fn
from utils import classification_metrics, load_config, seed_everything


ROOT = Path(__file__).resolve().parent
CONFIG_FILES = {
    "C16": "camelyon16_medical_ssl_config.yml",
    "C17": "camelyon17_medical_ssl_config.yml",
    "BRACS": "bracs_medical_ssl_config.yml",
}


class GaussianNoise(Dataset):
    def __init__(self, base, std=0.2):
        self.base = base
        self.std = std

    def __len__(self):
        return len(self.base)

    def __getitem__(self, index):
        item = dict(self.base[index])
        x = item["input"]
        item["input"] = x + self.std * torch.randn_like(x)
        return item


class Classifier1FC(nn.Module):
    def __init__(self, n_channels, n_classes, droprate=0.0):
        super().__init__()
        self.dropout = nn.Dropout(droprate) if droprate > 0 else nn.Identity()
        self.fc = nn.Linear(n_channels, n_classes)

    def forward(self, x):
        return self.fc(self.dropout(x))


class DimReduction(nn.Module):
    def __init__(self, n_channels, m_dim):
        super().__init__()
        self.fc = nn.Linear(n_channels, m_dim)
        self.relu = nn.ReLU()

    def forward(self, x):
        return self.relu(self.fc(x))


class AttentionGated(nn.Module):
    def __init__(self, L=256, D=128, K=1):
        super().__init__()
        self.attention_V = nn.Sequential(nn.Linear(L, D), nn.Tanh())
        self.attention_U = nn.Sequential(nn.Linear(L, D), nn.Sigmoid())
        self.attention_weights_w = nn.Linear(D, K)

    def forward(self, x):
        a_v = self.attention_V(x)
        a_u = self.attention_U(x)
        a = self.attention_weights_w(a_v * a_u)
        return a.transpose(1, 2)


class ABMIL(nn.Module):
    def __init__(self, conf):
        super().__init__()
        self.dimreduction = DimReduction(conf.D_feat, conf.D_inner)
        self.attention = AttentionGated(conf.D_inner, conf.D_attn, K=1)
        self.classifier = Classifier1FC(conf.D_inner, conf.n_class, conf.droprate)

    def forward(self, x, return_attention=False):
        med_feat = self.dimreduction(x)
        a_logits = self.attention(med_feat)
        attention = F.softmax(a_logits, dim=-1)
        bag_feat = torch.bmm(attention, med_feat).squeeze(1)
        logits = self.classifier(bag_feat)
        if return_attention:
            return logits, attention
        return logits


def train_one_epoch(model, criterion, data_loader, optimizer, device):
    model.train()
    total_loss = 0.0
    total_correct = 0
    total_examples = 0

    for data in data_loader:
        bag = data["input"].to(device, dtype=torch.float32)
        labels = data["label"].to(device, dtype=torch.long)
        optimizer.zero_grad()
        logits = model(bag)
        loss = criterion(logits, labels)
        loss.backward()
        optimizer.step()

        batch_size = labels.shape[0]
        total_loss += loss.item() * batch_size
        total_correct += (logits.argmax(dim=1) == labels).sum().item()
        total_examples += batch_size

    return {
        "loss": total_loss / total_examples,
        "acc": 100.0 * total_correct / total_examples,
    }


@torch.no_grad()
def evaluate(model, criterion, data_loader, device, conf):
    model.eval()
    total_loss = 0.0
    total_examples = 0
    probabilities_all = []
    labels_all = []

    for data in data_loader:
        bag = data["input"].to(device, dtype=torch.float32)
        labels = data["label"].to(device, dtype=torch.long)
        logits = model(bag)
        loss = criterion(logits, labels)
        probabilities = F.softmax(logits, dim=-1)
        total_loss += loss.item() * labels.shape[0]
        total_examples += labels.shape[0]
        probabilities_all.append(probabilities.cpu())
        labels_all.append(labels.cpu())

    probabilities_all = torch.cat(probabilities_all, dim=0)
    labels_all = torch.cat(labels_all, dim=0)
    metrics = classification_metrics(probabilities_all, labels_all, conf.n_class)
    metrics["loss"] = total_loss / total_examples
    return metrics


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", default="BRACS", choices=sorted(CONFIG_FILES))
    parser.add_argument("--noise-std", type=float, default=0.2)
    parser.add_argument("--droprate", type=float, default=None)
    parser.add_argument("--epochs", type=int, default=None)
    return parser.parse_args()


def main():
    args = parse_args()
    os.chdir(ROOT)
    conf = load_config(ROOT / "config" / CONFIG_FILES[args.dataset])
    if args.droprate is not None:
        conf.droprate = args.droprate
    seed = int(conf.seed)
    seed_everything(seed)

    train_data, val_data, test_data = build_HDF5_feat_dataset(conf.data_dir, conf)
    train_data = GaussianNoise(train_data, std=args.noise_std)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    batch_size = int(conf.B)
    num_workers = int(conf.n_worker)
    pin_memory = bool(conf.pin_memory) and torch.cuda.is_available()
    generator = torch.Generator()
    generator.manual_seed(seed)
    train_loader = DataLoader(
        train_data,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=pin_memory,
        collate_fn=mil_collate_fn,
        generator=generator,
    )
    val_loader = DataLoader(
        val_data,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
        collate_fn=mil_collate_fn,
    )
    test_loader = DataLoader(
        test_data,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
        collate_fn=mil_collate_fn,
    )

    seed_everything(seed)
    model = ABMIL(conf).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=conf.lr, weight_decay=conf.wd)
    num_epochs = int(conf.train_epoch if args.epochs is None else args.epochs)

    print(f"dataset={args.dataset} device={device} noise_std={args.noise_std} droprate={conf.droprate}")
    print(f"slides train={len(train_data)} val={len(val_data)} test={len(test_data)}")
    print(f"epochs={num_epochs} lr={conf.lr} n_class={conf.n_class}")

    best_score = float("-inf")
    best_epoch = -1
    best_model_state = None
    history = []

    for epoch in range(num_epochs):
        start = time.time()
        train_metrics = train_one_epoch(model, criterion, train_loader, optimizer, device)
        val_metrics = evaluate(model, criterion, val_loader, device, conf)
        score = val_metrics["auc"] + val_metrics["f1"]
        if score > best_score:
            best_score = score
            best_epoch = epoch
            best_model_state = copy.deepcopy(model.state_dict())
        row = {
            "epoch": epoch + 1,
            "seconds": round(time.time() - start, 1),
            "train_loss": train_metrics["loss"],
            "train_acc": train_metrics["acc"],
            "val_loss": val_metrics["loss"],
            "val_acc": val_metrics["acc"],
            "val_auc": val_metrics["auc"],
            "val_f1": val_metrics["f1"],
        }
        history.append(row)
        print(
            f"Epoch {row['epoch']:02d}/{num_epochs} | "
            f"train loss={row['train_loss']:.4f} acc={row['train_acc']:.2f} | "
            f"val loss={row['val_loss']:.4f} acc={row['val_acc']:.2f} "
            f"auc={row['val_auc']:.3f} f1={row['val_f1']:.3f} | "
            f"best={best_epoch + 1} | {row['seconds']:.1f}s"
        )

    model.load_state_dict(best_model_state)
    test_metrics = evaluate(model, criterion, test_loader, device, conf)
    print(f"Best validation epoch: {best_epoch + 1}")
    print(
        "Final test metrics: "
        + " ".join(f"{key}={value:.4f}" for key, value in test_metrics.items())
    )

    out_dir = ROOT / "results"
    out_dir.mkdir(exist_ok=True)
    tag = f"{args.dataset.lower()}_noise{args.noise_std:g}_drop{conf.droprate:g}"
    ckpt_path = out_dir / f"abmil_{tag}_best.pt"
    result_path = out_dir / f"{tag}.json"
    torch.save(best_model_state, ckpt_path)
    payload = {
        "dataset": args.dataset,
        "noise_std": args.noise_std,
        "droprate": conf.droprate,
        "best_epoch": best_epoch + 1,
        "best_score": best_score,
        "test_metrics": test_metrics,
        "history": history,
    }
    result_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(f"checkpoint: {ckpt_path}")
    print(f"metrics: {result_path}")


if __name__ == "__main__":
    main()
