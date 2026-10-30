import argparse
import copy
import json
import os
import time
from pathlib import Path

import h5py
import torch
from torch import nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

from dataloader import _map_label, build_HDF5_feat_dataset, mil_collate_fn
from utils import classification_metrics, load_config, seed_everything


ROOT = Path(__file__).resolve().parent
CONFIG_FILES = {"BRACS": "bracs_medical_ssl_config.yml"}


class AttnNetGated(nn.Module):
    def __init__(self, l_dim, d_dim, n_classes):
        super().__init__()
        self.attention_a = nn.Sequential(nn.Linear(l_dim, d_dim), nn.Tanh())
        self.attention_b = nn.Sequential(nn.Linear(l_dim, d_dim), nn.Sigmoid())
        self.attention_c = nn.Linear(d_dim, n_classes)

    def forward(self, h):
        a = self.attention_a(h) * self.attention_b(h)
        return self.attention_c(a), h


class CLAMMB(nn.Module):
    def __init__(self, n_feat, n_classes, embed_dim=256, attn_dim=128, k_sample=8, subtyping=True):
        super().__init__()
        self.n_classes = n_classes
        self.k_sample = k_sample
        self.subtyping = subtyping
        self.fc = nn.Sequential(nn.Linear(n_feat, embed_dim), nn.ReLU())
        self.attention = AttnNetGated(embed_dim, attn_dim, n_classes)
        self.classifiers = nn.ModuleList([nn.Linear(embed_dim, 1) for _ in range(n_classes)])
        self.instance_classifiers = nn.ModuleList(
            [nn.Linear(embed_dim, 2) for _ in range(n_classes)]
        )
        self.instance_loss_fn = nn.CrossEntropyLoss()

    def _cluster_loss(self, attention, h, classifier, positive):
        k = min(self.k_sample, h.shape[0] // 2)
        if k < 1:
            return h.new_zeros(())
        if positive:
            pos_ids = torch.topk(attention, k, dim=0)[1]
            neg_ids = torch.topk(-attention, k, dim=0)[1]
            instances = torch.cat([h[pos_ids], h[neg_ids]], dim=0)
            targets = torch.cat(
                [
                    torch.ones(k, device=h.device, dtype=torch.long),
                    torch.zeros(k, device=h.device, dtype=torch.long),
                ]
            )
        else:
            pos_ids = torch.topk(attention, k, dim=0)[1]
            instances = h[pos_ids]
            targets = torch.zeros(k, device=h.device, dtype=torch.long)
        return self.instance_loss_fn(classifier(instances), targets)

    def forward_bag(self, x, label=None, instance_eval=False):
        h = self.fc(x)
        attention, h = self.attention(h)
        attention = torch.softmax(attention.transpose(0, 1), dim=1)
        slide = attention @ h
        logits = torch.cat([self.classifiers[c](slide[c]) for c in range(self.n_classes)])
        inst_loss = logits.new_zeros(())
        if instance_eval and label is not None:
            losses = []
            for c in range(self.n_classes):
                positive = int(label.item()) == c
                if positive or self.subtyping:
                    losses.append(
                        self._cluster_loss(
                            attention[c], h, self.instance_classifiers[c], positive
                        )
                    )
            if losses:
                inst_loss = torch.stack(losses).mean()
        return logits, inst_loss

    def forward(self, x, labels=None, mask=None, instance_eval=False):
        logits = []
        inst_losses = []
        for i in range(x.shape[0]):
            patches = x[i] if mask is None else x[i][mask[i]]
            label = None if labels is None else labels[i]
            bag_logits, inst_loss = self.forward_bag(patches, label, instance_eval)
            logits.append(bag_logits)
            inst_losses.append(inst_loss)
        return torch.stack(logits, dim=0), torch.stack(inst_losses).mean()


def class_weights(dataset, n_class):
    with h5py.File(dataset.file_path, "r") as h5:
        labels = [
            _map_label(int(h5[name].attrs["label"]), dataset.dataset_name, n_class)
            for name in dataset.data_names
        ]
    counts = torch.bincount(torch.tensor(labels), minlength=n_class).float().clamp_min(1)
    return counts.sum() / (counts * n_class), counts.long()


def train_one_epoch(model, criterion, data_loader, optimizer, device, bag_weight):
    model.train()
    total_loss = 0.0
    total_correct = 0
    total_examples = 0
    for data in data_loader:
        bag = data["input"].to(device, dtype=torch.float32)
        labels = data["label"].to(device, dtype=torch.long)
        mask = data["mask"].to(device)
        optimizer.zero_grad()
        logits, inst_loss = model(bag, labels, mask, instance_eval=True)
        bag_loss = criterion(logits, labels)
        loss = bag_weight * bag_loss + (1.0 - bag_weight) * inst_loss
        loss.backward()
        optimizer.step()
        batch_size = labels.shape[0]
        total_loss += bag_loss.item() * batch_size
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
        mask = data["mask"].to(device)
        logits, _ = model(bag, mask=mask, instance_eval=False)
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
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--bag-weight", type=float, default=0.7)
    parser.add_argument("--k-sample", type=int, default=8)
    return parser.parse_args()


def main():
    args = parse_args()
    os.chdir(ROOT)
    conf = load_config(ROOT / "config" / CONFIG_FILES["BRACS"])
    seed = int(conf.seed)
    seed_everything(seed)
    train_data, val_data, test_data = build_HDF5_feat_dataset(conf.data_dir, conf)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    weights, counts = class_weights(train_data, int(conf.n_class))
    weights = weights.to(device)
    batch_size = int(conf.B)
    num_workers = int(conf.n_worker)
    pin_memory = bool(conf.pin_memory) and torch.cuda.is_available()
    generator = torch.Generator()
    generator.manual_seed(seed)
    loader_args = dict(
        batch_size=batch_size,
        num_workers=num_workers,
        pin_memory=pin_memory,
        collate_fn=mil_collate_fn,
    )
    train_loader = DataLoader(train_data, shuffle=True, generator=generator, **loader_args)
    val_loader = DataLoader(val_data, shuffle=False, **loader_args)
    test_loader = DataLoader(test_data, shuffle=False, **loader_args)

    seed_everything(seed)
    model = CLAMMB(int(conf.D_feat), int(conf.n_class), k_sample=args.k_sample).to(device)
    criterion = nn.CrossEntropyLoss(weight=weights)
    optimizer = torch.optim.AdamW(model.parameters(), lr=conf.lr, weight_decay=conf.wd)
    num_epochs = int(conf.train_epoch if args.epochs is None else args.epochs)
    print(
        f"dataset=BRACS device={device} k_sample={args.k_sample} "
        f"bag_weight={args.bag_weight} params={sum(p.numel() for p in model.parameters())}"
    )
    print(f"class counts={counts.tolist()}")
    print(f"class weights={[round(float(w), 3) for w in weights]}")
    print(f"slides train={len(train_data)} val={len(val_data)} test={len(test_data)}")

    best_score = float("-inf")
    best_epoch = -1
    best_model_state = None
    history = []
    for epoch in range(num_epochs):
        start = time.time()
        train_metrics = train_one_epoch(
            model, criterion, train_loader, optimizer, device, args.bag_weight
        )
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
    ckpt_path = out_dir / "bracs_clam_mb_best.pt"
    result_path = out_dir / "bracs_clam_mb.json"
    torch.save(best_model_state, ckpt_path)
    payload = {
        "dataset": "BRACS",
        "model": "CLAM-MB",
        "bag_weight": args.bag_weight,
        "k_sample": args.k_sample,
        "class_weights": [float(w) for w in weights.cpu()],
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
