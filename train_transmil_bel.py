import argparse
import copy
import json
import os
import time
from pathlib import Path

import torch
from torch import nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

from dataloader import build_HDF5_feat_dataset, mil_collate_fn
from train_transmil import SpatialSquareBag, TransMIL
from utils import classification_metrics, load_config, seed_everything


ROOT = Path(__file__).resolve().parent


class BagEmbeddingLoss(nn.Module):
    def __init__(self, n_classes, dim, margin=0.3, beta=0.95):
        super().__init__()
        self.n_classes = n_classes
        self.margin = margin
        self.beta = beta
        self.register_buffer("prototypes", torch.zeros(n_classes, dim))
        self.register_buffer("ready", torch.zeros(n_classes, dtype=torch.bool))

    def forward(self, embeddings, labels):
        losses = []
        for embedding, label in zip(embeddings, labels):
            label = int(label.item())
            current = F.normalize(embedding, dim=0)
            if not bool(self.ready[label]):
                continue
            positive = F.cosine_similarity(current, self.prototypes[label], dim=0)
            loss = 0.5 * (1.0 - positive)
            others = [c for c in range(self.n_classes) if c != label and bool(self.ready[c])]
            if others:
                negative = torch.stack(
                    [F.cosine_similarity(current, self.prototypes[c], dim=0) for c in others]
                )
                loss = loss + (negative - self.margin).clamp(min=0).sum() / (
                    2.0 * (self.n_classes - 1)
                )
            losses.append(loss)
        if not losses:
            return embeddings.sum() * 0.0
        return torch.stack(losses).mean()

    @torch.no_grad()
    def update(self, embeddings, labels):
        for embedding, label in zip(embeddings, labels):
            label = int(label.item())
            current = F.normalize(embedding.detach(), dim=0)
            if not bool(self.ready[label]):
                self.prototypes[label] = current
                self.ready[label] = True
            else:
                mixed = self.beta * self.prototypes[label] + (1.0 - self.beta) * current
                self.prototypes[label] = F.normalize(mixed, dim=0)


def train_one_epoch(model, criterion, bel, data_loader, optimizer, device, bel_weight):
    model.train()
    total_loss = 0.0
    total_correct = 0
    total_examples = 0
    for data in data_loader:
        bag = data["input"].to(device, dtype=torch.float32)
        labels = data["label"].to(device, dtype=torch.long)
        optimizer.zero_grad()
        logits, embedding = model(bag, return_embedding=True)
        ce_loss = criterion(logits, labels)
        bel_loss = bel(embedding, labels)
        loss = ce_loss + bel_weight * bel_loss
        loss.backward()
        optimizer.step()
        bel.update(embedding, labels)
        batch_size = labels.shape[0]
        total_loss += ce_loss.item() * batch_size
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
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--max-patches", type=int, default=4096)
    parser.add_argument("--bel-weight", type=float, default=0.2)
    parser.add_argument("--margin", type=float, default=0.3)
    parser.add_argument("--beta", type=float, default=0.95)
    return parser.parse_args()


def main():
    args = parse_args()
    os.chdir(ROOT)
    conf = load_config(ROOT / "config" / "bracs_medical_ssl_config.yml")
    seed = int(conf.seed)
    seed_everything(seed)
    train_base, val_data, test_data = build_HDF5_feat_dataset(conf.data_dir, conf)
    train_data = SpatialSquareBag(train_base, args.max_patches, train=True)
    val_data = SpatialSquareBag(val_data, args.max_patches, train=False)
    test_data = SpatialSquareBag(test_data, args.max_patches, train=False)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
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
    model = TransMIL(int(conf.D_feat), int(conf.n_class)).to(device)
    criterion = nn.CrossEntropyLoss()
    bel = BagEmbeddingLoss(int(conf.n_class), model.classifier.in_features, args.margin, args.beta)
    bel.to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=conf.lr, weight_decay=conf.wd)
    num_epochs = int(conf.train_epoch if args.epochs is None else args.epochs)
    print(
        f"dataset=BRACS device={device} max_patches={args.max_patches} "
        f"bel_weight={args.bel_weight} margin={args.margin} beta={args.beta}"
    )
    print(f"slides train={len(train_data)} val={len(val_data)} test={len(test_data)}")

    best_score = float("-inf")
    best_epoch = -1
    best_model_state = None
    history = []
    for epoch in range(num_epochs):
        start = time.time()
        train_metrics = train_one_epoch(
            model, criterion, bel, train_loader, optimizer, device, args.bel_weight
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
    ckpt_path = out_dir / "bracs_transmil4096_bel_best.pt"
    result_path = out_dir / "bracs_transmil4096_bel.json"
    torch.save(best_model_state, ckpt_path)
    payload = {
        "dataset": "BRACS",
        "model": "TransMIL+BEL",
        "max_patches": args.max_patches,
        "bel_weight": args.bel_weight,
        "margin": args.margin,
        "beta": args.beta,
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
