import argparse
import copy
import json
import math
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


class SpatialSquareBag(Dataset):
    def __init__(self, base, max_patches=4096, train=False):
        self.base = base
        self.max_patches = max_patches
        self.train = train

    def __len__(self):
        return len(self.base)

    def __getitem__(self, index):
        item = dict(self.base[index])
        x = item["input"]
        coords = item.get("coords")
        n = x.shape[0]
        if n > self.max_patches:
            if self.train:
                idx = torch.randperm(n)[: self.max_patches]
            else:
                idx = (torch.arange(self.max_patches) * n) // self.max_patches
            x = x[idx]
            if coords is not None:
                coords = coords[idx]
        if coords is not None and tuple(coords.shape) == (x.shape[0], 2):
            order = torch.argsort(coords[:, 0], stable=True)
            order = order[torch.argsort(coords[order, 1], stable=True)]
            x = x[order]
        item["input"] = x.contiguous()
        item["coords"] = None
        return item


def iterative_pinv(x, n_iter=6):
    abs_x = x.abs()
    col = abs_x.sum(dim=-1)
    row = abs_x.sum(dim=-2)
    denom = (col.amax(dim=-1) * row.amax(dim=-1)).clamp_min(1e-6)
    z = x.transpose(-1, -2) / denom[..., None, None]
    eye = torch.eye(x.shape[-1], device=x.device, dtype=x.dtype)
    for _ in range(n_iter):
        xz = x @ z
        z = 0.25 * z @ (13 * eye - xz @ (15 * eye - xz @ (7 * eye - xz)))
    return z


class NystromAttention(nn.Module):
    def __init__(self, dim, heads=8, num_landmarks=64, pinv_iterations=6, dropout=0.1):
        super().__init__()
        self.heads = heads
        self.head_dim = dim // heads
        self.num_landmarks = num_landmarks
        self.pinv_iterations = pinv_iterations
        self.scale = self.head_dim ** -0.5
        self.to_qkv = nn.Linear(dim, dim * 3, bias=False)
        self.res_conv = nn.Conv2d(
            heads,
            heads,
            kernel_size=(33, 1),
            padding=(16, 0),
            groups=heads,
        )
        self.to_out = nn.Sequential(nn.Linear(dim, dim), nn.Dropout(dropout))

    def forward(self, x):
        b, n, _ = x.shape
        qkv = self.to_qkv(x).view(b, n, 3, self.heads, self.head_dim)
        q, k, v = qkv.unbind(dim=2)
        q = q.permute(0, 2, 1, 3)
        k = k.permute(0, 2, 1, 3)
        v = v.permute(0, 2, 1, 3)
        q = q * self.scale

        landmarks = min(self.num_landmarks, q.shape[-2])
        pad = (landmarks - q.shape[-2] % landmarks) % landmarks
        if pad:
            q = F.pad(q, (0, 0, 0, pad))
            k = F.pad(k, (0, 0, 0, pad))
            v = F.pad(v, (0, 0, 0, pad))
        n_pad = q.shape[-2]
        seg = n_pad // landmarks
        q_land = q.view(b, self.heads, landmarks, seg, self.head_dim).mean(dim=3)
        k_land = k.view(b, self.heads, landmarks, seg, self.head_dim).mean(dim=3)

        attn1 = torch.softmax(q @ k_land.transpose(-1, -2), dim=-1)
        attn2 = torch.softmax(q_land @ k_land.transpose(-1, -2), dim=-1)
        attn3 = torch.softmax(q_land @ k.transpose(-1, -2), dim=-1)
        eye = torch.eye(landmarks, device=x.device, dtype=x.dtype)
        attn2 = attn2 + eye * 1e-5
        out = attn1 @ iterative_pinv(attn2, self.pinv_iterations) @ (attn3 @ v)
        out = out + self.res_conv(v)
        out = out[:, :, :n, :].permute(0, 2, 1, 3).reshape(b, n, -1)
        return self.to_out(out)


class TransLayer(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.attn = NystromAttention(dim)

    def forward(self, x):
        return x + self.attn(self.norm(x))


class PPEG(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.proj = nn.Conv2d(dim, dim, 7, 1, 3, groups=dim)
        self.proj1 = nn.Conv2d(dim, dim, 5, 1, 2, groups=dim)
        self.proj2 = nn.Conv2d(dim, dim, 3, 1, 1, groups=dim)

    def forward(self, x, height, width):
        cls_token, feat = x[:, :1], x[:, 1:]
        feat = feat.transpose(1, 2).reshape(x.shape[0], -1, height, width)
        feat = self.proj(feat) + feat + self.proj1(feat) + self.proj2(feat)
        feat = feat.flatten(2).transpose(1, 2)
        return torch.cat((cls_token, feat), dim=1)


class TransMIL(nn.Module):
    def __init__(self, n_feat, n_classes, dim=512):
        super().__init__()
        self.fc1 = nn.Sequential(nn.Linear(n_feat, dim), nn.ReLU())
        self.cls_token = nn.Parameter(torch.zeros(1, 1, dim))
        nn.init.normal_(self.cls_token, std=0.02)
        self.layer1 = TransLayer(dim)
        self.layer2 = TransLayer(dim)
        self.pos_layer = PPEG(dim)
        self.norm = nn.LayerNorm(dim)
        self.classifier = nn.Linear(dim, n_classes)

    def forward(self, x, return_embedding=False):
        h = self.fc1(x)
        n = h.shape[1]
        side = math.ceil(math.sqrt(n))
        target = side * side
        if target > n:
            reps = math.ceil(target / n)
            h = h.repeat(1, reps, 1)[:, :target, :]
        cls = self.cls_token.expand(h.shape[0], -1, -1)
        h = torch.cat((cls, h), dim=1)
        h = self.layer1(h)
        h = self.pos_layer(h, side, side)
        h = self.layer2(h)
        embedding = self.norm(h)[:, 0]
        logits = self.classifier(embedding)
        if return_embedding:
            return logits, embedding
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


class FocalLoss(nn.Module):
    def __init__(self, gamma=2.0):
        super().__init__()
        self.gamma = gamma

    def forward(self, logits, targets):
        log_prob = F.log_softmax(logits, dim=-1)
        prob = log_prob.exp()
        target_log_prob = log_prob.gather(1, targets.unsqueeze(1)).squeeze(1)
        target_prob = prob.gather(1, targets.unsqueeze(1)).squeeze(1)
        return (-((1.0 - target_prob) ** self.gamma) * target_log_prob).mean()


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", default="BRACS", choices=sorted(CONFIG_FILES))
    parser.add_argument("--max-patches", type=int, default=4096)
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--focal", action="store_true")
    parser.add_argument("--focal-gamma", type=float, default=2.0)
    return parser.parse_args()


def main():
    args = parse_args()
    os.chdir(ROOT)
    conf = load_config(ROOT / "config" / CONFIG_FILES[args.dataset])
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
    criterion = FocalLoss(args.focal_gamma) if args.focal else nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=conf.lr, weight_decay=conf.wd)
    num_epochs = int(conf.train_epoch if args.epochs is None else args.epochs)
    loss_name = f"focal{args.focal_gamma:g}" if args.focal else "ce"
    print(
        f"dataset={args.dataset} device={device} max_patches={args.max_patches} "
        f"loss={loss_name} params={sum(p.numel() for p in model.parameters())}"
    )
    print(f"slides train={len(train_data)} val={len(val_data)} test={len(test_data)}")

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
    tag = f"{args.dataset.lower()}_transmil{args.max_patches}_{loss_name}"
    ckpt_path = out_dir / f"{tag}_best.pt"
    result_path = out_dir / f"{tag}.json"
    torch.save(best_model_state, ckpt_path)
    payload = {
        "dataset": args.dataset,
        "model": "TransMIL",
        "loss": loss_name,
        "max_patches": args.max_patches,
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
