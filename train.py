"""
Train a five-class diabetic retinopathy grader (EfficientNet-B0) using the
dataset's OFFICIAL train / valid / test splits and their annotations.csv files.

Expected layout (images may sit directly in each split folder or in a subfolder):
dataset/Datasets/Diabetic Retinopathy/
    train/  annotations.csv  + images
    valid/  annotations.csv  + images
    test/   annotations.csv  + images

CSV columns used: "Image name", "Retinopathy grade" (0-4).

Run:
  python train.py --epochs 5 --batch-size 8
"""
import argparse
import json
import random
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from PIL import Image
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from torch.utils.data import DataLoader, Dataset
from torchvision import models, transforms

IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp"}
IMAGE_COL = "Image name"
GRADE_COL = "Retinopathy grade"
# Standard DR grading; confirm against your Kaggle dataset's description.
CLASS_NAMES = ["0 - No DR", "1 - Mild", "2 - Moderate", "3 - Severe", "4 - Proliferative DR"]


def seed_everything(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


class DRDataset(Dataset):
    """Reads labels from <split>/annotations.csv and finds each image under <split>/."""

    def __init__(self, split_dir: Path, transform=None):
        csv_path = split_dir / "annotations.csv"
        if not csv_path.exists():
            raise SystemExit(f"Missing {csv_path}")
        df = pd.read_csv(csv_path)
        df.columns = [c.strip() for c in df.columns]
        for col in (IMAGE_COL, GRADE_COL):
            if col not in df.columns:
                raise SystemExit(f"{csv_path} has no '{col}' column. Found: {list(df.columns)}")

        # Index every image under the split folder by file name (handles an images/ subfolder).
        index = {p.name.lower(): p for p in split_dir.rglob("*") if p.suffix.lower() in IMG_EXTS}

        samples, missing = [], 0
        for name, grade in zip(df[IMAGE_COL].astype(str).str.strip(), df[GRADE_COL]):
            path = index.get(name.lower())
            if path is None:
                missing += 1
                continue
            samples.append((path, int(grade)))
        if missing:
            print(f"WARNING: {missing} images listed in {csv_path.name} not found in {split_dir.name}/")
        if not samples:
            raise SystemExit(f"No labelled images found in {split_dir}")
        bad = sorted({g for _, g in samples if g not in range(5)})
        if bad:
            raise SystemExit(f"Unexpected grade values in {split_dir.name}: {bad}")
        self.samples = samples
        self.targets = [g for _, g in samples]
        self.transform = transform

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, i):
        path, label = self.samples[i]
        img = Image.open(path).convert("RGB")
        if self.transform:
            img = self.transform(img)
        return img, label


def build_model(num_classes, pretrained=True):
    weights = models.EfficientNet_B0_Weights.DEFAULT if pretrained else None
    model = models.efficientnet_b0(weights=weights)
    model.classifier[1] = nn.Linear(model.classifier[1].in_features, num_classes)
    return model


def evaluate(model, loader, device):
    model.eval()
    y_true, y_pred = [], []
    with torch.no_grad():
        for images, labels in loader:
            y_pred.extend(model(images.to(device)).argmax(1).cpu().tolist())
            y_true.extend(labels.tolist())
    return y_true, y_pred


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", default="dataset/Datasets/Diabetic Retinopathy")
    ap.add_argument("--epochs", type=int, default=5)
    ap.add_argument("--batch-size", type=int, default=8)
    ap.add_argument("--image-size", type=int, default=224)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    seed_everything(args.seed)
    root = Path(args.data_dir)
    if not root.exists():
        raise SystemExit(f"Dataset folder not found: {root.resolve()}")

    norm = transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    train_tfms = transforms.Compose([
        transforms.Resize((args.image_size, args.image_size)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(10),
        transforms.ColorJitter(brightness=0.08, contrast=0.08),
        transforms.ToTensor(), norm,
    ])
    eval_tfms = transforms.Compose([
        transforms.Resize((args.image_size, args.image_size)),
        transforms.ToTensor(), norm,
    ])

    train_ds = DRDataset(root / "train", train_tfms)
    val_ds = DRDataset(root / "valid", eval_tfms)
    test_ds = DRDataset(root / "test", eval_tfms)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    pin = device.type == "cuda"
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=0, pin_memory=pin)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, num_workers=0, pin_memory=pin)
    test_loader = DataLoader(test_ds, batch_size=args.batch_size, shuffle=False, num_workers=0, pin_memory=pin)

    n = len(CLASS_NAMES)
    counts = np.bincount(train_ds.targets, minlength=n)
    weights = len(train_ds) / (n * np.maximum(counts, 1))
    criterion = nn.CrossEntropyLoss(weight=torch.tensor(weights, dtype=torch.float32, device=device))

    model = build_model(n).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="max", factor=0.5, patience=1)

    out_dir = Path("models")
    out_dir.mkdir(exist_ok=True)
    ckpt_path = out_dir / "retinal_efficientnet.pth"

    print(f"\nDevice: {device}")
    print(f"Split sizes: train={len(train_ds)}, valid={len(val_ds)}, test={len(test_ds)}")
    print("Train class counts:", dict(zip(CLASS_NAMES, counts.tolist())))

    best_val = -1.0
    for epoch in range(args.epochs):
        model.train()
        running = 0.0
        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad(set_to_none=True)
            loss = criterion(model(images), labels)
            loss.backward()
            optimizer.step()
            running += loss.item() * images.size(0)

        vt, vp = evaluate(model, val_loader, device)
        val_acc = accuracy_score(vt, vp)
        scheduler.step(val_acc)
        print(f"Epoch {epoch+1}/{args.epochs} | loss={running/len(train_ds):.4f} | val_accuracy={val_acc*100:.2f}%")
        if val_acc > best_val:
            best_val = val_acc
            torch.save({
                "model_state_dict": model.state_dict(),
                "class_names": CLASS_NAMES,
                "image_size": args.image_size,
                "validation_accuracy": float(val_acc),
                "architecture": "efficientnet_b0",
            }, ckpt_path)

    # Final, one-time evaluation of the best checkpoint on the official test split.
    ckpt = torch.load(ckpt_path, map_location=device)
    model.load_state_dict(ckpt["model_state_dict"])
    y_true, y_pred = evaluate(model, test_loader, device)
    labels = list(range(n))
    report = classification_report(y_true, y_pred, labels=labels, target_names=CLASS_NAMES,
                                   output_dict=True, zero_division=0)
    metrics = {
        "test_accuracy": float(accuracy_score(y_true, y_pred)),
        "macro_precision": float(report["macro avg"]["precision"]),
        "macro_recall": float(report["macro avg"]["recall"]),
        "macro_f1": float(report["macro avg"]["f1-score"]),
        "classification_report": report,
        "confusion_matrix": confusion_matrix(y_true, y_pred, labels=labels).tolist(),
        "class_names": CLASS_NAMES,
        "test_images": len(y_true),
        "best_validation_accuracy": float(best_val),
    }
    (out_dir / "metrics.json").write_text(json.dumps(metrics, indent=2))

    print("\nHELD-OUT TEST METRICS (official test split)")
    print(f"Accuracy : {metrics['test_accuracy']*100:.2f}%")
    print(f"Precision: {metrics['macro_precision']*100:.2f}% (macro)")
    print(f"Recall   : {metrics['macro_recall']*100:.2f}% (macro)")
    print(f"F1-score : {metrics['macro_f1']*100:.2f}% (macro)")
    print(classification_report(y_true, y_pred, labels=labels, target_names=CLASS_NAMES, zero_division=0))
    print(f"Saved model:   {ckpt_path.resolve()}")
    print(f"Saved metrics: {(out_dir / 'metrics.json').resolve()}")


if __name__ == "__main__":
    main()