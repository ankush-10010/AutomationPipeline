"""
arcface_metric_train.py — Colab Training Script
================================================

Fine-tunes a projection head on top of frozen CLIP ViT-B-32 using ArcFace loss
for few-shot, open-set character recognition across all 30 Ben 10 classes.

Why ArcFace instead of softmax classification:
  - Softmax cross-entropy needs hundreds of examples per class to fit reliable
    decision boundaries. With 6 images of Eye Guy vs 1,600 of Ben, the model
    simply ignores Eye Guy.
  - ArcFace optimizes for angular separation in embedding space. Even 6 images
    get pushed away from everything else. The loss doesn't care about class
    frequency — it cares about margin between identities.
  - Open-set rejection is free: if a frame's embedding is far from ALL
    prototypes (below threshold τ), it's rejected as "unknown/background."
    No K+1 background class needed.

Pipeline:
  1. Pre-compute CLIP ViT-B-32 embeddings for all images (backbone frozen).
  2. Train a small projection head (512 → 256 → 128) with ArcFace loss.
  3. Compute per-class prototype embeddings in the new space.
  4. Save projection head weights + prototypes + class names.

Usage on Colab:
    !pip install -q sentence-transformers torch torchvision
    !gdown --folder "1D6blD6g_kycN3Y__KjtSGj9zP_sHZaBL"
    !python arcface_metric_train.py --dataset "Ready Dataset" --epochs 40

Downloads to bring back to your laptop:
    - arcface_head.pt       (projection head weights, ~500KB)
    - prototypes.npz        (per-class prototype embeddings)
"""

import argparse
import os
import sys
import random
import numpy as np
from pathlib import Path
from collections import Counter

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from PIL import Image


# ═══════════════════════════════════════════════════════════════════════════════
# Model Components
# ═══════════════════════════════════════════════════════════════════════════════

class ProjectionHead(nn.Module):
    """Small MLP that maps frozen CLIP embeddings into a discriminative space.

    Architecture: 512 → 256 (BN + ReLU + Dropout) → 128 (L2-normalized).
    Deliberately small — we're not learning visual features from scratch,
    just rotating the existing CLIP space to separate our 30 characters.
    """

    def __init__(self, input_dim=512, hidden_dim=256, output_dim=128, dropout=0.1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, output_dim),
        )

    def forward(self, x):
        return F.normalize(self.net(x), dim=1)


class ArcFaceLoss(nn.Module):
    """Additive Angular Margin Loss (ArcFace / CosFace hybrid).

    Instead of learning a softmax weight matrix, this learns a set of
    class-center vectors on the unit hypersphere. The angular margin `m`
    is added to the angle between the embedding and its true class center,
    forcing intra-class compactness and inter-class separation.

    Args:
        num_classes: Number of identity classes.
        embedding_dim: Dimension of the normalized embeddings.
        s: Scale factor (temperature). Higher = sharper probability peaks.
        m: Angular margin in radians. 0.50 is standard ArcFace.
    """

    def __init__(self, num_classes, embedding_dim, s=30.0, m=0.50):
        super().__init__()
        self.s = s
        self.m = m
        self.num_classes = num_classes
        # Class center vectors (learnable)
        self.W = nn.Parameter(torch.randn(num_classes, embedding_dim))
        nn.init.xavier_uniform_(self.W)

    def forward(self, embeddings, labels):
        # Normalize both embeddings and class centers to unit sphere
        W = F.normalize(self.W, dim=1)
        x = F.normalize(embeddings, dim=1)

        # Cosine similarity = dot product on unit sphere
        cos_theta = x @ W.t()  # (batch, num_classes)

        # Clamp for numerical stability before acos
        cos_theta = torch.clamp(cos_theta, -1.0 + 1e-7, 1.0 - 1e-7)

        # Convert to angle, add margin to target class angle, convert back
        theta = torch.acos(cos_theta)
        target_theta = theta[torch.arange(len(labels)), labels] + self.m
        target_cos = torch.cos(target_theta)

        # Replace target class logit with margin-penalized version
        one_hot = F.one_hot(labels, num_classes=self.num_classes).float()
        logits = cos_theta * (1.0 - one_hot) + target_cos.unsqueeze(1) * one_hot

        # Scale and compute cross-entropy
        return F.cross_entropy(self.s * logits, labels)


# ═══════════════════════════════════════════════════════════════════════════════
# Dataset
# ═══════════════════════════════════════════════════════════════════════════════

class EmbeddingDataset(Dataset):
    """Pre-computed CLIP embeddings + integer labels."""

    def __init__(self, embeddings, labels):
        self.embeddings = torch.tensor(embeddings, dtype=torch.float32)
        self.labels = torch.tensor(labels, dtype=torch.long)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return self.embeddings[idx], self.labels[idx]


# ═══════════════════════════════════════════════════════════════════════════════
# Pre-compute CLIP embeddings
# ═══════════════════════════════════════════════════════════════════════════════

def precompute_embeddings(dataset_dir: Path, clip_model):
    """Encode all images in the dataset through frozen CLIP.

    Returns:
        embeddings: np.ndarray (N, 512)
        labels: np.ndarray (N,) integer labels
        class_names: list of str, index-aligned with label integers
    """
    print("\n📸 Pre-computing CLIP embeddings for all images...")
    class_dirs = sorted([d for d in dataset_dir.iterdir() if d.is_dir()])
    class_names = [d.name for d in class_dirs]

    all_embeddings = []
    all_labels = []

    for class_idx, class_dir in enumerate(class_dirs):
        images = (
            list(class_dir.glob("*.jpg"))
            + list(class_dir.glob("*.png"))
            + list(class_dir.glob("*.jpeg"))
        )
        if not images:
            print(f"  ⚠️  {class_dir.name}: 0 images, skipping")
            continue

        # Load all images for this class
        pil_images = []
        for img_path in images:
            try:
                img = Image.open(img_path).convert("RGB")
                pil_images.append(img)
            except Exception:
                continue

        if not pil_images:
            continue

        # Encode in batches
        batch_size = 64
        class_embs = []
        for i in range(0, len(pil_images), batch_size):
            batch = pil_images[i : i + batch_size]
            embs = clip_model.encode(batch, show_progress_bar=False)
            class_embs.append(embs)

        class_embs = np.vstack(class_embs)

        # Normalize
        norms = np.linalg.norm(class_embs, axis=1, keepdims=True)
        norms[norms == 0] = 1
        class_embs = class_embs / norms

        all_embeddings.append(class_embs)
        all_labels.extend([class_idx] * len(class_embs))
        print(f"  ✅ {class_dir.name:20s}: {len(class_embs)} images encoded")

    embeddings = np.vstack(all_embeddings)
    labels = np.array(all_labels)
    print(f"\n  Total: {len(labels)} embeddings across {len(class_names)} classes")
    return embeddings, labels, class_names


# ═══════════════════════════════════════════════════════════════════════════════
# Training
# ═══════════════════════════════════════════════════════════════════════════════

def train(
    embeddings: np.ndarray,
    labels: np.ndarray,
    num_classes: int,
    epochs: int = 40,
    batch_size: int = 128,
    lr: float = 1e-3,
    hidden_dim: int = 256,
    output_dim: int = 128,
    arcface_s: float = 30.0,
    arcface_m: float = 0.50,
    dropout: float = 0.1,
    device: str = "auto",
):
    """Train the projection head with ArcFace loss."""

    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"\n🔧 Training on: {device}")

    # Build dataset with oversampling for class balance
    dataset = EmbeddingDataset(embeddings, labels)

    # WeightedRandomSampler: rare classes get sampled more often
    class_counts = Counter(labels.tolist())
    sample_weights = [1.0 / class_counts[l] for l in labels.tolist()]
    sampler = WeightedRandomSampler(sample_weights, num_samples=len(dataset), replacement=True)
    loader = DataLoader(dataset, batch_size=batch_size, sampler=sampler, drop_last=True)

    # Models
    input_dim = embeddings.shape[1]  # 512 for CLIP ViT-B-32
    head = ProjectionHead(input_dim, hidden_dim, output_dim, dropout).to(device)
    arcface = ArcFaceLoss(num_classes, output_dim, s=arcface_s, m=arcface_m).to(device)

    # Optimizer — train both the projection head and ArcFace class centers
    optimizer = torch.optim.AdamW(
        list(head.parameters()) + list(arcface.parameters()),
        lr=lr,
        weight_decay=1e-4,
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    print(f"  Projection head params: {sum(p.numel() for p in head.parameters()):,}")
    print(f"  ArcFace class centers:  {num_classes} × {output_dim}")
    print(f"  Oversampling: ON (balances {min(class_counts.values())} → {max(class_counts.values())} per class)")
    print(f"  Epochs: {epochs}, Batch size: {batch_size}, LR: {lr}")
    print()

    best_loss = float("inf")
    best_state = None

    for epoch in range(1, epochs + 1):
        head.train()
        arcface.train()
        total_loss = 0
        correct = 0
        total = 0

        for emb_batch, lbl_batch in loader:
            emb_batch = emb_batch.to(device)
            lbl_batch = lbl_batch.to(device)

            projected = head(emb_batch)
            loss = arcface(projected, lbl_batch)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

            # Track accuracy using nearest class center
            with torch.no_grad():
                W = F.normalize(arcface.W, dim=1)
                cos_sim = projected @ W.t()
                preds = cos_sim.argmax(dim=1)
                correct += (preds == lbl_batch).sum().item()
                total += len(lbl_batch)

        scheduler.step()
        avg_loss = total_loss / len(loader)
        acc = correct / total * 100

        if avg_loss < best_loss:
            best_loss = avg_loss
            best_state = {
                "head": head.state_dict(),
                "arcface": arcface.state_dict(),
            }

        if epoch % 5 == 0 or epoch == 1:
            print(f"  Epoch {epoch:3d}/{epochs}  loss={avg_loss:.4f}  acc={acc:.1f}%  lr={scheduler.get_last_lr()[0]:.6f}")

    print(f"\n  Best loss: {best_loss:.4f}")

    # Restore best
    head.load_state_dict(best_state["head"])
    arcface.load_state_dict(best_state["arcface"])

    return head, arcface


# ═══════════════════════════════════════════════════════════════════════════════
# Prototype Computation
# ═══════════════════════════════════════════════════════════════════════════════

def compute_prototypes(
    head: ProjectionHead,
    embeddings: np.ndarray,
    labels: np.ndarray,
    class_names: list,
    device: str = "auto",
    sub_prototype_threshold: int = 200,
    n_sub_prototypes: int = 3,
):
    """Compute per-class prototype vectors in the projected space.

    For classes with > sub_prototype_threshold images, uses k-means to
    create multiple sub-prototypes (handles multi-modal classes like Ben
    who has different outfits/forms).

    Returns dict with:
        prototypes: np.ndarray (P, 128) where P >= num_classes
        prototype_labels: np.ndarray (P,) integer class indices
        class_names: list of str
    """
    from sklearn.cluster import KMeans

    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"

    print("\n🎯 Computing prototypes in fine-tuned embedding space...")
    head.eval()

    # Project all embeddings through the trained head
    with torch.no_grad():
        all_emb = torch.tensor(embeddings, dtype=torch.float32).to(device)
        batch_size = 256
        projected = []
        for i in range(0, len(all_emb), batch_size):
            batch = all_emb[i : i + batch_size]
            proj = head(batch).cpu().numpy()
            projected.append(proj)
        projected = np.vstack(projected)

    all_prototypes = []
    all_proto_labels = []

    for class_idx, class_name in enumerate(class_names):
        mask = labels == class_idx
        class_embs = projected[mask]

        if len(class_embs) == 0:
            continue

        if len(class_embs) > sub_prototype_threshold:
            # Multi-modal class: use k-means sub-prototypes
            n_clusters = min(n_sub_prototypes, len(class_embs))
            km = KMeans(n_clusters=n_clusters, n_init=10, random_state=42)
            km.fit(class_embs)
            centroids = km.cluster_centers_
            # Normalize centroids
            norms = np.linalg.norm(centroids, axis=1, keepdims=True)
            norms[norms == 0] = 1
            centroids = centroids / norms
            all_prototypes.append(centroids)
            all_proto_labels.extend([class_idx] * n_clusters)
            print(f"  ✅ {class_name:20s}: {n_clusters} sub-prototypes (from {len(class_embs)} images)")
        else:
            # Single mean prototype
            mean_emb = class_embs.mean(axis=0)
            mean_emb = mean_emb / (np.linalg.norm(mean_emb) or 1)
            all_prototypes.append(mean_emb.reshape(1, -1))
            all_proto_labels.append(class_idx)
            print(f"  ✅ {class_name:20s}: 1 prototype (from {len(class_embs)} images)")

    prototypes = np.vstack(all_prototypes)
    prototype_labels = np.array(all_proto_labels)
    print(f"\n  Total prototypes: {len(prototypes)} across {len(class_names)} classes")

    return {
        "prototypes": prototypes,
        "prototype_labels": prototype_labels,
        "class_names": np.array(class_names, dtype=object),
    }


# ═══════════════════════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(
        description="ArcFace Metric Learning for Ben 10 Character Recognition"
    )
    parser.add_argument("--dataset", type=str, default="Ready Dataset",
                        help="Path to the character image dataset (folders of images)")
    parser.add_argument("--epochs", type=int, default=40)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--hidden-dim", type=int, default=256)
    parser.add_argument("--output-dim", type=int, default=128,
                        help="Dimension of the projected embedding space")
    parser.add_argument("--arcface-s", type=float, default=30.0,
                        help="ArcFace scale factor")
    parser.add_argument("--arcface-m", type=float, default=0.50,
                        help="ArcFace angular margin (radians)")
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--output-dir", type=str, default=".",
                        help="Where to save arcface_head.pt and prototypes.npz")
    args = parser.parse_args()

    dataset_dir = Path(args.dataset)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if not dataset_dir.exists():
        print(f"❌ Dataset not found: {dataset_dir}")
        sys.exit(1)

    # ── Step 1: Load CLIP and pre-compute embeddings ─────────────────────
    print("=" * 60)
    print("  STEP 1: Pre-computing CLIP embeddings (backbone frozen)")
    print("=" * 60)
    from sentence_transformers import SentenceTransformer
    clip_model = SentenceTransformer("clip-ViT-B-32")

    embeddings, labels, class_names = precompute_embeddings(dataset_dir, clip_model)
    num_classes = len(class_names)

    # Save raw embeddings for reuse
    np.savez(
        str(output_dir / "clip_raw_embeddings.npz"),
        embeddings=embeddings,
        labels=labels,
        class_names=np.array(class_names, dtype=object),
    )
    print(f"  Saved raw embeddings to {output_dir / 'clip_raw_embeddings.npz'}")

    # ── Step 2: Train projection head with ArcFace ───────────────────────
    print("\n" + "=" * 60)
    print("  STEP 2: Training ArcFace Projection Head")
    print("=" * 60)

    head, arcface = train(
        embeddings, labels, num_classes,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        hidden_dim=args.hidden_dim,
        output_dim=args.output_dim,
        arcface_s=args.arcface_s,
        arcface_m=args.arcface_m,
        dropout=args.dropout,
    )

    # Save projection head
    head_path = output_dir / "arcface_head.pt"
    torch.save({
        "state_dict": head.state_dict(),
        "input_dim": embeddings.shape[1],
        "hidden_dim": args.hidden_dim,
        "output_dim": args.output_dim,
        "dropout": args.dropout,
    }, str(head_path))
    print(f"\n  💾 Saved projection head to {head_path}")

    # ── Step 3: Compute prototypes ───────────────────────────────────────
    print("\n" + "=" * 60)
    print("  STEP 3: Computing Per-Class Prototypes")
    print("=" * 60)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    proto_data = compute_prototypes(
        head, embeddings, labels, class_names, device=device
    )

    proto_path = output_dir / "prototypes.npz"
    np.savez(str(proto_path), **proto_data)
    print(f"\n  💾 Saved prototypes to {proto_path}")

    # ── Summary ──────────────────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("  🎉 TRAINING COMPLETE!")
    print("=" * 60)
    print(f"\n  Files to download and bring to your laptop:")
    print(f"    1. {head_path}        (~500KB, projection head weights)")
    print(f"    2. {proto_path}       (~50KB, prototype embeddings)")
    print(f"\n  Place them in your project root and run:")
    print(f"    python scripts/prototype_inference.py")


if __name__ == "__main__":
    main()
