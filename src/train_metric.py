"""
Part B: fine-tune the selected Part A model for image retrieval.
"""

import csv
import json
import random
from pathlib import Path

import numpy as np
import torch
from torch import nn, optim

from config import METRIC_BATCH_SIZE, SEED
from data import get_metric_dataloader, get_retrieval_eval_dataloader
from losses import normalize_embeddings, triplet_margin_loss
from models import build_model

SELECTED_BACKBONE = "separable_cnn"
EMBEDDING_DIM = 32
EPOCHS = 20
LEARNING_RATE = 0.0001
TRIPLET_MARGIN = 1.0

SCRIPT_DIR = Path(__file__).resolve().parent
CHECKPOINT_DIR = SCRIPT_DIR / "checkpoints"
RESULTS_DIR = SCRIPT_DIR / "results"
PART_A_CHECKPOINT = CHECKPOINT_DIR / f"part_a_{SELECTED_BACKBONE}.pt"
PART_B_CHECKPOINT = CHECKPOINT_DIR / f"part_b_{SELECTED_BACKBONE}_triplet.pt"


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def load_part_a_weights(model: nn.Module, checkpoint_path: Path, device: torch.device) -> None:
    if not checkpoint_path.exists():
        raise FileNotFoundError(
            f"Could not find {checkpoint_path}. Run Part A first, or update PART_A_CHECKPOINT."
        )

    checkpoint = torch.load(checkpoint_path, map_location=device)
    part_a_state = checkpoint["model_state_dict"]
    part_b_state = model.state_dict()

    compatible_weights = {}
    for name, weight in part_a_state.items():
        if name in part_b_state and weight.shape == part_b_state[name].shape:
            compatible_weights[name] = weight

    part_b_state.update(compatible_weights)
    model.load_state_dict(part_b_state)
    print(f"Loaded {len(compatible_weights)} compatible tensors from {checkpoint_path}")


def train_one_epoch(
    model: nn.Module,
    train_loader: torch.utils.data.DataLoader,
    optimizer: optim.Optimizer,
    device: torch.device,
) -> float:
    model.train()
    total_loss = 0.0
    total = 0

    for anchor, positive, negative in train_loader:
        anchor = anchor.to(device)
        positive = positive.to(device)
        negative = negative.to(device)

        anchor_embeddings = normalize_embeddings(model(anchor))
        positive_embeddings = normalize_embeddings(model(positive))
        negative_embeddings = normalize_embeddings(model(negative))

        loss = triplet_margin_loss(
            anchor_embeddings,
            positive_embeddings,
            negative_embeddings,
            eps=TRIPLET_MARGIN,
        )

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item() * anchor.size(0)
        total += anchor.size(0)

    return total_loss / total


def collect_embeddings(
    model: nn.Module,
    data_loader: torch.utils.data.DataLoader,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    model.eval()
    all_images = []
    all_embeddings = []
    all_labels = []

    with torch.no_grad():
        for images, labels in data_loader:
            images = images.to(device)
            embeddings = normalize_embeddings(model(images))

            all_images.append(images.cpu())
            all_embeddings.append(embeddings.cpu())
            all_labels.append(labels.cpu())

    return torch.cat(all_images, dim=0), torch.cat(all_embeddings, dim=0), torch.cat(all_labels, dim=0)


def recall_at_1(embeddings: torch.Tensor, labels: torch.Tensor) -> float:
    distances = torch.cdist(embeddings, embeddings)
    distances.fill_diagonal_(float("inf"))

    nearest_indices = distances.argmin(dim=1)
    nearest_labels = labels[nearest_indices]
    correct = (nearest_labels == labels).sum().item()
    return correct / labels.numel()


def evaluate_retrieval(
    model: nn.Module,
    eval_loader: torch.utils.data.DataLoader,
    device: torch.device,
) -> float:
    _, embeddings, labels = collect_embeddings(model, eval_loader, device)
    return recall_at_1(embeddings, labels)


def save_results(history: list[dict], best_recall: float, best_epoch: int) -> None:
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    json_path = RESULTS_DIR / "part_b_results.json"
    csv_path = RESULTS_DIR / "part_b_results.csv"

    result = {
        "selected_backbone": SELECTED_BACKBONE,
        "embedding_dim": EMBEDDING_DIM,
        "loss": "triplet",
        "best_epoch": best_epoch,
        "best_recall_at_1": best_recall,
        "checkpoint": str(PART_B_CHECKPOINT),
        "history": history,
    }

    with json_path.open("w", encoding="utf-8") as file:
        json.dump(result, file, indent=2)

    with csv_path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=["epoch", "train_loss", "recall_at_1"])
        writer.writeheader()
        writer.writerows(history)

    print(f"\nSaved Part B results to {csv_path}")
    print(f"Saved full Part B history to {json_path}")


def main() -> None:
    set_seed(SEED)
    CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    model = build_model(SELECTED_BACKBONE, embedding_dim=EMBEDDING_DIM).to(device)
    load_part_a_weights(model, PART_A_CHECKPOINT, device)

    train_loader = get_metric_dataloader(loss_name="triplet", batch_size=METRIC_BATCH_SIZE)
    eval_loader = get_retrieval_eval_dataloader()
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

    best_recall = 0.0
    best_epoch = 0
    history = []

    for epoch in range(1, EPOCHS + 1):
        train_loss = train_one_epoch(model, train_loader, optimizer, device)
        recall = evaluate_retrieval(model, eval_loader, device)

        history.append(
            {
                "epoch": epoch,
                "train_loss": train_loss,
                "recall_at_1": recall,
            }
        )

        print(f"Epoch {epoch:02d}/{EPOCHS} | train loss {train_loss:.4f} | Recall@1 {recall:.4f}")

        if recall > best_recall:
            best_recall = recall
            best_epoch = epoch
            torch.save(
                {
                    "model_name": SELECTED_BACKBONE,
                    "embedding_dim": EMBEDDING_DIM,
                    "loss": "triplet",
                    "model_state_dict": model.state_dict(),
                    "recall_at_1": recall,
                    "epoch": epoch,
                },
                PART_B_CHECKPOINT,
            )

    save_results(history, best_recall, best_epoch)
    print(f"\nBest Recall@1: {best_recall:.4f} at epoch {best_epoch}")
    print(f"Saved best retrieval checkpoint to {PART_B_CHECKPOINT}")


if __name__ == "__main__":
    main()
