"""
Part B: fine-tune the selected Part A model for image retrieval.
"""

import csv
import copy
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
LEARNING_RATES = [0.001] # best value found after testing [0.001, 0.005,0.01, 0.02, 0.03, 0.04, 0.05]
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


def train_for_learning_rate(
    learning_rate: float,
    train_loader: torch.utils.data.DataLoader,
    eval_loader: torch.utils.data.DataLoader,
    device: torch.device,
) -> dict:
    model = build_model(SELECTED_BACKBONE, embedding_dim=EMBEDDING_DIM).to(device)
    load_part_a_weights(model, PART_A_CHECKPOINT, device)

    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    best_recall = 0.0
    best_epoch = 0
    best_state = copy.deepcopy(model.state_dict())
    history = []

    print(f"\nTraining retrieval model with learning rate {learning_rate}")

    for epoch in range(1, EPOCHS + 1):
        train_loss = train_one_epoch(model, train_loader, optimizer, device)
        recall = evaluate_retrieval(model, eval_loader, device)

        history.append(
            {
                "epoch": epoch,
                "learning_rate": learning_rate,
                "train_loss": train_loss,
                "recall_at_1": recall,
            }
        )

        print(f"Epoch {epoch:02d}/{EPOCHS} | train loss {train_loss:.4f} | Recall@1 {recall:.4f}")

        if recall > best_recall:
            best_recall = recall
            best_epoch = epoch
            best_state = copy.deepcopy(model.state_dict())

    return {
        "selected_backbone": SELECTED_BACKBONE,
        "embedding_dim": EMBEDDING_DIM,
        "learning_rate": learning_rate,
        "loss": "triplet",
        "best_epoch": best_epoch,
        "best_recall_at_1": best_recall,
        "checkpoint": "",
        "model_state_dict": best_state,
        "history": history,
    }


def save_best_checkpoint(best_result: dict) -> None:
    torch.save(
        {
            "model_name": SELECTED_BACKBONE,
            "embedding_dim": EMBEDDING_DIM,
            "learning_rate": best_result["learning_rate"],
            "loss": "triplet",
            "model_state_dict": best_result["model_state_dict"],
            "recall_at_1": best_result["best_recall_at_1"],
            "epoch": best_result["best_epoch"],
        },
        PART_B_CHECKPOINT,
    )
    best_result["checkpoint"] = str(PART_B_CHECKPOINT)


def save_results(results: list[dict], best_result: dict) -> None:
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    json_path = RESULTS_DIR / "part_b_results.json"
    csv_path = RESULTS_DIR / "part_b_results.csv"

    results_for_json = []
    for result in results:
        result_copy = result.copy()
        result_copy.pop("model_state_dict")
        results_for_json.append(result_copy)

    best_result_for_json = best_result.copy()
    best_result_for_json.pop("model_state_dict")

    with json_path.open("w", encoding="utf-8") as file:
        json.dump(
            {
                "best_result": best_result_for_json,
                "all_results": results_for_json,
            },
            file,
            indent=2,
        )

    with csv_path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(
            file,
            fieldnames=["learning_rate", "best_epoch", "best_recall_at_1", "checkpoint"],
        )
        writer.writeheader()
        for result in results:
            writer.writerow(
                {
                    "learning_rate": result["learning_rate"],
                    "best_epoch": result["best_epoch"],
                    "best_recall_at_1": result["best_recall_at_1"],
                    "checkpoint": result["checkpoint"],
                }
            )

    print(f"\nSaved Part B results to {csv_path}")
    print(f"Saved full Part B history to {json_path}")


def main() -> None:
    set_seed(SEED)
    CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    train_loader = get_metric_dataloader(loss_name="triplet", batch_size=METRIC_BATCH_SIZE)
    eval_loader = get_retrieval_eval_dataloader()

    results = []
    for learning_rate in LEARNING_RATES:
        result = train_for_learning_rate(learning_rate, train_loader, eval_loader, device)
        results.append(result)

    best_result = max(results, key=lambda item: item["best_recall_at_1"])
    save_best_checkpoint(best_result)

    save_results(results, best_result)
    print(f"\nBest Recall@1: {best_result['best_recall_at_1']:.4f} at epoch {best_result['best_epoch']}")
    print(f"Best learning rate: {best_result['learning_rate']}")
    print(f"Saved best retrieval checkpoint to {PART_B_CHECKPOINT}")


if __name__ == "__main__":
    main()
