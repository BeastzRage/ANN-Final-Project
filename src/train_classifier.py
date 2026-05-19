"""
Part A: train and compare the two provided classification backbones.
"""

import csv
import json
import random
from pathlib import Path

import numpy as np
import torch
from fvcore.nn import FlopCountAnalysis
from sklearn.metrics import f1_score
from torch import nn, optim

from config import CLASSIFICATION_BATCH_SIZE, PART_A_MODELS, SEED
from data import get_classification_dataloaders
from models import build_model

EPOCHS = 20
LEARNING_RATE = 0.001
OUTPUT_DIR = Path("results")
CHECKPOINT_DIR = Path("checkpoints")


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def count_parameters(model: nn.Module) -> int:
    return sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)


def count_flops(model: nn.Module, device: torch.device) -> int:
    model.eval()
    example_input = torch.randn(1, 1, 28, 28, device=device)
    return int(FlopCountAnalysis(model, example_input).total())


def train_one_epoch(model: nn.Module, train_loader: torch.utils.data.DataLoader, criterion: nn.Module,
                    optimizer: optim.Optimizer, device: torch.device) -> tuple[float, float]:
    model.train()
    total_loss = 0.0
    correct = 0
    total = 0

    for images, labels in train_loader:
        images = images.to(device)
        labels = labels.to(device)

        outputs = model(images)
        loss = criterion(outputs, labels)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item() * images.size(0)
        predictions = outputs.argmax(dim=1)
        correct += (predictions == labels).sum().item()
        total += labels.size(0)

    return total_loss / total, correct / total


def evaluate(model: nn.Module, data_loader: torch.utils.data.DataLoader, criterion: nn.Module, device: torch.device) -> tuple[float, float, float]:
    model.eval()
    total_loss = 0.0
    correct = 0
    total = 0
    all_labels = []
    all_predictions = []

    with torch.no_grad():
        for images, labels in data_loader:
            images = images.to(device)
            labels = labels.to(device)

            outputs = model(images)
            loss = criterion(outputs, labels)

            predictions = outputs.argmax(dim=1)
            total_loss += loss.item() * images.size(0)
            correct += (predictions == labels).sum().item()
            total += labels.size(0)

            all_labels.extend(labels.cpu().tolist())
            all_predictions.extend(predictions.cpu().tolist())

    macro_f1 = f1_score(all_labels, all_predictions, average="macro")
    return total_loss / total, correct / total, macro_f1


def train_model(
    model_name: str,
    train_loader: torch.utils.data.DataLoader,
    val_loader: torch.utils.data.DataLoader,
    test_loader: torch.utils.data.DataLoader,
    device: torch.device,
) -> dict:
    model = build_model(model_name).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

    print(f"\nTraining {model_name}")
    print(f"Parameters: {count_parameters(model):,}")
    print(f"FLOPs for one 28x28 input: {count_flops(model, device):,}")

    best_val_accuracy = 0.0
    best_checkpoint = CHECKPOINT_DIR / f"part_a_{model_name}.pt"

    history = []
    for epoch in range(1, EPOCHS + 1):
        train_loss, train_accuracy = train_one_epoch(model, train_loader, criterion, optimizer, device)
        val_loss, val_accuracy, val_macro_f1 = evaluate(model, val_loader, criterion, device)

        history.append(
            {
                "epoch": epoch,
                "train_loss": train_loss,
                "train_accuracy": train_accuracy,
                "val_loss": val_loss,
                "val_accuracy": val_accuracy,
                "val_macro_f1": val_macro_f1,
            }
        )

        print(
            f"Epoch {epoch:02d}/{EPOCHS} | "
            f"train loss {train_loss:.4f}, train acc {train_accuracy:.4f} | "
            f"val loss {val_loss:.4f}, val acc {val_accuracy:.4f}, val macro F1 {val_macro_f1:.4f}"
        )

        if val_accuracy > best_val_accuracy:
            best_val_accuracy = val_accuracy
            torch.save(
                {
                    "model_name": model_name,
                    "model_state_dict": model.state_dict(),
                    "val_accuracy": val_accuracy,
                    "epoch": epoch,
                },
                best_checkpoint,
            )

    checkpoint = torch.load(best_checkpoint, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])
    test_loss, test_accuracy, test_macro_f1 = evaluate(model, test_loader, criterion, device)

    result = {
        "model": model_name,
        "best_epoch": checkpoint["epoch"],
        "best_val_accuracy": checkpoint["val_accuracy"],
        "test_loss": test_loss,
        "test_accuracy": test_accuracy,
        "test_macro_f1": test_macro_f1,
        "parameters": count_parameters(model),
        "flops": count_flops(model, device),
        "checkpoint": str(best_checkpoint),
        "history": history,
    }
    return result


def save_results(results: list[dict], output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)

    json_path = output_dir / "part_a_results.json"
    csv_path = output_dir / "part_a_results.csv"

    with json_path.open("w", encoding="utf-8") as file:
        json.dump(results, file, indent=2)

    columns = [
        "model",
        "best_epoch",
        "best_val_accuracy",
        "test_accuracy",
        "test_macro_f1",
        "parameters",
        "flops",
        "checkpoint",
    ]
    with csv_path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=columns)
        writer.writeheader()
        for result in results:
            writer.writerow({column: result[column] for column in columns})

    print(f"\nSaved results to {csv_path}")
    print(f"Saved full training history to {json_path}")


def main() -> None:
    CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)

    set_seed(SEED)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    train_loader, val_loader, test_loader = get_classification_dataloaders(batch_size=CLASSIFICATION_BATCH_SIZE)

    results = []
    for model_name in PART_A_MODELS:
        result = train_model(model_name, train_loader, val_loader, test_loader, device)
        results.append(result)

    save_results(results, OUTPUT_DIR)

    selected = max(results, key=lambda item: item["best_val_accuracy"])
    print(f"\nSuggested Part A backbone for Part B: {selected['model']}")


if __name__ == "__main__":
    main()
