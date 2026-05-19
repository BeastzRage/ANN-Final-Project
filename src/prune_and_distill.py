"""
Part C: prune and distill the Part B retrieval model.

Notebook code reused:
- Global L1 pruning pattern: Practicals solutions/main_solution.ipynb.
- Knowledge-distillation training pattern: Practicals solutions/main_solution.ipynb.
- Triplet retrieval training/evaluation style: src/train_metric.py and
  Practicals solutions/deep_learning_optimization_solution.ipynb.

Run this file from PyCharm after Part B has produced its retrieval checkpoint.
"""

import copy
import csv
import json
from pathlib import Path

import torch
import torch.nn.functional as F
import torch.nn.utils.prune as prune
from fvcore.nn import FlopCountAnalysis
from torch import nn, optim

from config import COMPRESSION_STUDENT_MODEL, METRIC_BATCH_SIZE, SEED
from data import get_metric_dataloader, get_retrieval_eval_dataloader
from losses import normalize_embeddings, triplet_margin_loss
from models import build_model
from train_metric import (
    EMBEDDING_DIM,
    PART_B_CHECKPOINT,
    RESULTS_DIR,
    SELECTED_BACKBONE,
    evaluate_retrieval,
    set_seed,
)

PRUNING_AMOUNT = 0.70
PRUNING_FINETUNE_EPOCHS = 10
DISTILLATION_EPOCHS = 20
PRUNING_LEARNING_RATE = 0.00005
DISTILLATION_LEARNING_RATE = 0.0001
TRIPLET_MARGIN = 1.0
DISTILLATION_WEIGHT = 0.9  # 0 = only triplet loss, 1 = only teacher embedding matching

SCRIPT_DIR = Path(__file__).resolve().parent
CHECKPOINT_DIR = SCRIPT_DIR / "checkpoints"
PRUNED_CHECKPOINT = CHECKPOINT_DIR / f"part_c_pruned_{SELECTED_BACKBONE}.pt"
STUDENT_CHECKPOINT = CHECKPOINT_DIR / f"part_c_distilled_{COMPRESSION_STUDENT_MODEL}.pt"
RESULTS_PATH = RESULTS_DIR / "part_c_results.json"
RESULTS_TABLE_PATH = RESULTS_DIR / "part_c_results.csv"


def count_parameters(model: nn.Module) -> int:
    return sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)


def count_nonzero_parameters(model: nn.Module) -> int:
    total = 0
    for parameter in model.parameters():
        if parameter.requires_grad:
            total += torch.count_nonzero(parameter).item()
    return total


def count_flops(model: nn.Module, device: torch.device) -> int:
    model.eval()
    example_input = torch.randn(1, 1, 28, 28, device=device)
    return int(FlopCountAnalysis(model, example_input).total())


def load_retrieval_model(model_name: str, checkpoint_path: Path, device: torch.device) -> nn.Module:
    if not checkpoint_path.exists():
        raise FileNotFoundError(
            f"Could not find {checkpoint_path}. Run Part B first, or update PART_B_CHECKPOINT."
        )

    checkpoint = torch.load(checkpoint_path, map_location=device)
    model = build_model(model_name, embedding_dim=EMBEDDING_DIM).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    return model


def modules_to_prune(model: nn.Module) -> list[tuple[nn.Module, str]]:
    modules = []
    for module in model.modules():
        if isinstance(module, (nn.Conv2d, nn.Linear)):
            modules.append((module, "weight"))
    return modules


def apply_global_l1_pruning(model: nn.Module, amount: float) -> None:
    # Reuses the global_l1_prune pattern from main_solution.ipynb.
    prune.global_unstructured(modules_to_prune(model), pruning_method=prune.L1Unstructured, amount=amount)


def remove_pruning_masks(model: nn.Module) -> None:
    for module, parameter_name in modules_to_prune(model):
        if hasattr(module, parameter_name + "_orig"):
            prune.remove(module, parameter_name)


def train_triplet_epoch(
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


def run_pruning_experiment(
    teacher_model: nn.Module,
    train_loader: torch.utils.data.DataLoader,
    eval_loader: torch.utils.data.DataLoader,
    device: torch.device,
) -> dict:
    print("\nPart C.1: Global L1 pruning")
    pruned_model = copy.deepcopy(teacher_model).to(device)

    apply_global_l1_pruning(pruned_model, PRUNING_AMOUNT)
    recall_before_finetune = evaluate_retrieval(pruned_model, eval_loader, device)
    print(f"Recall@1 after pruning before fine-tuning: {recall_before_finetune:.4f}")

    optimizer = optim.Adam(pruned_model.parameters(), lr=PRUNING_LEARNING_RATE)
    best_recall = recall_before_finetune
    best_state = copy.deepcopy(pruned_model.state_dict())
    history = []

    for epoch in range(1, PRUNING_FINETUNE_EPOCHS + 1):
        train_loss = train_triplet_epoch(pruned_model, train_loader, optimizer, device)
        recall = evaluate_retrieval(pruned_model, eval_loader, device)

        history.append({"epoch": epoch, "train_loss": train_loss, "recall_at_1": recall})
        print(f"Pruning fine-tune epoch {epoch:02d}/{PRUNING_FINETUNE_EPOCHS} | "
              f"train loss {train_loss:.4f} | Recall@1 {recall:.4f}")

        if recall > best_recall:
            best_recall = recall
            best_state = copy.deepcopy(pruned_model.state_dict())

    pruned_model.load_state_dict(best_state)
    remove_pruning_masks(pruned_model)
    torch.save(
        {
            "model_name": SELECTED_BACKBONE,
            "embedding_dim": EMBEDDING_DIM,
            "pruning_method": "global_l1_unstructured",
            "pruning_amount": PRUNING_AMOUNT,
            "model_state_dict": pruned_model.state_dict(),
            "recall_at_1": best_recall,
        },
        PRUNED_CHECKPOINT,
    )

    parameters = count_parameters(pruned_model)
    nonzero_parameters = count_nonzero_parameters(pruned_model)
    return {
        "method": "global_l1_pruning",
        "checkpoint": str(PRUNED_CHECKPOINT),
        "pruning_amount": PRUNING_AMOUNT,
        "recall_before_finetune": recall_before_finetune,
        "best_recall_at_1": best_recall,
        "parameters": parameters,
        "nonzero_parameters": nonzero_parameters,
        "nonzero_fraction": nonzero_parameters / parameters,
        "flops": count_flops(pruned_model, device),
        "history": history,
    }


def distillation_loss(
    student_embeddings: torch.Tensor,
    teacher_embeddings: torch.Tensor,
) -> torch.Tensor:
    return F.mse_loss(student_embeddings, teacher_embeddings)


def train_distillation_epoch(
    student_model: nn.Module,
    teacher_model: nn.Module,
    train_loader: torch.utils.data.DataLoader,
    optimizer: optim.Optimizer,
    device: torch.device,
) -> tuple[float, float, float]:
    student_model.train()
    teacher_model.eval()

    total_loss = 0.0
    total_triplet_loss = 0.0
    total_teacher_loss = 0.0
    total = 0

    for anchor, positive, negative in train_loader:
        anchor = anchor.to(device)
        positive = positive.to(device)
        negative = negative.to(device)

        student_anchor = normalize_embeddings(student_model(anchor))
        student_positive = normalize_embeddings(student_model(positive))
        student_negative = normalize_embeddings(student_model(negative))

        with torch.no_grad():
            teacher_anchor = normalize_embeddings(teacher_model(anchor))
            teacher_positive = normalize_embeddings(teacher_model(positive))
            teacher_negative = normalize_embeddings(teacher_model(negative))

        triplet_loss = triplet_margin_loss(
            student_anchor,
            student_positive,
            student_negative,
            eps=TRIPLET_MARGIN,
        )
        teacher_loss = (
            distillation_loss(student_anchor, teacher_anchor)
            + distillation_loss(student_positive, teacher_positive)
            + distillation_loss(student_negative, teacher_negative)
        ) / 3
        loss = (1 - DISTILLATION_WEIGHT) * triplet_loss + DISTILLATION_WEIGHT * teacher_loss

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        batch_size = anchor.size(0)
        total_loss += loss.item() * batch_size
        total_triplet_loss += triplet_loss.item() * batch_size
        total_teacher_loss += teacher_loss.item() * batch_size
        total += batch_size

    return total_loss / total, total_triplet_loss / total, total_teacher_loss / total


def run_distillation_experiment(
    teacher_model: nn.Module,
    train_loader: torch.utils.data.DataLoader,
    eval_loader: torch.utils.data.DataLoader,
    device: torch.device,
) -> dict:
    print("\nPart C.2: Knowledge distillation")
    student_model = build_model(COMPRESSION_STUDENT_MODEL, embedding_dim=EMBEDDING_DIM).to(device)
    optimizer = optim.Adam(student_model.parameters(), lr=DISTILLATION_LEARNING_RATE)

    best_recall = 0.0
    best_state = copy.deepcopy(student_model.state_dict())
    history = []

    for epoch in range(1, DISTILLATION_EPOCHS + 1):
        train_loss, triplet_loss, teacher_loss = train_distillation_epoch(
            student_model,
            teacher_model,
            train_loader,
            optimizer,
            device,
        )
        recall = evaluate_retrieval(student_model, eval_loader, device)

        history.append(
            {
                "epoch": epoch,
                "train_loss": train_loss,
                "triplet_loss": triplet_loss,
                "teacher_embedding_loss": teacher_loss,
                "recall_at_1": recall,
            }
        )
        print(f"Distillation epoch {epoch:02d}/{DISTILLATION_EPOCHS} | "
              f"train loss {train_loss:.4f} | Recall@1 {recall:.4f}")

        if recall > best_recall:
            best_recall = recall
            best_state = copy.deepcopy(student_model.state_dict())

    student_model.load_state_dict(best_state)
    torch.save(
        {
            "model_name": COMPRESSION_STUDENT_MODEL,
            "teacher_model": SELECTED_BACKBONE,
            "embedding_dim": EMBEDDING_DIM,
            "distillation_weight": DISTILLATION_WEIGHT,
            "model_state_dict": student_model.state_dict(),
            "recall_at_1": best_recall,
        },
        STUDENT_CHECKPOINT,
    )

    return {
        "method": "knowledge_distillation",
        "checkpoint": str(STUDENT_CHECKPOINT),
        "student_model": COMPRESSION_STUDENT_MODEL,
        "teacher_model": SELECTED_BACKBONE,
        "best_recall_at_1": best_recall,
        "parameters": count_parameters(student_model),
        "nonzero_parameters": count_nonzero_parameters(student_model),
        "nonzero_fraction": 1.0,
        "flops": count_flops(student_model, device),
        "history": history,
    }


def save_part_c_results(results: list[dict]) -> None:
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    with RESULTS_PATH.open("w", encoding="utf-8") as file:
        json.dump(results, file, indent=2)

    columns = [
        "method",
        "best_recall_at_1",
        "parameters",
        "nonzero_parameters",
        "nonzero_fraction",
        "flops",
        "checkpoint",
    ]
    with RESULTS_TABLE_PATH.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=columns)
        writer.writeheader()
        for result in results:
            writer.writerow({column: result[column] for column in columns})

    print(f"\nSaved Part C table to {RESULTS_TABLE_PATH}")
    print(f"Saved full Part C history to {RESULTS_PATH}")


def main() -> None:
    set_seed(SEED)
    CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    train_loader = get_metric_dataloader(loss_name="triplet", batch_size=METRIC_BATCH_SIZE)
    eval_loader = get_retrieval_eval_dataloader()

    teacher_model = load_retrieval_model(SELECTED_BACKBONE, PART_B_CHECKPOINT, device)
    teacher_model.eval()

    teacher_recall = evaluate_retrieval(teacher_model, eval_loader, device)
    teacher_result = {
        "method": "part_b_teacher",
        "checkpoint": str(PART_B_CHECKPOINT),
        "best_recall_at_1": teacher_recall,
        "parameters": count_parameters(teacher_model),
        "nonzero_parameters": count_nonzero_parameters(teacher_model),
        "nonzero_fraction": 1.0,
        "flops": count_flops(teacher_model, device),
        "history": [],
    }
    print(f"\nPart B teacher Recall@1: {teacher_recall:.4f}")

    pruning_result = run_pruning_experiment(teacher_model, train_loader, eval_loader, device)
    distillation_result = run_distillation_experiment(teacher_model, train_loader, eval_loader, device)

    save_part_c_results([teacher_result, pruning_result, distillation_result])


if __name__ == "__main__":
    main()
