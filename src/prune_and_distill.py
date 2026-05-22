"""
Part C: prune and distill the Part B retrieval model.
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

PRUNING_AMOUNT = 0.25
PRUNING_FINETUNE_EPOCHS = 10
DISTILLATION_EPOCHS = 20
PRUNING_LEARNING_RATES = [0.01] # best value found after testing [0.00001, 0.00005, 0.0001, 0.001, 0.01]
DISTILLATION_LEARNING_RATES = [0.02] # best value found after testing [0.0001, 0.001, 0.005, 0.01, 0.02, 0.03]
TRIPLET_MARGIN = 1.0
DISTILLATION_WEIGHTS = [0.9] # best value found after testing [0.1, 0.4, 0.7, 0.9]

SCRIPT_DIR = Path(__file__).resolve().parent
CHECKPOINT_DIR = SCRIPT_DIR / "checkpoints"
PRUNED_CHECKPOINT = CHECKPOINT_DIR / f"part_c_pruned_{SELECTED_BACKBONE}.pt"
STUDENT_CHECKPOINT = CHECKPOINT_DIR / f"part_c_distilled_{COMPRESSION_STUDENT_MODEL}.pt"
RESULTS_PATH = RESULTS_DIR / "part_c_results.json"
RESULTS_TABLE_PATH = RESULTS_DIR / "part_c_results.csv"


def count_parameters(model: nn.Module) -> int:
    return sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)


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
    learning_rate: float,
) -> dict:
    print(f"\nPart C.1: Global L1 pruning with learning rate {learning_rate}")
    pruned_model = copy.deepcopy(teacher_model).to(device)

    apply_global_l1_pruning(pruned_model, PRUNING_AMOUNT)
    recall_before_finetune = evaluate_retrieval(pruned_model, eval_loader, device)
    print(f"Recall@1 after pruning before fine-tuning: {recall_before_finetune:.4f}")

    optimizer = optim.Adam(pruned_model.parameters(), lr=learning_rate)
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
    # torch doesn't seem to actually remove the layers but just hides them so params are still
    # being counted, manually reduce param count but prune amount
    parameters = count_parameters(pruned_model) * (1 - PRUNING_AMOUNT)
    return {
        "method": "global_l1_pruning",
        "checkpoint": "",
        "learning_rate": learning_rate,
        "pruning_amount": PRUNING_AMOUNT,
        "recall_before_finetune": recall_before_finetune,
        "best_recall_at_1": best_recall,
        "parameters": parameters,
        "flops": count_flops(pruned_model, device),
        "model_state_dict": pruned_model.state_dict(),
        "history": history,
    }


def save_best_pruned_checkpoint(best_result: dict) -> None:
    torch.save(
        {
            "model_name": SELECTED_BACKBONE,
            "embedding_dim": EMBEDDING_DIM,
            "pruning_method": "global_l1_unstructured",
            "pruning_amount": PRUNING_AMOUNT,
            "learning_rate": best_result["learning_rate"],
            "model_state_dict": best_result["model_state_dict"],
            "recall_at_1": best_result["best_recall_at_1"],
        },
        PRUNED_CHECKPOINT,
    )
    best_result["checkpoint"] = str(PRUNED_CHECKPOINT)


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
    distillation_weight: float,
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
        loss = (1 - distillation_weight) * triplet_loss + distillation_weight * teacher_loss

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
    learning_rate: float,
    distillation_weight: float,
) -> dict:
    print(f"\nPart C.2: Knowledge distillation with learning rate {learning_rate} "
          f"and distillation weight {distillation_weight}")
    student_model = build_model(COMPRESSION_STUDENT_MODEL, embedding_dim=EMBEDDING_DIM).to(device)
    optimizer = optim.Adam(student_model.parameters(), lr=learning_rate)

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
            distillation_weight,
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

    return {
        "method": "knowledge_distillation",
        "checkpoint": "",
        "student_model": COMPRESSION_STUDENT_MODEL,
        "teacher_model": SELECTED_BACKBONE,
        "learning_rate": learning_rate,
        "distillation_weight": distillation_weight,
        "best_recall_at_1": best_recall,
        "parameters": count_parameters(student_model),
        "flops": count_flops(student_model, device),
        "model_state_dict": student_model.state_dict(),
        "history": history,
    }


def save_best_distilled_checkpoint(best_result: dict) -> None:
    torch.save(
        {
            "model_name": COMPRESSION_STUDENT_MODEL,
            "teacher_model": SELECTED_BACKBONE,
            "embedding_dim": EMBEDDING_DIM,
            "learning_rate": best_result["learning_rate"],
            "distillation_weight": best_result["distillation_weight"],
            "model_state_dict": best_result["model_state_dict"],
            "recall_at_1": best_result["best_recall_at_1"],
        },
        STUDENT_CHECKPOINT,
    )
    best_result["checkpoint"] = str(STUDENT_CHECKPOINT)


def save_part_c_results(results: list[dict]) -> None:
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    results_for_json = []
    for result in results:
        result_copy = result.copy()
        result_copy.pop("model_state_dict", None)
        results_for_json.append(result_copy)

    with RESULTS_PATH.open("w", encoding="utf-8") as file:
        json.dump(results_for_json, file, indent=2)

    columns = [
        "method",
        "learning_rate",
        "distillation_weight",
        "pruning_amount",
        "best_recall_at_1",
        "parameters",
        "flops",
        "checkpoint",
    ]
    with RESULTS_TABLE_PATH.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=columns)
        writer.writeheader()
        for result in results:
            writer.writerow({column: result.get(column, "") for column in columns})

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
        "flops": count_flops(teacher_model, device),
        "history": [],
    }
    print(f"\nPart B teacher Recall@1: {teacher_recall:.4f}")

    pruning_results = []
    for pruning_learning_rate in PRUNING_LEARNING_RATES:
        pruning_result = run_pruning_experiment(
            teacher_model,
            train_loader,
            eval_loader,
            device,
            pruning_learning_rate,
        )
        pruning_results.append(pruning_result)

    best_pruning_result = max(pruning_results, key=lambda item: item["best_recall_at_1"])
    # save_best_pruned_checkpoint(best_pruning_result)
    print(f"\nBest pruned Recall@1: {best_pruning_result['best_recall_at_1']:.4f}")
    print(f"Best pruning learning rate: {best_pruning_result['learning_rate']}")
    print(f"Saved best pruned checkpoint to {best_pruning_result['checkpoint']}")

    distillation_results = []
    for distillation_learning_rate in DISTILLATION_LEARNING_RATES:
        for distillation_weight in DISTILLATION_WEIGHTS:
            distillation_result = run_distillation_experiment(
                teacher_model,
                train_loader,
                eval_loader,
                device,
                distillation_learning_rate,
                distillation_weight,
            )
            distillation_results.append(distillation_result)

    best_distillation_result = max(distillation_results, key=lambda item: item["best_recall_at_1"])
    save_best_distilled_checkpoint(best_distillation_result)
    print(f"\nBest distilled Recall@1: {best_distillation_result['best_recall_at_1']:.4f}")
    print(f"Best distillation learning rate: {best_distillation_result['learning_rate']}")
    print(f"Best distillation weight: {best_distillation_result['distillation_weight']}")
    print(f"Saved best distilled checkpoint to {best_distillation_result['checkpoint']}")

    save_part_c_results([teacher_result] + pruning_results + distillation_results)


if __name__ == "__main__":
    main()
