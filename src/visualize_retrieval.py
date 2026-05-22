"""
Part B retrieval visualization.
"""
import matplotlib.pyplot as plt
import torch

from data import get_retrieval_eval_dataloader
from train_metric import (
    EMBEDDING_DIM,
    PART_B_CHECKPOINT,
    RESULTS_DIR,
    SELECTED_BACKBONE,
    collect_embeddings,
)
from models import build_model


NUM_QUERIES = 5
NUM_NEIGHBORS = 5
OUTPUT_PATH = RESULTS_DIR / "part_b_retrieval_examples.png"


def load_retrieval_model(device: torch.device) -> torch.nn.Module:
    checkpoint = torch.load(PART_B_CHECKPOINT, map_location=device)
    model = build_model(SELECTED_BACKBONE, embedding_dim=EMBEDDING_DIM).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    return model


def main() -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = load_retrieval_model(device)
    eval_loader = get_retrieval_eval_dataloader()

    images, embeddings, labels = collect_embeddings(model, eval_loader, device)
    distances = torch.cdist(embeddings, embeddings)
    distances.fill_diagonal_(float("inf"))

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(NUM_QUERIES, NUM_NEIGHBORS + 1, figsize=(12, 10))

    for row in range(NUM_QUERIES):
        query_index = row
        neighbor_indices = distances[query_index].argsort()[:NUM_NEIGHBORS]
        shown_indices = [query_index] + neighbor_indices.tolist()

        for col, image_index in enumerate(shown_indices):
            ax = axes[row, col]
            ax.imshow(images[image_index].squeeze(0), cmap="gray")
            ax.axis("off")

            prefix = "query" if col == 0 else f"nn {col}"
            ax.set_title(f"{prefix}\nlabel {labels[image_index].item()}", fontsize=9)

    plt.tight_layout()
    plt.savefig(OUTPUT_PATH, dpi=150)
    plt.close()
    print(f"Saved retrieval examples to {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
