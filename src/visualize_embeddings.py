"""
Part B embedding-space visualization with t-SNE.

Notebook code reused:
- t-SNE visualization idea: Practicals solutions/deep_learning_optimization_solution.ipynb.
"""

import matplotlib.pyplot as plt
import torch
from sklearn.manifold import TSNE

from data import get_retrieval_eval_dataloader
from train_metric import (
    SEED,
    EMBEDDING_DIM,
    PART_B_CHECKPOINT,
    RESULTS_DIR,
    SELECTED_BACKBONE,
    collect_embeddings,
)
from models import build_model


OUTPUT_PATH = RESULTS_DIR / "part_b_tsne_embeddings.png"


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

    _, embeddings, labels = collect_embeddings(model, eval_loader, device)

    tsne = TSNE(n_components=2)
    reduced = tsne.fit_transform(embeddings.numpy())

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    plt.figure(figsize=(8, 6))
    scatter = plt.scatter(reduced[:, 0], reduced[:, 1], c=labels.numpy(), cmap="tab10", s=8)
    plt.colorbar(scatter, label="FashionMNIST class")
    plt.title("t-SNE embedding visualization")
    plt.tight_layout()
    plt.savefig(OUTPUT_PATH, dpi=150)
    plt.close()
    print(f"Saved embedding visualization to {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
