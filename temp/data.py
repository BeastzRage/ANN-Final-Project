"""
Shared data-loading utilities.

How students should use this file:
- reuse these functions and dataset classes in your own training scripts,
- keep the default seed and subset sizes so results stay comparable,
- feel free to change batch size arguments if your machine requires it.

What should normally stay fixed:
- the random seed used for subset selection,
- the subset sizes.

What can usually be changed safely:
- batch size,
- whether you add lightweight data augmentation inside your own experiments,
- DataLoader details such as `num_workers` if needed.

Typical usage examples:

Classification:
    train_loader, val_loader, test_loader = get_classification_dataloaders()

Metric learning:
    train_loader = get_metric_dataloader(loss_name="triplet")

Retrieval evaluation:
    eval_loader = get_retrieval_eval_dataloader()

This file defines the common experimental protocol. Your own model code and
training logic should live in separate files.
"""

import random
from collections import defaultdict
from typing import Sequence, Tuple

import torch
from torch.utils.data import DataLoader, Dataset, Subset
from torchvision import datasets, transforms

from config import (
    CLASSIFICATION_TRAIN_SIZE,
    CLASSIFICATION_VAL_SIZE,
    DATA_ROOT,
    METRIC_EVAL_SIZE,
    METRIC_TRAIN_SIZE,
    SEED,
)


def _build_indices_by_class(targets: Sequence[int]) -> dict:
    indices = defaultdict(list)
    for idx, label in enumerate(targets):
        indices[int(label)].append(idx)
    return indices


def _fixed_indices(dataset_size: int, take: int, seed: int) -> list[int]:
    generator = torch.Generator().manual_seed(seed)
    return torch.randperm(dataset_size, generator=generator).tolist()[:take]


def _loader_generator(seed: int) -> torch.Generator:
    return torch.Generator().manual_seed(seed)


def get_classification_dataloaders(
    batch_size: int = 64,
    train_size: int = CLASSIFICATION_TRAIN_SIZE,
    val_size: int = CLASSIFICATION_VAL_SIZE,
    seed: int = SEED,
) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """
    Build the shared classification train/validation/test loaders.

    Students should normally keep `train_size`, `val_size`, and `seed`
    unchanged. The main argument they may adjust is `batch_size`.
    """
    transform = transforms.ToTensor()
    train_dataset = datasets.FashionMNIST(root=DATA_ROOT, train=True, download=True, transform=transform)
    test_dataset = datasets.FashionMNIST(root=DATA_ROOT, train=False, download=True, transform=transform)

    subset_indices = _fixed_indices(len(train_dataset), train_size + val_size, seed)
    train_indices = subset_indices[:train_size]
    val_indices = subset_indices[train_size : train_size + val_size]

    train_loader = DataLoader(
        Subset(train_dataset, train_indices),
        batch_size=batch_size,
        shuffle=True,
        generator=_loader_generator(seed),
    )
    val_loader = DataLoader(Subset(train_dataset, val_indices), batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    return train_loader, val_loader, test_loader


class ContrastiveFashionDataset(Dataset):
    def __init__(self, train: bool = True, subset_size: int = METRIC_TRAIN_SIZE, seed: int = SEED) -> None:
        """
        Pair dataset for contrastive learning.

        The subset is chosen deterministically from the fixed seed so that
        students start from the same training pool.
        """
        transform = transforms.ToTensor()
        base = datasets.FashionMNIST(root=DATA_ROOT, train=train, download=True, transform=transform)
        if train:
            indices = _fixed_indices(len(base), subset_size, seed)
            self.base = Subset(base, indices)
            targets = [int(base.targets[idx]) for idx in indices]
        else:
            self.base = base
            targets = [int(target) for target in base.targets]
        self.targets = targets
        self.indices_by_class = _build_indices_by_class(self.targets)
        rng = random.Random(seed + (0 if train else 10_000))
        self.pairs = []
        for idx, label in enumerate(self.targets):
            same_class = rng.random() < 0.5
            if same_class:
                candidate_idx = rng.choice(self.indices_by_class[int(label)])
            else:
                negative_labels = [class_id for class_id in self.indices_by_class if class_id != int(label)]
                negative_label = rng.choice(negative_labels)
                candidate_idx = rng.choice(self.indices_by_class[negative_label])
            self.pairs.append((idx, candidate_idx))

    def __len__(self) -> int:
        return len(self.base)

    def __getitem__(self, idx: int):
        pair_a, candidate_idx = self.pairs[idx]
        image_a, label_a = self.base[pair_a]
        image_b, label_b = self.base[candidate_idx]
        target = float(label_a == label_b)
        return image_a, image_b, torch.tensor(target, dtype=torch.float32)


class TripletFashionDataset(Dataset):
    def __init__(self, train: bool = True, subset_size: int = METRIC_TRAIN_SIZE, seed: int = SEED) -> None:
        """
        Triplet dataset for metric learning.

        As above, the subset definition should normally remain fixed.
        """
        transform = transforms.ToTensor()
        base = datasets.FashionMNIST(root=DATA_ROOT, train=train, download=True, transform=transform)
        if train:
            indices = _fixed_indices(len(base), subset_size, seed)
            self.base = Subset(base, indices)
            targets = [int(base.targets[idx]) for idx in indices]
        else:
            self.base = base
            targets = [int(target) for target in base.targets]
        self.targets = targets
        self.indices_by_class = _build_indices_by_class(self.targets)
        rng = random.Random(seed + (1 if train else 10_001))
        self.triplets = []
        for idx, anchor_label in enumerate(self.targets):
            positive_idx = rng.choice(self.indices_by_class[int(anchor_label)])
            negative_labels = [class_id for class_id in self.indices_by_class if class_id != int(anchor_label)]
            negative_label = rng.choice(negative_labels)
            negative_idx = rng.choice(self.indices_by_class[negative_label])
            self.triplets.append((idx, positive_idx, negative_idx))

    def __len__(self) -> int:
        return len(self.base)

    def __getitem__(self, idx: int):
        anchor_idx, positive_idx, negative_idx = self.triplets[idx]
        anchor, _ = self.base[anchor_idx]
        positive, _ = self.base[positive_idx]
        negative, _ = self.base[negative_idx]
        return anchor, positive, negative


def get_metric_dataloader(
    loss_name: str,
    batch_size: int = 64,
    subset_size: int = METRIC_TRAIN_SIZE,
    seed: int = SEED,
) -> DataLoader:
    """
    Return the shared training DataLoader for metric learning.

    Students should usually keep `subset_size` and `seed` fixed and only
    change `batch_size` if needed.
    """
    if loss_name == "contrastive":
        dataset = ContrastiveFashionDataset(train=True, subset_size=subset_size, seed=seed)
    elif loss_name == "triplet":
        dataset = TripletFashionDataset(train=True, subset_size=subset_size, seed=seed)
    else:
        raise ValueError(f"Unsupported metric learning loss: {loss_name}")
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        generator=_loader_generator(seed),
    )


def get_retrieval_eval_dataloader(
    batch_size: int = 256,
    eval_size: int = METRIC_EVAL_SIZE,
    seed: int = SEED,
) -> DataLoader:
    """
    Return a fixed evaluation loader for retrieval metrics.

    The loader is built from a deterministic subset of the FashionMNIST test
    split so retrieval comparisons remain reproducible.
    """
    transform = transforms.ToTensor()
    test_dataset = datasets.FashionMNIST(root=DATA_ROOT, train=False, download=True, transform=transform)
    eval_indices = _fixed_indices(len(test_dataset), eval_size, seed)
    return DataLoader(Subset(test_dataset, eval_indices), batch_size=batch_size, shuffle=False)
