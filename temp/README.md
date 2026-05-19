# Suggested Source Layout

Students may choose their own structure, but a clean submission would typically include files such as:

- `config.py`
- `data.py`
- `models.py`
- `losses.py`
- `train_classifier.py`
- `train_metric.py`
- `prune_and_distill.py`
- `run_project.py`

`config.py`, `data.py`, and `models.py` are already provided and should normally be reused so that everyone follows the same data and model protocol.

The staged implementation should stay coherent:

- `train_classifier.py` handles Part A using the provided `StandardCNN` and your completed `SeparableCNN`.
- `train_metric.py` should load the selected Part A checkpoint and fine-tune that model for retrieval in Part B.
- `visualize_retrieval.py` should save qualitative nearest-neighbor retrieval examples for Part B.
- `visualize_embeddings.py` or an equivalent helper should save an embedding-space visualization for Part B.
- `prune_and_distill.py` should analyze both pruning and knowledge distillation on the Part B retrieval model for Part C.

You do not need to follow this exact layout for the rest if your structure is clean and reproducible.
