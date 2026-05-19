"""
Shared experiment configuration

Keep these fixed across all students to ensure comparability of results:
- `SEED`
- `PART_A_MODELS`
- `COMPRESSION_STUDENT_MODEL`
- `CLASSIFICATION_TRAIN_SIZE`
- `CLASSIFICATION_VAL_SIZE`
- `METRIC_TRAIN_SIZE`
- `METRIC_EVAL_SIZE`

These are normally safe to change without breaking comparability:
- batch sizes
- learning rates in your own training scripts
- number of epochs in your own training scripts

If you change any fixed setting, you should clearly justify it in your report.
"""

DATA_ROOT = "./data"

# Keep this fixed so all students sample the same subset and split.
SEED = 42

# Fixed model protocol:
# use these exact model choices so architecture changes do not dominate the
# comparison or make results hard to reproduce across students.
PART_A_MODELS = ("cnn", "separable_cnn")
COMPRESSION_STUDENT_MODEL = "compact_separable_cnn"

# Classification protocol:
# keep the train/validation sizes fixed for fair comparison across submissions.
# the batch size may be changed if needed for memory or runtime reasons.
CLASSIFICATION_BATCH_SIZE = 64
CLASSIFICATION_TRAIN_SIZE = 12000
CLASSIFICATION_VAL_SIZE = 2000

# Metric-learning protocol:
# keep the train and evaluation subset sizes fixed so retrieval experiments are
# comparable.
# the batch size may be changed if needed.
METRIC_BATCH_SIZE = 64
METRIC_TRAIN_SIZE = 10000
METRIC_EVAL_SIZE = 2000
