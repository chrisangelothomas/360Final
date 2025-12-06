import os
import ast
import pandas as pd
from datasets import Dataset, DatasetDict

from .config import DATA_DIR, TRAIN_FILE, VAL_FILE, TEST_FILE


def _load_split(filename: str) -> pd.DataFrame:
    """Load one split and do basic cleaning."""
    path = os.path.join(DATA_DIR, filename)
    df = pd.read_csv(path)

    # Drop annoying unnamed columns
    df = df[[c for c in df.columns if not c.startswith("Unnamed")]]

    # Keep only columns we care about
    keep_cols = [
        "title",
        "description",
        "ingredients",
        "directions",
        "category",
    ]
    existing = [c for c in keep_cols if c in df.columns]
    df = df[existing]

    # Drop rows with missing category
    df = df[df["category"].notna()].copy()

    # Build a single text field
    def build_text(row):
        parts = []
        for col in ["title", "description", "ingredients", "directions"]:
            if col not in row:
                continue
            val = row[col]
            if isinstance(val, float):
                # NaN
                continue
            if col == "directions" and isinstance(val, str):
                # Try to parse list like "['step1', 'step2', ...]"
                try:
                    parsed = ast.literal_eval(val)
                    if isinstance(parsed, list):
                        val = " ".join(parsed)
                except Exception:
                    # If parsing fails, fall back to raw string
                    pass
            if isinstance(val, str):
                parts.append(val)
        if len(parts) > 1:
            parts[0] = parts[0] + "."  # Add period after title
        return " ".join(parts)

    df["text"] = df.apply(build_text, axis=1)

    # Drop any rows where text ended up empty
    df = df[df["text"].str.strip().astype(bool)]

    return df[["text", "category"]]


def load_datasets() -> tuple[DatasetDict, dict, dict]:
    """
    Load train/val/test splits as a HuggingFace DatasetDict and
    return (datasets, label2id, id2label).
    """
    train_df = _load_split(TRAIN_FILE)
    val_df = _load_split(VAL_FILE)
    test_df = _load_split(TEST_FILE)

    # Build label mapping from TRAIN ONLY
    unique_labels = sorted(train_df["category"].unique())
    label2id = {label: i for i, label in enumerate(unique_labels)}
    id2label = {i: label for label, i in label2id.items()}

    # Map string labels to int ids
    def apply_label_map(df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()
        df["label"] = df["category"].map(label2id)
        # Drop rows with unknown labels (shouldn't happen if splits are consistent)
        df = df[df["label"].notna()]
        df["label"] = df["label"].astype(int)
        return df[["text", "label"]]

    train_df = apply_label_map(train_df)
    val_df = apply_label_map(val_df)
    test_df = apply_label_map(test_df)

    # Build HuggingFace datasets
    train_ds = Dataset.from_pandas(train_df, preserve_index=False)
    val_ds = Dataset.from_pandas(val_df, preserve_index=False)
    test_ds = Dataset.from_pandas(test_df, preserve_index=False)

    dataset_dict = DatasetDict(
        {
            "train": train_ds,
            "val": val_ds,
            "test": test_ds,
        }
    )

    return dataset_dict, label2id, id2label
