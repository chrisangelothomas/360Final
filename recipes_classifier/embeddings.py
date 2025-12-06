# recipes_classifier/embeddings.py
import os
import json
import numpy as np
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification, AutoConfig

from .config import OUTPUT_DIR, DATA_DIR
from .data import load_datasets


EMBEDDINGS_PATH = os.path.join(DATA_DIR, "recipe_embeddings.npz")


def _get_device():
    return "cuda" if torch.cuda.is_available() else "cpu"


def build_and_save_embeddings(batch_size: int = 32, max_length: int = 256):
    """
    Build an embedding index for all recipes (train + val + test)
    and save it to data/recipe_embeddings.npz
    """
    datasets, label2id, id2label = load_datasets()

    tokenizer = AutoTokenizer.from_pretrained(OUTPUT_DIR)
    
    # Load config first to get the correct label mappings
    config = AutoConfig.from_pretrained(OUTPUT_DIR)
    
    # The model was trained with 9 classes - explicitly set this
    # This prevents transformers from inferring num_labels from separate label files
    config.num_labels = 9
    
    # Ensure id2label and label2id match the 9-class model
    # Convert string keys to int for id2label if needed
    if hasattr(config, 'id2label') and config.id2label:
        if isinstance(list(config.id2label.keys())[0], str):
            config.id2label = {int(k): v for k, v in config.id2label.items()}
    
    # Load model with explicit num_labels to match the saved weights (9 classes)
    model = AutoModelForSequenceClassification.from_pretrained(
        OUTPUT_DIR,
        num_labels=9,
        id2label=config.id2label,
        label2id=config.label2id
    )
    model.eval()
    model.config.output_hidden_states = True  # we want hidden states for embeddings

    device = _get_device()
    model.to(device)

    all_embeddings = []
    all_texts = []
    all_labels = []
    all_categories = []
    all_splits = []

    for split_name in ["train", "val", "test"]:
        ds = datasets[split_name]
        n = len(ds)
        print(f"Building embeddings for split '{split_name}' with {n} examples...")

        for start in range(0, n, batch_size):
            end = min(start + batch_size, n)
            batch = ds[start:end]
            texts = batch["text"]
            labels = batch["label"]

            enc = tokenizer(
                texts,
                padding=True,
                truncation=True,
                max_length=max_length,
                return_tensors="pt",
            )
            enc = {k: v.to(device) for k, v in enc.items()}

            with torch.no_grad():
                outputs = model(**enc)
                # last hidden layer: (batch, seq_len, hidden_dim)
                hidden_states = outputs.hidden_states[-1]
                cls_embeddings = hidden_states[:, 0, :].cpu().numpy()

            all_embeddings.append(cls_embeddings)
            all_texts.extend(texts)
            all_labels.extend(labels)
            all_categories.extend([id2label[int(l)] for l in labels])
            all_splits.extend([split_name] * len(labels))

    embeddings = np.vstack(all_embeddings)
    all_texts = np.array(all_texts, dtype=object)
    all_labels = np.array(all_labels, dtype=int)
    all_categories = np.array(all_categories, dtype=object)
    all_splits = np.array(all_splits, dtype=object)

    os.makedirs(DATA_DIR, exist_ok=True)
    np.savez(
        EMBEDDINGS_PATH,
        embeddings=embeddings,
        texts=all_texts,
        labels=all_labels,
        categories=all_categories,
        splits=all_splits,
    )

    print(f"Saved embedding index to {EMBEDDINGS_PATH}")


if __name__ == "__main__":
    build_and_save_embeddings()
