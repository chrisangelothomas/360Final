import os
import numpy as np
import evaluate
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    Trainer,
    TrainingArguments,
)

from .config import (
    MODEL_CHECKPOINT,
    OUTPUT_DIR,
    NUM_EPOCHS,
    BATCH_SIZE,
    LEARNING_RATE,
    WEIGHT_DECAY,
)
from .data import load_datasets


def tokenize_function(tokenizer, examples):
    return tokenizer(
        examples["text"],
        truncation=True,
        padding="max_length",  # simple and consistent
        max_length=256,
    )


def get_metrics_fn():
    accuracy_metric = evaluate.load("accuracy")
    f1_metric = evaluate.load("f1")

    def compute_metrics(eval_pred):
        logits, labels = eval_pred
        preds = np.argmax(logits, axis=-1)
        accuracy = accuracy_metric.compute(
            predictions=preds, references=labels
        )
        f1 = f1_metric.compute(
            predictions=preds, references=labels, average="macro"
        )
        return {**accuracy, **f1}

    return compute_metrics


def train_and_evaluate():
    # 1. Load datasets + label mappings
    datasets, label2id, id2label = load_datasets()

    # 2. Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(MODEL_CHECKPOINT, use_fast=True)

    # 3. Tokenize datasets
    tokenized_datasets = datasets.map(
        lambda batch: tokenize_function(tokenizer, batch),
        batched=True,
        remove_columns=["text"],
    )

    # 4. Load model
    num_labels = len(label2id)
    model = AutoModelForSequenceClassification.from_pretrained(
        MODEL_CHECKPOINT,
        num_labels=num_labels,
        id2label=id2label,
        label2id=label2id,
    )

    # 5. Training arguments
    training_args = TrainingArguments(
        output_dir=OUTPUT_DIR,
        num_train_epochs=NUM_EPOCHS,
        per_device_train_batch_size=BATCH_SIZE,
        per_device_eval_batch_size=BATCH_SIZE,
        learning_rate=LEARNING_RATE,
        weight_decay=WEIGHT_DECAY,
        evaluation_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="accuracy",
        logging_steps=50,
    )

    compute_metrics = get_metrics_fn()

    # 6. Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_datasets["train"],
        eval_dataset=tokenized_datasets["val"],
        compute_metrics=compute_metrics,
    )

    # Disable wandb if your env tries to use it
    os.environ["WANDB_DISABLED"] = "true"

    # 7. Evaluate before fine-tuning (optional)
    print("Eval BEFORE training on test set:")
    print(trainer.evaluate(tokenized_datasets["test"]))

    # 8. Train
    trainer.train()

    # 9. Evaluate after training
    print("Eval AFTER training on test set:")
    print(trainer.evaluate(tokenized_datasets["test"]))

    # 10. Save model + tokenizer + label mappings
    trainer.save_model(OUTPUT_DIR)
    tokenizer.save_pretrained(OUTPUT_DIR)

    # Save label maps for later use (pairing / inference)
    import json

    with open(os.path.join(OUTPUT_DIR, "label2id.json"), "w") as f:
        json.dump(label2id, f)
    with open(os.path.join(OUTPUT_DIR, "id2label.json"), "w") as f:
        json.dump({int(k): v for k, v in id2label.items()}, f)
