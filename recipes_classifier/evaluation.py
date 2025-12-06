"""
Evaluation module to generate performance metrics and visualizations
without retraining the model.
"""
import os
import json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    confusion_matrix,
    classification_report,
    roc_curve,
    auc,
    precision_recall_curve,
    precision_recall_fscore_support,
    accuracy_score,
    f1_score,
    log_loss
)
from sklearn.preprocessing import label_binarize
from tqdm import tqdm

from .config import DATA_DIR
from .data import load_datasets
from .inference import RecipeClassifier


def evaluate_model(output_dir: str = None, batch_size: int = 32, split: str = "test"):
    """
    Evaluate the trained model and generate performance metrics.
    
    Args:
        output_dir: Directory to save results. Defaults to data/evaluation_results
        batch_size: Batch size for evaluation
        split: Dataset split to evaluate on ("test", "val", or "train")
    
    Returns:
        dict: Dictionary containing predictions, labels, confidences, and metrics
    """
    if output_dir is None:
        output_dir = os.path.join(DATA_DIR, "evaluation_results")
    os.makedirs(output_dir, exist_ok=True)
    
    print("=" * 60)
    print("MODEL EVALUATION")
    print("=" * 60)
    print(f"\nLoading datasets...")
    datasets, label2id, id2label = load_datasets()
    
    if split not in datasets:
        raise ValueError(f"Split '{split}' not found. Available: {list(datasets.keys())}")
    
    dataset = datasets[split]
    print(f"Evaluating on {split} set with {len(dataset)} examples...")
    
    print("\nLoading model...")
    classifier = RecipeClassifier()
    
    print(f"\nGenerating predictions (batch size: {batch_size})...")
    all_predictions = []
    all_labels = []
    all_confidences = []
    all_probabilities = []
    
    # Process in batches with progress bar
    for i in tqdm(range(0, len(dataset), batch_size), desc="Evaluating"):
        batch = dataset[i:i+batch_size]
        texts = batch["text"]
        labels = batch["label"]
        
        for text, label in zip(texts, labels):
            # Get prediction and confidence
            category, confidence = classifier.classify(text)
            pred_id = classifier.label2id[category]
            
            # Get full probability distribution
            proba = classifier.predict_proba(text)
            
            all_predictions.append(pred_id)
            all_labels.append(label)
            all_confidences.append(confidence)
            all_probabilities.append(proba)
    
    all_predictions = np.array(all_predictions)
    all_labels = np.array(all_labels)
    all_confidences = np.array(all_confidences)
    all_probabilities = np.array(all_probabilities)
    
    print(f"\nCalculating metrics...")
    
    # Use model's label space (should be 0-8 for 9 classes)
    model_class_ids = set(classifier.id2label.keys())
    max_model_class_id = max(model_class_ids)
    n_model_classes = len(model_class_ids)
    
    # Filter out labels that aren't in the model's label space
    valid_mask = np.array([label in model_class_ids for label in all_labels])
    
    if not np.all(valid_mask):
        n_filtered = np.sum(~valid_mask)
        print(f"Warning: Filtering out {n_filtered} samples with labels outside model's label space (0-{max_model_class_id})")
        all_labels = all_labels[valid_mask]
        all_predictions = all_predictions[valid_mask]
        all_confidences = all_confidences[valid_mask]
        all_probabilities = all_probabilities[valid_mask]
    
    # Verify probabilities shape matches number of classes
    n_prob_classes = all_probabilities.shape[1]
    if n_prob_classes != n_model_classes:
        print(f"Warning: Model outputs {n_prob_classes} classes but model has {n_model_classes} classes")
        print(f"Using {n_model_classes} classes from model")
    
    # Calculate overall accuracy
    accuracy = accuracy_score(all_labels, all_predictions)
    print(f"Overall Accuracy: {accuracy:.4f}")
    
    # Generate all visualizations and reports
    print("\nGenerating visualizations...")
    # Use model's id2label to ensure we only show the 9 classes
    class_names = [classifier.id2label[i] for i in sorted(classifier.id2label.keys())]
    
    # 1. Confusion Matrix
    plot_confusion_matrix(all_labels, all_predictions, class_names, output_dir)
    
    # 2. Classification Report
    save_classification_report(all_labels, all_predictions, class_names, output_dir)
    
    # 3. Confidence Distributions
    plot_confidence_distributions(
        all_confidences, all_labels, all_predictions, classifier.id2label, output_dir
    )
    
    # 4. ROC Curves
    plot_roc_curves(all_labels, all_probabilities, classifier.id2label, output_dir)
    
    # 5. Precision-Recall Curves (with F1)
    plot_precision_recall_curves(all_labels, all_probabilities, classifier.id2label, output_dir)
    
    # 6. Threshold-based F1 and Accuracy Curves
    plot_threshold_curves(all_labels, all_probabilities, classifier.id2label, output_dir)
    
    # 7. Per-class Metrics
    save_per_class_metrics(all_labels, all_predictions, classifier.id2label, output_dir)
    
    # 8. Summary metrics (including loss)
    save_summary_metrics(all_labels, all_predictions, all_confidences, accuracy, output_dir)
    
    # 9. Calculate and save loss
    calculate_and_save_loss(all_labels, all_probabilities, classifier.id2label, output_dir)
    
    print(f"\n{'=' * 60}")
    print(f"Evaluation complete! Results saved to: {output_dir}")
    print(f"{'=' * 60}")
    
    return {
        'predictions': all_predictions,
        'labels': all_labels,
        'confidences': all_confidences,
        'probabilities': all_probabilities,
        'accuracy': accuracy,
        'output_dir': output_dir
    }


def plot_confusion_matrix(true_labels, predictions, class_names, output_dir):
    """Plot and save confusion matrix (raw counts and normalized)."""
    # Explicitly specify labels to only include the classes we care about (0 to len(class_names)-1)
    labels = list(range(len(class_names)))
    cm = confusion_matrix(true_labels, predictions, labels=labels)
    
    # Raw confusion matrix
    plt.figure(figsize=(12, 10))
    sns.heatmap(
        cm,
        annot=True,
        fmt='d',
        cmap='Blues',
        xticklabels=class_names,
        yticklabels=class_names,
        cbar_kws={'label': 'Count'}
    )
    plt.title('Confusion Matrix', fontsize=16, fontweight='bold')
    plt.ylabel('True Label', fontsize=12)
    plt.xlabel('Predicted Label', fontsize=12)
    plt.xticks(rotation=45, ha='right')
    plt.yticks(rotation=0)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'confusion_matrix.png'), dpi=300, bbox_inches='tight')
    plt.close()
    
    # Normalized confusion matrix
    cm_normalized = cm.astype('float') / (cm.sum(axis=1)[:, np.newaxis] + 1e-10)
    plt.figure(figsize=(12, 10))
    sns.heatmap(
        cm_normalized,
        annot=True,
        fmt='.2f',
        cmap='Blues',
        xticklabels=class_names,
        yticklabels=class_names,
        cbar_kws={'label': 'Proportion'}
    )
    plt.title('Normalized Confusion Matrix', fontsize=16, fontweight='bold')
    plt.ylabel('True Label', fontsize=12)
    plt.xlabel('Predicted Label', fontsize=12)
    plt.xticks(rotation=45, ha='right')
    plt.yticks(rotation=0)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'confusion_matrix_normalized.png'), dpi=300, bbox_inches='tight')
    plt.close()
    
    print("  ✓ Confusion matrices saved")


def save_classification_report(true_labels, predictions, class_names, output_dir):
    """Save classification report to JSON and text files."""
    report_dict = classification_report(
        true_labels,
        predictions,
        target_names=class_names,
        output_dict=True,
        zero_division=0
    )
    
    # Save as JSON
    with open(os.path.join(output_dir, 'classification_report.json'), 'w') as f:
        json.dump(report_dict, f, indent=2)
    
    # Save as text
    report_text = classification_report(
        true_labels,
        predictions,
        target_names=class_names,
        zero_division=0
    )
    with open(os.path.join(output_dir, 'classification_report.txt'), 'w') as f:
        f.write(report_text)
    
    print("  ✓ Classification report saved")


def plot_confidence_distributions(confidences, true_labels, predictions, id2label, output_dir):
    """Plot confidence distributions for correct and incorrect predictions."""
    correct_mask = true_labels == predictions
    
    # Overall confidence distribution
    plt.figure(figsize=(12, 6))
    plt.hist(confidences[correct_mask], bins=50, alpha=0.7, label='Correct', color='green', edgecolor='black')
    plt.hist(confidences[~correct_mask], bins=50, alpha=0.7, label='Incorrect', color='red', edgecolor='black')
    plt.xlabel('Confidence Score', fontsize=12)
    plt.ylabel('Frequency', fontsize=12)
    plt.title('Confidence Distribution: Correct vs Incorrect Predictions', fontsize=14, fontweight='bold')
    plt.legend()
    plt.grid(alpha=0.3, axis='y')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'confidence_distribution.png'), dpi=300, bbox_inches='tight')
    plt.close()
    
    # Per-class confidence distribution
    n_classes = len(id2label)
    n_cols = 3
    n_rows = (n_classes + n_cols - 1) // n_cols
    
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(18, 6 * n_rows))
    if n_rows == 1:
        axes = axes.reshape(1, -1)
    axes = axes.flatten()
    
    for class_id in range(n_classes):
        class_mask = true_labels == class_id
        if class_mask.sum() > 0:
            axes[class_id].hist(confidences[class_mask], bins=30, alpha=0.7, color='blue', edgecolor='black')
            axes[class_id].set_title(f'{id2label[class_id]} (n={class_mask.sum()})', fontsize=10)
            axes[class_id].set_xlabel('Confidence')
            axes[class_id].set_ylabel('Frequency')
            axes[class_id].grid(alpha=0.3, axis='y')
        else:
            axes[class_id].axis('off')
    
    # Hide unused subplots
    for i in range(n_classes, len(axes)):
        axes[i].axis('off')
    
    plt.suptitle('Confidence Distribution by Class', fontsize=16, fontweight='bold')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'confidence_by_class.png'), dpi=300, bbox_inches='tight')
    plt.close()
    
    print("  ✓ Confidence distribution plots saved")


def plot_roc_curves(true_labels, probabilities, id2label, output_dir):
    """Plot ROC curves for each class (one-vs-rest)."""
    # Use the actual number of classes from probabilities shape
    n_classes = probabilities.shape[1]
    
    # Use sequential class indices 0 to n_classes-1 (matching probability array)
    class_indices = list(range(n_classes))
    
    # Binarize labels for multi-class ROC
    y_true_bin = label_binarize(true_labels, classes=class_indices)
    
    # Compute ROC curve and AUC for each class
    fpr = dict()
    tpr = dict()
    roc_auc = dict()
    
    for class_idx in class_indices:
        if y_true_bin[:, class_idx].sum() > 0:  # Only if class exists in test set
            fpr[class_idx], tpr[class_idx], _ = roc_curve(
                y_true_bin[:, class_idx], probabilities[:, class_idx]
            )
            roc_auc[class_idx] = auc(fpr[class_idx], tpr[class_idx])
        else:
            roc_auc[class_idx] = 0.0
    
    # Plot all ROC curves
    plt.figure(figsize=(10, 8))
    colors = plt.cm.tab10(np.linspace(0, 1, n_classes))
    
    for class_idx, color in zip(class_indices, colors):
        if class_idx in fpr:  # Only plot if class exists
            class_name = id2label.get(class_idx, f"Class {class_idx}")
            plt.plot(fpr[class_idx], tpr[class_idx], color=color, lw=2,
                    label=f'{class_name} (AUC = {roc_auc[class_idx]:.2f})')
    
    plt.plot([0, 1], [0, 1], 'k--', lw=2, label='Random (AUC = 0.50)')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate', fontsize=12)
    plt.ylabel('True Positive Rate', fontsize=12)
    plt.title('ROC Curves (One-vs-Rest)', fontsize=14, fontweight='bold')
    plt.legend(loc="lower right", fontsize=9)
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'roc_curves.png'), dpi=300, bbox_inches='tight')
    plt.close()
    
    print("  ✓ ROC curves saved")


def save_per_class_metrics(true_labels, predictions, id2label, output_dir):
    """Calculate and save per-class metrics."""
    precision, recall, f1, support = precision_recall_fscore_support(
        true_labels, predictions, average=None, zero_division=0
    )
    
    metrics = []
    for i, class_name in sorted(id2label.items()):
        metrics.append({
            'class': class_name,
            'precision': float(precision[i]),
            'recall': float(recall[i]),
            'f1_score': float(f1[i]),
            'support': int(support[i])
        })
    
    # Save as JSON
    with open(os.path.join(output_dir, 'per_class_metrics.json'), 'w') as f:
        json.dump(metrics, f, indent=2)
    
    # Save as CSV
    df = pd.DataFrame(metrics)
    df.to_csv(os.path.join(output_dir, 'per_class_metrics.csv'), index=False)
    
    print("  ✓ Per-class metrics saved")


def save_summary_metrics(true_labels, predictions, confidences, accuracy, output_dir):
    """Save summary metrics including confidence statistics."""
    correct_mask = true_labels == predictions
    
    summary = {
        'overall_accuracy': float(accuracy),
        'total_samples': int(len(true_labels)),
        'correct_predictions': int(correct_mask.sum()),
        'incorrect_predictions': int((~correct_mask).sum()),
        'confidence_statistics': {
            'mean': float(confidences.mean()),
            'std': float(confidences.std()),
            'min': float(confidences.min()),
            'max': float(confidences.max()),
            'median': float(np.median(confidences))
        },
        'confidence_by_accuracy': {
            'correct_mean': float(confidences[correct_mask].mean()) if correct_mask.sum() > 0 else 0.0,
            'incorrect_mean': float(confidences[~correct_mask].mean()) if (~correct_mask).sum() > 0 else 0.0
        }
    }
    
    with open(os.path.join(output_dir, 'summary_metrics.json'), 'w') as f:
        json.dump(summary, f, indent=2)
    
    print("  ✓ Summary metrics saved")


def plot_precision_recall_curves(true_labels, probabilities, id2label, output_dir):
    """Plot Precision-Recall curves with F1 scores for each class."""
    n_classes = probabilities.shape[1]
    class_indices = list(range(n_classes))
    
    # Binarize labels for multi-class
    y_true_bin = label_binarize(true_labels, classes=class_indices)
    
    # Compute Precision-Recall curve and F1 for each class
    precision = dict()
    recall = dict()
    pr_auc = dict()
    f1_scores = dict()
    
    for class_idx in class_indices:
        if y_true_bin[:, class_idx].sum() > 0:
            precision[class_idx], recall[class_idx], thresholds = precision_recall_curve(
                y_true_bin[:, class_idx], probabilities[:, class_idx]
            )
            pr_auc[class_idx] = auc(recall[class_idx], precision[class_idx])
            
            # Calculate F1 at each threshold
            f1_at_threshold = []
            for p, r in zip(precision[class_idx], recall[class_idx]):
                if p + r > 0:
                    f1_at_threshold.append(2 * (p * r) / (p + r))
                else:
                    f1_at_threshold.append(0.0)
            f1_scores[class_idx] = f1_at_threshold
        else:
            pr_auc[class_idx] = 0.0
    
    # Plot Precision-Recall curves
    plt.figure(figsize=(10, 8))
    colors = plt.cm.tab10(np.linspace(0, 1, n_classes))
    
    for class_idx, color in zip(class_indices, colors):
        if class_idx in precision:
            class_name = id2label.get(class_idx, f"Class {class_idx}")
            plt.plot(recall[class_idx], precision[class_idx], color=color, lw=2,
                    label=f'{class_name} (AUC = {pr_auc[class_idx]:.2f})')
    
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('Recall', fontsize=12)
    plt.ylabel('Precision', fontsize=12)
    plt.title('Precision-Recall Curves (One-vs-Rest)', fontsize=14, fontweight='bold')
    plt.legend(loc="lower left", fontsize=9)
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'precision_recall_curves.png'), dpi=300, bbox_inches='tight')
    plt.close()
    
    # Plot F1 vs Threshold for each class
    plt.figure(figsize=(12, 8))
    for class_idx, color in zip(class_indices, colors):
        if class_idx in f1_scores:
            class_name = id2label.get(class_idx, f"Class {class_idx}")
            # Get thresholds (one less than precision/recall arrays)
            if class_idx in precision and len(precision[class_idx]) > 1:
                # Use recall as x-axis (thresholds are implicit)
                plt.plot(recall[class_idx][:-1], f1_scores[class_idx][:-1], 
                        color=color, lw=2, label=f'{class_name}')
    
    plt.xlabel('Recall', fontsize=12)
    plt.ylabel('F1 Score', fontsize=12)
    plt.title('F1 Score vs Recall (One-vs-Rest)', fontsize=14, fontweight='bold')
    plt.legend(loc="best", fontsize=9)
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'f1_curves.png'), dpi=300, bbox_inches='tight')
    plt.close()
    
    print("  ✓ Precision-Recall and F1 curves saved")


def plot_threshold_curves(true_labels, probabilities, id2label, output_dir):
    """Plot how F1 and Accuracy change with different classification thresholds."""
    n_classes = probabilities.shape[1]
    thresholds = np.linspace(0.0, 1.0, 101)  # 100 threshold points
    
    # Calculate predictions at different thresholds
    # For multi-class, we use the max probability and compare to threshold
    macro_f1_scores = []
    weighted_f1_scores = []
    accuracies = []
    
    for threshold in thresholds:
        # Get predictions: class with highest probability if it exceeds threshold, else -1 (reject)
        max_probs = probabilities.max(axis=1)
        pred_classes = probabilities.argmax(axis=1)
        
        # Only accept predictions above threshold
        predictions = np.where(max_probs >= threshold, pred_classes, -1)
        
        # Filter out rejected predictions for accuracy calculation
        valid_mask = predictions != -1
        if valid_mask.sum() > 0:
            valid_preds = predictions[valid_mask]
            valid_labels = true_labels[valid_mask]
            acc = accuracy_score(valid_labels, valid_preds)
            accuracies.append(acc)
        else:
            accuracies.append(0.0)
        
        # For F1, we need to handle rejected predictions
        # Convert -1 to a valid class for F1 calculation (use most common class as fallback)
        if valid_mask.sum() < len(predictions):
            # Replace rejected with most common class
            most_common = np.bincount(true_labels).argmax()
            predictions[predictions == -1] = most_common
        
        # Calculate F1 scores
        macro_f1 = f1_score(true_labels, predictions, average='macro', zero_division=0)
        weighted_f1 = f1_score(true_labels, predictions, average='weighted', zero_division=0)
        macro_f1_scores.append(macro_f1)
        weighted_f1_scores.append(weighted_f1)
    
    # Plot F1 and Accuracy vs Threshold
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
    
    # F1 curves
    ax1.plot(thresholds, macro_f1_scores, 'b-', lw=2, label='Macro F1')
    ax1.plot(thresholds, weighted_f1_scores, 'r-', lw=2, label='Weighted F1')
    ax1.set_xlabel('Threshold', fontsize=12)
    ax1.set_ylabel('F1 Score', fontsize=12)
    ax1.set_title('F1 Score vs Classification Threshold', fontsize=14, fontweight='bold')
    ax1.legend()
    ax1.grid(alpha=0.3)
    ax1.set_xlim([0.0, 1.0])
    
    # Accuracy curve
    ax2.plot(thresholds, accuracies, 'g-', lw=2, label='Accuracy')
    ax2.set_xlabel('Threshold', fontsize=12)
    ax2.set_ylabel('Accuracy', fontsize=12)
    ax2.set_title('Accuracy vs Classification Threshold', fontsize=14, fontweight='bold')
    ax2.legend()
    ax2.grid(alpha=0.3)
    ax2.set_xlim([0.0, 1.0])
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'threshold_curves.png'), dpi=300, bbox_inches='tight')
    plt.close()
    
    print("  ✓ Threshold-based F1 and Accuracy curves saved")


def calculate_and_save_loss(true_labels, probabilities, id2label, output_dir):
    """Calculate and save loss metrics."""
    n_classes = probabilities.shape[1]
    
    # Calculate cross-entropy loss (log loss)
    # Convert true labels to one-hot encoding
    y_true_onehot = label_binarize(true_labels, classes=list(range(n_classes)))
    
    # Calculate log loss
    test_loss = log_loss(y_true_onehot, probabilities)
    
    # Also calculate per-class loss
    per_class_loss = []
    for class_idx in range(n_classes):
        class_mask = true_labels == class_idx
        if class_mask.sum() > 0:
            class_true = y_true_onehot[class_mask, class_idx]
            class_probs = probabilities[class_mask, class_idx]
            # For binary log loss, we need probabilities for the positive class
            class_loss = log_loss(class_true, class_probs)
            per_class_loss.append({
                'class': id2label.get(class_idx, f"Class {class_idx}"),
                'loss': float(class_loss),
                'samples': int(class_mask.sum())
            })
    
    # Save loss metrics
    loss_metrics = {
        'overall_test_loss': float(test_loss),
        'per_class_loss': per_class_loss,
        'note': 'This is the loss on the test set. Training loss curves require training history which is not available without retraining.'
    }
    
    with open(os.path.join(output_dir, 'loss_metrics.json'), 'w') as f:
        json.dump(loss_metrics, f, indent=2)
    
    # Create a simple visualization (single point since we don't have training history)
    plt.figure(figsize=(10, 6))
    classes = [item['class'] for item in per_class_loss]
    losses = [item['loss'] for item in per_class_loss]
    
    plt.bar(range(len(classes)), losses, color='steelblue', edgecolor='black')
    plt.xlabel('Class', fontsize=12)
    plt.ylabel('Cross-Entropy Loss', fontsize=12)
    plt.title('Test Loss by Class', fontsize=14, fontweight='bold')
    plt.xticks(range(len(classes)), classes, rotation=45, ha='right')
    plt.grid(alpha=0.3, axis='y')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'loss_by_class.png'), dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"  ✓ Loss metrics saved (Test Loss: {test_loss:.4f})")
    print("    Note: Training loss curves require training history. Current loss is only on test set.")


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Evaluate recipe classification model")
    parser.add_argument(
        "--split",
        type=str,
        default="test",
        choices=["train", "val", "test"],
        help="Dataset split to evaluate on (default: test)"
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="Output directory for results (default: data/evaluation_results)"
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=32,
        help="Batch size for evaluation (default: 32)"
    )
    
    args = parser.parse_args()
    
    evaluate_model(
        output_dir=args.output_dir,
        batch_size=args.batch_size,
        split=args.split
    )

