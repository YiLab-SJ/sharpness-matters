"""
Utility file for evaluation of classification performance
"""

import os
import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset
from scipy.stats import norm
import scipy.stats as st
from sklearn.metrics import accuracy_score, roc_auc_score, f1_score, confusion_matrix
from tqdm import tqdm
from pathlib import Path
from typing import Dict, List, Tuple, Union, Type
from sharpness_matters.config.load_config import cfg

HOME = Path(__file__).resolve().parent.parent
CHECKPOINT_DIR = cfg.pneumonia.model.checkpoint_dir


def auroc_confidence_interval(
    y_true: np.ndarray, y_scores: np.ndarray, confidence=0.95, n_bootstrap=2000
):
    """
    Compute the confidence interval for AUROC using bootstrapping.
    Args:
        y_true (np.ndarray): Ground truth binary labels (0 or 1).
        y_scores (np.ndarray): Predicted scores (probabilities or raw scores).
        confidence (float): Confidence level (default: 0.95).
        n_bootstrap (int): Number of bootstrap samples (default: 1000).
    Returns:
        (float, float): Lower and upper bounds of the AUROC confidence interval.
    """
    bootstrapped_aurocs = []
    n = len(y_true)
    # Perform bootstrapping
    for _ in range(n_bootstrap):
        indices = np.random.choice(np.arange(n), size=n, replace=True)
        if len(np.unique(y_true[indices])) < 2:
            continue  # Skip if there's only one class in the resampled data
        bootstrapped_aurocs.append(roc_auc_score(y_true[indices], y_scores[indices]))
    # Compute confidence interval from percentiles
    lower_percentile = (1 - confidence) / 2 * 100
    upper_percentile = (1 + confidence) / 2 * 100
    ci_lower = np.percentile(bootstrapped_aurocs, lower_percentile)
    ci_upper = np.percentile(bootstrapped_aurocs, upper_percentile)

    return ci_lower, ci_upper


def delong_roc_test(y_true: np.ndarray, scores1: np.ndarray, scores2: np.ndarray):
    """
    Perform DeLong's test to compare two AUROCs.
    Parameters:
    y_true: Ground truth binary labels
    scores1: Predicted scores from the first classifier
    scores2: Predicted scores from the second classifier
    Returns:
    z_stat: Z-statistic
    p_value: P-value for the significance test
    """
    # Compute AUROCs
    auc1 = roc_auc_score(y_true, scores1)
    auc2 = roc_auc_score(y_true, scores2)

    def compute_delong_covariance(scores, y_true):
        order = np.argsort(scores)[::-1]
        scores = scores[order]
        y_true = y_true[order]

        unique_scores = np.unique(scores)
        m, n = np.sum(y_true == 1), np.sum(y_true == 0)
        auc_partial_sums = np.zeros_like(unique_scores)

        for idx, score in enumerate(unique_scores):
            y_positive = np.sum(y_true[scores == score] == 1)
            auc_partial_sums[idx] = y_positive / (m * n)

        covariance = np.cov(auc_partial_sums)
        return covariance

    # Compute covariance matrices
    cov1 = compute_delong_covariance(scores1, y_true)
    cov2 = compute_delong_covariance(scores2, y_true)
    diff = auc1 - auc2
    var = cov1 + cov2
    z_stat = diff / np.sqrt(var)
    p_value = 2 * (1 - norm.cdf(abs(z_stat)))
    return z_stat, p_value


# Evaluation function for a single fold
def evaluate_fold(fold_number: int, val_loader: DataLoader, model: torch.nn.Module):
    """
    Evaluate a single cross-validation fold, collecting model sigmoid outputs and labels.
    Args:
        fold_number (int): Index of the current fold (for logging only).
        val_loader (DataLoader): Validation data loader yielding batches with keys "data" and "label".
        model (torch.nn.Module): Trained (or current) model returning raw logits of shape (B, 1) or (B,).
    Returns:
        dict: {
            "logits": np.ndarray of shape (N,), sigmoid probabilities for each sample,
            "labels": np.ndarray of shape (N,), ground-truth labels
        }
    """

    all_labels = []
    all_logits = []
    # Evaluate on validation data
    with torch.no_grad():
        for batch in tqdm(val_loader):
            pixel_values = batch["data"].to(
                "cuda" if torch.cuda.is_available() else "cpu"
            )
            labels = batch["label"].to("cuda" if torch.cuda.is_available() else "cpu")
            logits = model(pixel_values).squeeze(1)
            preds = torch.sigmoid(logits).cpu().numpy()
            all_labels.extend(labels.cpu().numpy())
            all_logits.append(preds)
    # Stack logits for this fold
    all_logits = np.concatenate(all_logits, axis=0)
    print(f"Fold {fold_number} - Collected {len(all_logits)} logits")
    return {"logits": all_logits, "labels": np.array(all_labels)}


def eval_res(
    dataset: Dataset,
    img_size: int,
    model_name: str,
    model_class: Type[torch.nn.Module],
    logger,
):
    """
    Evaluate a 5-fold cross-validated classification model and aggregate ensemble metrics.
    Loads each fold's best checkpoint, collects logits and labels, computes per-fold
    (acc, auc, f1) and ensemble metrics by averaging logits across folds.
    Parameters
    ---------
    dataset : torch.utils.data.Dataset
        Validation dataset (same for all folds).
    img_size : int
        Image size suffix used in checkpoint path construction.
    model_name : str
        Base model name used in directory structure.
    model_class : Type[torch.nn.Module]
        LightningModule class with `load_from_checkpoint`.
    logger : logging.Logger
        Logger for debug output.
    Returns
    -------
    all_labels : np.ndarray
        Ground-truth labels (shared across folds).
    all_logits_ensemble : np.ndarray
        Averaged logits across the 5 folds (ensemble output).
    ensemble_metrics : dict
        Dict with keys: 'acc', 'auc', 'f1', 'auroc_ci'.
    """
    k_folds = 5
    results = []
    val_loader = DataLoader(dataset, batch_size=32, num_workers=8, shuffle=False)
    # Iterate through folds
    all_logits_ensemble = []
    all_labels = None
    metrics = {"acc": [], "auc": [], "f1": []}
    print(f"Evaluating {model_name} model with image size {img_size}...")
    # Compute fold-wise metrics
    for fold in range(k_folds):
        print(f"\nEvaluating Fold {fold + 1}/{k_folds}...")
        # Path to the best model checkpoint for this fold
        name = os.listdir(
            f"{CHECKPOINT_DIR}/{model_name}/{model_name}_{img_size}_logs/lightning_logs/version_{fold}/checkpoints/"
        )[0]
        checkpoint_path = f"{CHECKPOINT_DIR}/{model_name}/{model_name}_{img_size}_logs/lightning_logs/version_{fold}/checkpoints/{name}"
        if not os.path.exists(checkpoint_path):
            print(f"Checkpoint not found for fold {fold}: {checkpoint_path}")
            continue
        # Evaluate this fold
        model = model_class.load_from_checkpoint(checkpoint_path)
        model.eval()
        model.to("cuda" if torch.cuda.is_available() else "cpu")
        fold_results = evaluate_fold(fold, val_loader, model)
        all_logits_ensemble.append(fold_results["logits"])
        fold_preds = (fold_results["logits"] > 0.5).astype(int)
        acc = accuracy_score(fold_results["labels"], fold_preds)
        auc = roc_auc_score(fold_results["labels"], fold_results["logits"])
        f1 = f1_score(fold_results["labels"], fold_preds)
        # Store labels once (labels are the same for all folds)
        if all_labels is None:
            all_labels = fold_results["labels"]
        metrics["acc"].append(acc)
        metrics["auc"].append(auc)
        metrics["f1"].append(f1)
    # Compute average metrics from all folds
    for key in metrics:
        metrics[key] = np.round(np.mean(metrics[key]), 4)
    # Combine logits from all folds (average for ensemble)
    all_logits_ensemble = np.mean(all_logits_ensemble, axis=0)
    # Compute ensemble metrics
    all_preds = (all_logits_ensemble > 0.5).astype(int)
    accuracy = accuracy_score(all_labels, all_preds)
    auc = roc_auc_score(all_labels, all_logits_ensemble)
    f1 = f1_score(all_labels, all_preds)
    cm = confusion_matrix(all_labels, all_preds)
    ci = auroc_confidence_interval(all_labels, all_logits_ensemble)
    # Save metrics as dict for ease of reporting csv
    ensemble_metrics = {}
    ensemble_metrics["acc"] = np.round(accuracy, 4)
    ensemble_metrics["auc"] = np.round(auc, 4)
    ensemble_metrics["f1"] = np.round(f1, 4)
    ensemble_metrics["auroc_ci"] = np.round(ci, 4)
    # Log metrics
    logger.debug("\nEnsemble Evaluation Results:")
    logger.debug(f"Accuracy: {accuracy:.4f}")
    logger.debug(f"AUC: {auc:.4f}")
    logger.debug(f"AUROC CI: {ci}")
    logger.debug(f"F1: {f1:.4f}")
    logger.debug(f"Confusion Matrix: {cm}")
    return all_labels, all_logits_ensemble, ensemble_metrics
