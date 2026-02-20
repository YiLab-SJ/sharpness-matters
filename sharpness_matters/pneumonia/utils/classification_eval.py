"""
Utility functions for evaluating classification performance
"""

import os
import glob
import numpy as np
import torch
from tqdm import tqdm
import pytorch_lightning as pl
from torch.utils.data import DataLoader
from sklearn.metrics import accuracy_score, roc_auc_score, f1_score
from scipy.stats import norm
from typing import Dict, List, Tuple, Union, Optional, Any
from pathlib import Path

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


def delong_roc_test(
    y_true: np.ndarray, scores1: np.ndarray, scores2: np.ndarray
) -> Tuple[float, float]:
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
def evaluate_fold(
    model_class: pl.LightningModule,
    val_loader: DataLoader,
    checkpoint_path: str,
    device: torch.device = torch.device("cuda" if torch.cuda.is_available() else "cpu"),
) -> Dict[str, Any]:
    """
    Evaluate a single cross-validation fold by loading a model checkpoint and generating
    prediction probabilities and corresponding ground-truth labels on the provided validation
    data loader.
    Parameters
    ----------
    model_class : pl.LightningModule
        The PyTorch LightningModule class (not an instance) from which to load the checkpoint.
    fold_number : int
        The index of the cross-validation fold (currently unused inside the function).
    val_loader : torch.utils.data.DataLoader
        DataLoader yielding batches with keys "data" (input tensor) and "label" (target tensor).
    checkpoint_path : str
        Filesystem path to the model checkpoint to be loaded via `load_from_checkpoint`.
    Returns
    -------
    Dict[str, Any]
        A dictionary with:
            "logits": np.ndarray
                Array of predicted probabilities (after sigmoid) for each sample in the
                validation set, shape (N,).
            "labels": np.ndarray
                Array of ground-truth labels corresponding to the predictions, shape (N,).
    """
    # Load model
    model = model_class.load_from_checkpoint(checkpoint_path, map_location=device)
    model.eval()
    model.to(device)

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

    # Combine logits for this fold
    all_logits = np.concatenate(all_logits, axis=0)
    return {"logits": all_logits, "labels": np.array(all_labels)}


def eval_res(
    val_loader: DataLoader,
    model_class: pl.LightningModule,
    logger,
    img_size: int = 512,
    model_name: str = "resnet",
):  # -> Tuple[np.ndarray, np.ndarray, Dict[str, float]]:
    """
    Evaluate a 5-fold ensemble of saved model checkpoints on a validation DataLoader.
    Loads the best checkpoint from each of 5 folds, obtains per-sample logits, computes
    per-fold metrics (accuracy, AUROC, F1), then averages logits across folds to form an
    ensemble prediction and computes ensemble metrics (accuracy, AUROC, F1, AUROC CI).
    Parameters:
        val_loader (DataLoader): Validation data loader (labels assumed consistent across folds).
        model_class (pl.LightningModule): LightningModule class used to load checkpoints.
        logger: Logger with .debug and .info methods for progress output.
        img_size (int): Image size used in training (part of checkpoint path). Default 512.
        model_name (str): Model name prefix (part of checkpoint path). Default 'resnet'.
    Returns:
        np.ndarray: Ground-truth labels (shape: [N]).
        np.ndarray: Ensemble-averaged logits (shape: [N]).
        dict: Ensemble metrics with keys:
              'acc' (float), 'auc' (float), 'f1' (float), 'auroc_ci' (float),
              plus an info log of per-fold averaged metrics.
    """
    k_folds = 5
    # Iterate through folds
    all_logits_ensemble = []
    all_labels = None
    metrics = {"acc": [], "auc": [], "f1": []}

    for fold in range(k_folds):
        logger.debug(f"\nEvaluating Fold {fold + 1}/{k_folds}...")

        # Path to the best model checkpoint for this fold
        name = os.listdir(
            f"{CHECKPOINT_DIR}/{model_name}/{model_name}_{img_size}_logs/lightning_logs/version_{fold}/checkpoints/"
        )[0]
        checkpoint_path = f"{CHECKPOINT_DIR}/{model_name}/{model_name}_{img_size}_logs/lightning_logs/version_{fold}/checkpoints/{name}"

        if not os.path.exists(checkpoint_path):
            logger.debug(f"Checkpoint not found for fold {fold}: {checkpoint_path}")
            continue

        # Evaluate this fold
        fold_results = evaluate_fold(model_class, val_loader, checkpoint_path)
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

    for key in metrics:
        metrics[key] = np.round(np.mean(metrics[key]), 4)

    # Combine logits from all folds (average for ensemble)
    all_logits_ensemble = np.mean(all_logits_ensemble, axis=0)

    # Compute ensemble metrics
    all_preds = (all_logits_ensemble > 0.5).astype(int)
    accuracy = accuracy_score(all_labels, all_preds)
    auc = roc_auc_score(all_labels, all_logits_ensemble)
    f1 = f1_score(all_labels, all_preds)
    ci = auroc_confidence_interval(all_labels, all_logits_ensemble)
    ensemble_metrics = {}
    ensemble_metrics["acc"] = np.round(accuracy, 4)
    ensemble_metrics["auc"] = np.round(auc, 4)
    ensemble_metrics["f1"] = np.round(f1, 4)
    ensemble_metrics["auroc_ci"] = np.round(ci, 4)
    logger.info(f"Fold metrics: {metrics}")
    return all_labels, all_logits_ensemble, ensemble_metrics
