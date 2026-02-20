"""Cross-validation evaluation for pneumonia classification models."""

import os
import torch
import glob
import numpy as np
from torch import nn
from sklearn.metrics import roc_auc_score
from scipy.stats import norm
import pandas as pd
from typing import Dict
from pathlib import Path
import pytorch_lightning as pl
from torch.utils.data import DataLoader, Subset
from sklearn.metrics import accuracy_score, roc_auc_score, f1_score
from sklearn.model_selection import KFold

from sharpness_matters.pneumonia.datasets.rsna_pneumonia import ChestXrayHoldout
from sharpness_matters.pneumonia.models.cnn import CNNBinaryClassifier
from sharpness_matters.pneumonia.utils.logger_utils import initialize_logger
from sharpness_matters.config.load_config import cfg

HOME = Path(__file__).resolve().parent.parent
logger = initialize_logger(level="info")
CHECKPOINT_DIR = cfg.pneumonia.model.checkpoint_dir
TUNED_CHECKPOINT_DIR = cfg.pneumonia.model.tuned_checkpoint_dir


# Evaluation function for a single fold
def evaluate_fold(
    val_loader: DataLoader, checkpoint_path: str
) -> Dict[str, np.ndarray]:
    """Evaluate a single fold using the provided validation DataLoader and model checkpoint.
    Parameters:
        val_loader (DataLoader): DataLoader for the validation dataset.
        checkpoint_path (str): Path to the model checkpoint file.
    Returns:
        Dict[str, np.ndarray]: Dictionary containing logits and labels for the validation set.
    """
    # Load model
    model = CNNBinaryClassifier.load_from_checkpoint(checkpoint_path)
    model.eval()
    model.to("cuda" if torch.cuda.is_available() else "cpu")

    all_labels = []
    all_logits = []

    # Evaluate on validation data
    with torch.no_grad():
        for batch in val_loader:
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


# Main script
def eval_res(
    img_size: int = 512, model_name: str = "resnet", tuning: str = "untuned"
) -> Dict[str, float]:
    """Evaluate classification performance across K-Fold cross-validation.
    Parameters:
        img_size (int): Image size for resizing.
        model_name (str): Model architecture name ('resnet' or 'densenet').
        tuning (str): Whether to use 'tuned' or 'untuned' model checkpoints.
    Returns:
        Dict[str, float]: Dictionary containing averaged metrics (accuracy, AUC, F1, loss).
    """
    seed = 42
    batch_size = 2
    DATA_DIR = "/data/rsna_pneumonia"
    train_dicom_dir = os.path.join(DATA_DIR, "stage_2_train_images/")
    test_dicom_dir = os.path.join(DATA_DIR, "stage_2_test_images/")
    train_glob = f"{train_dicom_dir}/*.dcm"
    test_glob = f"{test_dicom_dir}/*.dcm"
    train_names = [f for f in sorted(glob.glob(train_glob))]
    test_names = [f for f in sorted(glob.glob(test_glob))]

    dataset = ChestXrayHoldout(
        train_names,
        "/data/rsna_pneumonia/stage_2_train_labels.csv",
        split="train",
        img_size=img_size,
        mode="classification",
    )

    num_folds = 5
    # K-Fold Cross-Validation
    kf = KFold(n_splits=num_folds, shuffle=True, random_state=seed)
    fold_results = []
    checkpoint_paths = []
    # Ensemble predictions across folds
    all_logits_ensemble = []
    all_labels = None
    metrics = {"acc": [], "auc": [], "f1": [], "loss": []}
    for fold, (train_idx, val_idx) in enumerate(kf.split(dataset)):
        # Split dataset into training and validation subsets
        train_subset = Subset(dataset, train_idx)
        val_subset = Subset(dataset, val_idx)

        # Dataloaders
        class_counts = train_subset.dataset.class_count
        pos_weight = class_counts[0] / class_counts[1]
        val_loader = DataLoader(val_subset, batch_size=batch_size, num_workers=2)
        if tuning == "tuned":
            checkpoint_dir = f"{TUNED_CHECKPOINT_DIR}/{model_name}/{model_name}_{img_size}_logs/lightning_logs/version_{fold}/checkpoints/"
        elif tuning == "untuned":
            checkpoint_dir = f"{CHECKPOINT_DIR}/{model_name}/{model_name}_{img_size}_logs/lightning_logs/version_{fold}/checkpoints/"
        else:
            raise ("Invalid tuning argument. Choose one of tuned, untuned.")
        if not os.path.exists(checkpoint_dir):
            logger.warning(f"Checkpoint not found for fold {fold}: {checkpoint_dir}")
            continue
        checkpoint_name = os.listdir(checkpoint_dir)[0]
        checkpoint_path = os.path.join(checkpoint_dir, checkpoint_name)

        logger.info(f"\nEvaluating Fold {fold + 1}/{num_folds}...")
        fold_results = evaluate_fold(val_loader, checkpoint_path)
        # all_logits_ensemble.append(fold_results["logits"])
        fold_preds = (fold_results["logits"] > 0.5).astype(int)
        acc = accuracy_score(fold_results["labels"], fold_preds)
        auc = roc_auc_score(fold_results["labels"], fold_results["logits"])
        f1 = f1_score(fold_results["labels"], fold_preds)
        criterion = nn.BCEWithLogitsLoss()
        bce_loss = criterion(
            torch.tensor(fold_results["logits"], dtype=torch.float32),
            torch.tensor(fold_results["labels"], dtype=torch.float32),
        )
        # Store labels once (labels are the same for all folds)
        if all_labels is None:
            all_labels = fold_results["labels"]
        metrics["acc"].append(acc)
        metrics["auc"].append(auc)
        metrics["f1"].append(f1)
        metrics["loss"].append(bce_loss)

    for key in metrics:
        metrics[key] = (
            f"{np.round(np.mean(metrics[key]),4)} +- {np.round(np.max(metrics[key]) - np.min(metrics[key]),4)}"
        )

    logger.info(f"Fold metrics: {metrics}")
    return metrics


if __name__ == "__main__":
    seed = 42
    pl.seed_everything(seed)
    generator = torch.Generator()
    generator.manual_seed(seed)
    model_names = ["resnet", "densenet"]
    tuning = "tuned"
    for model_name in model_names:
        out = []
        for img_size in [64, 128, 224, 512, 768, 1024]:
            res = eval_res(img_size, model_name, tuning)
            res["img_size"] = img_size
            out.append(res)
            df = pd.DataFrame(out)
            columns_order = ["img_size"] + [
                col for col in df.columns if col != "img_size"
            ]
            df = df[columns_order]
            df.to_csv(f"/cnvrg/output/cv_{model_name}_{tuning}.csv", index=False)
