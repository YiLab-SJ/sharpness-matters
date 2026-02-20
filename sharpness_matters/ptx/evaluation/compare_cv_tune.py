import os
import torch
import glob
from pathlib import Path
import numpy as np
import pandas as pd
from torch import nn
import pytorch_lightning as pl
from torch.utils.data import DataLoader, Subset
from sklearn.metrics import accuracy_score, roc_auc_score, f1_score
from scipy.stats import norm
from sklearn.model_selection import KFold
from typing import Dict, Any, Tuple, List
import click

from sharpness_matters.ptx.datasets.siim_acr_classification_dataset import (
    ChestXrayDataset,
)
from sharpness_matters.ptx.models.cnn import CNNBinaryClassifier
from sharpness_matters.ptx.utils.logging_utils import initialize_logger
from sharpness_matters.config.load_config import cfg

HOME = Path(__file__).resolve().parent.parent
logger = initialize_logger(level="debug")
DATA_DIR = cfg.ptx.siim_acr.root_dir
CHECKPOINT_DIR = cfg.ptx.model.checkpoint_dir
TUNED_CHECKPOINT_DIR = cfg.ptx.model.tuned_checkpoint_dir


def evaluate_fold(
    val_loader: DataLoader, checkpoint_path: str
) -> Dict[str, np.ndarray]:
    """
    Evaluate model performance on a single fold.

    Args:
        val_loader (DataLoader): Validation data loader
        checkpoint_path (str): Path to model checkpoint

    Returns:
        Dict[str, np.ndarray]: Dictionary containing logits and labels
    """
    # Load model
    model = CNNBinaryClassifier.load_from_checkpoint(
        checkpoint_path, map_location="cuda" if torch.cuda.is_available() else "cpu"
    )
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


def eval_res(
    img_size: int = 512, model_name: str = "resnet", use_tuned: bool = True
) -> Dict[str, str]:
    """
    Evaluate model performance using K-fold cross-validation.

    Args:
        img_size (int): Input image size for model evaluation
        model_name (str): Model architecture ('resnet' or 'densenet')

    Returns:
        Dict[str, str]: Dictionary containing averaged metrics with standard deviations
    """
    seed = 42
    batch_size = 4
    train_dicom_dir = os.path.join(DATA_DIR, "dicom-images-train")
    train_glob = f"{train_dicom_dir}/*/*/*.dcm"
    train_names = [f for f in sorted(glob.glob(train_glob))]
    dataset = ChestXrayDataset(
        train_names, f"{DATA_DIR}/train-rle.csv", split="test", img_size=img_size
    )
    num_folds = 5
    # K-Fold Cross-Validation
    kf = KFold(n_splits=num_folds, shuffle=True, random_state=seed)
    fold_results = []
    # Ensemble predictions across folds
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

        if use_tuned:
            checkpoint_dir = f"{TUNED_CHECKPOINT_DIR}/{model_name}/{model_name}_{img_size}_logs/lightning_logs/version_{fold}/checkpoints/"
        else:
            checkpoint_dir = f"{CHECKPOINT_DIR}/{model_name}/{model_name}_{img_size}_logs/lightning_logs/version_{fold}/checkpoints/"
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


@click.option(
    "--use_tuned", type=bool, default=True, help="Whether to use tuned models"
)
def main(use_tuned: bool):
    seed = 42
    pl.seed_everything(seed)
    generator = torch.Generator()
    generator.manual_seed(seed)
    model_names = ["densenet", "resnet"]
    for model_name in model_names:
        out = []
        for img_size in [64, 128, 224, 512, 768, 1024]:
            logger.info(f"Evaluating model {model_name} at image size {img_size}")
            res = eval_res(img_size, model_name, use_tuned)
            res["img_size"] = img_size
            out.append(res)
            df = pd.DataFrame(out)
            columns_order = ["img_size"] + [
                col for col in df.columns if col != "img_size"
            ]
            df = df[columns_order]
            if use_tuned:
                df.to_csv(f"{HOME}/output/cv_{model_name}_tuned.csv", index=False)
            else:
                df.to_csv(f"{HOME}/output/cv_{model_name}.csv", index=False)


if __name__ == "__main__":
    main()
