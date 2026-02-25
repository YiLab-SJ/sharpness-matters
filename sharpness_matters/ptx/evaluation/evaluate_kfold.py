"""
Module for evaluating chest X-ray classification performance across multiple resolutions.

This module uses a chest X-ray dataset along with a binary classification model (CNNBinaryClassifier)
to evaluate classification performance metrics such as accuracy, AUC, and F1 score across different
image resolutions and model architectures (e.g., DenseNet, ResNet). It includes a custom collate function
to handle bounding box information in the dataset and fetches evaluation results per resolution.
The predicted logits and ground truth labels are saved for further analysis.

Usage:
    Run this module directly:
        python3 evaluate_kfold.py --validation_mode=<validation_dataset>
    Output CSV files with evaluation metrics for each model will be saved in 'classification_{model_name}.csv'.
"""

import os
import glob
import numpy as np
import torch
import pytorch_lightning as pl
from torch.utils.data import ConcatDataset, Dataset
import pandas as pd
import sys
import click
from pathlib import Path
from typing import Dict, Any, Tuple

from sharpness_matters.ptx.datasets.siim_acr_classification_dataset import (
    ChestXrayDataset,
)
from sharpness_matters.ptx.datasets.vin_dataset import VinDataset
from sharpness_matters.ptx.datasets.ptx498_dataset import OODTestDataset
from sharpness_matters.ptx.models.cnn import CNNBinaryClassifier
from sharpness_matters.ptx.utils.classification_eval import eval_res
from sharpness_matters.ptx.utils.logging_utils import initialize_logger
from sharpness_matters.config.load_config import cfg

logger = initialize_logger(level="debug")
DATA_DIR = cfg.ptx.siim_acr.root_dir
HOME = Path(__file__).resolve().parent.parent
logger.debug(f"Output Root: {HOME}")


def fetch_results_per_resolution(
    img_size: int = 512, model_name: str = "resnet", validation_mode: str = "holdout"
) -> Dict[str, Any]:
    """
    Evaluate model performance for a specific image resolution and validation dataset.

    Args:
        img_size (int): Input image size for model evaluation
        model_name (str): Model architecture ('resnet' or 'densenet')
        validation_mode (str): Validation dataset type ('holdout' or 'ood')

    Returns:
        Dict[str, Any]: Dictionary containing evaluation metrics
    """
    # Extract training files required to initialize dataset
    if validation_mode == "holdout":
        train_dicom_dir = os.path.join(DATA_DIR, "dicom-images-train")
        train_glob = f"{train_dicom_dir}/*/*/*.dcm"
        train_names = [f for f in sorted(glob.glob(train_glob))]
        dataset = ChestXrayDataset(
            train_names, f"{DATA_DIR}/train-rle.csv", split="test", img_size=img_size
        )
    elif validation_mode == "ood":
        dataset = VinDataset(img_size=img_size)
        ptx_dataset = OODTestDataset(img_size=img_size)
        dataset = ConcatDataset([dataset, ptx_dataset])
    else:
        raise ValueError("Invalid validation mode: Choose one of holdout, ood")
    # Compute evaluation metrics for given resolution
    all_labels, all_logits_ensemble, ensemble_metrics = eval_res(
        dataset, img_size, model_name, CNNBinaryClassifier, logger
    )
    # Save predictions to avoid re-running for stat analysis
    os.makedirs(f"{HOME}/output/classif_preds/", exist_ok=True)
    np.save(
        f"{HOME}/output/classif_preds/{validation_mode}pred_{model_name}_{img_size}.npy",
        all_logits_ensemble,
    )
    np.save(f"{HOME}/output/classif_preds/{validation_mode}gt.npy", all_labels)
    return ensemble_metrics


@click.command()
@click.option(
    "--validation_mode",
    type=str,
    required=True,
    help="Choose validation dataset: One of holdout, ood",
)
def main(validation_mode: str) -> None:
    """
    Main function to evaluate model performance across multiple resolutions and architectures.

    Args:
        validation_mode (str): Validation dataset type ('holdout' or 'ood')
    """
    seed = 42
    pl.seed_everything(seed)
    generator = torch.Generator()
    generator.manual_seed(seed)
    model_names = ["resnet", "densenet"]
    for model_name in model_names:
        out = []
        for img_size in [64, 128, 224, 512, 768, 1024]:
            res = fetch_results_per_resolution(img_size, model_name, validation_mode)
            res["img_size"] = img_size
            out.append(res)
            df = pd.DataFrame(out)
            columns_order = ["img_size"] + [
                col for col in df.columns if col != "img_size"
            ]
            df = df[columns_order]
            df.to_csv(
                f"{HOME}/output/classification_{model_name}_{validation_mode}.csv",
                index=False,
            )


if __name__ == "__main__":
    main()
