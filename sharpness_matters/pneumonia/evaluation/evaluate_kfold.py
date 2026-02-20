"""
Module for evaluating chest X-ray classification performance across multiple resolutions.

This module uses a chest X-ray dataset along with a binary classification model (CNNBinaryClassifier)
to evaluate classification performance metrics such as accuracy, AUC, and F1 score across different
image resolutions and model architectures (e.g., DenseNet, ResNet). It includes a custom collate function
to handle bounding box information in the dataset and fetches evaluation results per resolution.
The predicted logits and ground truth labels are saved for further analysis.

Usage:
    Run this module directly:
        python3 sharpness_matters/pneumonia/datasets/evaluate_kfold.py
    Output CSV files with evaluation metrics for each model will be saved in '/cnvrg/output/classification_{model_name}.csv'.
"""

import os
import torch
import glob
from pathlib import Path
import click
import numpy as np
from torch import nn
import pytorch_lightning as pl
from torch.utils.data import DataLoader
from sklearn.model_selection import KFold
from typing import Dict, List, Union, Tuple, Any
import pandas as pd

from sharpness_matters.pneumonia.datasets.rsna_pneumonia import ChestXrayHoldout
from sharpness_matters.pneumonia.datasets.siim_covid import ChestXrayOOD
from sharpness_matters.pneumonia.models.cnn import CNNBinaryClassifier
from sharpness_matters.pneumonia.utils.classification_eval import eval_res
from sharpness_matters.pneumonia.utils.logger_utils import initialize_logger
from sharpness_matters.config.load_config import cfg

logger = initialize_logger(level="debug")
HOME = Path(__file__).resolve().parent.parent
RSNA_PNEUMONIA_DATA_DIR = cfg.pneumonia.rsna_pneumonia.root_dir
SIIM_COVID_DATA_DIR = cfg.pneumonia.siim_covid.root_dir


def collate_fn(
    batch: List[Dict[str, Any]]
) -> Dict[str, Union[torch.Tensor, List[Any]]]:
    """
    Custom collate function to handle variable numbers of bounding boxes.
    Parameters:
        batch (List[Dict[str, Any]]): List of samples from the dataset, each a dictionary with keys 'data', 'bbox', and 'label'.
    Returns:
        Dict[str, Union[torch.Tensor, List[Any]]]: Collated batch with stacked images
    """
    images = []
    bboxes = []
    labels = []
    for item in batch:
        images.append(item["data"])
        bboxes.append(item["bbox"])
        labels.append(item["label"])
    # Stack images (assuming all images have the same shape)
    images = torch.stack(images, dim=0)
    labels = torch.stack(labels, dim=0)
    # Keep bounding boxes as a list of tensors (or pad if necessary)
    return {"data": images, "bbox": bboxes, "label": labels}


def fetch_eval_per_resolution(
    img_size: int, model_name: str, validation_mode: str
) -> Dict[str, float]:
    """
    Evaluate and log classification performance for a given image resolution and model.

    This function loads the dataset at the specified resolution, creates a DataLoader,
    and computes ensemble evaluation metrics (accuracy, AUC, and F1) using the CNNBinaryClassifier.
    It saves the ensemble predictions and ground truth labels to disk and logs the metrics.

    Parameters:
        img_size (int): Target image resolution (height/width) for evaluation.
        model_name (str): Name of the model architecture (e.g., 'densenet', 'resnet').

    Returns:
        dict: A dictionary containing ensemble evaluation metrics ('acc', 'auc', 'f1').
    """
    seed = 42
    pl.seed_everything(seed)
    if validation_mode == "holdout":
        train_dicom_dir = os.path.join(RSNA_PNEUMONIA_DATA_DIR, "stage_2_train_images/")
        train_glob = f"{train_dicom_dir}/*.dcm"
        train_names = list(sorted(glob.glob(train_glob)))
        # Define dataset and loader
        dataset = ChestXrayHoldout(
            train_names,
            f"{RSNA_PNEUMONIA_DATA_DIR}/stage_2_train_labels.csv",
            split="test",
            img_size=img_size,
            mode="classification",
        )
        val_loader = DataLoader(dataset, batch_size=2, num_workers=8)
    elif validation_mode == "ood":
        train_dicom_dir = os.path.join(SIIM_COVID_DATA_DIR, "covid_train/")
        train_glob = f"{train_dicom_dir}/*/*/*.dcm"
        train_names = [f for f in sorted(glob.glob(train_glob))]
        dataset = ChestXrayOOD(
            train_names,
            [
                f"{SIIM_COVID_DATA_DIR}/covid_study_level.csv",
                f"{SIIM_COVID_DATA_DIR}/covid_image_level.csv",
            ],
            img_size=img_size,
        )
        val_loader = DataLoader(
            dataset, batch_size=16, num_workers=32, shuffle=False, collate_fn=collate_fn
        )
    else:
        raise ("Invalid validation mode: Choose one of holdout, ood")
    all_labels, all_logits_ensemble, ensemble_metrics = eval_res(
        val_loader, CNNBinaryClassifier, logger, img_size, model_name
    )
    # Save predictions to avoid re-running for statistical analysis
    os.makedirs(f"/{HOME}/output/classif_preds/", exist_ok=True)
    np.save(
        f"/{HOME}/output/classif_preds/{validation_mode}pred_{model_name}_{img_size}.npy",
        all_logits_ensemble,
    )
    np.save(f"/{HOME}/output/classif_preds/{validation_mode}gt.npy", all_labels)
    # Print ensemble metrics
    logger.info("\nEnsemble Evaluation Results:")
    logger.info(f"Accuracy: {ensemble_metrics['acc']:.4f}")
    logger.info(f"AUC: {ensemble_metrics['auc']:.4f}")
    logger.info(f"F1: {ensemble_metrics['f1']:.4f}")
    return ensemble_metrics


@click.command()
@click.option(
    "--validation_mode",
    type=str,
    required=True,
    help="Choose validation dataset: One of holdout, ood",
)
def main(validation_mode):
    seed = 42
    pl.seed_everything(seed)
    generator = torch.Generator()
    generator.manual_seed(seed)
    model_names = ["densenet", "resnet"]
    for model_name in model_names:
        out = []
        for img_size in [64, 128, 224, 512, 768, 1024]:
            res = fetch_eval_per_resolution(img_size, model_name, validation_mode)
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
