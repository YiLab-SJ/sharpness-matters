"""
Module for performing saliency evaluation experiments on the Holdout Dataset.

This module evaluates saliency maps produced by different model architectures (e.g., DenseNet, ResNet). 
It selects a subset of the dataset based on the specified evaluation mode, runs predictions in chunks
(Computation is done chunk wise to prevent crashes due to CPU memory over-utilization)
for each cross-validation fold, computes metrics such as Intersection over Union (IoU),
Precision (%-IN), and coverage, and finally summarizes the results.

Usage:
    Run this module directly to execute the experiments:
        python3 evaluate_saliency.py
    Output is stored as a csv named "saliency_thresholded_{model_name}_{threshold}.csv"
"""

import os
import pytorch_lightning as pl
from pathlib import Path
import torch
import numpy as np
import glob
import sys
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
import pandas as pd
import gc
import shutil
from typing import Dict, Any, List, Union

from sharpness_matters.ptx.datasets.siim_acr_dataset import ChestXrayDataset
from sharpness_matters.ptx.models.cnn import CNNBinaryClassifier
from sharpness_matters.ptx.utils.logging_utils import initialize_logger
from sharpness_matters.ptx.utils.saliency_eval import (
    compute_scores_chunked,
    predict_fold_chunked,
)
from sharpness_matters.config.load_config import cfg

logger = initialize_logger(level="debug")
HOME = Path(__file__).resolve().parent.parent
CHECKPOINT_DIR = cfg.ptx.model.checkpoint_dir


def collate_fn(
    batch: List[Dict[str, Any]]
) -> Dict[str, Union[torch.Tensor, List[Any]]]:
    """
    Simple collate function that stacks tensors for key 'data' and aggregates other fields in lists.

    Args:
        batch (List[Dict[str, Any]]): List of sample dictionaries from dataset

    Returns:
        Dict[str, Union[torch.Tensor, List[Any]]]: Collated batch with stacked tensors and aggregated lists
    """
    collated = {}
    for key in batch[0]:
        if key == "data":
            collated[key] = torch.stack([b[key] for b in batch], dim=0)
        else:
            collated[key] = [b[key] for b in batch]
    return collated


def run_experiment(
    dataset: Dataset,
    img_size: int,
    model_name: str,
    threshold: float = 0.5,
    chunk_size: int = 5,
    device: torch.device = torch.device("cuda" if torch.cuda.is_available() else "cpu"),
) -> Dict[str, Union[str, float]]:
    """
    Run saliency evaluation experiment for holdout dataset with specified model and parameters.

    Args:
        dataset (Dataset): PyTorch dataset containing chest X-ray samples
        img_size (int): Input image size for the model
        model_name (str): Name of the model architecture ('resnet' or 'densenet')
        threshold (float): Threshold for saliency map evaluation
        chunk_size (int): Size of chunks for batch processing to manage memory

    Returns:
        Dict[str, Union[str, float]]: Dictionary containing evaluation metrics (IoU, precision, coverage, etc.)
    """
    # For chunked prediction, use a DataLoader with batch_size=1.
    pred_loader = DataLoader(
        dataset, batch_size=4, shuffle=False, num_workers=8, pin_memory=True
    )
    # Create a temporary cache directory for chunk predictions.
    cache_dir = f"{HOME}/output/chunk_predictions_{model_name}_{img_size}"
    os.makedirs(cache_dir, exist_ok=True)
    # For each fold, load the checkpoint and save chunked predictions to cache dir.
    for fold in range(5):
        ckpt_dir = f"{CHECKPOINT_DIR}/{model_name}/{model_name}_{img_size}_logs/lightning_logs/version_{fold}/checkpoints/"
        ckpt_file = os.listdir(ckpt_dir)[0]
        checkpoint_path = os.path.join(ckpt_dir, ckpt_file)
        if device.type == "cpu":
            logger.warning("Device is CPU; saliency evaluation may be slow.")
        if "densenet" in model_name or "resnet" in model_name:
            model = CNNBinaryClassifier.load_from_checkpoint(
                checkpoint_path, map_location=device
            )
            if "densenet" in model_name:
                target_layers = [model.model.features.denseblock4]
            if "resnet" in model_name:
                target_layers = [model.model.layer4[-1].conv3]
        model.eval()
        model.to(device)
        predict_fold_chunked(
            pred_loader,
            model,
            target_layers,
            fold,
            chunk_size=chunk_size,
            output_dir=cache_dir,
        )
        del model
        gc.collect()

    # Compute ensemble scores from the saved chunk predictions.
    iou, precision, coverage = compute_scores_chunked(
        dataset,
        cache_dir,
        model_name,
        img_size,
        collate_fn,
        logger,
        validation_mode="holdout",
        threshold=threshold,
        chunk_size=chunk_size,
    )
    # Summarize the evaluation metrics
    complete_miss = sum(1 for v in iou.values() if v == 0)
    complete_miss_percent = f"{np.round(100 * complete_miss / len(dataset), 2)}%"
    iou_values = np.array(list(iou.values()))
    precision_values = np.array(list(precision.values()))
    coverage = np.mean(list(coverage.values()))
    summary = {
        "iou": f"{iou_values.mean():.4f} +- {iou_values.std():.4f}",
        "precision": f"{precision_values.mean():.4f} +- {precision_values.std():.4f}",
        "coverage": coverage,
        "num_accurate": f"{np.round(100 * len(iou) / len(dataset), 2)}%",
        "complete_miss": complete_miss_percent,
    }
    # Save computed metrics
    save_path = f"{HOME}/output/saliency_eval/{model_name}"
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    np.save(os.path.join(save_path, f"iou_{img_size}_{threshold}"), iou)
    np.save(os.path.join(save_path, f"precision_{img_size}_{threshold}"), precision)
    # Delete the temporary cache directory after processing.
    if os.path.exists(cache_dir):
        shutil.rmtree(cache_dir)
    gc.collect()
    return summary


if __name__ == "__main__":
    seed = 42
    batch_size = 5
    pl.seed_everything(seed)
    generator = torch.Generator()
    generator.manual_seed(seed)

    DATA_DIR = cfg.ptx.siim_acr.root_dir
    train_dicom_dir = os.path.join(DATA_DIR, "dicom-images-train")
    train_glob = f"{train_dicom_dir}/*/*/*.dcm"
    train_names = [f for f in sorted(glob.glob(train_glob))]

    img_size = 64
    model_names = ["resnet", "densenet"]
    thresholds = [0.25, 0.75, 0.5]
    for threshold in thresholds:
        for model_name in model_names:
            out = []
            for img_size in [64, 128, 224, 512, 768, 1024]:
                torch.cuda.empty_cache()
                dataset = ChestXrayDataset(
                    train_names,
                    f"{DATA_DIR}/train-rle.csv",
                    split="test",
                    img_size=img_size,
                )
                logger.info(
                    f"Running: Image size: {img_size}, Threshold: {threshold}, Model: {model_name}"
                )
                res = run_experiment(
                    dataset, img_size, model_name, threshold=threshold, chunk_size=10
                )
                res["img_size"] = img_size
                out.append(res)
                logger.debug(f"Results: {res}")
                df = pd.DataFrame(out)
                columns_order = ["img_size"] + [
                    col for col in df.columns if col != "img_size"
                ]
                df = df[columns_order]
                df.to_csv(
                    f"{HOME}/output/saliency_thresholded_{model_name}_{threshold}.csv",
                    index=False,
                )
                logger.debug(
                    f"Saved {HOME}/output/saliency_thresholded_{model_name}_{threshold}.csv"
                )
