"""
Module for performing saliency evaluation experiments on the Holdout Dataset.

This module evaluates saliency maps produced by different model architectures (e.g., DenseNet, ResNet). 
It selects a subset of the dataset based on the specified evaluation mode, runs predictions in chunks
(Computation is done chunk wise to prevent crashes due to CPU memory over-utilization)
for each cross-validation fold, computes metrics such as Intersection over Union (IoU),
Precision (%-IN), and coverage, and finally summarizes the results.

Usage:
    Run this module directly to execute the experiments:
        python3 sharpness_matters/pneumonia/evaluation/evaluate_saliency.py
    Output is stored as a csv named "saliency_bbox_{model_name}_ood_{threshold}.csv"
"""

import os
import glob
import pandas as pd
import numpy as np
import torch
import pytorch_lightning as pl
from torch.utils.data import DataLoader, Subset
from tqdm import tqdm
import matplotlib.pyplot as plt
import gc
import shutil
from pathlib import Path
from typing import Dict, Any, List, Union
from sharpness_matters.pneumonia.datasets.rsna_pneumonia import ChestXrayHoldout
from sharpness_matters.pneumonia.models.cnn import CNNBinaryClassifier
from sharpness_matters.pneumonia.utils.saliency_eval import (
    predict_fold_chunked,
    compute_scores_chunked,
)
from sharpness_matters.pneumonia.utils.logger_utils import initialize_logger
from sharpness_matters.config.load_config import cfg

logger = initialize_logger(level="debug")
HOME = Path(__file__).resolve().parent.parent
DATA_DIR = cfg.pneumonia.rsna_pneumonia.root_dir
CHECKPOINT_DIR = cfg.pneumonia.model.checkpoint_dir


def collate_fn(
    batch: List[Dict[str, Any]]
) -> Dict[str, Union[torch.Tensor, List[Any]]]:
    """
    Custom collate function to handle variable numbers of bounding boxes.
    Parameters:
        batch (List[Dict[str, Any]]): List of sample dictionaries from dataset
    Returns:
        Dict[str, Union[torch.Tensor, List[Any]]]: Collated batch with stacked tensors and aggregated lists
    """
    images = []
    bboxes = []
    labels = []
    pids = []
    for item in batch:
        images.append(item["data"])
        labels.append(item["label"])
        bboxes.append(item["bbox"])
        pids.append(item["pid"])
    images = torch.stack(images, dim=0)
    labels = torch.stack(labels, dim=0)
    return {"data": images, "bbox": bboxes, "label": labels, "pid": pids}


def run_experiment(
    dataset: ChestXrayHoldout,
    img_size: int,
    model_name: str,
    chunk_size: int = 250,
    threshold: float = 0.5,
    device: torch.device = torch.device("cuda" if torch.cuda.is_available() else "cpu"),
) -> Dict[str, Union[str, float]]:
    """
    Run a saliency evaluation experiment using chunked predictions.

    This function selects a subset of the dataset based on the evaluation mode, creates a DataLoader,
    and for each cross-validation fold loads the appropriate model checkpoint (for 'deit', 'densenet', or 'resnet').
    It performs chunked predictions using `predict_fold_chunked`, computes ensemble metrics (IoU, percentage inclusion,
    coverage) via `compute_scores_chunked`, cleans up temporary files, and returns a summary of the metrics.

    Parameters:
        dataset (Dataset): The dataset to evaluate.
        img_size (int): Target image size for evaluation.
        eval_mode (str): Dataset selection mode ('all', 'all_pos', 'first_quartile', 'interquartile', 'last_quartile').
        model_name (str): Model type to use ('deit', 'densenet', or 'resnet').
        chunk_size (int, optional): Number of samples per prediction chunk (default: 500).
        threshold (float, optional): Threshold for metric computation (default: 0.05).

    Returns:
        dict: Summary metrics including 'iou', 'precision', 'coverage', 'num_accurate', and 'complete_miss'.
    """
    # Use a small batch size for chunked predictions with our custom collate_fn.
    pred_loader = DataLoader(
        dataset,
        batch_size=4,
        shuffle=False,
        num_workers=8,
        pin_memory=True,
        collate_fn=collate_fn,
    )
    cache_dir = f"{HOME}/output/chunk_predictions_holdout"
    os.makedirs(cache_dir, exist_ok=True)
    # Process each fold in chunks.
    for fold in range(5):
        if device.type == "cpu":
            logger.warning("Device is CPU; saliency evaluation may be slow.")
        if "densenet" in model_name or "resnet" in model_name:
            checkpoint_dir = f"{CHECKPOINT_DIR}/{model_name}/{model_name}_{img_size}_logs/lightning_logs/version_{fold}/checkpoints/"
            ckpt_file = os.listdir(checkpoint_dir)[0]
            checkpoint_path = os.path.join(checkpoint_dir, ckpt_file)
            model = CNNBinaryClassifier.load_from_checkpoint(
                checkpoint_path, map_location=device
            )
            model.eval()
            if "densenet" in model_name:
                target_layers = [model.model.features.denseblock4]
            elif "resnet" in model_name:
                target_layers = [model.model.layer4[-1].conv3]
        else:
            raise Exception(
                "Unknown model name, only 'densenet' and 'resnet' are supported."
            )
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
    # Compute ensemble scores using the chunked predictions.
    iou, precision, coverage = compute_scores_chunked(
        dataset,
        cache_dir,
        model_name,
        img_size,
        validation_mode="holdout",
        collate_fn=collate_fn,
        logger=logger,
        chunk_size=chunk_size,
        threshold=threshold,
    )

    complete_miss = sum(1 for v in iou.values() if v == 0)
    complete_miss_percent = f"{np.round(100 * complete_miss / len(dataset), 2)}%"
    iou_values = np.array(list(iou.values()))
    precision_values = np.array(list(precision.values()))
    coverage_values = np.array(list(coverage.values()))
    summary = {
        "iou": f"{iou_values.mean():.4f} +- {iou_values.std():.4f}",
        "precision": f"{precision_values.mean():.4f} +- {precision_values.std():.4f}",
        "coverage": f"{coverage_values.mean():.4f}",
        "num_accurate": f"{np.round(100 * len(iou) / len(dataset), 2)}%",
        "complete_miss": complete_miss_percent,
    }
    save_path = f"{HOME}/output/saliency_eval/{model_name}"
    os.makedirs(save_path, exist_ok=True)
    np.save(os.path.join(save_path, f"bboxiou_{img_size}_{threshold}.npy"), iou)
    np.save(
        os.path.join(save_path, f"bboxprecision_{img_size}_{threshold}.npy"), precision
    )
    if os.path.exists(cache_dir):
        shutil.rmtree(cache_dir)
    gc.collect()
    return summary


if __name__ == "__main__":
    seed = 42
    batch_size = 1  # Not used in chunked prediction (handled by pred_loader)
    pl.seed_everything(seed)
    torch.manual_seed(seed)
    train_dicom_dir = os.path.join(DATA_DIR, "stage_2_train_images/")
    train_glob = f"{train_dicom_dir}/*.dcm"
    train_names = list(sorted(glob.glob(train_glob)))
    model_names = ["densenet", "resnet"]
    for threshold in [0.25, 0.75, 0.5]:
        for model_name in model_names:
            out = []
            for img_size in [64, 128, 224, 512, 768, 1024]:
                torch.cuda.empty_cache()
                dataset = ChestXrayHoldout(
                    train_names,
                    f"{DATA_DIR}/stage_2_train_labels.csv",
                    split="test",
                    img_size=img_size,
                )
                logger.info(
                    f"Running: Image size: {img_size}, Model: {model_name}, Threshold: {threshold}"
                )
                res = run_experiment(
                    dataset, img_size, model_name, chunk_size=250, threshold=threshold
                )
                res["img_size"] = img_size
                out.append(res)
                df = pd.DataFrame(out)
                columns_order = ["img_size"] + [
                    col for col in df.columns if col != "img_size"
                ]
                df = df[columns_order]
                df.to_csv(
                    f"{HOME}/output/saliency_bbox_{model_name}_{threshold}.csv",
                    index=False,
                )
