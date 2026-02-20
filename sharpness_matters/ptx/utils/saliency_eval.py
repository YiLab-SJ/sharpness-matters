"""
Utility file for saliency map evaluation
"""

import os
import glob
import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset
from PIL import Image
from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.image import preprocess_image, show_cam_on_image
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
from sklearn.metrics import jaccard_score
from tqdm import tqdm
from pathlib import Path
from typing import Dict, Tuple, List, Union, Optional, Any, Callable
import matplotlib.pyplot as plt

HOME = Path(__file__).resolve().parent.parent


def visualize_and_save(
    image: Image.Image,
    rollout_map: np.ndarray,
    save_path: str = "attention_rollout.png",
) -> Image.Image:
    """
    Save the aggregated attention map overlayed on the image.

    Args:
        image (Image.Image): Input PIL image
        rollout_map (np.ndarray): Aggregated attention map
        save_path (str): Path to save the overlayed image

    Returns:
        Image.Image: Blended image with attention overlay
    """
    rollout_map_resized = rollout_map / rollout_map.max()  # Normalize
    rollout_map_resized = np.kron(
        rollout_map_resized, np.ones((16, 16))
    )  # Upscale to 224x224
    # Convert rollout map to RGB heatmap
    cmap = plt.get_cmap("jet")
    heatmap = cmap(rollout_map_resized)  # Apply colormap
    heatmap = (heatmap[:, :, :3] * 255).astype(np.uint8)  # Convert to RGB
    # Convert image and heatmap to numpy
    original_image = image.convert("RGB")
    overlay = Image.blend(original_image, Image.fromarray(heatmap), alpha=0.4)
    return overlay


def tensor_to_pil(input_data: Union[torch.Tensor, np.ndarray]) -> Image.Image:
    """
    Convert tensor or numpy array to PIL Image.

    Args:
        input_data (Union[torch.Tensor, np.ndarray]): Input tensor or array

    Returns:
        Image.Image: Converted PIL Image
    """
    # If the input is a torch tensor, convert to numpy
    if isinstance(input_data, torch.Tensor):
        array = input_data.detach().cpu().numpy()
    elif isinstance(input_data, np.ndarray):
        array = input_data
    else:
        raise TypeError("Input must be a torch.Tensor or numpy.ndarray")
    # Check if the array is in channel-first format (C, H, W) and convert it to (H, W, C)
    if array.ndim == 3 and array.shape[0] in [1, 3]:
        array = np.transpose(array, (1, 2, 0))
    # If the array contains floats, assume range [0, 1] and convert to [0, 255]
    if np.issubdtype(array.dtype, np.floating):
        array = (array * 255).astype(np.uint8)
    return Image.fromarray(array)


def overlay_map(
    overlayviz: Union[torch.Tensor, np.ndarray],
    segviz: Union[torch.Tensor, np.ndarray],
    alpha: float = 0.5,
) -> Image.Image:
    """
    Overlay two images with specified transparency.

    Args:
        overlayviz (Union[torch.Tensor, np.ndarray]): Base image tensor/array
        segviz (Union[torch.Tensor, np.ndarray]): Overlay image tensor/array
        alpha (float): Transparency factor for blending

    Returns:
        Image.Image: Blended PIL Image
    """
    # Convert both tensors to PIL Images
    base_img = tensor_to_pil(overlayviz)
    overlay_img = tensor_to_pil(segviz)
    # Ensure both images are the same size (resize overlay if necessary)
    if overlay_img.size != base_img.size:
        overlay_img = overlay_img.resize(base_img.size)
    # Convert images to RGB (in case they aren't)
    base_img = base_img.convert("RGB")
    overlay_img = overlay_img.convert("RGB")
    # Blend the images using the provided alpha (0.0 to 1.0)
    blended = Image.blend(base_img, overlay_img, alpha)
    return blended


def calculate_precision(
    ground_truth: np.ndarray, segmentation: np.ndarray, threshold: float = 0.5
) -> float:
    """
    Calculate the Effective Heat Ratio (EHR) between a ground truth mask and a segmentation map.

    Args:
        ground_truth (np.ndarray): Binary ground truth mask (H, W), values are 0 or 1
        segmentation (np.ndarray): Predicted segmentation map (H, W), with values between 0 and 1
        threshold (float): Threshold to binarize the segmentation map

    Returns:
        float: Precision value
    """
    if ground_truth.shape != segmentation.shape:
        raise ValueError("Ground truth and segmentation must have the same shape.")
    binary_segmentation = (segmentation >= threshold).astype(np.float32)
    threshold_area_inside_gt = np.sum(binary_segmentation * ground_truth)
    complete_threshold_area = np.sum(binary_segmentation)
    if complete_threshold_area == 0:
        return 0.0
    precision = threshold_area_inside_gt / complete_threshold_area
    return precision


def collage_maker(image1: Image.Image, image2: Image.Image, name: str) -> None:
    """
    Create a horizontal collage of two images and save it.

    Args:
        image1 (Image.Image): First PIL image
        image2 (Image.Image): Second PIL image
        name (str): Output filename for the collage
    """
    i1 = np.array(image1)
    i2 = np.array(image2)
    collage = np.hstack([i1, i2])
    image = Image.fromarray(collage)
    image.save(name)


def predict_fold_chunked(
    val_loader: DataLoader,
    model: torch.nn.Module,
    target_layers: List[Any],
    fold: int,
    chunk_size: int = 5,
    output_dir: str = f"{HOME}/output/chunk_predictions",
) -> None:
    """
    Process predictions and saliency maps in chunks for a specific fold and save them to disk.

    This function iterates over the validation DataLoader, computes model predictions and GradCAM saliency maps
    for each batch, and aggregates results into chunks. Once the number of processed samples reaches the chunk_size,
    the logits and saliency maps are saved as .npy files to the specified output directory, with filenames indicating
    the fold and chunk index.

    Args:
        val_loader (DataLoader): DataLoader providing validation batches
        model (torch.nn.Module): The model used for prediction
        target_layers (List[Any]): List of target layers for computing GradCAM
        fold (int): The current fold index (used in output filenames)
        chunk_size (int): Number of samples per chunk
        output_dir (str): Directory where chunk files will be saved
    """
    os.makedirs(output_dir, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.eval()
    # Initialize tracking variables
    all_logits = {}
    all_sm = {}
    sample_count = 0
    chunk_index = 0
    # Initialize GradCAM with the provided model and target layers.
    with GradCAM(model=model, target_layers=target_layers) as cam:
        for idx, batch in tqdm(
            enumerate(val_loader), total=len(val_loader), desc=f"Fold {fold}"
        ):
            pixel_values = batch["data"].to(device)
            pid = batch["pid"]
            with torch.no_grad():
                logits = torch.sigmoid(model(pixel_values).squeeze(1)).cpu().numpy()
            # Since we have binary classificaiton, we will always get gradients corresponding to the positive class,
            # when using an index of 0 for ClassifierOutputTarget
            targets = [ClassifierOutputTarget(0) for _ in range(pixel_values.shape[0])]
            pixel_values.requires_grad = True
            grayscale_cam = cam(input_tensor=pixel_values, targets=targets)
            for k, p in enumerate(pid):
                all_logits[p] = logits[k]
                all_sm[p] = np.expand_dims(grayscale_cam[k], axis=0).transpose(1, 2, 0)
                sample_count += 1
                # If the current chunk is full, save it to disk.
                if sample_count == chunk_size:
                    logits_path = os.path.join(
                        output_dir, f"fold_{fold}_logits_chunk_{chunk_index}.npy"
                    )
                    sm_path = os.path.join(
                        output_dir, f"fold_{fold}_sm_chunk_{chunk_index}.npy"
                    )
                    np.save(logits_path, all_logits, allow_pickle=True)
                    np.save(sm_path, all_sm, allow_pickle=True)
                    chunk_index += 1
                    sample_count = 0
                    all_logits = {}
                    all_sm = {}
        # Save any remaining samples that did not fill up the last chunk.
        if sample_count > 0:
            logits_path = os.path.join(
                output_dir, f"fold_{fold}_logits_chunk_{chunk_index}.npy"
            )
            sm_path = os.path.join(
                output_dir, f"fold_{fold}_sm_chunk_{chunk_index}.npy"
            )
            np.save(logits_path, all_logits, allow_pickle=True)
            np.save(sm_path, all_sm, allow_pickle=True)
    return


def compute_scores_chunked(
    val_dataset: Dataset,
    cache_dir: str,
    model_name: str,
    img_size: int,
    collate_fn: Callable,
    logger: Any,
    validation_mode: str = "holdout",
    threshold: float = 0.5,
    chunk_size: int = 5,
) -> Tuple[Dict[str, float], Dict[str, float], Dict[str, float]]:
    """
    Load ensemble predictions from disk in chunks, compute evaluation metrics, and save segmentation visualizations.

    This function aggregates model logits and GradCAM saliency maps from pre-saved
    chunk files across multiple folds, computes metrics such as Intersection over Union (IoU),
    percentage inclusion, and coverage for each sample in the validation dataset, and
    generates visualizations of the segmentation and ground truth masks.

    Args:
        val_dataset (Dataset): Dataset containing validation data, labels, and ground truth masks
        cache_dir (str): Directory where chunked prediction files are stored
        model_name (str): Name of the evaluated model (used for directory structure)
        img_size (int): The image resolution used during evaluation
        collate_fn (Callable): Function to collate samples into batches
        logger (Any): Logger instance for logging errors and information
        validation_mode (str): Mode of validation ('holdout' or 'ood')
        threshold (float): Threshold to binarize saliency maps
        chunk_size (int): Batch size for processing chunks

    Returns:
        Tuple[Dict[str, float], Dict[str, float], Dict[str, float]]: Three dictionaries (iou_all, percentin_all, coverage_all) keyed by sample ID, containing the IoU scores, percentage inclusion, and coverage metrics, respectively
    """
    # Set up directories to save ground truth, segmentation, and mask visualizations
    gt_save_path = (
        f"{HOME}/output/{model_name}_maps/maps_{validation_mode}/maps_{img_size}/gt"
    )
    seg_save_path = (
        f"{HOME}/output/{model_name}_maps/maps_{validation_mode}/maps_{img_size}/sm"
    )
    mask_save_path = f"{HOME}/output/{model_name}_maps/maps_{validation_mode}/maps_{img_size}/masks_{threshold}"
    os.makedirs(gt_save_path, exist_ok=True)
    os.makedirs(seg_save_path, exist_ok=True)
    os.makedirs(mask_save_path, exist_ok=True)
    # Create a DataLoader for chunked processing.
    val_loader = DataLoader(
        val_dataset,
        batch_size=chunk_size,
        shuffle=False,
        num_workers=8,
        pin_memory=True,
        collate_fn=collate_fn,
    )
    iou_all = {}
    percentin_all = {}
    coverage_all = {}
    operating_point = 0.5
    # Determine number of chunks using files from fold 0
    fold0_logits_files = sorted(
        glob.glob(os.path.join(cache_dir, f"fold_0_logits_chunk_*.npy"))
    )
    num_chunks = len(fold0_logits_files)
    chunk_index = 0
    # Process each chunk from the DataLoader
    for batch in tqdm(
        val_loader, total=len(val_loader), desc="Computing scores chunk-by-chunk"
    ):
        ensemble_logits_chunk = {}
        ensemble_sm_chunk = {}
        # For each fold, load the corresponding chunk files and aggregate predictions
        for fold in range(5):
            logits_file = os.path.join(
                cache_dir, f"fold_{fold}_logits_chunk_{chunk_index}.npy"
            )
            sm_file = os.path.join(cache_dir, f"fold_{fold}_sm_chunk_{chunk_index}.npy")
            if not os.path.exists(logits_file) or not os.path.exists(sm_file):
                continue
            fold_logits = np.load(logits_file, allow_pickle=True).item()
            fold_sm = np.load(sm_file, allow_pickle=True).item()
            for pid, logit in fold_logits.items():
                ensemble_logits_chunk.setdefault(pid, []).append(logit)
            for pid, sm in fold_sm.items():
                ensemble_sm_chunk.setdefault(pid, []).append(sm)
        # Average predictions across folds for each sample
        for pid in ensemble_logits_chunk.keys():
            ensemble_logits_chunk[pid] = np.mean(ensemble_logits_chunk[pid], axis=0)
        for pid in ensemble_sm_chunk.keys():
            ensemble_sm_chunk[pid] = np.mean(ensemble_sm_chunk[pid], axis=0)

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        pixel_values = batch["data"].to(device)
        gt = batch["label"]
        pids = batch["pid"]
        ground_truth_masks = batch["mask"]
        # Process each sample in the batch
        for k, pid in enumerate(pids):
            # Handling mispredictions
            try:
                if ensemble_logits_chunk[pid] > operating_point and gt[k] == 0:
                    iou_all[pid] = 0
                    percentin_all[pid] = 0
                    sm = ensemble_sm_chunk[pid]
                    binary_masks = (
                        (torch.tensor(sm) > threshold).int().cpu().squeeze(-1).numpy()
                    )
                    coverage = np.sum(binary_masks) / binary_masks.size
                    coverage_all[pid] = coverage
                    continue
                if ensemble_logits_chunk[pid] < operating_point and gt[k] == 1:
                    iou_all[pid] = 0
                    percentin_all[pid] = 0
                    continue
                if ensemble_logits_chunk[pid] < operating_point and gt[k] == 0:
                    continue
                # if gt[k] == 0:
                # continue
            except Exception as e:
                logger.error(e)
            sm = ensemble_sm_chunk[pid]
            # Visualize saliency map
            rgb_img = np.repeat(sm, 3, axis=-1)
            segviz = show_cam_on_image(rgb_img, sm, use_rgb=True)
            binary_mask_viz = (
                pixel_values[k].permute(1, 2, 0).repeat(1, 1, 3).detach().cpu()
            )
            overlayviz = binary_mask_viz.clone()
            segviz_overlay = overlay_map(overlayviz, segviz, alpha=0.5)
            # Visualize GT mask
            gtviz = val_dataset.visualize_segmentation(
                binary_mask_viz, ground_truth_masks[k].squeeze(0)
            )
            seg_save_file = os.path.join(seg_save_path, f"{pid}.jpg")
            mask_save_file = os.path.join(mask_save_path, f"{pid}.jpg")
            gt_save_file = os.path.join(gt_save_path, f"{pid}.jpg")
            segviz_overlay.save(seg_save_file)
            Image.fromarray(gtviz).save(gt_save_file)
            # Threshold saliency map to binary mask
            binary_masks = (
                (torch.tensor(sm) > threshold).int().cpu().squeeze(-1).numpy()
            )
            # Alternate approach (commented out) for converting saliency map to binary mask (topk most salient pixles)
            # binary_masks = (torch.tensor(sm) > threshold).int().flatten().cpu().numpy()
            # Visualize and save binary mask
            maskviz = val_dataset.visualize_segmentation(
                binary_mask_viz, binary_masks * 255.0
            )
            Image.fromarray(maskviz).save(mask_save_file)
            gt_mask_np = (
                (np.array(ground_truth_masks[k]) / 255.0).astype(np.uint8).flatten()
            )
            # Compute quantitative metrics for saliency map evaluation
            coverage = np.sum(binary_masks) / binary_masks.size
            binary_masks = binary_masks.flatten()
            iou_all[pid] = jaccard_score(gt_mask_np, binary_masks)
            percentin_all[pid] = calculate_precision(gt_mask_np, binary_masks)
            coverage_all[pid] = coverage
        chunk_index += 1
    return iou_all, percentin_all, coverage_all


def compute_scores_chunked_neg(
    val_dataset: Dataset,
    cache_dir: str,
    model_name: str,
    img_size: int,
    collate_fn: Callable,
    logger: Any,
    validation_mode: str = "holdout",
    threshold: float = 0.5,
    chunk_size: int = 5,
) -> Tuple[Dict[str, float], Dict[str, float], Dict[str, float]]:
    """
    Load ensemble predictions from disk in chunks, compute evaluation metrics for negative cases, and save segmentation visualizations.

    This function aggregates model logits and GradCAM saliency maps from pre-saved
    chunk files across multiple folds, computes metrics such as Intersection over Union (IoU),
    percentage inclusion, and coverage for each sample in the validation dataset, and
    generates visualizations of the segmentation and ground truth masks.

    Args:
        val_dataset (Dataset): Dataset containing validation data, labels, and ground truth masks
        cache_dir (str): Directory where chunked prediction files are stored
        model_name (str): Name of the evaluated model (used for directory structure)
        img_size (int): The image resolution used during evaluation
        collate_fn (Callable): Function to collate samples into batches
        logger (Any): Logger instance for logging errors and information
        validation_mode (str): Mode of validation ('holdout' or 'ood')
        threshold (float): Threshold to binarize saliency maps
        chunk_size (int): Batch size for processing chunks

    Returns:
        Tuple[Dict[str, float], Dict[str, float], Dict[str, float]]: Three dictionaries (iou_all, percentin_all, coverage_all) keyed by sample ID, containing the IoU scores, percentage inclusion, and coverage metrics, respectively
    """
    # Set up directories to save ground truth, segmentation, and mask visualizations
    gt_save_path = (
        f"{HOME}/output/{model_name}_maps_neg/maps_{validation_mode}/maps_{img_size}/gt"
    )
    seg_save_path = (
        f"{HOME}/output/{model_name}_maps_neg/maps_{validation_mode}/maps_{img_size}/sm"
    )
    mask_save_path = f"{HOME}/output/{model_name}_maps_neg/maps_{validation_mode}/maps_{img_size}/masks_{threshold}"
    os.makedirs(gt_save_path, exist_ok=True)
    os.makedirs(seg_save_path, exist_ok=True)
    os.makedirs(mask_save_path, exist_ok=True)
    # Create a DataLoader for chunked processing.
    val_loader = DataLoader(
        val_dataset,
        batch_size=chunk_size,
        shuffle=False,
        num_workers=8,
        pin_memory=True,
        collate_fn=collate_fn,
    )
    iou_all = {}
    percentin_all = {}
    coverage_all = {}
    operating_point = 0.5
    # Determine number of chunks using files from fold 0
    fold0_logits_files = sorted(
        glob.glob(os.path.join(cache_dir, f"fold_0_logits_chunk_*.npy"))
    )
    num_chunks = len(fold0_logits_files)
    chunk_index = 0
    # Process each chunk from the DataLoader
    for batch in tqdm(
        val_loader, total=len(val_loader), desc="Computing scores chunk-by-chunk"
    ):
        ensemble_logits_chunk = {}
        ensemble_sm_chunk = {}
        # For each fold, load the corresponding chunk files and aggregate predictions
        for fold in range(5):
            logits_file = os.path.join(
                cache_dir, f"fold_{fold}_logits_chunk_{chunk_index}.npy"
            )
            sm_file = os.path.join(cache_dir, f"fold_{fold}_sm_chunk_{chunk_index}.npy")
            if not os.path.exists(logits_file) or not os.path.exists(sm_file):
                continue
            fold_logits = np.load(logits_file, allow_pickle=True).item()
            fold_sm = np.load(sm_file, allow_pickle=True).item()
            for pid, logit in fold_logits.items():
                ensemble_logits_chunk.setdefault(pid, []).append(logit)
            for pid, sm in fold_sm.items():
                ensemble_sm_chunk.setdefault(pid, []).append(sm)
        # Average predictions across folds for each sample
        for pid in ensemble_logits_chunk.keys():
            ensemble_logits_chunk[pid] = np.mean(ensemble_logits_chunk[pid], axis=0)
        for pid in ensemble_sm_chunk.keys():
            ensemble_sm_chunk[pid] = np.mean(ensemble_sm_chunk[pid], axis=0)

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        pixel_values = batch["data"].to(device)
        gt = batch["label"]
        pids = batch["pid"]
        ground_truth_masks = batch["mask"]
        # Process each sample in the batch
        for k, pid in enumerate(pids):
            # Handling mispredictions
            try:
                if ensemble_logits_chunk[pid] > operating_point:
                    continue
            except Exception as e:
                logger.error(e)
            sm = ensemble_sm_chunk[pid]
            logit = ensemble_logits_chunk[pid]
            # Visualize saliency map
            rgb_img = np.repeat(sm, 3, axis=-1)
            segviz = show_cam_on_image(rgb_img, sm, use_rgb=True)
            binary_mask_viz = (
                pixel_values[k].permute(1, 2, 0).repeat(1, 1, 3).detach().cpu()
            )
            overlayviz = binary_mask_viz.clone()
            segviz_overlay = overlay_map(overlayviz, segviz, alpha=0.5)
            # Visualize GT mask
            gtviz = val_dataset.visualize_segmentation(
                binary_mask_viz, ground_truth_masks[k].squeeze(0)
            )
            seg_save_file = os.path.join(seg_save_path, f"{pid}_{logit}.jpg")
            mask_save_file = os.path.join(mask_save_path, f"{pid}.jpg")
            gt_save_file = os.path.join(gt_save_path, f"{pid}.jpg")
            segviz_overlay.save(seg_save_file)
            Image.fromarray(gtviz).save(gt_save_file)
            # Threshold saliency map to binary mask
            binary_masks = (
                (torch.tensor(sm) > threshold).int().cpu().squeeze(-1).numpy()
            )
            # Alternate approach (commented out) for converting saliency map to binary mask (topk most salient pixles)
            # binary_masks = (torch.tensor(sm) > threshold).int().flatten().cpu().numpy()
            # Visualize and save binary mask
            maskviz = val_dataset.visualize_segmentation(
                binary_mask_viz, binary_masks * 255.0
            )
            Image.fromarray(maskviz).save(mask_save_file)
            gt_mask_np = (
                (np.array(ground_truth_masks[k]) / 255.0).astype(np.uint8).flatten()
            )
            # Compute quantitative metrics for saliency map evaluation
            coverage = np.sum(binary_masks) / binary_masks.size
            binary_masks = binary_masks.flatten()
            iou_all[pid] = jaccard_score(gt_mask_np, binary_masks)
            percentin_all[pid] = calculate_precision(gt_mask_np, binary_masks)
            coverage_all[pid] = coverage
        chunk_index += 1
    return iou_all, percentin_all, coverage_all
