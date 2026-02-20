"""
Utility functions for evaluating Saliency Maps
"""

import os
import glob
import numpy as np
from PIL import Image
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset
from matplotlib import pyplot as plt
from tqdm import tqdm
from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.image import show_cam_on_image
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
from sklearn.metrics import jaccard_score
from scipy.ndimage import label
from typing import List, Tuple, Union, Callable
from pathlib import Path

HOME = Path(__file__).resolve().parent.parent


def seg_to_bbox_mask(seg_mask: np.ndarray) -> List[List[float]]:
    """
    Given a binary segmentation mask, find all connected regions and compute
    their bounding boxes. The output format is:

        [[x1, x2, ...],
         [y1, y2, ...],
         [width1, width2, ...],
         [height1, height2, ...]]

    For each bounding box:
        x: minimum column index (left coordinate)
        y: minimum row index (top coordinate)
        width: (max column - min column + 1)
        height: (max row - min row + 1)

    Parameters:
        seg_mask (np.ndarray): A 2D binary segmentation mask.

    Returns:
        list: A list containing four lists [x_list, y_list, width_list, height_list].
              Returns [[], [], [], []] if no segmentation regions are found.
    """
    # Label the connected components in the segmentation mask.
    labeled_array, num_features = label(seg_mask)

    norm_x_list = []
    norm_y_list = []
    norm_width_list = []
    norm_height_list = []

    # Get image dimensions.
    height, width = seg_mask.shape

    for i in range(1, num_features + 1):
        # Find indices for the i-th connected region.
        rows, cols = np.where(labeled_array == i)
        if rows.size == 0 or cols.size == 0:
            continue

        # Determine the bounding box in pixel coordinates.
        min_row, max_row = np.min(rows), np.max(rows)
        min_col, max_col = np.min(cols), np.max(cols)

        # Normalize the bounding box coordinates.
        norm_x = min_col / width
        norm_y = min_row / height
        norm_w = (max_col - min_col + 1) / width
        norm_h = (max_row - min_row + 1) / height

        norm_x_list.append(norm_x)
        norm_y_list.append(norm_y)
        norm_width_list.append(norm_w)
        norm_height_list.append(norm_h)

    return [norm_x_list, norm_y_list, norm_width_list, norm_height_list]


def topk_salient_mask(sm: np.ndarray, topk: Union[int, float]) -> np.ndarray:
    """
    Generate a binary mask where only the top salient pixels are set to 1.
    The `topk` parameter can be a percentage (float between 0 and 1) or an integer count.
    The output is a flattened NumPy array.
    """
    if isinstance(sm, torch.Tensor):
        sm = sm.detach().cpu().numpy()
    else:
        sm = np.asarray(sm)

    flat_sm = sm.flatten()
    n_pixels = flat_sm.size

    if isinstance(topk, float) and topk < 1:
        k = int(np.round(n_pixels * topk))
    else:
        k = int(topk)

    k = max(1, min(k, n_pixels))

    topk_indices = np.argpartition(-flat_sm, kth=k - 1)[:k]

    binary_mask = np.zeros_like(flat_sm, dtype=np.int64)
    binary_mask[topk_indices] = 1

    return binary_mask


def tensor_to_pil(input_data: Union[torch.Tensor, np.ndarray]) -> Image.Image:
    """
    Convert a tensor or numpy array to a PIL Image.
    """
    if isinstance(input_data, torch.Tensor):
        array = input_data.detach().cpu().numpy()
    elif isinstance(input_data, np.ndarray):
        array = input_data
    else:
        raise TypeError("Input must be a torch.Tensor or numpy.ndarray")

    if array.ndim == 3 and array.shape[0] in [1, 3]:
        array = np.transpose(array, (1, 2, 0))

    if np.issubdtype(array.dtype, np.floating):
        array = (array * 255).astype(np.uint8)

    return Image.fromarray(array)


def overlay_map(
    overlayviz: Union[torch.Tensor, np.ndarray],
    segviz: Union[torch.Tensor, np.ndarray],
    alpha: float = 0.5,
) -> Image.Image:
    """
    Overlay a segmentation map on top of an image.
    """
    # Convert both inputs to PIL Images
    base_img = tensor_to_pil(overlayviz)
    overlay_img = tensor_to_pil(segviz)

    if overlay_img.size != base_img.size:
        overlay_img = overlay_img.resize(base_img.size)

    base_img = base_img.convert("RGB")
    overlay_img = overlay_img.convert("RGB")

    blended = Image.blend(base_img, overlay_img, alpha)
    return blended


def bbox_to_image_mask_normalized(
    bboxes: List[List[float]], shape: Tuple[int, int]
) -> torch.Tensor:
    """
    Convert a list of normalized bounding boxes into a single binary mask for the image.
    """
    height, width = shape
    x_coords, y_coords, widths, heights = bboxes
    mask = np.zeros((height, width), dtype=np.uint8)
    for x, y, w, h in zip(x_coords, y_coords, widths, heights):
        x_min, x_max = int(x * width), int((x + w) * width)
        y_min, y_max = int(y * height), int((y + h) * height)
        mask[y_min:y_max, x_min:x_max] = 1
    return torch.tensor(mask)


def visualize_and_save(
    image: Image.Image, rollout_map: np.ndarray, save_path="attention_rollout.png"
):
    """
    Save the aggregated attention map overlayed on the image.
    """
    rollout_map_resized = rollout_map / rollout_map.max()
    rollout_map_resized = np.kron(rollout_map_resized, np.ones((16, 16)))
    cmap = plt.get_cmap("jet")
    heatmap = cmap(rollout_map_resized)
    heatmap = (heatmap[:, :, :3] * 255).astype(np.uint8)
    original_image = image.convert("RGB")
    overlay = Image.blend(original_image, Image.fromarray(heatmap), alpha=0.4)
    return overlay


def calculate_precision(
    ground_truth: np.ndarray, segmentation: np.ndarray, threshold: float = 0.5
) -> float:
    """
    Calculate the precision between a ground truth mask and a segmentation map.
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
    Create a side-by-side collage of two images and save to disk.
    """
    i1 = np.array(image1)
    i2 = np.array(image2)
    collage = np.hstack([i1, i2])
    image = Image.fromarray(collage)
    image.save(name)


def predict_fold_chunked(
    val_loader: DataLoader,
    model: nn.Module,
    target_layers: List[nn.Module],
    fold: int,
    chunk_size: int = 500,
    output_dir: str = f"{HOME}/output/chunk_predictions",
):
    """
    Process a validation DataLoader in chunks, computing model logits and Grad-CAM saliency maps,
    and persist each chunk (dict pid -> value) as .npy files.

    Args:
        val_loader (DataLoader): Iterable yielding batches with keys "data" (tensor) and "pid" (IDs).
        model (nn.Module): Trained binary classification model; sigmoid applied to its outputs.
        target_layers (List[nn.Module]): Layers to target for Grad-CAM computation.
        fold (int): Fold index used in output filenames.
        chunk_size (int, optional): Number of samples per saved chunk. Defaults to 500.
        output_dir (str, optional): Directory to store chunked .npy files. Created if missing.

    Side Effects:
        Writes two files per chunk:
            fold_{fold}_logits_chunk_{i}.npy (dict pid -> float logit)
            fold_{fold}_sm_chunk_{i}.npy (dict pid -> saliency map array of shape (H, W, 1))

    Notes:
        - Assumes binary classification; Grad-CAM targets class index 0.
        - Any remainder (< chunk_size) after iteration is saved as a final chunk.
        - Moves model to CUDA if available and runs in eval mode.
        - Saliency maps are stored with an added channel dimension (H, W, 1).

    Returns:
        None
    """
    os.makedirs(output_dir, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # Load model to device and set to eval mode
    model.to(device)
    model.eval()
    # Initialize variables for tracking logits and saliency maps
    all_logits = {}
    all_sm = {}
    sample_count = 0
    chunk_index = 0
    with GradCAM(model=model, target_layers=target_layers) as cam:
        for idx, batch in tqdm(
            enumerate(val_loader), total=len(val_loader), desc=f"Fold {fold}"
        ):
            pixel_values = batch["data"].to(device)
            pid = batch["pid"]
            pixel_values.requires_grad = True
            # Since this in binary classification, we will always get saliency maps for the positive class
            targets = [ClassifierOutputTarget(0) for _ in range(pixel_values.shape[0])]
            grayscale_cam = cam(input_tensor=pixel_values, targets=targets)
            with torch.no_grad():
                logits = torch.sigmoid(model(pixel_values).squeeze(1)).cpu().numpy()
            for k, p in enumerate(pid):
                all_logits[p] = logits[k]
                # Expand dims and transpose to (H, W, 1) as needed
                all_sm[p] = np.expand_dims(grayscale_cam[k], axis=0).transpose(1, 2, 0)
                sample_count += 1
                # Handle edge cases for the last batch
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
        # Save logtis and saliency maps
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
    validation_mode: str,
    collate_fn: Callable,
    logger,
    chunk_size: int = 500,
    threshold: float = 0.5,
):
    """
    Compute saliency evaluation metrics (IoU, precision within GT, coverage) for a validation
    dataset by loading precomputed per-fold logits and saliency maps chunk-by-chunk, forming
    an ensemble (mean across folds), and deriving bounding boxes from thresholded saliency maps.

    Parameters:
        val_dataset (Dataset): Validation dataset wrapped with necessary fields (data, label, pid, bbox).
        cache_dir (str): Directory containing cached numpy files: fold_{i}_logits_chunk_{j}.npy and fold_{i}_sm_chunk_{j}.npy.
        model_name (str): Model identifier used for output directory naming.
        img_size (int): Image size specifier (used in path construction).
        validation_mode (str): Either 'ood' or 'holdout'; selects output subdirectory.
        collate_fn (Callable): Collate function for DataLoader.
        logger: Logger object for error reporting.
        chunk_size (int, optional): Batch size for chunked processing. Default is 500.
        threshold (float, optional): Saliency threshold for binary mask generation. Default is 0.5.

    Returns:
        tuple(dict, dict, dict):
            - iou_all: pid -> Jaccard score between GT mask (from bbox) and predicted bbox mask.
            - precision_all: pid -> Precision (fraction of predicted positive pixels that are GT positive).
            - coverage_all: pid -> Fraction of pixels above saliency threshold prior to bbox fitting.

    """
    if not (validation_mode == "ood" or validation_mode == "holdout"):
        raise ValueError("Invalid validation mode, choose one of ood, holdout")
    # Define and create directories for saving ground truth maps, saliency maps, and bounding box visualizations
    gt_save_path = (
        f"{HOME}/output/{model_name}_maps/maps_{validation_mode}/maps_{img_size}/gt"
    )
    sm_save_path = (
        f"{HOME}/output/{model_name}_maps/maps_{validation_mode}/maps_{img_size}/sm"
    )
    bbox_save_path = (
        f"{HOME}/output/{model_name}_maps/maps_{validation_mode}/maps_{img_size}/bbox"
    )
    os.makedirs(gt_save_path, exist_ok=True)
    os.makedirs(sm_save_path, exist_ok=True)
    os.makedirs(bbox_save_path, exist_ok=True)
    # Create a DataLoader for processing the validation dataset in chunks
    val_loader = DataLoader(
        val_dataset,
        batch_size=chunk_size,
        shuffle=False,
        num_workers=8,
        pin_memory=True,
        collate_fn=collate_fn,
    )
    # Initialize variables for tracking logits and saliency maps
    iou_all = {}
    precision_all = {}
    coverage_all = {}
    operating_point = 0.5
    # Determine number of chunks based on one fold's files
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
        # Aggregate predictions from each fold for the current chunk
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
        # Compute the average prediction across folds for each sample
        for pid in ensemble_logits_chunk.keys():
            ensemble_logits_chunk[pid] = np.mean(ensemble_logits_chunk[pid], axis=0)
        for pid in ensemble_sm_chunk.keys():
            ensemble_sm_chunk[pid] = np.mean(ensemble_sm_chunk[pid], axis=0)
        # Move batch data to the appropriate device
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        pixel_values = batch["data"].to(device)
        gt = batch["label"]
        pids = batch["pid"]
        # Process each sample in the current batch
        for k, pid in enumerate(pids):
            # Handling mispredictions:
            try:
                if ensemble_logits_chunk[pid] > operating_point and gt[k] == 0:
                    iou_all[pid] = 0
                    precision_all[pid] = 0
                    sm = ensemble_sm_chunk[pid]
                    binary_masks = (
                        (torch.tensor(sm) > threshold).int().cpu().squeeze(-1).numpy()
                    )
                    coverage = np.sum(binary_masks) / binary_masks.size
                    coverage_all[pid] = coverage
                    continue
                if ensemble_logits_chunk[pid] < operating_point and gt[k] == 1:
                    iou_all[pid] = 0
                    precision_all[pid] = 0
                    continue
                if ensemble_logits_chunk[pid] < operating_point and gt[k] == 0:
                    continue
            except Exception as e:
                logger.error(e)
            # Process saliency map and bounding box visualization for valid predictions
            sm = ensemble_sm_chunk[pid]
            bbox = batch["bbox"][k].to(device)
            ground_truth_masks = bbox_to_image_mask_normalized(
                bbox, pixel_values.shape[2:]
            ).unsqueeze(-1)
            rgb_img = np.repeat(sm, 3, axis=-1)
            segviz = show_cam_on_image(rgb_img, sm, use_rgb=True)
            binary_mask_viz = pixel_values[k].permute(1, 2, 0).repeat(1, 1, 3)
            overlayviz = binary_mask_viz.clone()
            segviz_overlay = overlay_map(overlayviz, segviz, alpha=0.5)
            # Visualize ground truth bounding boxes (assumes the dataset provides this method)
            gtviz = val_dataset.visualize_bboxes(
                binary_mask_viz.detach().cpu().numpy(), bbox
            )
            # Save the visualizations
            seg_save_file = os.path.join(sm_save_path, f"{pid}.jpg")
            gt_save_file = os.path.join(gt_save_path, f"{pid}.jpg")
            bbox_save_file = os.path.join(bbox_save_path, f"{pid}.jpg")
            segviz_overlay.save(seg_save_file)
            gtviz.save(gt_save_file)
            # Convert the saliency map to a binary mask using the threshold.
            binary_masks = (
                (torch.tensor(sm) > threshold).int().cpu().squeeze(-1).numpy()
            )
            # Alternative approach (commented out): use top-k salient regions.
            # binary_masks = topk_salient_mask(sm, 0.05).int().cpu().numpy()
            # Fit a bounding box to the mask extracted from the saliency map
            # This step is necessary since we are evaluating bounding boxes extracted from saliency maps
            bboxpred = seg_to_bbox_mask(binary_masks)
            # Visualize and save the predicted bounding boxes on the original image.
            bboxviz = val_dataset.visualize_bboxes(
                binary_mask_viz.detach().cpu().numpy(), bboxpred
            )
            bboxviz.save(bbox_save_file)
            # Compute coverage as the fraction of pixels above the threshold.
            coverage = np.sum(binary_masks) / binary_masks.size
            # Convert the predicted bounding boxes back to a normalized binary mask.
            binary_masks = (
                bbox_to_image_mask_normalized(bboxpred, binary_masks.shape)
                .flatten()
                .cpu()
                .numpy()
            )
            ground_truth_masks = ground_truth_masks.int().cpu().flatten().numpy()
            # Calculate iou and precision
            iou_all[pid] = jaccard_score(ground_truth_masks, binary_masks)
            precision_all[pid] = calculate_precision(ground_truth_masks, binary_masks)
            coverage_all[pid] = coverage
        chunk_index += 1
    return iou_all, precision_all, coverage_all
