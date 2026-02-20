import os
from pathlib import Path
import torch
from torch.utils.data import Dataset
import numpy as np
from PIL import Image
import subprocess
import glob
import pandas as pd
from tqdm import tqdm
import cv2
from torchvision import transforms
import pickle
import nibabel as nib
from typing import Dict, Any, Tuple, Union, List, Optional

from sharpness_matters.ptx.utils.logging_utils import initialize_logger
from sharpness_matters.config.load_config import cfg

HOME = Path(__file__).resolve().parent.parent
logger = initialize_logger(level="info")


class OODTestDataset(Dataset):
    """
    PyTorch Dataset for PTX498 out-of-distribution test data with NIfTI images.
    """

    def __init__(
        self,
        root_dir: str = cfg.ptx.ptx498.root_dir,
        img_size: int = 224,
        stats_path: str = f"{HOME}/output/cache/stats_ptx498.pkl",
    ):
        """
        Initialize PTX498 dataset.

        Args:
            root_dir (str): Root directory containing Site folders with NIfTI files
            img_size (int): Target image size for resizing
            stats_path (str): Path to save/load dataset statistics
        """
        self.root_dir = root_dir
        self.img_size = img_size
        self.parse_images()
        self.transform = transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Resize((img_size, img_size), antialias=True),
            ]
        )
        self.resize = transforms.Resize((img_size, img_size), antialias=True)
        self.mean, self.std = self.get_global_mean_std(stats_path)

    def get_global_mean_std(self, stats_path: str) -> Tuple[float, float]:
        """
        Compute or load global mean and standard deviation for the dataset.

        Args:
            stats_path (str): Path to save/load statistics file

        Returns:
            Tuple[float, float]: Global mean and standard deviation
        """
        if os.path.exists(stats_path):
            with open(stats_path, "rb") as f:
                data = pickle.load(f)
                return data["mean"], data["std"]
        mean, std = 0, 0
        for id in tqdm(self.image_ids, desc="Computing stats for the dataset"):
            site = id.split("_")[0]
            number = id.split("_")[1]
            image_path = os.path.join(
                self.root_dir, f"Site{site}", f"{number}.4.img.nii.gz"
            )
            mask_path = os.path.join(
                self.root_dir, f"Site{site}", f"{number}.5.mask.nii.gz"
            )
            mask = nib.load(mask_path).get_fdata()
            image = nib.load(image_path).get_fdata()
            mean_ = image.mean()
            std_ = image.std()
            mean += mean_
            std += std_
        mean = mean / len(self.image_ids)
        std = std / len(self.image_ids)
        stats_dict = {"mean": mean, "std": std}
        # Save the dictionary to a file
        with open(stats_path, "wb") as f:
            pickle.dump(stats_dict, f)
        return mean, std

    def z_normalize(self, xray_array: np.ndarray) -> np.ndarray:
        """
        Perform Z-normalization on an input X-ray NumPy array.

        Args:
            xray_array (np.ndarray): Input X-ray image array

        Returns:
            np.ndarray: Z-normalized and min-max scaled X-ray image array
        """
        # Compute mean and standard deviation of the array
        # mean = np.mean(xray_array)
        # std = np.std(xray_array)

        # Avoid division by zero
        if self.std == 0:
            logger.warning(
                "Standard deviation of the input array is zero. Cannot perform Z-normalization."
            )
            return xray_array

        # Apply Z-normalization
        normalized_array = (xray_array - self.mean) / self.std
        # Min-Max scaling to [0, 1]
        min_val = np.min(normalized_array)
        max_val = np.max(normalized_array)
        scaled_array = (normalized_array - min_val) / (max_val - min_val)
        return scaled_array

    def parse_images(self) -> None:
        """
        Parse and collect image IDs from all sites in the dataset.
        Populates self.image_ids with format 'Site_ID'.
        """
        sites = ["A", "B", "C"]
        self.image_ids = []
        for site in sites:
            all_paths = os.listdir(os.path.join(self.root_dir, f"Site{site}"))
            ids = [i.split(".")[0] for i in all_paths]
            ids = set(ids)
            for id in ids:
                self.image_ids.append(f"{site}_{id}")

    def visualize_segmentation(
        self,
        image: np.ndarray,
        mask: np.ndarray,
        alpha: float = 0.5,
        mask_color: Tuple[float, float, float] = (1, 0, 0),
    ) -> np.ndarray:
        """
        Visualize a segmentation mask overlaid on an image.

        Args:
            image (np.ndarray): The base image (H, W, 3)
            mask (np.ndarray): The segmentation mask (H, W), binary or multi-class
            alpha (float): Transparency of the mask overlay (0-1)
            mask_color (Tuple[float, float, float]): RGB color of the mask overlay

        Returns:
            np.ndarray: Image with overlaid segmentation mask
        """
        # Ensure the image and mask have the same dimensions
        if image.shape[:2] != mask.shape:
            raise ValueError(
                f"Image and mask must have the same height and width. Image shape: {image.shape}, Mask shape: {mask.shape}"
            )
        image = np.array(image * 255).astype(np.uint8)
        # Convert binary mask to RGB if necessary
        if len(mask.shape) == 2:  # Single-channel binary or multi-class mask
            colored_mask = np.zeros_like(image, dtype=np.uint8)
            for c in range(3):
                colored_mask[..., c] = mask * mask_color[c]  # Apply mask color
        else:
            colored_mask = mask  # Assume already RGB for multi-class

        # Overlay the mask on the image
        overlaid_image = cv2.addWeighted(image, 1 - alpha, colored_mask, alpha, 0)
        return overlaid_image

    def __len__(self) -> int:
        return len(self.image_ids)

    def __getitem__(
        self, idx: int
    ) -> Dict[str, Union[torch.Tensor, np.ndarray, int, str]]:
        """
        Get a single sample from the dataset.

        Args:
            idx (int): Index of the sample to retrieve

        Returns:
            Dict[str, Union[torch.Tensor, np.ndarray, int, str]]: Dictionary containing:
                - data: Processed image tensor
                - mask: Segmentation mask array
                - label: Binary label (always 1 for pneumothorax)
                - pid: Patient ID string
        """
        # Extract image and mask paths
        image_id = self.image_ids[idx]
        site = image_id.split("_")[0]
        number = image_id.split("_")[1]
        image_path = os.path.join(
            self.root_dir, f"Site{site}", f"{number}.4.img.nii.gz"
        )
        mask_path = os.path.join(
            self.root_dir, f"Site{site}", f"{number}.5.mask.nii.gz"
        )
        # Load from paths
        mask = nib.load(mask_path).get_fdata()
        if len(mask.shape) < 3:
            mask = np.expand_dims(mask, -1)
        mask = self.resize(torch.tensor(mask).permute(2, 0, 1))
        image = nib.load(image_path).get_fdata().astype(np.float32)
        image = np.expand_dims(image, -1)
        image = self.z_normalize(image)
        image = self.transform(image)  # This is only a resize, mask was resized earlier
        # For some reason, the images and masks were rotated by 90 degrees in the original dataset
        image = transforms.functional.rotate(image, angle=90)
        mask = transforms.functional.rotate(mask, angle=90)
        return {
            "data": image,
            "mask": np.squeeze(mask, -1) * 255.0,
            "label": 1,
            "pid": f"{site}_{number}",
        }


if __name__ == "__main__":
    dataset = OODTestDataset()
    for idx in range(len(dataset)):
        batch = dataset[idx]
        logger.debug(f"Image id: {dataset.image_ids[idx]}")
        img = batch["data"]
        mask = batch["mask"]
        logger.debug(f"Image type: {type(img)}, Shape: {img.shape}")
        logger.debug(f"Mean: {img.mean()}, Max: {img.max()}, Min: {img.min()}")
        logger.debug(f"Mask type: {type(mask)}, Shape: {mask.shape}")
        logger.debug(f"Mean: {mask.mean()}, Max: {mask.max()}, Min: {mask.min()}")
        # Visualize segmentation and save
        out_dir = Path(HOME) / "output" / "seg_viz"
        out_dir.mkdir(parents=True, exist_ok=True)

        # Prepare image (H,W,3) in [0,1]
        if torch.is_tensor(img):
            img_np = img.squeeze().cpu().numpy()  # (H,W)
        else:
            img_np = np.squeeze(img)
        img_rgb = np.stack([img_np] * 3, axis=-1)  # (H,W,3)

        # Prepare mask (H,W) binary
        mask_arr = mask
        if torch.is_tensor(mask_arr):
            mask_arr = mask_arr.squeeze().cpu().numpy()
        else:
            mask_arr = np.squeeze(mask_arr)

        seg_viz = dataset.visualize_segmentation(
            image=img_rgb, mask=mask_arr, alpha=0.4, mask_color=(1, 0, 0)
        )

        save_path = out_dir / f"{batch['pid']}.png"
        cv2.imwrite(str(save_path), cv2.cvtColor(seg_viz, cv2.COLOR_RGB2BGR))
        logger.debug(f"Saved segmentation visualization to {save_path}")
        break
