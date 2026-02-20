import os
from pathlib import Path
import torch
from torch.utils.data import Dataset
import pydicom
import numpy as np
from PIL import Image
import subprocess
import glob
import pandas as pd
from tqdm import tqdm
import cv2
from torchvision import transforms
import pickle
from sklearn.model_selection import train_test_split
from typing import Dict, Any, Tuple, Union, List, Optional


from sharpness_matters.ptx.utils.logging_utils import initialize_logger
from sharpness_matters.config.load_config import cfg

HOME_DIR = Path(__file__).resolve().parent.parent
logger = initialize_logger(level="info")


class ChestXrayDataset(Dataset):
    """
    PyTorch Dataset for SIIM-ACR chest X-ray classification (binary pneumothorax detection).
    """

    def __init__(
        self,
        file_paths: List[str],
        label_dir: str,
        img_size: int = 224,
        split: str = "train",
        transform: Optional[Any] = None,
        test_size: float = 0.1,
        split_seed: int = 42,
        stats_dir: str = os.path.join(
            HOME_DIR, "output", "cache", "siim_acr_stats.pkl"
        ),
        label_cache_dir: str = os.path.join(HOME_DIR, "output", "cache", "label.pkl"),
    ):
        """
        Initialize SIIM-ACR chest X-ray classification dataset.

        Args:
            file_paths (List[str]): List of paths to DICOM files
            label_dir (str): Path to CSV file with labels
            img_size (int): Target image size for resizing
            split (str): Dataset split ('train' or 'test')
            transform (Optional[Any]): Image transformations to apply
            test_size (float): Fraction of data for test split
            split_seed (int): Random seed for train/test split
            stats_dir (str): Path to cached dataset statistics
            label_cache_dir (str): Path to cached labels
        """
        self.file_paths = file_paths
        label_df = pd.read_csv(label_dir)
        image_id_arr = label_df["ImageId"].unique()
        self.image_ids = list(image_id_arr)
        if os.path.exists(label_cache_dir) and os.path.exists(stats_dir):
            try:
                with open(label_cache_dir, "rb") as handle:
                    self.labels = pickle.load(handle)
            except Exception as e:
                logger.error(e)
            try:
                with open(stats_dir, "rb") as f:
                    self.global_stats = pickle.load(f)
            except Exception as e:
                logger.error(e)
        else:
            logger.info(
                f"Cache dir for saved labels and dataset stats not found. Generating split and labels."
            )
            self.labels, self.global_stats = self.process_labels(label_df)
            with open(label_cache_dir, "wb") as handle:
                pickle.dump(self.labels, handle)
            with open(stats_dir, "wb") as f:
                pickle.dump(self.global_stats, f)
        self.mean, self.std = self.global_stats["mean"], self.global_stats["std"]

        indices = np.arange(len(self.image_ids))
        labels = []
        for i in tqdm(self.image_ids, desc=f"Generating {split} split"):
            labels.append(self.labels[i])
        train_idx, test_idx = train_test_split(
            indices,
            stratify=labels,
            test_size=test_size,
            random_state=split_seed,
        )
        if split == "train":
            indices = train_idx
        else:
            indices = test_idx
        self.image_ids = list(np.array(self.image_ids)[indices])
        self.labels = {i: self.labels[i] for i in self.image_ids}
        pos, neg = 0, 0
        for val in self.labels.values():
            if val == 1:
                pos += 1
            else:
                neg += 1
        self.class_count = {1: pos, 0: neg}
        logger.info(f"Class counts: {self.class_count}")
        # Transformations
        self.preprocess = transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Resize((img_size, img_size), antialias=True),
            ]
        )
        self.transform = transform
        logger.info(f"Initialzed dataset of size {len(self.labels)}")

    def __len__(self) -> int:
        return len(self.image_ids)

    def __getitem__(self, idx: int) -> Dict[str, Union[torch.Tensor, str]]:
        image_id = self.image_ids[idx]
        # Load the DICOM file
        file_idx = list(
            filter(
                lambda x: image_id in self.file_paths[x], range(len(self.file_paths))
            )
        )[0]
        dicom_path = self.file_paths[file_idx]
        image, pid = self.load_dicom(dicom_path)

        # Apply transformations
        image = self.z_normalize(image)
        if self.transform:
            image = self.transform(image)
        else:
            image = self.preprocess(image)

        # image = torch.tensor(image, dtype=torch.float32) # Add channel dimension
        image = image.float()
        label = torch.tensor(self.labels[image_id], dtype=torch.long)
        return {"data": image, "label": label, "pid": image_id}

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

    def get_weights(self) -> List[float]:
        """
        Calculate class weights for balanced sampling.

        Returns:
            List[float]: List of weights for each sample based on class frequency
        """
        weights = []
        for val in self.labels.values():
            weights.append(1 / self.class_count[val])
        return weights

    def process_labels(
        self, df: pd.DataFrame
    ) -> Tuple[Dict[str, int], Dict[str, float]]:
        """
        Process labels from CSV file and compute global statistics.

        Args:
            df (pd.DataFrame): DataFrame containing image IDs and RLE encodings

        Returns:
            Tuple[Dict[str, int], Dict[str, float]]: Labels dictionary and statistics dictionary
        """
        image_id_arr = df["ImageId"].unique()
        labels = {}
        pos, neg = 0, 0
        mean, std = 0, 0
        for index, image_id in tqdm(enumerate(image_id_arr), total=len(image_id_arr)):
            idx = list(
                filter(
                    lambda x: image_id in self.file_paths[x],
                    range(len(self.file_paths)),
                )
            )
            dataset = pydicom.dcmread(self.file_paths[idx[0]])
            image_data = dataset.pixel_array
            mean += image_data.mean()
            std += image_data.std()
            record_arr = df[df["ImageId"] == image_id]
            # Visualize patient has multi segment
            label = 0
            for _, row in record_arr.iterrows():
                if row[" EncodedPixels"] != " -1":
                    label = 1
            labels[image_id] = label
        mean = mean / len(image_id_arr)
        std = std / len(image_id_arr)
        stats = {"mean": mean, "std": std}
        return labels, stats

    @staticmethod
    def load_dicom(file_path: str) -> Tuple[np.ndarray, Optional[str]]:
        """
        Load a DICOM file and extract the pixel array.

        Args:
            file_path (str): Path to the DICOM file

        Returns:
            Tuple[np.ndarray, Optional[str]]: Pixel data array and patient ID
        """
        dicom = pydicom.dcmread(file_path)
        metadata = dicom.file_meta
        patient_id = dicom.get("PatientID", None)
        image = dicom.pixel_array  # Get the raw pixel data
        return image, patient_id

    def visualize_as_jpg(self, idx: int, output_path: str) -> None:
        """
        Save the specified DICOM image as a JPG for visualization.

        Args:
            idx (int): Index of the image in the dataset
            output_path (str): Path to save the JPG file
        """
        # Load and preprocess the image
        image_id = self.image_ids[idx]
        file_idx = list(
            filter(
                lambda x: image_id in self.file_paths[x], range(len(self.file_paths))
            )
        )[0]
        dicom_path = self.file_paths[file_idx]
        image, _ = self.load_dicom(dicom_path)
        image = self.z_normalize(image)
        image = self.preprocess(image).squeeze(0).cpu().numpy()

        # Convert to 8-bit image (range [0, 255])
        image_8bit = (image * 255).astype(np.uint8)

        # Save as JPG using Pillow
        image_pil = Image.fromarray(image_8bit)
        image_pil = image_pil.resize((1024, 1024))
        image_pil.save(output_path)
        logger.info(f"Image saved as JPG at: {output_path}")

    def visualize_segmentation(
        self,
        image: np.ndarray,
        mask: np.ndarray,
        alpha: float = 0.5,
        mask_color: Tuple[float, float, float] = (0, 0, 1),
    ) -> np.ndarray:
        """
        Visualize a segmentation mask overlaid on an image.

        Args:
            image (np.ndarray): The base image array
            mask (np.ndarray): The segmentation mask array
            alpha (float): Transparency of the mask overlay (0-1)
            mask_color (Tuple[float, float, float]): RGB color of the mask overlay

        Returns:
            np.ndarray: Image with overlaid segmentation mask
        """
        # Ensure the image and mask have the same dimensions
        if image.shape[:2] != mask.shape:
            raise ValueError("Image and mask must have the same height and width.")
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


if __name__ == "__main__":
    DATA_DIR = cfg.ptx.siim_acr.root_dir
    train_dicom_dir = os.path.join(DATA_DIR, "dicom-images-train")
    test_dicom_dir = os.path.join(DATA_DIR, "dicom-images-test")
    csv_path = os.path.join(DATA_DIR, "train-rle.csv")
    train_glob = f"{train_dicom_dir}/*/*/*.dcm"
    test_glob = f"{test_dicom_dir}/*/*/*.dcm"
    train_names = [f for f in sorted(glob.glob(train_glob))]
    test_names = [f for f in sorted(glob.glob(test_glob))]
    logger.debug(f"Home dir: {HOME_DIR}")
    if not os.path.exists(f"{HOME_DIR}/output/samples"):
        os.mkdir(f"{HOME_DIR}/output/samples")
    if not os.path.exists(f"{HOME_DIR}/output/masks"):
        os.mkdir(f"{HOME_DIR}/output/masks")

    img_size = 224
    train_ds = ChestXrayDataset(train_names, csv_path, split="train", img_size=img_size)
    test_ds = ChestXrayDataset(train_names, csv_path, split="test", img_size=img_size)
    logger.debug(f"Len train: {len(train_ds)}, Len Test: {len(test_ds)}")
    ds = test_ds
    train_image_ids = train_ds.image_ids
    test_image_ids = test_ds.image_ids
    logger.debug(
        f"Number of overlapping indices: {len(list(set(train_image_ids) & set(test_image_ids)))}"
    )
    pids = []
    for idx in tqdm(range(len(ds))):
        if idx > 500:
            break
        batch = ds[idx]
        img = batch["data"]
        label = batch["label"]
        logger.debug(f"PID: {batch['pid']}")
        # print(f"Label : {label}")
        # pids.append(batch['pid'])
        if label == 0:
            continue
        ds.visualize_as_jpg(idx, f"/{HOME_DIR}/output/samples/sample_{idx}.jpg")
    logger.debug(f"Number of unique pids: {len(set(pids))}")
