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

HOME_DIR = Path(__file__).resolve().parent.parent

from sharpness_matters.ptx.utils.logging_utils import initialize_logger
from sharpness_matters.config.load_config import cfg

logger = initialize_logger(level="debug")


def mask2rle(img: np.ndarray, width: int, height: int) -> str:
    """
    Convert binary mask to run-length encoding (RLE) format.

    Args:
        img (np.ndarray): Binary mask image
        width (int): Image width
        height (int): Image height

    Returns:
        str: RLE encoded string
    """
    rle = []
    lastColor = 0
    currentPixel = 0
    runStart = -1
    runLength = 0

    for x in range(width):
        for y in range(height):
            currentColor = img[x][y]
            if currentColor != lastColor:
                if currentColor == 255:
                    runStart = currentPixel
                    runLength = 1
                else:
                    rle.append(str(runStart))
                    rle.append(str(runLength))
                    runStart = -1
                    runLength = 0
                    currentPixel = 0
            elif runStart > -1:
                runLength += 1
            lastColor = currentColor
            currentPixel += 1

    return " ".join(rle)


def rle2mask(rle: str, width: int, height: int) -> np.ndarray:
    """
    Convert run-length encoding (RLE) format to binary mask.

    Args:
        rle (str): RLE encoded string
        width (int): Image width
        height (int): Image height

    Returns:
        np.ndarray: Binary mask array
    """
    mask = np.zeros(width * height)
    array = np.asarray([int(x) for x in rle.split()])
    starts = array[0::2]
    lengths = array[1::2]
    current_position = 0
    for index, start in enumerate(starts):
        current_position += start
        mask[current_position : current_position + lengths[index]] = 255
        current_position += lengths[index]
    return mask.reshape(width, height)


class ChestXrayDataset(Dataset):
    """
    PyTorch Dataset for SIIM-ACR chest X-ray classification with pneumothorax detection.
    """

    def __init__(
        self,
        file_paths: List[str],
        label_dir: str,
        img_size: int = 224,
        split: str = "test",
        transform: Optional[Any] = None,
        test_size: float = 0.1,
        split_seed: int = 42,
        stats_dir: str = os.path.join(
            HOME_DIR, "output", "cache", "siim_acr_stats.pkl"
        ),
        label_cache_dir: str = os.path.join(HOME_DIR, "output", "cache", "label.pkl"),
    ):
        """
        Initialize SIIM-ACR chest X-ray dataset.

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
        if os.path.exists(stats_dir) and os.path.exists(label_cache_dir):
            with open(stats_dir, "rb") as f:
                self.global_stats = pickle.load(f)
            self.mean, self.std = self.global_stats["mean"], self.global_stats["std"]
            with open(label_cache_dir, "rb") as f:
                tmp_labels = pickle.load(f)
        else:
            raise RuntimeError(
                f"Required cache files not found.\n"
                f"Expected stats at: {stats_dir}\n"
                f"Expected labels at: {label_cache_dir}\n"
                f"Please run siim_acr_classification_dataset.py first to generate the stratified train/test split and statistics."
            )

        labels = []
        for i in tqdm(self.image_ids, desc=f"Generating {split} split"):
            labels.append(tmp_labels[i])
        indices = np.arange(len(self.image_ids))
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
        self.labels = self.process_labels(label_df)
        self.labels = {i: self.labels[i] for i in self.image_ids}
        pos, neg = 0, 0
        for i in self.labels.values():
            val = i["label"]
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
        self.resize = transforms.Resize((img_size, img_size), antialias=True)
        logger.info(f"Initialzed dataset of size {len(self.labels)}")

    def __len__(self) -> int:
        return len(self.labels)

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
        mask = torch.tensor(self.labels[image_id]["mask"], dtype=torch.long)
        mask = self.resize(mask.unsqueeze(0)).squeeze(0)
        label = torch.tensor(self.labels[image_id]["label"], dtype=torch.long)

        # Apply transformations
        image = self.z_normalize(image)
        if self.transform:
            image = self.transform(image)
        else:
            image = self.preprocess(image)
        image = image.float()
        return {"data": image, "mask": mask, "label": label, "pid": pid}

    def process_labels(
        self, df: pd.DataFrame
    ) -> Dict[str, Dict[str, Union[np.ndarray, int]]]:
        """
        Process labels from CSV file and create masks.

        Args:
            df (pd.DataFrame): DataFrame containing image IDs and RLE encodings

        Returns:
            Dict[str, Dict[str, Union[np.ndarray, int]]]: Dictionary mapping image IDs to labels and masks
        """
        image_id_arr = df["ImageId"].unique()
        labels = {}
        for index, image_id in tqdm(
            enumerate(image_id_arr), total=len(image_id_arr), desc="Processing labels"
        ):
            if image_id not in self.image_ids:
                continue
            idx = list(
                filter(
                    lambda x: image_id in self.file_paths[x],
                    range(len(self.file_paths)),
                )
            )
            dataset = pydicom.dcmread(self.file_paths[idx[0]])
            image_data = dataset.pixel_array
            record_arr = df[df["ImageId"] == image_id]
            # Visualize patient has multi segment
            label = 0
            mask = np.zeros((1024, 1024))
            for _, row in record_arr.iterrows():
                if row[" EncodedPixels"] != " -1":
                    mask_ = rle2mask(row[" EncodedPixels"], 1024, 1024).T
                    mask[mask_ == 255] = 255
                    label = 1
            labels[image_id] = {"mask": mask, "label": label}
        return labels

    def z_normalize(self, xray_array: np.ndarray) -> np.ndarray:
        """
        Perform Z-normalization on an input X-ray NumPy array.

        Args:
            xray_array (np.ndarray): Input X-ray image array

        Returns:
            np.ndarray: Z-normalized and min-max scaled X-ray image array
        """
        # Compute mean and standard deviation of the array
        self.mean = np.mean(xray_array)
        self.std = np.std(xray_array)

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
        # metadata = dicom.file_meta
        # logger.debug(f"Metadata: {dicom}")
        patient_id = dicom.get("PatientID", None)
        image = dicom.pixel_array  # Get the raw pixel data
        return image, patient_id

    def extract_demographics(
        self, image_id: str
    ) -> Dict[str, Optional[Union[str, int]]]:
        """
        Extract age and sex metadata from a DICOM file.

        Args:
            image_id (str): The image ID to extract demographics for

        Returns:
            Dict[str, Optional[Union[str, int]]]: Dictionary containing age, sex, and patient_id
        """
        # Find the file path for this image_id
        file_idx = list(
            filter(
                lambda x: image_id in self.file_paths[x], range(len(self.file_paths))
            )
        )
        if not file_idx:
            logger.error(f"Could not find file path for image_id: {image_id}")
            return {"image_id": image_id, "age": None, "sex": None, "patient_id": None}

        dicom_path = self.file_paths[file_idx[0]]
        try:
            dicom = pydicom.dcmread(dicom_path)

            age = dicom.get("PatientAge", None)
            sex = dicom.get("PatientSex", None)
            patient_id = dicom.get("PatientID", None)

            # Clean age data - handle various age formats
            if age is not None:
                try:
                    if isinstance(age, str):
                        # Remove common suffixes and clean the string
                        age_str = age.strip().upper()
                        if age_str.endswith("Y"):
                            age_str = age_str[:-1]
                        elif age_str.endswith("YEARS"):
                            age_str = age_str[:-5]
                        elif age_str.endswith("YRS"):
                            age_str = age_str[:-3]

                        # Try to extract numeric part
                        import re

                        age_match = re.search(r"\d+", age_str)
                        if age_match:
                            age = int(age_match.group())
                        else:
                            age = None
                    else:
                        # Try to convert directly to int
                        age = int(age)

                    # Validate age range (reasonable human age)
                    if age is not None and (age < 0 or age > 120):
                        age = None

                except (ValueError, TypeError, AttributeError):
                    age = None

            # Clean sex data - standardize to M/F
            if sex and isinstance(sex, str):
                sex = sex.upper().strip()
                if sex not in ["M", "F"]:
                    sex = None

            return {
                "image_id": image_id,
                "age": age,
                "sex": sex,
                "patient_id": patient_id,
            }
        except Exception as e:
            logger.error(f"Error extracting demographics for {image_id}: {e}")
            return {"image_id": image_id, "age": None, "sex": None, "patient_id": None}

    def get_all_demographics(self) -> List[Dict[str, Optional[Union[str, int]]]]:
        """
        Extract demographics for all images in the dataset.

        Returns:
            List[Dict[str, Optional[Union[str, int]]]]: List of dictionaries containing demographics for each image
        """
        demographics = []
        for image_id in tqdm(self.image_ids, desc="Extracting demographics"):
            demo_data = self.extract_demographics(image_id)
            demo_data["label"] = self.labels[image_id][
                "label"
            ]  # Add the pneumothorax label
            demographics.append(demo_data)
        return demographics

    def visualize_as_jpg(self, image: torch.Tensor, idx: int, output_path: str) -> None:
        """
        Save the specified DICOM image as a JPG for visualization.

        Args:
            image (torch.Tensor): Image tensor to save
            idx (int): Index of the image in the dataset
            output_path (str): Path to save the JPG file
        """
        # Convert to 8-bit image (range [0, 255])
        image = image.numpy()
        image_8bit = (image * 255).astype(np.uint8)

        # Save as JPG using Pillow
        image_pil = Image.fromarray(image_8bit, "L")
        image_pil.save(output_path)
        logger.info(f"Image saved as JPG at: {output_path}")

    def visualize_segmentation(
        self,
        image: torch.Tensor,
        mask: torch.Tensor,
        alpha: float = 0.5,
        mask_color: Tuple[float, float, float] = (1, 0, 0),
    ) -> np.ndarray:
        """
        Visualize a segmentation mask overlaid on an image.

        Args:
            image (torch.Tensor): The base image tensor
            mask (torch.Tensor): The segmentation mask tensor
            alpha (float): Transparency of the mask overlay (0-1)
            mask_color (Tuple[float, float, float]): RGB color of the mask overlay

        Returns:
            np.ndarray: Image with overlaid segmentation mask
        """
        # Ensure the image and mask have the same dimensions
        image = cv2.resize(
            image.numpy(),
            dsize=(mask.shape[0], mask.shape[1]),
            interpolation=cv2.INTER_CUBIC,
        )
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
    if not os.path.exists(f"{HOME_DIR}/output/samples"):
        os.mkdir(f"{HOME_DIR}/output/samples")
    if not os.path.exists(f"{HOME_DIR}/output/masks"):
        os.mkdir(f"{HOME_DIR}/output/masks")
    img_size = 1024
    ds = ChestXrayDataset(train_names, csv_path, img_size=img_size)
    for idx in range(500):
        batch = ds[idx]
        img = batch["data"]
        label = batch["label"]
        mask = batch["mask"]
        pid = batch["pid"]
        if label == 0:
            continue
        if len(mask.shape) > 1:
            ds.visualize_as_jpg(
                img.squeeze(0), idx, f"{HOME_DIR}/output/samples/{pid}.jpg"
            )
            seg = ds.visualize_segmentation(
                img.permute(1, 2, 0).expand(*img.shape[1:], 3), mask
            )
            Image.fromarray(seg).save(f"{HOME_DIR}/output/masks/{pid}.jpg")
            logger.debug(f"Saved {pid} to {HOME_DIR}/output/masks/{pid}.jpg")
