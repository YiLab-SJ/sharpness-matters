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
from pydicom.pixel_data_handlers.util import apply_voi_lut

from sharpness_matters.ptx.utils.logging_utils import initialize_logger
from sharpness_matters.config.load_config import cfg

logger = initialize_logger(level="debug")
HOME = Path(__file__).resolve().parent.parent


class VinDataset(Dataset):
    def __init__(
        self,
        root_dir: str = cfg.ptx.vin_dr.root_dir,
        img_size: int = 224,
        transform=None,
        stats_dir=f"{HOME}/output/cache/vin_stats.pkl",
    ):
        self.root_dir = root_dir
        # If either vin500neg.txt or vin500pos.txt do not exist, raise an exception
        if not os.path.exists(f"{HOME}/output/vin500neg.txt") or not os.path.exists(
            f"{HOME}/output/vin500pos.txt"
        ):
            raise FileNotFoundError(
                f"Required files vin500neg.txt or vin500pos.txt not found in {HOME}/output/. "
                "Run create_vin_dr_subset.py first to generate these files."
            )
        # Read image IDs from vin500neg.txt and vin500pos.txt
        with open(f"{HOME}/output/vin500neg.txt", "r") as file:
            data = file.read()
        self.image_ids = data.replace("\n", " ").split(" ")[:-1]
        with open(f"{HOME}/output/vin500pos.txt", "r") as file:
            data = file.read()
        pos = data.replace("\n", " ").split(" ")[:-1]
        logger.debug(f"Pos: {len(pos)}, Neg: {len(self.image_ids)}")
        self.image_ids = self.image_ids + pos
        self.labels = {}
        for id in self.image_ids:
            if id in pos:
                self.labels[id] = 1
            else:
                self.labels[id] = 0
        self.transform = transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Resize((img_size, img_size), antialias=True),
            ]
        )
        self.mean, self.std = self.get_global_mean_std(stats_dir)
        logger.info(f"Len of Vin dataset: {len(self.image_ids)}")

    def __len__(self):
        return len(self.image_ids)

    def __getitem__(self, idx: int) -> dict:
        image_id = self.image_ids[idx]
        dicom_path = os.path.join(self.root_dir, f"{self.image_ids[idx]}.dicom")
        image, pid = self.load_dicom(dicom_path)
        image = self.z_normalize(image)
        image = self.transform(image)
        return {"data": image, "label": self.labels[image_id], "mask": image}

    def get_global_mean_std(self, stats_path: str) -> tuple:
        if os.path.exists(stats_path):
            with open(stats_path, "rb") as f:
                data = pickle.load(f)
                return data["mean"], data["std"]
        mean, std = 0, 0
        for idx, id in tqdm(
            enumerate(self.image_ids),
            desc="Computing stats for the dataset",
            total=len(self.image_ids),
        ):
            dicom_path = os.path.join(self.root_dir, f"{self.image_ids[idx]}.dicom")
            image, pid = self.load_dicom(dicom_path)
            mean += image.mean()
            std += image.std()
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
            xray_array (np.ndarray): Input X-ray image array.
        Returns:
            np.ndarray: Z-normalized X-ray image array.
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

    @staticmethod
    def load_dicom(file_path: str) -> np.ndarray:
        """
        Load a DICOM file and extract the pixel array.

        Parameters:
        - file_path (str): Path to the DICOM file.

        Returns:
        - numpy.ndarray: The pixel data as a NumPy array.
        """
        dicom = pydicom.dcmread(file_path)
        dicom.BitsStored = (
            16  # This was added due to a warning recommending this change
        )
        patient_id = dicom.get("PatientID", None)
        image = dicom.pixel_array  # Get the raw pixel data
        # image = apply_voi_lut(dicom.pixel_array, dicom)
        image = image.astype(np.float32)
        return image, patient_id

    def extract_demographics(self, image_id: str) -> dict:
        """
        Extract age and sex metadata from a DICOM file.

        Parameters:
        - image_id (str): The image ID to extract demographics for.

        Returns:
        - dict: Dictionary containing age, sex, and patient_id
        """
        dicom_path = os.path.join(self.root_dir, f"{image_id}.dicom")
        try:
            dicom = pydicom.dcmread(dicom_path)
            dicom.BitsStored = 16

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
            else:
                age = None
                logger.warning(f"Couldn't find age for image {image_id}")

            # Clean sex data - standardize to M/F
            if sex and isinstance(sex, str):
                sex = sex.upper().strip()
                if sex not in ["M", "F"]:
                    sex = None
            else:
                logger.warning(f"Couldn't find sex")

            return {
                "image_id": image_id,
                "age": age,
                "sex": sex,
                "patient_id": patient_id,
            }
        except Exception as e:
            logger.error(f"Error extracting demographics for {image_id}: {e}")
            return {"image_id": image_id, "age": None, "sex": None, "patient_id": None}

    def get_all_demographics(self) -> list:
        """
        Extract demographics for all images in the dataset.

        Returns:
        - list: List of dictionaries containing demographics for each image
        """
        demographics = []
        for image_id in tqdm(self.image_ids, desc="Extracting demographics"):
            demo_data = self.extract_demographics(image_id)
            demo_data["label"] = self.labels[image_id]  # Add the pneumothorax label
            demographics.append(demo_data)
        return demographics

    def visualize_as_jpg(self, idx: int, output_path: str):
        """
        Save the specified DICOM image as a JPG for visualization.

        Parameters:
        - idx (int): Index of the image in the dataset.
        - output_path (str): Path to save the JPG file.
        """
        # Load and preprocess the image
        image_id = self.image_ids[idx]
        dicom_path = os.path.join(self.root_dir, f"{self.image_ids[idx]}.dicom")
        image, pid = self.load_dicom(dicom_path)
        image = self.z_normalize(image)

        image = self.transform(image).squeeze(0).numpy()

        # Convert to 8-bit image (range [0, 255])
        image_8bit = (image * 255.0).astype(np.uint8)

        # Save as JPG using Pillow
        image_pil = Image.fromarray(image_8bit)
        image_pil.save(output_path)
        logger.info(f"Image saved as JPG at: {output_path}")


if __name__ == "__main__":
    img_size = 1024
    dataset = VinDataset(img_size=img_size)
    logger.debug(dataset[0]["data"].shape)
    # dataset.visualize_as_jpg(0, '/cnvrg/test.jpg')
