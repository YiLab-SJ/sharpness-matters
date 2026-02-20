import os
from pathlib import Path
import torch
from torch.utils.data import Dataset
import pydicom
import numpy as np
from PIL import Image
import glob
import pandas as pd
from tqdm import tqdm
import cv2
from torchvision import transforms
import pickle
import ast
from typing import Dict, List, Tuple, Union, Optional, Any

from sharpness_matters.pneumonia.utils.logger_utils import initialize_logger
from sharpness_matters.config.load_config import cfg

HOME = Path(__file__).resolve().parent.parent
DATA_DIR = cfg.pneumonia.siim_covid.root_dir
logger = initialize_logger(level="info")


class ChestXrayOOD(Dataset):
    """
    PyTorch Dataset for SIIM COVID-19 chest X-ray detection with opacity/pneumonia labels and bounding boxes.
    """

    def __init__(
        self,
        file_paths: List[str],
        label_dirs: List[str],
        img_size: int = 224,
        transform: Optional[Any] = None,
        stats_dir: str = f"{HOME}/output/cache/siim_covid_stats.pkl",
        label_cache_dir: str = f"{HOME}/output/cache/covid_label.pkl",
    ):
        """
        Initialize SIIM COVID-19 chest X-ray dataset.

        Args:
            file_paths (List[str]): List of paths to DICOM files
            label_dirs (List[str]): List containing paths to study-level and image-level CSV files
            img_size (int): Target image size for resizing
            transform (Optional[Any]): Image transformations to apply
            stats_dir (str): Path to cached dataset statistics
        """
        self.file_paths = file_paths
        study_label_df = pd.read_csv(label_dirs[0])
        image_label_df = pd.read_csv(label_dirs[1])
        study_id_arr = study_label_df["id"].unique()
        image_id_arr = image_label_df["id"].unique()
        self.study_ids = list(study_id_arr)
        image_ids = list(image_id_arr)
        self.study_ids = [i.split("_")[0] for i in self.study_ids]
        image_ids = [i.split("_")[0] for i in image_ids]
        # Select an image id for each study id
        self.study_image_mapping = {}
        for image_id in image_ids:
            study_id = image_label_df.loc[
                image_label_df["id"] == f"{image_id}_image", "StudyInstanceUID"
            ].values[0]
            self.study_image_mapping[study_id] = image_id.split("_")[0]

        if os.path.exists(label_cache_dir):
            with open(label_cache_dir, "rb") as handle:
                annotations = pickle.load(handle)
                self.labels = annotations["labels"]
                self.bboxes = annotations["bboxs"]
            with open(stats_dir, "rb") as f:
                self.global_stats = pickle.load(f)
        else:
            annotations, self.global_stats = self.process_labels(
                image_df=image_label_df, study_df=study_label_df, arr=self.study_ids
            )
            self.labels = annotations["labels"]
            self.bboxes = annotations["bboxs"]
            with open(label_cache_dir, "wb") as handle:
                pickle.dump(annotations, handle)
            with open(stats_dir, "wb") as f:
                pickle.dump(self.global_stats, f)
        self.mean, self.std = self.global_stats["mean"], self.global_stats["std"]
        logger.debug(f"Mean : {self.mean}, Std: {self.std}")
        logger.debug(f"Labels: {len(self.labels)}")
        logger.debug(f"Label length: {len(self.labels)}")

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
        """
        Get the total number of samples in the dataset.

        Returns:
            int: Number of samples
        """
        return len(self.study_ids)

    def __getitem__(self, idx: int) -> Dict[str, Union[torch.Tensor, str]]:
        """
        Get a single sample from the dataset.

        Args:
            idx (int): Index of the sample

        Returns:
            Dict[str, Union[torch.Tensor, str]]: Dictionary containing image data, label, study ID, and bboxes
        """
        image_id = self.study_ids[idx]
        # Load the DICOM file
        file_idx = list(
            filter(
                lambda x: image_id in self.file_paths[x], range(len(self.file_paths))
            )
        )[0]
        dicom_path = self.file_paths[file_idx]
        image, pid, pi = self.load_dicom(dicom_path)

        # Apply transformations
        image = self.z_normalize(image)
        if pi == "MONOCHROME1":
            image = 1 - image
        if self.transform:
            image = self.transform(image)
        else:
            image = self.preprocess(image)

        image = image.float()
        label = torch.tensor(self.labels[image_id], dtype=torch.long)
        bbox = torch.tensor(np.array(self.bboxes[image_id]))
        return {"data": image, "label": label, "pid": image_id, "bbox": bbox}

    def z_normalize(self, xray_array: np.ndarray) -> np.ndarray:
        """
        Perform Z-normalization on an input X-ray NumPy array.

        Args:
            xray_array (np.ndarray): Input X-ray image array

        Returns:
            np.ndarray: Z-normalized and min-max scaled X-ray image array
        """
        # Compute mean and standard deviation of the array
        mean = np.mean(xray_array)
        std = np.std(xray_array)
        # mean, std = self.mean, self.std

        # Avoid division by zero
        if self.std == 0:
            logger.warning(
                "Standard deviation of the input array is zero. Cannot perform Z-normalization."
            )
            return xray_array

        # Apply Z-normalization
        normalized_array = (xray_array - mean) / std
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
        self, study_df: pd.DataFrame, image_df: pd.DataFrame, arr: List[str]
    ) -> Tuple[Dict[str, Any], Dict[str, float]]:
        """
        Process labels and bounding boxes from study and image CSV files and compute global statistics.

        Args:
            study_df (pd.DataFrame): DataFrame containing study-level labels
            image_df (pd.DataFrame): DataFrame containing image-level labels and bounding boxes
            arr (List[str]): List of study IDs to process

        Returns:
            Tuple[Dict[str, Any], Dict[str, float]]: Annotations dictionary and statistics dictionary
        """
        labels = {}
        bboxs = {}
        pos, neg = 0, 0
        mean, std = 0, 0

        for index, study_id in tqdm(enumerate(arr), total=len(arr)):
            study_record = study_df[study_df["id"] == f"{study_id}_study"]
            image_id = self.study_image_mapping[study_id]
            idx = list(
                filter(
                    lambda x: image_id in self.file_paths[x],
                    range(len(self.file_paths)),
                )
            )
            dicom_data = pydicom.dcmread(self.file_paths[idx[0]])
            meta = dicom_data.file_meta
            image_data = dicom_data.pixel_array
            mean += image_data.mean()
            std += image_data.std()
            height, width = image_data.shape
            # labels[study_id] = 1-study_record["Negative for Pneumonia"].values[0]
            boxes = image_df[image_df["id"] == f"{image_id}_image"]["boxes"].values[0]
            label = (
                image_df[image_df["id"] == f"{image_id}_image"]["label"]
                .values[0]
                .split(" ")[0]
            )
            logger.debug(f"Study ID : {study_id}")
            logger.debug(f"Label: {label}")
            labels[study_id] = 1 if label == "opacity" else 0
            logger.debug(labels[study_id])
            if labels[study_id] == 1:
                bbox_list = ast.literal_eval(boxes)
                boxes = {
                    "x": [bbox["x"] for bbox in bbox_list],
                    "y": [bbox["y"] for bbox in bbox_list],
                    "width": [bbox["width"] for bbox in bbox_list],
                    "height": [bbox["height"] for bbox in bbox_list],
                }
                bbox = [
                    [i / width for i in boxes["x"]],
                    [i / height for i in boxes["y"]],
                    [i / width for i in boxes["width"]],
                    [i / height for i in boxes["height"]],
                ]
            else:
                bbox = [[], [], [], []]
            bboxs[study_id] = bbox
        mean = mean / len(arr)
        std = std / len(arr)
        stats_dir = {"mean": mean, "std": std}
        annotations = {"labels": labels, "bboxs": bboxs}
        return annotations, stats_dir

    @staticmethod
    def load_dicom(file_path: str) -> Tuple[np.ndarray, Optional[str], str]:
        """
        Load a DICOM file and extract the pixel array.

        Args:
            file_path (str): Path to the DICOM file

        Returns:
            Tuple[np.ndarray, Optional[str], str]: Pixel data array, patient ID, and photometric interpretation
        """
        dicom = pydicom.dcmread(file_path)
        pi = dicom.PhotometricInterpretation
        patient_id = dicom.get("PatientID", None)
        image = dicom.pixel_array  # Get the raw pixel data
        return image, patient_id, pi

    def extract_demographics(self, study_id: str) -> Dict[str, Union[str, int, float]]:
        """
        Extract demographic information from the DICOM file.

        Args:
            study_id (str): Study ID to extract demographics for

        Returns:
            Dict[str, Union[str, int, float]]: Dictionary containing demographic information
        """
        # Get corresponding image_id for this study_id
        if study_id not in self.study_image_mapping:
            logger.error(f"Could not find image mapping for study_id: {study_id}")
            return {"study_id": study_id, "age": None, "sex": None, "patient_id": None}

        image_id = self.study_image_mapping[study_id]

        # Find the file path for this image_id
        file_idx = list(
            filter(
                lambda x: image_id in self.file_paths[x], range(len(self.file_paths))
            )
        )
        if not file_idx:
            logger.error(f"Could not find file path for image_id: {image_id}")
            return {"study_id": study_id, "age": None, "sex": None, "patient_id": None}

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
                "study_id": study_id,
                "age": age,
                "sex": sex,
                "patient_id": patient_id,
            }
        except Exception as e:
            logger.error(f"Error extracting demographics for {study_id}: {e}")
            return {"study_id": study_id, "age": None, "sex": None, "patient_id": None}

    def get_all_demographics(self) -> List[Dict[str, Union[str, int, float]]]:
        """
        Get demographic information for all patients in the dataset.

        Returns:
            List[Dict[str, Union[str, int, float]]]: List of dictionaries containing demographic
                information for each patient
        """
        demographics = []
        for study_id in tqdm(self.study_ids, desc="Extracting demographics"):
            demo_data = self.extract_demographics(study_id)
            demo_data["label"] = self.labels[study_id]  # Add the COVID/opacity label
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
        image_id = self.study_ids[idx]
        file_idx = list(
            filter(
                lambda x: image_id in self.file_paths[x], range(len(self.file_paths))
            )
        )[0]
        dicom_path = self.file_paths[file_idx]
        image, _, pi = self.load_dicom(dicom_path)
        image = self.z_normalize(image)
        if pi == "MONOCHROME1":
            image = 1 - image
        image = self.preprocess(image).squeeze(0).cpu().numpy()
        logger.debug(f"Visualized image shape: {image.shape}")

        # Convert to 8-bit image (range [0, 255])
        image_8bit = (image * 255).astype(np.uint8)

        # Save as JPG using Pillow
        image_pil = Image.fromarray(image_8bit)
        image_pil = image_pil.resize((1024, 1024))
        image_pil.save(output_path)
        logger.info(f"Image saved as JPG at: {output_path}")

    def visualize_bboxes(
        self, image: np.ndarray, bboxes: list, thickness: int = 2
    ) -> np.ndarray:
        """
        Visualize bounding boxes on an image.

        Parameters:
        - image (numpy.ndarray): The base image.
        - bboxes (list): Normalized bounding boxes [[x1, x2, ...], [y1, y2, ...], [w1, w2, ...], [h1, h2, ...]].

        Returns:
        - numpy.ndarray: Image with bounding boxes drawn.
        """
        image = (image * 255).astype(np.uint8)
        height, width = image.shape[:2]
        x_list, y_list, w_list, h_list = bboxes

        for x, y, w, h in zip(x_list, y_list, w_list, h_list):
            start_point = (int(x * width), int(y * height))
            end_point = (int((x + w) * width), int((y + h) * height))
            image = cv2.rectangle(image, start_point, end_point, (255, 0, 0), thickness)
        image_pil = Image.fromarray(image)
        image_pil = image_pil.resize((1024, 1024))
        return image_pil


if __name__ == "__main__":
    train_dicom_dir = os.path.join(DATA_DIR, "covid_train/")
    train_glob = f"{train_dicom_dir}/*/*/*.dcm"
    train_names = [f for f in sorted(glob.glob(train_glob))]
    if not os.path.exists(f"{HOME}/output/samples"):
        os.mkdir(f"{HOME}/output/samples")
    if not os.path.exists(f"{HOME}/output/masks"):
        os.mkdir(f"{HOME}/output/masks")
    img_size = 1024
    logger.info(f"Len of train names: {len(train_names)}")
    ds = ChestXrayOOD(
        train_names,
        [f"{DATA_DIR}/covid_study_level.csv", f"{DATA_DIR}/covid_image_level.csv"],
        img_size=img_size,
    )
    pids = []
    for idx in tqdm(range(len(ds))):
        if idx > 500:
            break
        batch = ds[idx]
        img = batch["data"]
        label = batch["label"]
        bbox = batch["bbox"]
        if label == 0:
            continue
        ds.visualize_as_jpg(idx, f"{HOME}/output/samples/sample_{idx}.jpg")
        seg = ds.visualize_bboxes(
            img.permute(1, 2, 0).expand(*img.shape[1:], 3).numpy(), bbox
        )
        seg.save(f"{HOME}/output/masks/seg_{idx}.jpg")
    logger.debug(f"Number of unique pids: {len(set(pids))}")
