import os
import glob
import pickle
from pathlib import Path
from PIL import Image
import numpy as np
import pandas as pd
import cv2
import torch
from torch.utils.data import Dataset
from torchvision import transforms
from sklearn.model_selection import train_test_split
import pydicom
from tqdm import tqdm
from typing import Dict, List, Tuple, Union, Optional, Any

from sharpness_matters.pneumonia.utils.logger_utils import initialize_logger
from sharpness_matters.config.load_config import cfg

logger = initialize_logger(level="debug")
HOME = Path(__file__).resolve().parent.parent
DATA_DIR = cfg.pneumonia.rsna_pneumonia.root_dir


class ChestXrayHoldout(Dataset):
    """
    PyTorch Dataset for RSNA pneumonia detection from chest X-ray DICOM files.
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
        stats_dir: str = f"{HOME}/output/cache/siim_acr_stats.pkl",
        label_cache_dir: str = f"{HOME}/output/cache/label.pkl",
        mode: str = "saliency",
    ):
        """
        Initialize RSNA chest X-ray pneumonia dataset.

        Args:
            file_paths (List[str]): List of paths to DICOM files
            label_dir (str): Path to CSV file with labels and bounding boxes
            img_size (int): Target image size for resizing
            split (str): Dataset split ('train' or 'test')
            transform (Optional[Any]): Image transformations to apply
            test_size (float): Fraction of data for test split
            split_seed (int): Random seed for train/test split
            stats_dir (str): Path to cached dataset statistics
            mode (str): Dataset mode ('saliency' includes bboxes)
        """
        self.file_paths = file_paths
        label_df = pd.read_csv(label_dir)
        image_id_arr = label_df["patientId"].unique()
        self.image_ids = list(image_id_arr)
        logger.debug(f"Len image id arr: {len(image_id_arr)}")

        logger.debug(
            f"Len of image ids: {len(self.image_ids)}, Len of non unique image ids: {len(label_df['patientId'])}"
        )
        if os.path.exists(label_cache_dir) and os.path.exists(stats_dir):
            with open(label_cache_dir, "rb") as handle:
                annotations = pickle.load(handle)
                self.labels = annotations["labels"]
                self.bboxes = annotations["bboxs"]
            with open(stats_dir, "rb") as f:
                self.global_stats = pickle.load(f)
        else:
            os.makedirs(os.path.dirname(label_cache_dir), exist_ok=True)
            os.makedirs(os.path.dirname(stats_dir), exist_ok=True)
            annotations, self.global_stats = self.process_labels(
                label_df, self.image_ids
            )
            self.labels = annotations["labels"]
            self.bboxes = annotations["bboxs"]
            with open(label_cache_dir, "wb") as handle:
                pickle.dump(annotations, handle)
            with open(stats_dir, "wb") as f:
                pickle.dump(self.global_stats, f)
            logger.info(f"Processed and cached labels and statistics at {label_cache_dir} and {stats_dir}")
        self.mean, self.std = self.global_stats["mean"], self.global_stats["std"]
        logger.debug(f"Mean : {self.mean}, Std: {self.std}")
        logger.debug(f"Labels: {len(self.labels)}")
        logger.debug(f"Label length: {len(self.labels)}")
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
        self.mode = mode
        logger.info(f"Initialzed dataset of size {len(self.labels)}")

    def __len__(self) -> int:
        """
        Get the total number of samples in the dataset.

        Returns:
            int: Number of samples
        """
        return len(self.image_ids)

    def __getitem__(self, idx: int) -> Dict[str, Union[torch.Tensor, str]]:
        """
        Get a single sample from the dataset.

        Args:
            idx (int): Index of the sample

        Returns:
            Dict[str, Union[torch.Tensor, str]]: Dictionary containing image data, label, patient ID, and optionally bboxes
        """
        image_id = self.image_ids[idx]
        # Load the DICOM file
        file_idx = list(
            filter(
                lambda x: image_id in self.file_paths[x], range(len(self.file_paths))
            )
        )[0]
        dicom_path = self.file_paths[file_idx]
        image, _ = self.load_dicom(dicom_path)

        # Apply transformations
        image = self.z_normalize(image)
        if self.transform:
            image = self.transform(image)
        else:
            image = self.preprocess(image)

        # image = torch.tensor(image, dtype=torch.float32) # Add channel dimension
        image = image.float()
        label = torch.tensor(self.labels[image_id], dtype=torch.long)
        bbox = torch.tensor(np.array(self.bboxes[image_id]))
        if self.mode == "saliency":
            return {"data": image, "label": label, "pid": image_id, "bbox": bbox}
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
        mean = np.mean(xray_array)
        std = np.std(xray_array)
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
        self, df: pd.DataFrame, image_id_arr: List[str]
    ) -> Tuple[Dict[str, Any], Dict[str, float]]:
        """
        Process labels and bounding boxes from CSV file and compute global statistics.

        Args:
            df (pd.DataFrame): DataFrame containing patient IDs, labels, and bounding box coordinates
            image_id_arr (List[str]): List of image/patient IDs to process

        Returns:
            Tuple[Dict[str, Any], Dict[str, float]]: Annotations dictionary and statistics dictionary
        """
        labels = {}
        all_bboxs = {}
        mean, std = 0, 0
        for _, image_id in tqdm(enumerate(image_id_arr), total=len(image_id_arr)):
            record_arr = df[df["patientId"] == image_id]
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
            height, width = image_data.shape
            labels[image_id] = np.max(record_arr["Target"].tolist())
            if labels[image_id] == 1:
                bbox = [
                    record_arr["x"].values / width,
                    record_arr["y"].values / height,
                    record_arr["width"].values / width,
                    record_arr["height"].values / height,
                ]
            else:
                bbox = [[], [], [], []]
            all_bboxs[image_id] = bbox
        mean = mean / len(image_id_arr)
        std = std / len(image_id_arr)
        stats_dir = {"mean": mean, "std": std}
        annotations = {"labels": labels, "bboxs": all_bboxs}
        return annotations, stats_dir

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
            demo_data["label"] = self.labels[image_id]  # Add the pneumonia label
            demographics.append(demo_data)
        return demographics

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
        logger.debug(f"Visualized image shape: {image.shape}")

        # Convert to 8-bit image (range [0, 255])
        image_8bit = (image * 255).astype(np.uint8)

        # Save as JPG using Pillow
        image_pil = Image.fromarray(image_8bit)
        image_pil = image_pil.resize((1024, 1024))
        image_pil.save(output_path)
        logger.info(f"Image saved as JPG at: {output_path}")

    def visualize_bboxes(
        self, image: np.ndarray, bboxes: List[List[float]], thickness: int = 2
    ) -> Image.Image:
        """
        Visualize bounding boxes on an image.

        Args:
            image (np.ndarray): The base image array
            bboxes (List[List[float]]): Normalized bounding boxes [[x1, x2, ...], [y1, y2, ...], [w1, w2, ...], [h1, h2, ...]]
            thickness (int): Line thickness for bounding box rectangles

        Returns:
            Image.Image: PIL Image with bounding boxes drawn
        """
        image = (image * 255).astype(np.uint8)
        height, width = image.shape[:2]

        for x, y, w, h in zip(*bboxes):
            start_point = (int(x * width), int(y * height))
            end_point = (int((x + w) * width), int((y + h) * height))
            image = cv2.rectangle(image, start_point, end_point, (255, 0, 0), thickness)
        image_pil = Image.fromarray(image)
        image_pil = image_pil.resize((1024, 1024))
        return image_pil


if __name__ == "__main__":
    train_dicom_dir = os.path.join(DATA_DIR, "stage_2_train_images/")
    test_dicom_dir = os.path.join(DATA_DIR, "stage_2_test_images/")
    train_glob = f"{train_dicom_dir}/*.dcm"
    test_glob = f"{test_dicom_dir}/*.dcm"
    train_names = list(sorted(glob.glob(train_glob)))
    test_names = list(sorted(glob.glob(test_glob)))
    if not os.path.exists(f"{HOME}/output/samples"):
        os.mkdir(f"{HOME}/output/samples")
    if not os.path.exists(f"{HOME}/output/masks"):
        os.mkdir(f"{HOME}/output/masks")
    IMG_SIZE = 1024
    train_ds = ChestXrayHoldout(
        train_names,
        f"{DATA_DIR}/stage_2_train_labels.csv",
        split="train",
        img_size=IMG_SIZE,
    )
    test_ds = ChestXrayHoldout(
        train_names,
        f"{DATA_DIR}/stage_2_train_labels.csv",
        split="test",
        img_size=IMG_SIZE,
    )
    logger.info(f"Len train: {len(train_ds)}, Len Test: {len(test_ds)}")
    ds = test_ds
    train_image_ids = train_ds.image_ids
    test_image_ids = test_ds.image_ids
    logger.debug(
        f"Number of overlapping indices: {len(list(set(train_image_ids) & set(test_image_ids)))}"
    )
    pids = []
    for idx in tqdm(range(len(ds))):
        batch = ds[idx]
        img = batch["data"]
        label = batch["label"]
        bbox = batch["bbox"]
        # print(f"Label : {label}")
        # pids.append(batch['pid'])
        if label == 0:
            continue
        # ds.visualize_as_jpg(idx, f'{HOME}/output/samples/sample_{idx}.jpg')
        ds.visualize_as_jpg(idx, f"{HOME}/output/samples/sample_{idx}.jpg")
        seg = ds.visualize_bboxes(
            img.permute(1, 2, 0).expand(*img.shape[1:], 3).numpy(), bbox
        )
        seg.save(f"{HOME}/output/masks/seg_{idx}.jpg")
    logger.debug(f"Number of unique pids: {len(set(pids))}")
