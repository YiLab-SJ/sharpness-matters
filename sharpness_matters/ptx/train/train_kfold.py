import numpy as np
import os
import glob
from pathlib import Path
import click
import pytorch_lightning as pl
from torchvision import transforms
from pytorch_lightning import Trainer
from torch.utils.data import DataLoader, Subset, Dataset
from sklearn.model_selection import KFold
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.callbacks.early_stopping import EarlyStopping

from sharpness_matters.ptx.datasets.siim_acr_classification_dataset import (
    ChestXrayDataset,
)
from sharpness_matters.ptx.models.cnn import CNNBinaryClassifier
from sharpness_matters.ptx.utils.logging_utils import initialize_logger
from sharpness_matters.config.load_config import cfg

HOME = Path(__file__).resolve().parent.parent
DATA_DIR = cfg.ptx.siim_acr.root_dir
logger = initialize_logger(level="debug")
CHECKPOINT_DIR = cfg.ptx.model.checkpoint_dir


# Training function with K-Fold Cross-Validation
def train_with_kfold(
    dataset: Dataset,
    k_folds: int = 5,
    batch_size: int = 32,
    num_epochs: int = 20,
    lr: float = 1e-3,
    img_size: int = 512,
    seed: int = 42,
    val_check_interval: float = 0.5,
    patience: int = 5,
    model_name: str = "densenet",
):
    pl.seed_everything(seed)

    # K-Fold Cross-Validation
    kf = KFold(n_splits=k_folds, shuffle=True, random_state=seed)
    fold_results = []

    for fold, (train_idx, val_idx) in enumerate(kf.split(dataset)):
        logger.info(f"\nTraining Fold {fold + 1}/{k_folds}...")

        # Split dataset into training and validation subsets
        train_subset = Subset(dataset, train_idx)
        val_subset = Subset(dataset, val_idx)

        # Dataloaders
        class_counts = train_subset.dataset.class_count
        pos_weight = class_counts[0] / class_counts[1]
        train_loader = DataLoader(
            train_subset, batch_size=batch_size, shuffle=True, num_workers=2
        )
        val_loader = DataLoader(val_subset, batch_size=batch_size, num_workers=2)

        # Initialize model
        if "densenet" in model_name:
            model = CNNBinaryClassifier(
                lr=lr,
                img_size=img_size,
                model_name="densenet121",
                pos_weight=pos_weight,
            )
        elif "resnet" in model_name:
            model = CNNBinaryClassifier(
                lr=lr,
                img_size=img_size,
                model_name="resnet152.a2_in1k",
                pos_weight=pos_weight,
            )
        else:
            raise ValueError(
                f"Unsupported model_name '{model_name}'. Expected one of ['densenet', 'resnet']."
            )

        # Define callbacks
        callbacks = []
        checkpoint_callback = ModelCheckpoint(
            filename="last-checkpoint",
        )
        callbacks.append(checkpoint_callback)
        callbacks.append(
            EarlyStopping(
                monitor="val/loss",  # Metric to monitor
                patience=patience
                / val_check_interval,  # Number of epochs with no improvement before stopping
                mode="min",  # Minimize the validation loss
                verbose=True,
            )
        )

        # Trainer
        trainer = Trainer(
            max_epochs=num_epochs,
            devices=[0],
            default_root_dir=f"{CHECKPOINT_DIR}/{model_name}/{model_name}_{img_size}_logs",
            callbacks=callbacks,
            accelerator="auto",  # Automatically use GPU if available
            precision=32,  # Mixed precision if GPU available
            log_every_n_steps=10,
            val_check_interval=val_check_interval,
        )

        # Train the model
        trainer.fit(model, train_loader, val_loader)

        # Validate and log results
        val_results = trainer.validate(
            model, val_loader, ckpt_path=checkpoint_callback.best_model_path
        )
        fold_results.append(val_results[0])  # Each entry is a dictionary with metrics

    # Aggregate results
    avg_val_loss = np.mean([result["val/loss"] for result in fold_results])
    avg_val_acc = np.mean([result["val/acc"] for result in fold_results])
    logger.info(f"\nK-Fold Cross-Validation Results:")
    logger.info(f"Average Validation Loss: {avg_val_loss:.4f}")
    logger.info(f"Average Validation Accuracy: {avg_val_acc:.4f}")


@click.command()
@click.option(
    "--model_name",
    "-m",
    type=str,
    required=True,
    help="Choose between resnet or densenet",
)
def main(model_name):
    train_dicom_dir = os.path.join(DATA_DIR, "dicom-images-train")
    test_dicom_dir = os.path.join(DATA_DIR, "dicom-images-test")
    logger.debug(f"Test Len: {len(os.listdir(test_dicom_dir))}")
    logger.debug(f"Train Len: {len(os.listdir(train_dicom_dir))}")
    train_glob = f"{train_dicom_dir}/*/*/*.dcm"
    test_glob = f"{test_dicom_dir}/*/*/*.dcm"
    train_names = [f for f in sorted(glob.glob(train_glob))]
    batch_size = 8
    lr = 1e-4
    num_epochs = 20
    for img_size in [64, 128, 224, 512, 768, 1024]:
        train_transforms = transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Resize((img_size, img_size)),
                transforms.RandomHorizontalFlip(p=0.5),  # Random horizontal flip
                transforms.RandomRotation(
                    degrees=15
                ),  # Random rotation within 15 degrees
            ]
        )
        ds = ChestXrayDataset(
            train_names, f"{DATA_DIR}/train-rle.csv", transform=train_transforms
        )
        train_with_kfold(
            ds,
            k_folds=5,
            batch_size=batch_size,
            num_epochs=num_epochs,
            lr=lr,
            img_size=img_size,
            model_name=model_name,
        )


if __name__ == "__main__":
    main()
