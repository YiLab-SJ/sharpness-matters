import torch
import numpy as np
from pathlib import Path
import os
import click
import glob
import pytorch_lightning as pl
from torch import nn
from sklearn.model_selection import KFold
from pytorch_lightning import LightningModule, Trainer, seed_everything
from torch.utils.data import (
    DataLoader,
    Subset,
    Dataset,
    TensorDataset,
    WeightedRandomSampler,
)
from sklearn.model_selection import KFold
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
from torchvision import transforms
from tqdm import tqdm

from sharpness_matters.pneumonia.datasets.rsna_pneumonia import ChestXrayHoldout
from sharpness_matters.pneumonia.models.cnn import CNNBinaryClassifier
from sharpness_matters.config.load_config import cfg

HOME = Path(__file__).resolve().parent.parent
DATA_DIR = cfg.pneumonia.rsna.root_dir
CHECKPOINT_DIR = cfg.pneumonia.model.checkpoint_dir


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
    """
    Train a CNN binary classifier using K-fold cross-validation.
    Parameters:
        dataset (Dataset): Full dataset implementing __len__ and __getitem__; must expose class_count for pos_weight.
        k_folds (int): Number of folds for KFold splitting.
        batch_size (int): Mini-batch size for DataLoaders.
        num_epochs (int): Maximum number of epochs per fold.
        lr (float): Learning rate for optimizer.
        img_size (int): Image size passed to model initialization.
        seed (int): Random seed for reproducibility.
        val_check_interval (float): Fraction of an epoch between validation runs.
        patience (int): Early stopping patience (scaled by val_check_interval internally).
        model_name (str): Model backbone identifier; selects between DenseNet and ResNet variants.
    Behavior:
        - Splits dataset into k_folds with stratified shuffling (via KFold).
        - For each fold: initializes model, trains with PyTorch Lightning Trainer, applies checkpointing and early stopping.
        - Computes and prints average validation loss and accuracy across folds.
    Returns:
        None. Side effects include training logs, checkpoint files, and printed aggregate metrics.
    """
    pl.seed_everything(seed)
    # K-Fold Cross-Validation
    kf = KFold(n_splits=k_folds, shuffle=True, random_state=seed)
    fold_results = []

    for fold, (train_idx, val_idx) in enumerate(kf.split(dataset)):
        print(f"\nTraining Fold {fold + 1}/{k_folds}...")

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
            raise Exception(
                "Invalid model name, only densenet and resnet are supported"
            )

        # Define callbacks
        callbacks = []
        checkpoint_callback = ModelCheckpoint(
            filename="last-checkpoint",
            # monitor="val/loss",
            # mode="min",
            # save_top_k=1)
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
            devices=[1],
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
    print(f"\nK-Fold Cross-Validation Results:")
    print(f"Average Validation Loss: {avg_val_loss:.4f}")
    print(f"Average Validation Accuracy: {avg_val_acc:.4f}")


@click.command()
@click.option(
    "--model_name",
    "-m",
    type=str,
    required=True,
    help="Choose between resnet or densenet",
)
def main(model_name):
    train_dicom_dir = os.path.join(DATA_DIR, "stage_2_train_images/")
    train_glob = f"{train_dicom_dir}/*.dcm"
    train_names = [f for f in sorted(glob.glob(train_glob))]
    batch_size = 8
    lr = 1e-4
    num_epochs = 20
    for model_name in ["resnet"]:
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
            ds = ChestXrayHoldout(
                train_names,
                f"{DATA_DIR}/stage_2_train_labels.csv",
                transform=train_transforms,
                split="train",
                img_size=img_size,
                mode="classification",
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
