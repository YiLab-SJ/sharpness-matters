import pickle
import torch
import numpy as np
import os
import glob
import pytorch_lightning as pl
from pathlib import Path
from Typing import Tuple, Dict, Any
import ray
from ray import tune
from ray.tune.schedulers import ASHAScheduler
from ray.train import RunConfig, ScalingConfig, CheckpointConfig
from ray.train.torch import TorchTrainer
from ray.tune.search.optuna import OptunaSearch
from ray.tune.integration.pytorch_lightning import TuneReportCallback
from ray.train.lightning import (
    RayDDPStrategy,
    RayLightningEnvironment,
    RayTrainReportCallback,
    prepare_trainer,
)
from pytorch_lightning import Trainer
from torch.utils.data import DataLoader, Subset, Dataset
from torchvision import transforms
from sklearn.model_selection import train_test_split

from sharpness_matters.ptx.datasets.siim_acr_classification_dataset import (
    ChestXrayDataset,
)
from sharpness_matters.ptx.models.cnn import CNNBinaryClassifier
from sharpness_matters.ptx.utils.logging_utils import initialize_logger
from sharpness_matters.config.load_config import cfg

HOME = Path(__file__).resolve().parent.parent
DATA_DIR = cfg.ptx.siim_acr.root_dir
logger = initialize_logger(level="debug")


def stratified_split(
    dataset: Dataset, val_size: float = 0.2, random_state: int = 42
) -> Tuple[Subset, Subset]:
    """
    Splits a PyTorch dataset into stratified train and validation subsets.

    Args:
        dataset (Dataset): The PyTorch dataset to split.
        val_size (float): Proportion of the dataset to include in the validation split.
        random_state (int): Random seed for reproducibility.

    Returns:
        train_subset (Subset): Training subset of the dataset.
        val_subset (Subset): Validation subset of the dataset.
    """
    # Extract targets (labels) from the dataset
    targets = [
        dataset[i]["label"] for i in range(len(dataset))
    ]  # Assumes labels are at index 1

    # Perform stratified split
    train_idx, val_idx = train_test_split(
        np.arange(len(targets)),
        test_size=val_size,
        stratify=targets,
        random_state=random_state,
    )

    # Create Subsets for training and validation
    train_subset = Subset(dataset, train_idx)
    val_subset = Subset(dataset, val_idx)

    return train_subset, val_subset


# Train Function for Ray Tune
def train_model(config: Dict[str, Any]):
    global train_cfg
    pl.seed_everything(train_cfg["seed"])
    generator = torch.Generator()
    generator.manual_seed(train_cfg["seed"])
    # Generate synthetic dataset
    dataset = ChestXrayDataset(
        train_cfg["train_names"],
        f"{DATA_DIR}/train-rle.csv",
        transform=train_cfg["train_transforms"],
        split="train",
        img_size=img_size,
    )
    class_counts = dataset.class_count
    pos_weight = class_counts[0] / class_counts[1]
    train_dataset, val_dataset = stratified_split(dataset, val_size=0.2)

    train_loader = DataLoader(
        train_dataset, batch_size=config["batch_size"], shuffle=True
    )
    val_loader = DataLoader(val_dataset, batch_size=config["batch_size"])

    # Initialize the model
    if "densenet" in train_cfg["model_name"] or "resnet" in train_cfg["model_name"]:
        model = CNNBinaryClassifier(
            lr=config["lr"],
            model_name=train_cfg["model_name"],
            alpha=1.0,
            gamma=config["gamma"],
            pos_weight=pos_weight,
        )
    else:
        raise ValueError(
            f"Unsupported model_name '{train_cfg['model_name']}'. Expected one of ['densenet', 'resnet']."
        )
    # Define the Tune callback to report metrics
    tune_callback = TuneReportCallback(
        {
            "val/loss": "val/loss",
            "val/acc": "val/acc",
            "val/auc": "val/auc",
            "val/f1": "val/f1",
        },
        on="validation_end",
    )
    raycallback = RayTrainReportCallback()

    # Trainer
    trainer = Trainer(
        max_epochs=train_cfg["num_epochs"],
        accelerator="auto",
        callbacks=[tune_callback, raycallback],
        log_every_n_steps=10,
        plugins=[RayLightningEnvironment()],
        strategy=RayDDPStrategy(find_unused_parameters=True),
    )
    trainer = prepare_trainer(trainer)

    # Train the model
    trainer.fit(model, train_loader, val_loader)


# Main Function to Run Ray Tune
def main(train_cfg: Dict[str, Any], total_num_gpus: int = 1, total_num_cpus: int = 32):
    name = f"tune/{train_cfg['model_name']}/tune_{train_cfg['model_name']}_{img_size}"
    if "densenet" in train_cfg["model_name"]:
        train_cfg["model_name"] = "densenet121"
    elif "resnet" in train_cfg["model_name"]:
        train_cfg["model_name"] = "resnet152.a2_in1k"
    save_dir = f"{HOME}/output/"
    ray.init(num_gpus=total_num_gpus, num_cpus=total_num_cpus)
    search_space = {
        "lr": tune.loguniform(1e-5, 1e-2),  # Learning rate
        "batch_size": tune.choice([4, 8, 16]),  # Batch size
        "dropout": tune.choice([0.0]),
        "gamma": tune.choice([0]),
    }
    scheduler = ASHAScheduler(
        max_t=train_cfg["num_epochs"], grace_period=1, reduction_factor=2
    )
    # Make sure that the num workers * GPU per trial doesn't exceed total gpus
    # Num workers seems to be for both CPU and GPU
    scaling_cfg = ScalingConfig(
        num_workers=1,
        use_gpu=True,
        # accelerator_type="A100",
        resources_per_worker={"CPU": 4, "GPU": 0.25},  # Note that this is per trial
        # resources_per_worker={"CPU": 8, "GPU": 1.} # Note that this is per trial
    )
    run_cfg = RunConfig(
        name=name,
        storage_path=save_dir,
        checkpoint_config=CheckpointConfig(
            num_to_keep=2,
            checkpoint_score_attribute="val/f1",
            checkpoint_score_order="max",
        ),
        verbose=1,
    )
    search_algo = OptunaSearch(metric="val/f1", mode="max")
    ray_trainer = TorchTrainer(
        train_model, scaling_config=scaling_cfg, run_config=run_cfg
    )
    analysis = tune.Tuner(
        ray_trainer,
        tune_config=tune.TuneConfig(
            num_samples=train_cfg["num_samples"],
            metric="val/f1",
            mode="max",
            scheduler=scheduler,
            search_alg=search_algo,
        ),
        run_config=RunConfig(
            storage_path=save_dir,
            name=name,
            checkpoint_config=CheckpointConfig(
                num_to_keep=2,
                checkpoint_score_attribute="val/f1",
                checkpoint_score_order="max",
            ),
        ),
        param_space={
            "train_loop_config": search_space
        },  # This gets passed as an argument to the train fn.
    )
    analysis.fit()
    results = analysis.get_results()
    print(f"Best Result: {results.get_best_result()}")
    with open(f'results_{train_cfg["model_name"]}.pickle', "wb") as handle:
        pickle.dump(results, handle, protocol=pickle.HIGHEST_PROTOCOL)


if __name__ == "__main__":
    train_dicom_dir = os.path.join(DATA_DIR, "dicom-images-train")
    train_glob = f"{train_dicom_dir}/*/*/*.dcm"
    train_names = [f for f in sorted(glob.glob(train_glob))]
    img_size = 512
    train_transforms = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Resize((img_size, img_size)),
            transforms.RandomHorizontalFlip(p=0.5),  # Random horizontal flip
            transforms.RandomRotation(degrees=15),  # Random rotation within 15 degrees
        ]
    )
    train_cfg = {}
    train_cfg["seed"] = 42
    train_cfg["num_epochs"] = 25
    train_cfg["num_samples"] = 10
    train_cfg["train_transforms"] = train_transforms
    train_cfg["train_names"] = train_names
    train_cfg["image_size"] = img_size
    # models = ['deit', 'densenet121', 'resnet152.a2_in1k', 'dino']
    models = ["resnet"]
    for m in models:
        train_cfg["model_name"] = m
        main(total_num_gpus=2, total_num_cpus=32, train_cfg=train_cfg)
        # subprocess.run(['ray', 'stop', '--force'])
