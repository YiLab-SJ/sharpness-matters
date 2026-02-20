from sklearn.metrics import roc_auc_score, f1_score
import torch
import numpy as np
import timm
import os
import pytorch_lightning as pl
from torch import nn
from typing import Dict, Union


class CNNBinaryClassifier(pl.LightningModule):
    def __init__(
        self,
        lr: float = 1e-3,
        fold: int = 0,
        img_size: int = 512,
        model_name: str = "densenet121",
        pos_weight: float = 1,
        alpha: float = 0.5,
        gamma: float = 2,
    ):
        """
        Choose one of the following model names:
            - densenet121
            - resnet152.a2_in1k
        """
        super().__init__()
        self.save_hyperparameters()
        self.model = timm.create_model(
            model_name, pretrained=True, num_classes=1, in_chans=1
        )
        self.criterion = nn.BCEWithLogitsLoss(
            pos_weight=torch.tensor(pos_weight)
        )  # Binary cross-entropy loss with logits
        self.val_preds, self.val_labels = [], []

    def forward(self, x):
        return self.model(x)

    def training_step(
        self, batch: Dict[str, torch.Tensor], batch_idx: int
    ) -> torch.Tensor:
        x = batch["data"]
        y = batch["label"]
        logits = self(x).squeeze(1)
        loss = self.criterion(logits, y.float())
        self.log("train_loss", loss)
        return loss

    def validation_step(
        self, batch: Dict[str, torch.Tensor], batch_idx: int
    ) -> Dict[str, torch.Tensor]:
        x = batch["data"]
        y = batch["label"]
        logits = self(x).squeeze(1)
        loss = self.criterion(logits, y.float())
        preds = torch.sigmoid(logits)
        self.val_preds.append(preds.detach().cpu())
        self.val_labels.append(y.detach().cpu())
        acc = ((preds > 0.5) == y).float().mean()
        self.log("val/loss", loss, prog_bar=True)
        self.log("val/acc", acc, prog_bar=True)
        return {"val/loss": loss, "val/acc": acc}

    def on_validation_epoch_end(self):
        # Concatenate all predictions and labels
        preds = torch.cat(self.val_preds).numpy()
        labels = torch.cat(self.val_labels).numpy()

        # Compute AUC
        try:
            auc = roc_auc_score(labels, preds)
            f1 = f1_score(labels, (preds > 0.5))
        except ValueError:
            auc = float(
                "nan"
            )  # Handle edge cases (e.g., single class in validation set)
            f1 = float("nan")

        self.log("val/auc", auc, prog_bar=True)  # Log AUC
        self.log("val/f1", f1, prog_bar=True)
        self.val_preds.clear()  # Clear for the next epoch
        self.val_labels.clear()

    def configure_optimizers(self) -> torch.optim.Optimizer:
        return torch.optim.AdamW(self.parameters(), lr=self.hparams.lr)


if __name__ == "__main__":
    model = CNNBinaryClassifier(model_name="resnet152.a2_in1k")
    print(model)
