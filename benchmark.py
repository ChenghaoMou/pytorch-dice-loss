from typing import Optional

import numpy as np
import pytorch_lightning as pl
import torch
import torch.nn as nn
from imblearn.datasets import fetch_datasets
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, Dataset, random_split
from torchmetrics import AUROC, F1Score

from pytorch_imbalance_loss.dice_loss import DiceLoss, SelfAdjustingDiceLoss
from pytorch_imbalance_loss.focal_loss import FocalLoss


class ImbalancedDataset(Dataset):

    def __init__(self, data) -> None:
        super().__init__()
        self.data = data
    
    def __len__(self) -> int:
        return len(self.data["x"])
    
    def __getitem__(self, idx) -> tuple:
        return {key: self.data[key][idx] for key in self.data}
    
    @classmethod
    def collate_fn(batch):
        return {key: torch.stack([item[key] for item in batch]) for key in batch[0]}


class ImbalancedDataModule(pl.LightningDataModule):
    def __init__(self, data, batch_size: int = 32):
        super().__init__()
        self.data = data
        self.batch_size = batch_size

    def setup(self, stage: Optional[str] = None):
        
        x, y = self.data["x"], self.data["y"]
        y = np.unique(y,return_inverse=True)[1]
        x = torch.from_numpy(x).float()
        y = torch.from_numpy(y).long()
        # trainx, valx, trainy, valy= train_test_split(x, y, test_size=40_000, random_state=42, stratify=y)
        trainx, valx = x[:-40_000], x[-40_000:]
        trainy, valy = y[:-40_000], y[-40_000:]
        self.train = {"x": trainx, "y": trainy}
        self.val = {"x": valx, "y": valy}

    def train_dataloader(self):
        return DataLoader(ImbalancedDataset(self.train), batch_size=self.batch_size, num_workers=10)

    def val_dataloader(self):
        return DataLoader(ImbalancedDataset(self.val), batch_size=self.batch_size, num_workers=10)


class MLP(pl.LightningModule):

    def __init__(self, input_size, hidden_size, output_size, loss):
        super().__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.act1 = nn.ReLU()
        self.fc2 = nn.Linear(hidden_size, output_size)
        self.loss = loss
        # self.metric = F1Score(num_classes=output_size, average="macro")
        self.metric = AUROC(num_classes=output_size, average="macro")

    def forward(self, x):
        x = self.fc1(x)
        x = self.act1(x)
        x = self.fc2(x)
        return x
    
    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=1e-2)
        return optimizer
    
    def training_step(self, batch, batch_idx):
        x, y = batch["x"], batch["y"]
        y_hat = self(x)
        loss = self.loss(y_hat, y)
        return loss
    
    def validation_step(self, batch, batch_idx):
        
        x, y = batch["x"], batch["y"]
       
        y_hat = self(x)
        val_loss = self.loss(y_hat, y)
        self.log("val_loss", val_loss, prog_bar=True)
        # self.log("val_f1", self.metric(y_hat.argmax(-1), y), prog_bar=True)
        return {"y_hat": y_hat, "y": y}
    
    def validation_epoch_end(self, outputs) -> None:
        
        y_hat = torch.cat([output["y_hat"] for output in outputs])
        y = torch.cat([output["y"] for output in outputs])
        self.log("val_metric_final", self.metric(y_hat, y), prog_bar=True)
        self.log("val_loss_final", self.loss(y_hat, y), prog_bar=True)

if __name__ == "__main__":

    import pandas as pd
    from pytorch_lightning.callbacks.early_stopping import EarlyStopping

    credit = pd.read_csv("https://datahub.io/machine-learning/creditcard/r/creditcard.csv")
    credit = credit.drop(columns=['Time'])
    features = credit.drop(columns=['Class'])

    trainer = pl.Trainer(enable_checkpointing=False, callbacks=[EarlyStopping(monitor="val_loss", mode="min", patience=5)], max_epochs=10, deterministic=True)
    data = ImbalancedDataModule({"x": features.values, "y": credit["Class"].map(lambda x: 0 if x == "'0'" else 1).values}, batch_size=16)
    model = MLP(input_size=29, hidden_size=32, output_size=2, loss=DiceLoss())
    trainer.fit(model, data)
    print("Done")
    model = MLP(input_size=29, hidden_size=32, output_size=2, loss=FocalLoss(alpha=None, gamma=2))
    trainer.fit(model, data)
    print("Done")
    trainer = pl.Trainer(enable_checkpointing=False, callbacks=[EarlyStopping(monitor="val_loss", mode="min", patience=5)], max_epochs=10, deterministic=True)
    # data = ImbalancedDataModule(fetch_datasets()["pen_digits"], batch_size=32)
    model = MLP(input_size=29, hidden_size=32, output_size=2, loss=FocalLoss(alpha=None, gamma=1))
    trainer.fit(model, data)
    print("Done")
    trainer = pl.Trainer(enable_checkpointing=False, callbacks=[EarlyStopping(monitor="val_loss", mode="min", patience=5)], max_epochs=10, deterministic=True)
    # data = ImbalancedDataModule(fetch_datasets()["pen_digits"], batch_size=32)
    model = MLP(input_size=29, hidden_size=32, output_size=2, loss=FocalLoss(alpha=None, gamma=0))
    trainer.fit(model, data)
    