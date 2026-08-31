import torch
import torch.nn as nn
from src.training.base_trainer import BaseTrainer

class ModelTrainer(BaseTrainer):
    def _step(self, batch: tuple) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        x_batch, y_batch = batch
        x_batch = x_batch.to(self.device)
        y_batch = y_batch.to(self.device)

        logits = self.model(x_batch)
        loss = self.criterion(logits, y_batch)
        preds = (torch.sigmoid(logits) >= 0.5).float()
        return loss, preds, y_batch

    def _predict_batch(self, batch: tuple) -> torch.Tensor:
        x_batch = batch[0].to(self.device)
        logits = self.model(x_batch)
        return torch.sigmoid(logits)
