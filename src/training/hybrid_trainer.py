import torch
import torch.nn as nn
from src.training.base_trainer import BaseTrainer

class HybridModelTrainer(BaseTrainer):
    def _step(self, batch: tuple) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        tab_x, home_seq, away_seq, y_batch = batch
        tab_x = tab_x.to(self.device)
        home_seq = home_seq.to(self.device)
        away_seq = away_seq.to(self.device)
        y_batch = y_batch.to(self.device)

        logits = self.model(tab_x, home_seq, away_seq)
        loss = self.criterion(logits, y_batch)
        preds = (torch.sigmoid(logits) >= 0.5).float()
        return loss, preds, y_batch

    def _predict_batch(self, batch: tuple) -> torch.Tensor:
        tab_x, home_seq, away_seq = (
            batch[0].to(self.device),
            batch[1].to(self.device),
            batch[2].to(self.device)
        )
        logits = self.model(tab_x, home_seq, away_seq)
        return torch.sigmoid(logits)
