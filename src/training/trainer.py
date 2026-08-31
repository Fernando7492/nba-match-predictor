from pathlib import Path
import copy
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.optim import Optimizer

class ModelTrainer:
    def __init__(
        self,
        model: nn.Module,
        optimizer: Optimizer,
        criterion: nn.Module | None = None,
        device: str | torch.device | None = None,
        patience: int = 15,
        save_path: Path | str | None = None
    ):
        self.device = torch.device(device if device else ("cuda" if torch.cuda.is_available() else "cpu"))
        self.model = model.to(self.device)
        self.optimizer = optimizer
        self.criterion = criterion or nn.BCEWithLogitsLoss()
        self.patience = patience
        self.save_path = Path(save_path) if save_path else None
        self.best_model_state: dict | None = None

    def train_epoch(self, loader: DataLoader) -> tuple[float, float]:
        self.model.train()
        total_loss = 0.0
        correct = 0
        total = 0

        for x_batch, y_batch in loader:
            x_batch = x_batch.to(self.device)
            y_batch = y_batch.to(self.device)

            self.optimizer.zero_grad()
            logits = self.model(x_batch)
            loss = self.criterion(logits, y_batch)
            loss.backward()
            self.optimizer.step()

            total_loss += loss.item() * len(y_batch)
            preds = (torch.sigmoid(logits) >= 0.5).float()
            correct += (preds == y_batch).sum().item()
            total += len(y_batch)

        return total_loss / total, correct / total

    def evaluate(self, loader: DataLoader) -> tuple[float, float]:
        self.model.eval()
        total_loss = 0.0
        correct = 0
        total = 0

        with torch.no_grad():
            for x_batch, y_batch in loader:
                x_batch = x_batch.to(self.device)
                y_batch = y_batch.to(self.device)

                logits = self.model(x_batch)
                loss = self.criterion(logits, y_batch)

                total_loss += loss.item() * len(y_batch)
                preds = (torch.sigmoid(logits) >= 0.5).float()
                correct += (preds == y_batch).sum().item()
                total += len(y_batch)

        return total_loss / total, correct / total

    def fit(self, train_loader: DataLoader, val_loader: DataLoader, epochs: int = 100) -> dict[str, list[float]]:
        history: dict[str, list[float]] = {
            "train_loss": [],
            "val_loss": [],
            "train_acc": [],
            "val_acc": []
        }

        best_val_loss = float("inf")
        epochs_no_improve = 0

        for epoch in range(epochs):
            train_loss, train_acc = self.train_epoch(train_loader)
            val_loss, val_acc = self.evaluate(val_loader)

            history["train_loss"].append(train_loss)
            history["val_loss"].append(val_loss)
            history["train_acc"].append(train_acc)
            history["val_acc"].append(val_acc)

            if val_loss < best_val_loss:
                best_val_loss = val_loss
                epochs_no_improve = 0
                self.best_model_state = copy.deepcopy(self.model.state_dict())
                if self.save_path:
                    self.save_path.parent.mkdir(parents=True, exist_ok=True)
                    torch.save(self.best_model_state, self.save_path)
            else:
                epochs_no_improve += 1
                if epochs_no_improve >= self.patience:
                    break

        if self.best_model_state is not None:
            self.model.load_state_dict(self.best_model_state)

        return history

    def predict_proba(self, loader: DataLoader) -> torch.Tensor:
        self.model.eval()
        probas: list[torch.Tensor] = []
        with torch.no_grad():
            for x_batch, _ in loader:
                x_batch = x_batch.to(self.device)
                logits = self.model(x_batch)
                p = torch.sigmoid(logits)
                probas.append(p.cpu())
        return torch.cat(probas, dim=0)
