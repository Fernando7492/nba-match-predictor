from abc import ABC, abstractmethod
from pathlib import Path
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

class BaseTrainer(ABC):
    def __init__(
        self,
        model: nn.Module,
        optimizer: torch.optim.Optimizer,
        criterion: nn.Module | None = None,
        scheduler: torch.optim.lr_scheduler._LRScheduler | None = None,
        patience: int = 10,
        save_path: Path | None = None,
        device: str | torch.device = "cpu",
        max_grad_norm: float = 1.0,
        use_amp: bool = False
    ):
        self.device = torch.device(device) if isinstance(device, str) else device
        self.model = model.to(self.device)
        self.optimizer = optimizer
        self.criterion = criterion or nn.BCEWithLogitsLoss()
        self.scheduler = scheduler
        self.patience = patience
        self.save_path = save_path
        self.max_grad_norm = max_grad_norm
        self.use_amp = use_amp and self.device.type == "cuda"
        self.scaler = torch.amp.GradScaler("cuda", enabled=self.use_amp)

    @abstractmethod
    def _step(self, batch: tuple) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        pass

    @abstractmethod
    def _predict_batch(self, batch: tuple) -> torch.Tensor:
        pass

    def train_epoch(self, loader: DataLoader) -> tuple[float, float]:
        self.model.train()
        total_loss = 0.0
        correct = 0
        total = 0

        for batch in loader:
            self.optimizer.zero_grad(set_to_none=True)
            
            with torch.amp.autocast("cuda", enabled=self.use_amp):
                loss, preds, y_batch = self._step(batch)

            self.scaler.scale(loss).backward()
            self.scaler.unscale_(self.optimizer)
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=self.max_grad_norm)
            self.scaler.step(self.optimizer)
            self.scaler.update()

            total_loss += loss.item() * len(y_batch)
            correct += (preds == y_batch).sum().item()
            total += len(y_batch)

        return total_loss / total, correct / total

    def evaluate(self, loader: DataLoader) -> tuple[float, float]:
        self.model.eval()
        total_loss = 0.0
        correct = 0
        total = 0

        with torch.no_grad():
            for batch in loader:
                with torch.amp.autocast("cuda", enabled=self.use_amp):
                    loss, preds, y_batch = self._step(batch)

                total_loss += loss.item() * len(y_batch)
                correct += (preds == y_batch).sum().item()
                total += len(y_batch)

        return total_loss / total, correct / total

    def fit(self, train_loader: DataLoader, val_loader: DataLoader, epochs: int = 50) -> dict[str, list[float]]:
        history: dict[str, list[float]] = {
            "train_loss": [], "train_acc": [],
            "val_loss": [], "val_acc": []
        }
        best_val_loss = float("inf")
        patience_counter = 0

        for epoch in range(epochs):
            tr_loss, tr_acc = self.train_epoch(train_loader)
            val_loss, val_acc = self.evaluate(val_loader)

            if self.scheduler:
                if isinstance(self.scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                    self.scheduler.step(val_loss)
                else:
                    self.scheduler.step()

            history["train_loss"].append(tr_loss)
            history["train_acc"].append(tr_acc)
            history["val_loss"].append(val_loss)
            history["val_acc"].append(val_acc)

            if val_loss < best_val_loss:
                best_val_loss = val_loss
                patience_counter = 0
                if self.save_path:
                    self._save_checkpoint(self.save_path)
            else:
                patience_counter += 1
                if patience_counter >= self.patience:
                    break

        if self.save_path and self.save_path.exists():
            self._load_checkpoint(self.save_path)

        return history

    def predict_proba(self, loader: DataLoader) -> torch.Tensor:
        self.model.eval()
        probas = []
        with torch.no_grad():
            for batch in loader:
                with torch.amp.autocast("cuda", enabled=self.use_amp):
                    prob = self._predict_batch(batch)
                probas.append(prob.cpu())
        return torch.cat(probas, dim=0)

    def _save_checkpoint(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(self.model.state_dict(), path)

    def _load_checkpoint(self, path: Path) -> None:
        self.model.load_state_dict(torch.load(path, map_location=self.device, weights_only=True))
