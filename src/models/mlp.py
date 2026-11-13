from functools import partial

import numpy as np
import torch
from scipy import sparse
from sklearn.base import BaseEstimator, RegressorMixin
from torch import nn
from torch.utils.data import DataLoader, TensorDataset, random_split
from tqdm import tqdm

from ..utils import set_seed


class MLP(nn.Module):
    """Fully connected net: [Linear -> BatchNorm -> LeakyReLU -> Dropout] per hidden size, then Linear."""

    def __init__(self, inp, out, hidden_sizes=(512, 256), dropout=0.0):
        super().__init__()
        layers = []
        for before, after in zip([inp, *hidden_sizes], hidden_sizes):
            layers.extend(
                [
                    nn.Linear(before, after),
                    nn.BatchNorm1d(after),
                    nn.LeakyReLU(),
                    nn.Dropout(dropout),
                ]
            )

        layers.append(nn.Linear(hidden_sizes[-1], out))
        self.model = nn.Sequential(*layers)

    def forward(self, x):
        return self.model(x)


class MLPRegressor(RegressorMixin, BaseEstimator):
    """sklearn-compatible wrapper around MLP.

    fit holds out validation_fraction of the rows to drive the scheduler and pick
    the best epoch. With refit_full=True a fresh network is then trained on all rows
    for that many epochs, replaying the recorded learning rates.
    """

    def __init__(
        self,
        hidden_sizes=(512, 256),
        dropout=0.0,
        epoches=20,
        batch_size=64,
        seed=0,
        optimizer=None,
        scheduler=None,
        loss_fn=None,
        validation_fraction=0.15,
        refit_full=True,
    ):
        """optimizer and scheduler are factories (Hydra `_partial_: true`):
        optimizer(params) -> torch optimizer, scheduler(optimizer) -> lr scheduler.
        Defaults: AdamW(lr=1e-3), no scheduler, MSELoss.
        """
        self.hidden_sizes = hidden_sizes
        self.dropout = dropout
        self.epoches = epoches
        self.batch_size = batch_size
        self.seed = seed
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.loss_fn = loss_fn
        self.validation_fraction = validation_fraction
        self.refit_full = refit_full

    def _to_tensor(self, X):
        if sparse.issparse(X):
            X = X.toarray()
        return torch.as_tensor(np.asarray(X), dtype=torch.float32)

    def _make_loader(self, dataset):
        return DataLoader(
            dataset,
            batch_size=self.batch_size,
            shuffle=True,
            drop_last=True,  # BatchNorm fails on a batch of size 1
        )

    def _train_epoch(self, model, loader, optimizer, loss_fn):
        model.train()
        epoch_loss = 0.0
        for xb, yb in loader:
            optimizer.zero_grad()
            loss = loss_fn(model(xb), yb)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
        return epoch_loss / len(loader)

    def fit(self, X, y):
        """Train on X, y (dense or sparse); see the class docstring for the two stages."""
        set_seed(self.seed)
        X = self._to_tensor(X)
        y = torch.as_tensor(np.asarray(y, dtype=np.float32)).unsqueeze(1)

        make_optimizer = self.optimizer or partial(torch.optim.AdamW, lr=1e-3)
        loss_fn = self.loss_fn or nn.MSELoss()
        dataset = TensorDataset(X, y)

        # stage 1: train on part of the rows, validate on the rest
        trainset, valset = random_split(
            dataset, (1 - self.validation_fraction, self.validation_fraction)
        )
        X_val, y_val = X[valset.indices], y[valset.indices]

        train_loader = self._make_loader(trainset)

        self.model_ = MLP(X.shape[1], 1, self.hidden_sizes, self.dropout)
        optimizer = make_optimizer(self.model_.parameters())
        scheduler = self.scheduler(optimizer) if self.scheduler is not None else None

        lr_history, val_history = [], []
        epoch_loop = tqdm(range(self.epoches), desc="epoches", leave=False)
        for epoch in epoch_loop:
            lr_history.append(optimizer.param_groups[0]["lr"])

            # train
            epoch_loss = self._train_epoch(
                self.model_, train_loader, optimizer, loss_fn
            )

            # eval
            self.model_.eval()
            with torch.no_grad():
                val_loss = loss_fn(self.model_(X_val), y_val).item()
            val_history.append(val_loss)

            # scheduler step
            if isinstance(scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                scheduler.step(val_loss)
            elif scheduler is not None:
                scheduler.step()
            epoch_loop.set_postfix(
                loss=epoch_loss, val_loss=val_loss, lr=lr_history[-1]
            )

        self.best_epoch_ = int(np.argmin(val_history)) + 1
        self.val_history_ = val_history

        # stage 2: fresh network on all rows for best_epoch_ epochs, same lr per epoch
        if self.refit_full:
            set_seed(self.seed)
            full_loader = self._make_loader(dataset)
            self.model_ = MLP(X.shape[1], 1, self.hidden_sizes, self.dropout)
            optimizer = make_optimizer(self.model_.parameters())
            for lr in tqdm(lr_history[: self.best_epoch_], desc="refit", leave=False):
                for group in optimizer.param_groups:
                    group["lr"] = lr
                self._train_epoch(self.model_, full_loader, optimizer, loss_fn)

        return self

    def predict(self, X):
        """Predict with the trained network."""
        X = self._to_tensor(X)
        self.model_.eval()
        with torch.no_grad():
            pred = self.model_(X)
        return pred.squeeze(1).numpy()
