"""轻量 MLP：直接输出完整预测时域，按验证误差早停。"""
import random
import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset
from .base import ForecastModel

class MLPForecast(ForecastModel):
    def __init__(self, horizon=24, seed=42, hidden=64, lr=0.001, epochs=40, patience=5, batch_size=128, device=None):
        self.horizon, self.seed, self.hidden = horizon, seed, hidden
        self.lr, self.epochs, self.patience, self.batch_size = lr, epochs, patience, batch_size
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")

    def fit(self, X, y, X_val=None, y_val=None):
        X = self.check_X(X); X_val = self.check_X(X_val)
        y = np.asarray(y, dtype=np.float32); y_val = np.asarray(y_val, dtype=np.float32)
        random.seed(self.seed); np.random.seed(self.seed); torch.manual_seed(self.seed)
        if torch.cuda.is_available(): torch.cuda.manual_seed_all(self.seed)
        self.model = nn.Sequential(nn.Flatten(), nn.Linear(X.shape[1] * X.shape[2], self.hidden), nn.ReLU(),
                                   nn.Linear(self.hidden, self.hidden), nn.ReLU(), nn.Linear(self.hidden, self.horizon)).to(self.device)
        loader = DataLoader(TensorDataset(torch.from_numpy(X), torch.from_numpy(y)), batch_size=self.batch_size, shuffle=True,
                            generator=torch.Generator().manual_seed(self.seed))
        optim = torch.optim.Adam(self.model.parameters(), lr=self.lr)
        xv = torch.from_numpy(X_val).to(self.device); yv = torch.from_numpy(y_val).to(self.device)
        best = float("inf"); remaining = self.patience
        for epoch in range(self.epochs):
            self.model.train()
            for xb, yb in loader:
                xb, yb = xb.to(self.device), yb.to(self.device)
                optim.zero_grad(); loss = ((self.model(xb) - yb) ** 2).mean(); loss.backward(); optim.step()
            self.model.eval()
            with torch.no_grad(): score = ((self.model(xv) - yv) ** 2).mean().item()
            if score < best - 1e-5:
                best = score; self.best_epoch = epoch + 1
                state = {k: v.detach().cpu().clone() for k, v in self.model.state_dict().items()}
                remaining = self.patience
            else:
                remaining -= 1
                if not remaining: break
        self.validation_mse = best
        self.model.load_state_dict(state); self.model.eval()
        return self

    def predict(self, X):
        X = self.check_X(X)
        self.model.eval()
        with torch.no_grad():
            return self.model(torch.from_numpy(X).to(self.device)).cpu().numpy()
