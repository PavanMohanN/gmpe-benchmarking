"""SpectralMLP: a multi-output network for the median and log-sigma at all 27 IMs.

* Trunk: [Linear -> LayerNorm -> SiLU -> Dropout] x depth; two linear heads (mu, log sigma).
* Loss: masked Gaussian NLL (NaN targets, e.g. SA beyond T_max, contribute nothing)
  + ``smooth_lambda`` x mean squared second difference of mu and log sigma across
  adjacent SA periods (ordered by period), so spectra stay smooth.
* Targets are standardised per IM with train_fit statistics; outputs are mapped back.
* Optimiser AdamW + CosineAnnealingLR; early stopping on an event-grouped
  validation split *inside* train_fit. ``cal`` and ``test`` are never seen.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
import torch
from torch import nn

from gmpe_bench.models.base import N_IMS, BaseSpectralModel, grouped_holdout, sa_columns
from gmpe_bench.models.features import FeaturePipeline

LOG_SIGMA_MIN, LOG_SIGMA_MAX = np.log(0.05), np.log(2.5)


@dataclass
class MLPConfig:
    hidden: int = 256
    depth: int = 3
    dropout: float = 0.1
    lr: float = 1e-3
    weight_decay: float = 1e-4
    epochs: int = 300
    batch_size: int = 512
    patience: int = 30
    val_frac: float = 0.15
    smooth_lambda: float = 0.05
    device: str = "auto"


def resolve_torch_device(preferred: str = "auto") -> torch.device:
    if preferred == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(preferred)


def masked_gaussian_nll(mu: torch.Tensor, log_sigma: torch.Tensor, y: torch.Tensor,
                        mask: torch.Tensor) -> torch.Tensor:
    """Mean Gaussian NLL over entries where ``mask`` is True (constant term dropped)."""
    y0 = torch.where(mask, y, mu.detach())  # NaN-free; masked entries give zero gradient
    nll = log_sigma + 0.5 * ((y0 - mu) * torch.exp(-log_sigma)) ** 2
    w = mask.to(nll.dtype)
    return (nll * w).sum() / w.sum().clamp_min(1.0)


def spectral_smoothness(pred: torch.Tensor, sa_idx: torch.Tensor) -> torch.Tensor:
    """Mean squared second difference across adjacent SA periods."""
    s = pred[:, sa_idx]
    if s.shape[1] < 3:
        return pred.new_zeros(())
    d2 = s[:, 2:] - 2.0 * s[:, 1:-1] + s[:, :-2]
    return (d2**2).mean()


class SpectralNet(nn.Module):
    def __init__(self, n_in: int, n_out: int, hidden: int, depth: int, dropout: float) -> None:
        super().__init__()
        layers: list[nn.Module] = []
        width = n_in
        for _ in range(depth):
            layers += [nn.Linear(width, hidden), nn.LayerNorm(hidden), nn.SiLU(), nn.Dropout(dropout)]
            width = hidden
        self.trunk = nn.Sequential(*layers)
        self.mu = nn.Linear(width, n_out)
        self.log_sigma = nn.Linear(width, n_out)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        h = self.trunk(x)
        return self.mu(h), self.log_sigma(h)


class SpectralMLP(BaseSpectralModel):
    name = "SpectralMLP"

    def __init__(self, config: MLPConfig | None = None, station_encoding: bool = True, seed: int = 0) -> None:
        super().__init__(seed)
        self.config = config or MLPConfig()
        self.pipeline = FeaturePipeline(station_encoding=station_encoding, seed=seed)
        self.device = resolve_torch_device(self.config.device)
        self.net: SpectralNet | None = None
        self.y_mean = np.zeros(N_IMS)
        self.y_std = np.ones(N_IMS)
        self.history: list[tuple[int, float, float]] = []
        self.best_epoch = 0
        self.val_eqids_: np.ndarray = np.array([])

    def _loss(self, x: torch.Tensor, y: torch.Tensor, m: torch.Tensor, sa_idx: torch.Tensor,
              scale: torch.Tensor, shift: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        assert self.net is not None
        mu_s, ls_s = self.net(x)
        ls_s = ls_s.clamp(LOG_SIGMA_MIN - 3, LOG_SIGMA_MAX + 3)
        nll = masked_gaussian_nll(mu_s, ls_s, y, m)
        mu = mu_s * scale + shift           # ln units for the smoothness penalty
        ls = ls_s + torch.log(scale)
        smooth = spectral_smoothness(mu, sa_idx) + spectral_smoothness(ls, sa_idx)
        return nll, smooth

    def fit(self, df: pd.DataFrame, offset: np.ndarray | None = None) -> SpectralMLP:
        c = self.config
        torch.manual_seed(self.seed)
        np.random.seed(self.seed)
        Y = self._targets(df, offset)
        X = self.pipeline.fit_transform(df)
        self.y_mean = np.nan_to_num(np.nanmean(Y, axis=0))
        sd = np.nanstd(Y, axis=0)
        self.y_std = np.where(np.isfinite(sd) & (sd > 1e-6), sd, 1.0)
        Ys = (Y - self.y_mean) / self.y_std

        eq = df["eqid"].to_numpy()
        val = grouped_holdout(eq, c.val_frac, self.seed)
        self.val_eqids_ = np.unique(eq[val])
        dev = self.device
        t = lambda a: torch.as_tensor(np.asarray(a), dtype=torch.float32, device=dev)  # noqa: E731
        Xt, Yt = t(X), t(np.nan_to_num(Ys))
        Mt = torch.as_tensor(np.isfinite(Ys), device=dev)
        tr_idx = torch.as_tensor(np.flatnonzero(~val), device=dev)
        va_idx = torch.as_tensor(np.flatnonzero(val), device=dev)
        sa_idx = torch.as_tensor(sa_columns(), device=dev)
        scale, shift = t(self.y_std), t(self.y_mean)

        self.net = SpectralNet(X.shape[1], N_IMS, c.hidden, c.depth, c.dropout).to(dev)
        opt = torch.optim.AdamW(self.net.parameters(), lr=c.lr, weight_decay=c.weight_decay)
        sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=c.epochs)
        gen = torch.Generator(device="cpu").manual_seed(self.seed)
        best, best_state, wait = np.inf, None, 0
        self.history = []
        for epoch in range(c.epochs):
            self.net.train()
            perm = tr_idx[torch.randperm(tr_idx.numel(), generator=gen).to(dev)]
            for start in range(0, perm.numel(), c.batch_size):
                b = perm[start:start + c.batch_size]
                nll, smooth = self._loss(Xt[b], Yt[b], Mt[b], sa_idx, scale, shift)
                loss = nll + c.smooth_lambda * smooth
                opt.zero_grad(set_to_none=True)
                loss.backward()
                nn.utils.clip_grad_norm_(self.net.parameters(), 5.0)
                opt.step()
            sched.step()
            self.net.eval()
            with torch.no_grad():
                src = va_idx if va_idx.numel() else tr_idx
                v_nll, v_s = self._loss(Xt[src], Yt[src], Mt[src], sa_idx, scale, shift)
                v = float(v_nll + c.smooth_lambda * v_s)
            self.history.append((epoch, loss.item(), v))
            if v < best - 1e-5:
                best, wait, self.best_epoch = v, 0, epoch
                best_state = {k: x.detach().clone() for k, x in self.net.state_dict().items()}
            else:
                wait += 1
                if wait >= c.patience:
                    break
        if best_state is not None:
            self.net.load_state_dict(best_state)
        self.fitted = True
        return self

    def predict(self, df: pd.DataFrame, offset: np.ndarray | None = None) -> tuple[np.ndarray, np.ndarray]:
        self._check()
        assert self.net is not None
        X = torch.as_tensor(self.pipeline.transform(df), dtype=torch.float32, device=self.device)
        self.net.eval()
        mus, lss = [], []
        with torch.no_grad():
            for start in range(0, X.shape[0], 8192):
                m, s = self.net(X[start:start + 8192])
                mus.append(m.cpu().numpy())
                lss.append(s.cpu().numpy())
        mu = np.concatenate(mus) * self.y_std + self.y_mean if mus else np.zeros((0, N_IMS))
        ls = np.concatenate(lss) + np.log(self.y_std) if lss else np.zeros((0, N_IMS))
        sigma = np.exp(np.clip(ls, LOG_SIGMA_MIN, LOG_SIGMA_MAX))
        if offset is not None:
            mu = mu + offset
        return mu.astype(float), sigma.astype(float)
