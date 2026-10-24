"""Neural quantile forecasters (multi-horizon, multi-quantile, direct output)."""
import torch
import torch.nn as nn
import torch.nn.functional as F

from data import H, L, QUANTILES

NQ = len(QUANTILES)
P_CH, F_CH = 7, 5   # past and future input channels (see data.batch)


def monotone(raw):
    """Map raw outputs [B, H, NQ] to non-crossing quantiles: median + cumulative softplus gaps."""
    mid = NQ // 2
    med = raw[..., mid:mid + 1]
    up = torch.cumsum(F.softplus(raw[..., mid + 1:]), dim=-1)
    down = torch.flip(torch.cumsum(F.softplus(torch.flip(raw[..., :mid], [-1])), dim=-1), [-1])
    return torch.cat([med - down, med, med + up], dim=-1)


class CausalBlock(nn.Module):
    def __init__(self, ch, dilation, k=3, p=0.1):
        super().__init__()
        self.pad = (k - 1) * dilation
        self.c1 = nn.Conv1d(ch, ch, k, dilation=dilation)
        self.c2 = nn.Conv1d(ch, ch, k, dilation=dilation)
        self.drop = nn.Dropout(p)

    def forward(self, x):
        y = F.gelu(self.c1(F.pad(x, (self.pad, 0))))
        y = self.drop(F.gelu(self.c2(F.pad(y, (self.pad, 0)))))
        return x + y


class PVTCQN(nn.Module):
    """PV-aware temporal convolutional quantile network (proposed).

    A dilated causal TCN encodes the 48 h history of net load, PV, consumption and clear-sky
    potential. A per-horizon decoder combines that context with the known future clear-sky PV
    potential and calendar at each target step, so the model can place the solar dip at the
    right time of day for every horizon. Outputs non-crossing quantiles.
    """

    def __init__(self, ch=48, dil=(1, 2, 4, 8, 16, 32), dec=64, p=0.1):
        super().__init__()
        self.inp = nn.Conv1d(P_CH, ch, 1)
        self.blocks = nn.Sequential(*[CausalBlock(ch, d, p=p) for d in dil])
        self.ctx = nn.Linear(ch, dec)
        self.hemb = nn.Embedding(H, dec)
        self.fut = nn.Linear(F_CH, dec)
        self.dec = nn.Sequential(nn.GELU(), nn.Linear(dec, dec), nn.GELU(), nn.Dropout(p),
                                 nn.Linear(dec, NQ))
        # skip path from the same time yesterday (seasonal lag), learned per horizon
        self.seasonal = nn.Parameter(torch.zeros(H, NQ))

    def forward(self, past, fut):
        z = self.blocks(self.inp(past))[:, :, -1]               # [B, ch]
        c = self.ctx(z)[:, None, :]                               # [B, 1, dec]
        f = self.fut(fut.transpose(1, 2))                         # [B, H, dec]
        e = self.hemb.weight[None]                                # [1, H, dec]
        raw = self.dec(c + f + e)                                 # [B, H, NQ]
        lag = past[:, 0, -H:]                                     # net load 24 h before each target
        raw = raw + lag[..., None] * self.seasonal[None]
        return monotone(raw)


class MLPQ(nn.Module):
    """Fully connected multi-horizon quantile baseline on flattened inputs."""

    def __init__(self, hid=256, p=0.1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(P_CH * L + F_CH * H, hid), nn.GELU(), nn.Dropout(p),
            nn.Linear(hid, hid), nn.GELU(), nn.Dropout(p), nn.Linear(hid, H * NQ))

    def forward(self, past, fut):
        x = torch.cat([past.flatten(1), fut.flatten(1)], 1)
        return monotone(self.net(x).view(-1, H, NQ))


class LSTMGauss(nn.Module):
    """DeepAR-style baseline: LSTM encoder, Gaussian output per horizon (mean, std)."""

    def __init__(self, hid=64):
        super().__init__()
        self.enc = nn.LSTM(P_CH, hid, num_layers=2, batch_first=True, dropout=0.1)
        self.dec = nn.Sequential(nn.Linear(hid + F_CH, hid), nn.GELU(), nn.Linear(hid, 2))

    def forward(self, past, fut):
        _, (h, _) = self.enc(past.transpose(1, 2))
        c = h[-1][:, None, :].expand(-1, H, -1)
        out = self.dec(torch.cat([c, fut.transpose(1, 2)], -1))
        mu, sd = out[..., 0], F.softplus(out[..., 1]) + 1e-3
        return mu, sd

    def quantiles(self, past, fut):
        mu, sd = self(past, fut)
        zq = torch.distributions.Normal(0, 1).icdf(torch.tensor(QUANTILES))
        return mu[..., None] + sd[..., None] * zq


def pinball(q, y, m):
    """Mean pinball loss. q [B, H, NQ], y [B, H], m [B, H] mask."""
    tau = torch.tensor(QUANTILES, device=q.device)
    d = y[..., None] - q
    loss = torch.maximum(tau * d, (tau - 1) * d).mean(-1)
    return (loss * m).sum() / m.sum().clamp(min=1)


def gauss_nll(mu, sd, y, m):
    nll = torch.log(sd) + 0.5 * ((y - mu) / sd) ** 2
    return (nll * m).sum() / m.sum().clamp(min=1)


MODELS = {"pv-tcqn": PVTCQN, "mlp-q": MLPQ, "lstm-gauss": LSTMGauss}
