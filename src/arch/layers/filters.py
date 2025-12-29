import torch
import torch.nn.functional as F
import math
import torch.nn as nn


# ---- (A) Gaussian DoG in time domain ----
def _gauss1d_kernel(sigma: float, device, dtype):
    rad = int(3 * sigma) + 1
    x = torch.arange(-rad, rad + 1, device=device, dtype=dtype)
    k = torch.exp(-(x**2) / (2 * sigma * sigma))
    k = (k / k.sum()).view(1, 1, -1)
    return k


@torch.no_grad()
def dog_gauss_bands(x, K=6, sig0=1.0, ratio=1.6, causal=False):
    B, C, L = x.shape
    sigmas = [sig0 * (ratio**i) for i in range(K - 1)]  # K-1 smooth levels
    smooth = []
    prev = x
    for s in sigmas:
        k = _gauss1d_kernel(s, x.device, x.dtype).expand(C, 1, -1)
        pad = k.size(-1) // 2
        if causal:
            y = F.conv1d(
                F.pad(prev, (k.size(-1) - 1, 0), mode="replicate"), k, groups=C
            )
        else:
            y = F.conv1d(F.pad(prev, (pad, pad), mode="reflect"), k, groups=C)
        smooth.append(y)
        prev = y
    bands = [torch.nan_to_num(x - smooth[0])]
    for i in range(1, len(smooth)):
        bands.append(smooth[i - 1] - smooth[i])
    bands = [torch.nan_to_num(b) for b in bands]
    bands.append(torch.nan_to_num(smooth[-1]))
    return torch.stack(bands, dim=2)  # [B,C,K,L]


# ---- (B) EMA ladder (causal IIR) ----
@torch.no_grad()
def ema_ladder_bands(x, K=4, tau0=8.0, grow=3.0):
    B, C, L = x.shape
    levels = K - 1
    taus = [tau0 * (grow**i) for i in range(levels)]
    alphas = [1.0 - math.exp(-1.0 / t) for t in taus]
    lows, prev = [], x
    for a in alphas:
        y = torch.empty_like(prev)
        y[..., 0] = prev[..., 0]
        for t in range(1, L):
            y[..., t] = a * prev[..., t] + (1 - a) * y[..., t - 1]
        lows.append(y)
        prev = y
    bands = [torch.nan_to_num(x - lows[0])]
    for i in range(1, len(lows)):
        bands.append(lows[i - 1] - lows[i])
    bands = [torch.nan_to_num(b) for b in bands]
    bands.append(torch.nan_to_num(lows[-1]))
    return torch.stack(bands, dim=2)


# ---- (C) Binomial (Gaussian-like) FIR pyramid ----
def _binomial_kernel(k, device, dtype):
    # Pascal row of length k (odd k recommended)
    from math import comb

    n = k - 1
    taps = torch.tensor([comb(n, i) for i in range(k)], device=device, dtype=dtype)
    taps = (taps / taps.sum()).view(1, 1, -1)
    return taps


@torch.no_grad()
def binomial_pyramid_bands(x, K=4, k0=3, kgrow=2):
    B, C, L = x.shape
    levels = K - 1
    lows, prev = [], x
    for i in range(levels):
        k = int(round(k0 * (kgrow**i))) | 1
        w = _binomial_kernel(k, x.device, x.dtype).expand(C, 1, -1)
        pad = (k - 1) // 2
        low = F.conv1d(F.pad(prev, (pad, pad), mode="reflect"), w, groups=C)
        lows.append(low)
        prev = low
    bands = [torch.nan_to_num(x - lows[0])]
    for i in range(1, len(lows)):
        bands.append(lows[i - 1] - lows[i])
    bands = [torch.nan_to_num(b) for b in bands]
    bands.append(torch.nan_to_num(lows[-1]))
    return torch.stack(bands, dim=2)


# ---- (D) FFT filterbanks: dyadic cosine / rectangular / Butterworth-like ----
def _rfft_omega(L, device, dtype):
    f = torch.fft.rfftfreq(L, d=1.0, device=device)
    return (2 * torch.pi * f).to(dtype)  # [Lh]


def _fft_apply(x, W):  # x[B,C,L], W[K,Lh] -> [B,C,K,L]
    X = torch.fft.rfft(x, dim=-1)  # [B,C,Lh]
    Y = X.unsqueeze(2) * W.view(1, 1, *W.shape)  # [B,C,K,Lh]
    y = torch.fft.irfft(Y, n=x.size(-1), dim=-1)
    return torch.nan_to_num(y)


def _ensure_pu(W, eps=1e-8):  # partition of unity over freq
    S = W.sum(0, keepdim=True)
    return torch.where(S > eps, W / S.clamp_min(eps), W)


@torch.no_grad()
def decomp_haar(x, levels=None):
    return dwt_haar_bands_full(x, levels)  # [B,C,levels+1,L]


def _haar_filters(C, dev, dt):
    s = 2**-0.5
    # analysis (down, stride=2)
    h0 = torch.tensor([s, s], device=dev, dtype=dt).view(1, 1, 2).repeat(C, 1, 1)  # low
    h1 = (
        torch.tensor([-s, s], device=dev, dtype=dt).view(1, 1, 2).repeat(C, 1, 1)
    )  # high
    # synthesis (up, stride=2)
    g0 = torch.tensor([s, s], device=dev, dtype=dt).view(1, 1, 2).repeat(C, 1, 1)  # low
    g1 = (
        torch.tensor([-s, s], device=dev, dtype=dt).view(1, 1, 2).repeat(C, 1, 1)
    )  # high
    # g1 = torch.tensor([s, -s], device=dev, dtype=dt).view(1,1,2).repeat(C,1,1)  # high
    return h0, h1, g0, g1


@torch.no_grad()
def dwt_haar_multilevel(x, levels):
    """
    x: [B,C,L]  →  approx: [B,C,L/2^levels], details: list of length `levels` with shapes [B,C,L/2^(ℓ+1)]
    Saves per-level lengths for exact inverse.
    """
    B, C, L0 = x.shape
    dev, dt = x.device, x.dtype
    h0, h1, g0, g1 = _haar_filters(C, dev, dt)

    details, lengths = [], [L0]
    a = x
    for level in range(levels):
        if a.size(-1) & 1:  # pad 1 on the right if odd
            a = F.pad(a, (0, 1), mode="reflect")
        lo = F.conv1d(a, h0, stride=2, groups=C)
        hi = F.conv1d(a, h1, stride=2, groups=C)
        details.append(hi)  # d1, d2, ..., d_levels
        a = lo
        lengths.append(a.size(-1))  # L after this level
    return a, details, lengths, (g0, g1)  # keep g0/g1 for synthesis


@torch.no_grad()
def dwt_haar_bands_full(x, levels, thresholds: torch.Tensor | None = None):
    """
    Get crisp, same-length bands that sum to x:
    returns bands: [B,C,levels+1,L]  where bands[...,0] is detail@level1, ..., bands[...,-2] detail@level=levels, bands[...,-1] final approx.
    """
    a, ds, lens, (g0, g1) = dwt_haar_multilevel(x, levels)

    # Optional soft-threshold on detail coefficients per level
    if thresholds is not None and levels > 0:
        dev, dt = x.device, x.dtype
        thr = thresholds.to(dev, dt)
        D_list = []
        for j, d in enumerate(ds):
            t = thr[j].view(1, 1, 1)  # scalar threshold for this level
            D_list.append(torch.sign(d) * torch.relu(torch.abs(d) - t))
        ds = D_list
    B, C, L0 = x.shape
    bands = []

    # contribution of detail at level `j` up to original length
    for j in range(levels):
        y = ds[j]
        # first upsample with high-pass synth once (its own level),
        # then low-pass synth all the way to the top
        for lvl in range(j, -1, -1):
            filt = g1 if lvl == j else g0
            y = F.conv_transpose1d(y, filt, stride=2, groups=C)
            y = y[..., : lens[lvl]]
        bands.append(y)  # [B,C,L0]

    # contribution of final approximation
    y = a
    for lvl in range(levels - 1, -1, -1):
        y = F.conv_transpose1d(y, g0, stride=2, groups=C)
        y = y[..., : lens[lvl]]
    bands.append(y)  # [B,C,L0]

    y = torch.stack(bands, dim=2)
    return y


@torch.no_grad()
def idwt_haar_multilevel(approx, details, lengths, g0g1):
    """
    exact inverse; returns [B,C,L0]
    """
    g0, g1 = g0g1
    a = approx
    # go back from deepest to root
    for lvl in range(len(details) - 1, -1, -1):
        d = details[lvl]
        y_lo = F.conv_transpose1d(a, g0, stride=2, groups=a.size(1))
        y_hi = F.conv_transpose1d(d, g1, stride=2, groups=a.size(1))
        a = (y_lo + y_hi)[..., : lengths[lvl]]  # crop to pre-downsample length
    return a


@torch.no_grad()
def diffuse_multiscale_plus(
    x,
    K=8,
    steps0=1,
    grow=2,
    ks0=3,
    kgrow=2,  # kernel size growth
    dil0=1,
    dilgrow=1,  # dilation growth (set 2 for wavelet-y)
    schedule=None,  # e.g., [1,1,2,3,5,8,...]
    causal=False,
    renorm=False,
):
    B, C, L = x.shape
    dev, dt = x.device, x.dtype

    def make_kernel(k):
        k = int(k) | 1  # force odd
        w = torch.ones(1, 1, k, device=dev, dtype=dt)
        w = w / w.sum()
        return w.repeat(C, 1, 1)  # depthwise

    def smooth(inp, w, dil, n):
        y = inp
        for _ in range(n):
            if causal:
                pad = dil * (w.size(-1) - 1)
                y = F.conv1d(
                    F.pad(y, (pad, 0), mode="replicate"), w, dilation=dil, groups=C
                )
            else:
                pad = dil * (w.size(-1) - 1) // 2
                y = F.conv1d(
                    F.pad(y, (pad, pad), mode="reflect"), w, dilation=dil, groups=C
                )
        return y

    lows, low_prev = [], x
    for lvl in range(max(1, K - 1)):
        steps = (
            schedule[lvl] if schedule is not None else max(1, int(steps0 * (grow**lvl)))
        )
        ks = max(3, int(round(ks0 * (kgrow**lvl))) | 1)
        dil = max(1, int(round(dil0 * (dilgrow**lvl))))
        w = make_kernel(ks)
        low = smooth(low_prev, w, dil, steps)
        lows.append(low)
        low_prev = low

    # bands
    bands = [torch.nan_to_num(x - lows[0])]
    for i in range(1, len(lows)):
        bands.append(lows[i - 1] - lows[i])
    bands = [torch.nan_to_num(b) for b in bands]
    bands.append(torch.nan_to_num(lows[-1]))
    imfs = torch.stack(bands, dim=2)  # [B,C,K,L]

    if renorm:
        s = imfs.flatten(-1).std(dim=-1, keepdim=True).clamp_min(1e-5)
        imfs = imfs / s

    return torch.nan_to_num(imfs)


@torch.no_grad()
def fft_dyadic_cosine_bands(x, K=4, transition=0.1):
    B, C, L = x.shape
    w = _rfft_omega(L, x.device, x.dtype)
    levels = K - 1
    W = []
    hi = torch.pi

    def rcos01(z):
        return 0.5 - 0.5 * torch.cos(torch.clamp(z, 0, 1) * torch.pi)

    for j in range(levels):
        lo = torch.pi / (2 ** (j + 1))
        span = hi - lo
        t = transition * span
        band = torch.zeros_like(w)
        band[(w >= lo + t) & (w <= hi - t)] = 1.0
        if t > 0:
            idx = (w >= lo) & (w < lo + t)
            band[idx] = rcos01((w[idx] - lo) / t)
            idx = (w > hi - t) & (w <= hi)
            band[idx] = rcos01((hi - w[idx]) / t)
        W.append(band)
        hi = lo
    lp = torch.zeros_like(w)
    t = transition * hi
    lp[w <= (hi - t)] = 1.0
    if t > 0:
        idx = (w > hi - t) & (w <= hi)
        lp[idx] = rcos01((hi - w[idx]) / t)
    W.append(lp)
    W = _ensure_pu(torch.stack(W, 0))  # [K,Lh]
    return _fft_apply(x, W)  # [B,C,K,L]


@torch.no_grad()
def fft_dyadic_rect_bands(x, K=4):
    B, C, L = x.shape
    w = _rfft_omega(L, x.device, x.dtype)
    levels = K - 1
    W = []
    hi = torch.pi
    for j in range(levels):
        lo = torch.pi / (2 ** (j + 1))
        W.append(((w >= lo) & (w <= hi)).to(x.dtype))
        hi = lo
    W.append((w <= hi).to(x.dtype))
    W = _ensure_pu(torch.stack(W, 0))
    return _fft_apply(x, W)


@torch.no_grad()
def fft_butterworth_diff_bands(x, K=4, order=4, w0=None):
    """
    Butterworth low-pass L(w; wc)=1/sqrt(1+(w/wc)^(2n)), then bands via differences:
      W0 = 1 - L(wc0), W1 = L(wc0)-L(wc1), ..., WK-1 = L(wcK-2)
    wc increases geometrically from w0 to ~pi.
    """
    B, C, L = x.shape
    w = _rfft_omega(L, x.device, x.dtype)
    levels = K - 1
    if w0 is None:
        w0 = torch.pi / (2**levels)  # base cutoff
    wcs = torch.tensor(
        [float(w0) * (2**i) for i in range(levels)], device=x.device, dtype=x.dtype
    ).clamp_max(torch.pi * 0.999)
    Lps = 1.0 / torch.sqrt(
        1.0 + (w.view(1, -1) / wcs.view(-1, 1)).pow(2 * order)
    )  # [levels,Lh]
    Ws = []
    Ws.append(1.0 - Lps[0])
    for i in range(levels - 1):
        Ws.append(Lps[i] - Lps[i + 1])
    Ws.append(Lps[-1])
    W = _ensure_pu(torch.stack(Ws, 0))  # [K,Lh]
    return _fft_apply(x, W)


class LearnableConvFilter(nn.Module):
    """
    Depthwise Conv1d filter bank.
    """

    def __init__(
        self,
        C: int,
        K: int = 1,
        kernel_size: int = 31,
        dilation: int = 1,
        causal: bool = False,
        pad_mode: str = "reflect",
        bias: bool = False,
    ):
        super().__init__()
        self.C = int(C)
        self.K = int(K)
        self.ks = int(kernel_size) | 1
        self.dil = int(dilation)
        self.causal = bool(causal)
        self.pad_mode = pad_mode

        # groups=C => depthwise；out_channels = C*K
        self.conv = nn.Conv1d(
            in_channels=self.C,
            out_channels=self.C * self.K,
            kernel_size=self.ks,
            dilation=self.dil,
            groups=self.C,
            bias=bias,
            padding=0,
        )

        self.reset_parameters_box()

    @torch.no_grad()
    def reset_parameters_box(self):
        w = torch.ones(self.C * self.K, 1, self.ks)
        w = w / w.sum(dim=-1, keepdim=True)
        self.conv.weight.copy_(w)
        if self.conv.bias is not None:
            self.conv.bias.zero_()

    def _pad(self, x):
        if self.causal:
            pad = self.dil * (self.ks - 1)
            return F.pad(x, (pad, 0), mode="replicate")
        else:
            pad = self.dil * (self.ks - 1) // 2
            if self.pad_mode == "reflect" and pad >= x.size(-1):
                return F.pad(x, (pad, pad), mode="replicate")
            return F.pad(x, (pad, pad), mode=self.pad_mode)

    def forward(self, x):  # [B,C,L]
        B, C, L = x.shape
        # assert C == self.C, f"Expected C={self.C}, got {C}"
        x = self._pad(x)
        y = self.conv(x)  # [B, C*K, L]
        y = y.view(B, C, self.K, L)  # [B, C, K, L]
        return y
