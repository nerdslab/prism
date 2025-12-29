# --- comprehensive plotting utilities for Prism analysis ---
import os, math
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch.distributed as dist
import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple, Union
import torch
from datetime import datetime


def _is_master():
    return (not dist.is_available()) or (not dist.is_initialized()) or dist.get_rank() == 0

def _mkdir_once(path):
    if _is_master():
        os.makedirs(path, exist_ok=True)

def _to_np(x):
    # x: torch tensor on any device -> np
    return x.detach().float().cpu().numpy()


def save_individual_bands_plot(bands_left, bands_right, weights_left, weights_right, x_ref=None, y=None, forecast_out=None, 
                             per_band_forecast=None, meta=None, side_tag="root", tag="", b_idx=0, c_idx=0, 
                             max_points=1200, cmap="viridis", original_future=None, stats=None, save_dir=None, 
                             seq_len=336, pred_len=96, current_epoch=0):
    """
    Three-row layout, left→right sorted by descending weights:
      row1: bands; row2: cumulative forecast vs true future; row3: cumulative reconstruction vs input.
    """
    if not _is_master():
        return
    # Ensure save_dir defaults to exp_prism/bands_imgs if not provided
    if not save_dir or str(save_dir).strip() == "":
        save_dir = os.path.join("./temp", "logs", "exp_prism", "bands_imgs")
    _mkdir_once(save_dir)

    B, C, K_left, Lp = bands_left.shape
    _, _, K_right, _ = bands_right.shape
    
    # Pick indices from arguments (wrap safely)
    b = (b_idx if isinstance(b_idx, int) else 0) % B
    c = (c_idx if isinstance(c_idx, int) else 0) % C

    # Convert to numpy
    bands_left_np = _to_np(bands_left[b, c])         # [K_left, Lp]
    bands_right_np = _to_np(bands_right[b, c])       # [K_right, Lp]
    weights_left_np = _to_np(weights_left[b, c])     # [K_left]
    weights_right_np = _to_np(weights_right[b, c])   # [K_right]

    # Combine all bands and weights, regardless of left/right origin
    all_bands = np.concatenate([bands_left_np, bands_right_np], axis=0)  # [K_total, Lp]
    all_weights = np.concatenate([weights_left_np, weights_right_np], axis=0)  # [K_total]
    all_origins = ['L'] * K_left + ['R'] * K_right  # Track origin for each band
    
    # Sort by descending weights (highest weight on the left)
    sorted_indices = np.argsort(all_weights)[::-1]
    sorted_bands = all_bands[sorted_indices]
    sorted_weights = all_weights[sorted_indices]
    sorted_origins = [all_origins[i] for i in sorted_indices]

    # Pad bands to seq_len for reconstruction
    # def pad_band_left(band, target_len):
    #     if len(band) >= target_len:
    #         return band[:target_len]
    #     else:
    #         # Pad from the left with first value
    #         first_val = band[0] if len(band) > 0 else 0
    #         return np.pad(band, (target_len - len(band), 0), mode='constant', constant_values=first_val)
    
    # def pad_band_right(band, target_len):
    #     if len(band) >= target_len:
    #         return band[:target_len]
    #     else:
    #         # Pad from the right with last value
    #         last_val = band[-1] if len(band) > 0 else 0
    #         return np.pad(band, (0, target_len - len(band)), mode='constant', constant_values=last_val)
    def pad_band_left(band, target_len):
        if len(band) >= target_len:
            return band[:target_len]
        else:
            # Pad from the left with zeros
            return np.pad(band, (target_len - len(band), 0), mode='constant', constant_values=0)

    def pad_band_right(band, target_len):
        if len(band) >= target_len:
            return band[:target_len]
        else:
            # Pad from the right with zeros
            return np.pad(band, (0, target_len - len(band)), mode='constant', constant_values=0)
    # Pad bands based on their origin (left or right) - use seq_len for reconstruction
    padded_bands = []
    unpadded_bands = []  # Keep original bands for band plots
    for i, (band, origin) in enumerate(zip(sorted_bands, sorted_origins)):
        if origin == 'L':
            # earlier segment: pad right
            padded_band = pad_band_right(band, seq_len)
        else:
            # later segment: pad left
            padded_band = pad_band_left(band, seq_len)
        padded_bands.append(padded_band)
        unpadded_bands.append(band)  # Keep original unpadded band
    padded_bands = np.array(padded_bands)
    unpadded_bands = np.array(unpadded_bands)
    
    # In test stage, y now contains only the future sequence (pred_len)
    original_future_np = _to_np(original_future[b, :, c])  # [pred_len] - select feature c for all time steps
    # Get per-band forecast data (no padding needed)
    per_band_forecast_np = _to_np(per_band_forecast[b, c, :, :])  # [K_total, pred_len]
    

    
    # How many to show
    total_bands = len(sorted_bands)
    max_show_bands = min(total_bands, 20)

    # Three-row transposed layout: rows fixed (3), columns = N bands
    n_rows = 3
    n_cols = max_show_bands

    fig = plt.figure(figsize=(2.8 * n_cols, 7.5))

    # Larger font sizes
    LABEL_FS = 16
    TICK_FS = 15
    LEGEND_FS = 15
    ANNO_FS = 15

    # Fixed palette for consistency across rows
    professional_colors = [
        '#2E86AB', '#A23B72', '#F18F01', '#C73E1D', '#6A994E', '#7209B7',
        '#F72585', '#4CC9F0', '#FF6B35', '#4ECDC4', '#45B7D1', '#96CEB4'
    ]

    time_steps_forecast = np.arange(pred_len)
    time_steps_recon = np.arange(seq_len)

    # Precompute cumulative sums in descending order
    cum_forecasts = []
    cum_recons = []
    running_forecast = np.zeros_like(time_steps_forecast, dtype=np.float32)
    running_recon = np.zeros_like(time_steps_recon, dtype=np.float32)
    for i in range(max_show_bands):
        orig_idx = sorted_indices[i]
        if orig_idx < len(per_band_forecast_np):
            running_forecast = running_forecast + per_band_forecast_np[orig_idx]
        cum_forecasts.append(running_forecast.copy())
        running_recon = running_recon + padded_bands[i]
        cum_recons.append(running_recon.copy())

    # -------- NEW: compute shared y-limits for Row 1 (bands) --------
    if max_show_bands > 0:
        # Concatenate only the bands we will actually show
        bands_for_ylim = [unpadded_bands[i] for i in range(max_show_bands) if len(unpadded_bands[i]) > 0]
        if len(bands_for_ylim) > 0:
            all_vals = np.concatenate(bands_for_ylim)
            band_ymin = np.nanmin(all_vals)
            band_ymax = np.nanmax(all_vals)
            # If degenerate (flat), expand slightly
            if not np.isfinite(band_ymin) or not np.isfinite(band_ymax):
                band_ymin, band_ymax = -1.0, 1.0
            elif band_ymin == band_ymax:
                eps = 1e-6 if band_ymin == 0 else 0.05 * abs(band_ymin)
                band_ymin -= eps
                band_ymax += eps
        else:
            band_ymin, band_ymax = -1.0, 1.0
    else:
        band_ymin, band_ymax = -1.0, 1.0
    # ----------------------------------------------------------------

    # Row 1: bands (with shared ylim)
    for i in range(max_show_bands):
        ax = plt.subplot(n_rows, n_cols, 1 + i)
        band_data = unpadded_bands[i]
        color = professional_colors[i % len(professional_colors)]
        ax.plot(np.arange(len(band_data)), band_data, color=color, linewidth=1.0, alpha=0.9)
        ax.set_ylim(band_ymin, band_ymax)  # <<< shared y-range across all band subplots
        weight = sorted_weights[i]
        origin = sorted_origins[i]
        orig_idx = sorted_indices[i]
        ax.text(0.5, 1.02, f'{origin}{orig_idx}  w={weight:.3f}', transform=ax.transAxes,
                ha='center', va='bottom', fontsize=ANNO_FS)
        ax.tick_params(labelsize=TICK_FS)
        if i == 0:
            ax.set_ylabel('Band', fontsize=LABEL_FS)


    # Row 3: cumulative reconstruction vs true input
    x_ref_np = _to_np(x_ref[b, c]) if x_ref is not None else None
    
    for i in range(max_show_bands):
        ax = plt.subplot(n_rows, n_cols, 1 * n_cols + 1 + i)
        ax.plot(time_steps_recon, x_ref_np, color='#10B981', linewidth=1.0, alpha=0.9, label='True Input')
        ax.plot(time_steps_recon, cum_recons[i], color='#F72585', linewidth=1.2, alpha=0.95, label='Cumulative\n Reconstruction')
        ax.tick_params(labelsize=TICK_FS)
        if i == 0:
            ax.set_ylabel('Reconstruct', fontsize=LABEL_FS)
            ax.legend(loc='upper right', fontsize=LEGEND_FS, frameon=False, handlelength=2, borderaxespad=0.2)


    # Row 2: cumulative forecast vs true future
    for i in range(max_show_bands):
        ax = plt.subplot(n_rows, n_cols, 2 * n_cols + 1 + i)
        ax.plot(time_steps_forecast, cum_forecasts[i], color='#8B5CF6', linewidth=1.2, alpha=0.95, label='Cumulative\n Forecast')
        if original_future_np is not None:
            ax.plot(time_steps_forecast, original_future_np, color='#10B981', linewidth=1.0, alpha=0.9, label='True Future')
        ax.tick_params(labelsize=TICK_FS)
        if i == 0:
            ax.set_ylabel('Forecast', fontsize=LABEL_FS)
            ax.legend(loc='upper right', fontsize=LEGEND_FS, frameon=False, handlelength=2, borderaxespad=0.2)



    # No global titles per request
    plt.tight_layout()

    filename = f"{side_tag}_{tag}_bands_gridT_B{b}_C{c}_L{K_left}_R{K_right}_S{seq_len}_P{pred_len}_sample{b}.png"
    out_path = os.path.join(save_dir, filename)
    plt.savefig(out_path, dpi=150, bbox_inches='tight')
    plt.close()
    