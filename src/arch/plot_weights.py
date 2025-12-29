#!/usr/bin/env python3
"""
Plot PRISM weights for pred_len=720 across four ETT datasets.
This script loads the pred_len=720 checkpoints (no regularization tags).
"""

import argparse
import os
import re
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import torch
from torch.utils.data import DataLoader
import torch.nn.functional as F

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

import sys

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))
if str(ROOT / "src") not in sys.path:
    sys.path.append(str(ROOT / "src"))

from src.arch.prism import prism
from src.data.data_loader import Dataset_ETT_hour, Dataset_ETT_minute


def load_state_dict_compatible(model: torch.nn.Module, ckpt_path: Path) -> None:
    raw = None
    try:
        try:
            from torch.serialization import add_safe_globals
            import numpy as _np

            add_safe_globals([_np.core.multiarray.scalar])
        except Exception:
            pass
        raw = torch.load(str(ckpt_path), map_location="cpu", weights_only=True)
    except Exception:
        raw = torch.load(str(ckpt_path), map_location="cpu", weights_only=False)
    sd = raw.get("state_dict", raw.get("model_state_dict", raw))
    sd = {(k[7:] if k.startswith("module.") else k): v for k, v in sd.items()}
    cur = model.state_dict()
    kept = {k: v for k, v in sd.items() if (k in cur and v.shape == cur[k].shape)}
    cur.update(kept)
    model.load_state_dict(cur, strict=False)


def build_val_loader(
    dataset: str, seq_len: int, pred_len: int, batch_size: int = 512
) -> DataLoader:
    root_path = str(ROOT / "temp" / "datasets" / "ETT-small")
    if dataset in ("ETTh1", "ETTh2"):
        data_path = f"{dataset}.csv"
        ds = Dataset_ETT_hour(
            root_path=root_path,
            data_path=data_path,
            flag="test",  # Use test set
            size=[seq_len, seq_len, pred_len],
            features="M",
            target="OT",
            timeenc=1,
        )
    elif dataset in ("ETTm1", "ETTm2"):
        data_path = f"{dataset}.csv"
        ds = Dataset_ETT_minute(
            root_path=root_path,
            data_path=data_path,
            flag="test",  # Use test set
            size=[seq_len, seq_len, pred_len],
            features="M",
            target="OT",
            timeenc=1,
        )
    else:
        raise ValueError(f"Unknown dataset: {dataset}")
    return DataLoader(ds, batch_size=batch_size, shuffle=False, drop_last=False)


@torch.no_grad()
def eval_mean_wlr_for_ckpt(
    ckpt_path: Path, meta: Dict, device: str, K_IMP: int
) -> Tuple[np.ndarray, np.ndarray, float, float]:
    """Evaluate PRISM model and capture weights."""

    model = prism(
        input_dim=7,
        hidden_dim=meta.get("seq_len", 336),
        num_components=K_IMP,
        dropout=meta.get("dropout", 0.25),
        seq_len=meta.get("seq_len", 336),
        pred_len=meta.get("pred_len", 96),
        save_dir=str(ckpt_path.parent),
        overlap=0,
        decomp_kind="haar",
        tree_depth=meta.get("tree_depth", 2),
        use_last_layer_only=True,
    ).to(device)

    # Load checkpoint with strict=True
    # Load checkpoint with strict=False and handle shape mismatches
    checkpoint = torch.load(ckpt_path, map_location=device)

    # Handle both old format (state_dict) and new format (dict with model_state_dict)
    if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
        # New format: comprehensive checkpoint
        model_state_dict = checkpoint["model_state_dict"]
        print(
            f"Loaded comprehensive checkpoint from epoch {checkpoint.get('epoch', 'unknown')}"
        )
        print(f"Validation loss: {checkpoint.get('val_loss', 'unknown')}")
    else:
        # Old format: direct state_dict
        model_state_dict = checkpoint

    # Handle DDP checkpoint format (remove 'module.' prefix)
    if any(key.startswith("module.") for key in model_state_dict.keys()):
        model_state_dict = {
            key.replace("module.", ""): value for key, value in model_state_dict.items()
        }

    # Filter out weights with shape mismatches
    model_state = model.state_dict()
    filtered_checkpoint = {}
    for key, value in model_state_dict.items():
        if key in model_state:
            if model_state[key].shape == value.shape:
                filtered_checkpoint[key] = value
            # Skip weights with shape mismatches

    model.load_state_dict(filtered_checkpoint, strict=False)
    model.eval()

    # Set epoch for plotting internals
    try:
        model.set_current_epoch(0)
    except Exception:
        try:
            setattr(model.backbone.tree, "_current_epoch", 0)
        except Exception:
            pass

    loader = build_val_loader(meta["dataset"], meta["seq_len"], meta["pred_len"])

    wL_sums = None
    wR_sums = None
    count = 0
    mse_sum = 0.0
    mae_sum = 0.0
    num_batches = 0

    for bx, by in loader:
        bx = bx.to(device=device, dtype=torch.float32)
        by = by.to(device=device, dtype=torch.float32)
        pred_len = int(meta.get("pred_len", 96))
        y_target = by[:, -pred_len:, :]
        y_target_t = y_target.transpose(1, 2).contiguous()

        # Forward pass
        preds = model.forecast(bx)

        # Handle prediction shape
        if preds.dim() == 3:
            C = bx.shape[2]
            H = pred_len
            shape = list(preds.shape)
            try:
                c_dim = shape.index(C)
            except ValueError:
                c_dim = 1 if shape[1] == y_target_t.shape[1] else 2
            try:
                h_dim = shape.index(H)
            except ValueError:
                h_dim = 2 if shape[2] == y_target_t.shape[2] else 1
            if not (c_dim == 1 and h_dim == 2):
                order = [0, c_dim, h_dim]
                preds = preds.permute(*order).contiguous()

        # Try to capture actual weights from the model
        wL = None
        wR = None

        # Check if model has stored weights
        if (
            hasattr(model.backbone.tree, "_current_wL")
            and model.backbone.tree._current_wL is not None
        ):
            wL = model.backbone.tree._current_wL
        elif (
            hasattr(model.backbone.tree, "_last_wL")
            and model.backbone.tree._last_wL is not None
        ):
            wL = model.backbone.tree._last_wL
        elif (
            hasattr(model.backbone.tree, "_captured_wL")
            and model.backbone.tree._captured_wL is not None
        ):
            wL = model.backbone.tree._captured_wL

        if (
            hasattr(model.backbone.tree, "_current_wR")
            and model.backbone.tree._current_wR is not None
        ):
            wR = model.backbone.tree._current_wR
        elif (
            hasattr(model.backbone.tree, "_last_wR")
            and model.backbone.tree._last_wR is not None
        ):
            wR = model.backbone.tree._last_wR
        elif (
            hasattr(model.backbone.tree, "_captured_wR")
            and model.backbone.tree._captured_wR is not None
        ):
            wR = model.backbone.tree._captured_wR

        # If no weights found, try to access weights from tree levels
        if wL is None or wR is None:
            if hasattr(model.backbone.tree, "levels"):
                for level in model.backbone.tree.levels:
                    if hasattr(level, "wL") and level.wL is not None:
                        wL = level.wL
                    if hasattr(level, "wR") and level.wR is not None:
                        wR = level.wR
                    if wL is not None and wR is not None:
                        break

        # If still no weights, raise error instead of creating dummy weights
        if wL is None or wR is None:
            raise RuntimeError(
                f"No weights found in model for {ckpt_path.name}. "
                f"Expected _current_wL and _current_wR buffers to be available."
            )

        wL_dummy = wL
        wR_dummy = wR

        wL_mean = wL_dummy.mean(dim=(0, 1)).detach().cpu().numpy()
        wR_mean = wR_dummy.mean(dim=(0, 1)).detach().cpu().numpy()

        if wL_sums is None:
            wL_sums = np.zeros_like(wL_mean)
        if wR_sums is None:
            wR_sums = np.zeros_like(wR_mean)
        wL_sums += wL_mean
        wR_sums += wR_mean
        count += 1

        mse_sum += float(
            F.mse_loss(
                preds.detach().cpu(), y_target_t.detach().cpu(), reduction="mean"
            ).item()
        )
        mae_sum += float(
            F.l1_loss(
                preds.detach().cpu(), y_target_t.detach().cpu(), reduction="mean"
            ).item()
        )
        num_batches += 1

    if count == 0:
        raise RuntimeError(f"No batches processed for {ckpt_path.name}")

    avg_mse = (mse_sum / num_batches) if num_batches > 0 else float("nan")
    avg_mae = (mae_sum / num_batches) if num_batches > 0 else float("nan")
    return wL_sums / count, wR_sums / count, avg_mse, avg_mae


def parse_pred720_fname_from_checkpoint(fname: str, dataset: str) -> Dict:
    """Parse pred_len=720 checkpoint filename."""
    pattern = (
        rf"^({dataset})_pred(\d+)_cont(\d+)_s(\d+)_M_w(\d+)_L(\d+)_d([\d.]+)_"
        rf"loss_(\w+)_p(\d+)_checkpoint\.pth$"
    )
    match = re.search(pattern, fname)
    if not match:
        return {}
    return {
        "dataset": match.group(1),
        "pred_len": int(match.group(2)),
        "seq_len": int(match.group(3)),
        "seed": int(match.group(4)),
        "num_wavelets": int(match.group(5)),
        "tree_depth": int(match.group(6)),
        "dropout": float(match.group(7)),
        "loss": match.group(8),
        "patience": int(match.group(9)),
    }


def _score_meta(meta: Dict) -> int:
    score = 0
    if meta.get("loss") == "mae":
        score += 10
    if meta.get("patience") == 15:
        score += 5
    if meta.get("num_wavelets") == 8:
        score += 3
    if meta.get("tree_depth") == 2:
        score += 2
    if meta.get("seed") == 14:
        score += 1
    return score


def select_pred720_checkpoint(ckpt_dir: Path) -> Tuple[Path, Dict] | None:
    """Select the best pred_len=720 checkpoint in a checkpoint directory."""
    dataset = ckpt_dir.parent.name.split("_")[0]
    candidates = []
    for ckpt_file in ckpt_dir.glob(f"{dataset}_pred720_cont*_s*_M_w*_L*_d*_loss_*_p*_checkpoint.pth"):
        meta = parse_pred720_fname_from_checkpoint(ckpt_file.name, dataset)
        if meta and meta.get("pred_len") == 720:
            candidates.append((ckpt_file, meta))
    if not candidates:
        return None
    candidates.sort(key=lambda item: (_score_meta(item[1]), item[0].name), reverse=True)
    return candidates[0]


def band_labels(num_bands: int) -> List[str]:
    return [f"L{i}" for i in range(num_bands)]


def make_pred720_plot(df: pd.DataFrame, out_png: Path) -> None:
    """Create plot for pred_len=720 weights across datasets."""
    datasets = sorted(df["dataset"].unique())

    if df.empty:
        print("No data found!")
        return

    fig, axes = plt.subplots(2, 2, figsize=(16, 12), squeeze=False)

    for i, dataset in enumerate(datasets):
        row = i // 2
        col = i % 2
        ax = axes[row, col]

        dataset_data = df[df["dataset"] == dataset]

        if dataset_data.empty:
            ax.axis("off")
            continue

        # Get weight columns
        wL_cols = [
            c
            for c in dataset_data.columns
            if c.startswith("wL_") and c.split("_")[1].isdigit()
        ]
        if not wL_cols:
            ax.axis("off")
            continue

        max_idx = max(int(c.split("_")[1]) for c in wL_cols)
        K = max_idx + 1
        labels_k = band_labels(K)

        x_band = np.arange(K) * 1.2
        width = 0.35

        # Plot mean weights - group wL_i and wR_i together
        wL_entropy = [
            dataset_data.get(f"wL_{i}", pd.Series(dtype=float)).mean() for i in range(K)
        ]
        wR_entropy = [
            dataset_data.get(f"wR_{i}", pd.Series(dtype=float)).mean() for i in range(K)
        ]

        # Create grouped bars: wL_0, wR_0, wL_1, wR_1, etc.
        grouped_values = []
        grouped_labels = []
        for i in range(K):
            grouped_values.extend([wL_entropy[i], wR_entropy[i]])
            grouped_labels.extend([f"wL_{i}", f"wR_{i}"])

        x_positions = np.arange(len(grouped_values))
        colors = ["#4C72B0" if "wL" in label else "#55A868" for label in grouped_labels]

        ax.bar(x_positions, grouped_values, width=0.6, color=colors, alpha=0.8)

        # Set x-axis labels
        ax.set_xticks(x_positions)
        ax.set_xticklabels(grouped_labels, rotation=45, ha="right")

        ax.set_title(f"{dataset} - pred_len=720 Weights")
        ax.set_ylabel("Mean weight")
        ax.grid(axis="y", alpha=0.2)

        # Add legend manually
        from matplotlib.patches import Patch

        legend_elements = [
            Patch(facecolor="#4C72B0", label="wL"),
            Patch(facecolor="#55A868", label="wR"),
        ]
        ax.legend(handles=legend_elements)

    fig.suptitle("PRISM Weights (pred_len=720) Across ETT Datasets")
    fig.tight_layout(rect=[0, 0, 0.98, 0.94])

    fig.savefig(out_png, dpi=150)
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--ckpt_base_dir",
        type=str,
        default="./temp/logs/exp_prism",
        help="Base directory containing checkpoint directories",
    )
    args = ap.parse_args()

    ckpt_base_dir = Path(args.ckpt_base_dir)
    if not ckpt_base_dir.exists():
        raise FileNotFoundError(f"Checkpoint base dir not found: {ckpt_base_dir}")

    out_dir = ROOT / "temp" / "plots" / "pred720_weights"
    out_dir.mkdir(parents=True, exist_ok=True)

    all_items = []
    for dataset in ["ETTh1", "ETTh2", "ETTm1", "ETTm2"]:
        dataset_dir = ckpt_base_dir / f"{dataset}_cont336_M" / "checkpoint_dir"
        if dataset_dir.exists():
            item = select_pred720_checkpoint(dataset_dir)
            if item:
                all_items.append(item)
                print(f"Selected pred720 checkpoint for {dataset}: {item[0].name}")
            else:
                print(f"No pred720 checkpoint found for {dataset}")

    if not all_items:
        print("No pred720 checkpoints found!")
        return

    print(f"Total pred720 checkpoints found: {len(all_items)}")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    # print(f"Using device: {device}")

    rows = []
    for ckpt_path, meta in all_items:
        try:
            print(f"Processing {ckpt_path.name}")
            wL, wR, avg_mse, avg_mae = eval_mean_wlr_for_ckpt(
                ckpt_path, meta, device=device, K_IMP=8
            )
        except Exception as e:
            print(f"Skipping {ckpt_path.name}: {e}")
            continue

        row = {
            "ckpt": ckpt_path.name,
            **meta,
            "mse": float(avg_mse),
            "mae": float(avg_mae),
        }

        for i, v in enumerate(wL):
            row[f"wL_{i}"] = float(v)
        for i, v in enumerate(wR):
            row[f"wR_{i}"] = float(v)
        rows.append(row)

    if not rows:
        print("No checkpoints processed successfully.")
        return

    print(f"Successfully processed {len(rows)} checkpoints")

    df = pd.DataFrame(rows)

    print(f"DataFrame shape: {df.shape}")
    csv_path = out_dir / "pred720_weights_summary.csv"
    df.to_csv(csv_path, index=False)

    # Clean previous images
    for f in out_dir.glob("*.png"):
        try:
            f.unlink()
        except Exception:
            pass

    make_pred720_plot(df, out_dir / "pred720_weights_comparison.png")


if __name__ == "__main__":
    main()
