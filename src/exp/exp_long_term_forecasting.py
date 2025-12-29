#!/usr/bin/env python3
"""
Optimized long-term forecasting experiment - no more amateur hour code duplication
"""
import os
import time
from pathlib import Path
from typing import Any, Dict, List, Tuple
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import optim
from src.data.data_loader import Dataset_ETT_hour, Dataset_ETT_minute, Dataset_Custom
from torch.utils.data import DataLoader

# Local imports
# from src.utils.metrics import metric
# from src.data.data_factory import data_provider
from src.exp.exp_basic import Exp_Basic
from src.arch.prism import prism
from src.utils.tools import EarlyStopping

# WandB import (optional)
try:
    import wandb

    WANDB_AVAILABLE = True
except ImportError:
    WANDB_AVAILABLE = False


import os, math
import numpy as np
import matplotlib

matplotlib.use("Agg")


def tic():
    torch.cuda.synchronize()
    return time.time()


def toc(t0, tag):
    torch.cuda.synchronize()
    print(f"[TIMER] {tag}: {time.time()-t0:.3f}s")
    return time.time()


def MAE(pred, true):
    return np.mean(np.abs(pred - true))


def MSE(pred, true):
    return np.mean((pred - true) ** 2)


def adjust_learning_rate(optimizer, epoch, args):
    if args.train.lradj == 1:
        lr_adjust = {epoch: args.train.learning_rate * (0.95 ** (epoch // 1))}

    elif args.train.lradj == 2:
        lr_adjust = {
            0: 0.0001,
            5: 0.0005,
            10: 0.001,
            20: 0.0001,
            30: 0.00005,
            40: 0.00001,
            70: 0.000001,
        }

    if epoch in lr_adjust.keys():
        lr = lr_adjust[epoch]
        for param_group in optimizer.param_groups:
            param_group["lr"] = lr
        print("Updating learning rate to {}".format(lr))
    else:
        for param_group in optimizer.param_groups:
            lr = param_group["lr"]
    return lr


class Exp_Long_Term_Forecast(Exp_Basic):
    """Optimized long-term forecasting experiment - no more monolithic mess."""

    def __init__(self, args: Any) -> None:
        self.precision_mode = args.train.precision_mode
        super().__init__(args)

        # Create checkpoint directory once during initialization
        # Handle case when save_dir is not yet set
        if hasattr(self, "save_dir") and self.save_dir is not None:
            ckpt_dir = self.save_dir / "checkpoint_dir"
            try:
                os.makedirs(ckpt_dir, exist_ok=True)
            except Exception:
                pass

    def _build_model(self) -> torch.nn.Module:
        """Build model - no additional regularization hooks."""
        model = prism(
            input_dim=self.args.model.params.enc_in,
            hidden_dim=self.args.model.params.hidden_size,
            num_components=getattr(self.args, "num_components", 6),
            dropout=self.args.model.params.dropout,
            seq_len=self.args.data.context_length,
            pred_len=getattr(
                self.args.data, "prediction_length", getattr(self.args, "pred_len", 96)
            ),
            save_dir=self.args.log_dir,
            decomp_kind=self.args.decomp_kind,
            overlap=self.args.overlap,
            tree_depth=getattr(self.args.model.params, "tree_depth", 2),
            use_last_layer_only=getattr(
                self.args.model.params, "use_last_layer_only", False
            ),
        )
        # Hook up save_dir if the base class set it
        model.save_dir = getattr(self, "save_dir", None)
        return model.float()

    def _get_data(self, flag: str) -> Tuple[Any, Any]:
        """Get data loader."""
        args = self.args
        data_dict = {
            "ETTh1": Dataset_ETT_hour,
            "ETTh2": Dataset_ETT_hour,
            "ETTm1": Dataset_ETT_minute,
            "ETTm2": Dataset_ETT_minute,
            "weather": Dataset_Custom,
            "ECL": Dataset_Custom,
            "electricity": Dataset_Custom,
            "Solar": Dataset_Custom,
            "traffic": Dataset_Custom,
            "exchange": Dataset_Custom,
            "illness": Dataset_Custom,
        }
        # data = "Dataset_Custom" if args.data.dataset in {"traffic", "weather2", "electricity", "weather", "exchange_rate"} else args.data.dataset
        Data = data_dict[args.data.dataset]
        timeenc = 0 if args.data.embed != "timeF" else 1

        drop_last_cfg = bool(getattr(args.data, "drop_last", True))
        if flag == "test":
            shuffle_flag = False
            drop_last = drop_last_cfg
            batch_size = args.data.batch_size
        elif flag == "val":
            shuffle_flag = True
            drop_last = drop_last_cfg
            if args.data.dataset in {"exchange"}:
                batch_size = 32
            else:
                batch_size = args.data.batch_size
        else:
            # Train should never drop the last incomplete batch
            shuffle_flag = True
            drop_last = False
            batch_size = args.data.batch_size
        data_set = Data(
            root_path=args.data.root_path,
            data_path=args.data.data_path,
            flag=flag,
            size=[
                args.data.context_length,
                args.data.label_length,
                args.data.prediction_length,
            ],
            features=args.data.features,
            target=args.data.target,
            timeenc=timeenc,
        )
        print(flag, len(data_set))
        data_loader = DataLoader(
            data_set,
            batch_size=batch_size,
            shuffle=shuffle_flag,
            num_workers=args.data.num_workers,
            drop_last=drop_last,
        )
        return data_loader

    # return #data_provider(self.args, flag)

    def _select_optimizer(self) -> torch.optim.Optimizer:
        """Select optimizer."""
        return optim.Adam(self.model.parameters(), lr=self.args.lr)

    def vali(self, loader: Any) -> Dict[str, float]:
        """Validation method - plots bands once at i==5 (first epoch batch trigger)."""
        self.model.eval()
        total_loss, mses, maes = [], [], []

        with torch.no_grad():
            for i, (bx, by) in enumerate(loader):
                bx = bx.float().to(self.args.rank)
                by = by.float()
                preds = self.model.module.forecast(bx)  # Forecast future
                # recon_x = self.model.module.reconstruct(bx, tk, bxm)  # Forecast future
                # recon_y = self.model.module.reconstruct(by, tk, bym)  # Forecast future
                # recon_mse = F.mse_loss(recon_x, bx, reduction="mean") + F.mse_loss(recon_y, by, reduction="mean")
                if self.args.train.loss == "mae":
                    val_loss = F.l1_loss(
                        preds.detach().cpu(), by.detach().cpu(), reduction="mean"
                    )
                else:  # mse
                    val_loss = F.mse_loss(
                        preds.detach().cpu(), by.detach().cpu(), reduction="mean"
                    )
                mse = MSE(preds.detach().cpu().numpy(), by.detach().cpu().numpy())
                mae = MAE(preds.detach().cpu().numpy(), by.detach().cpu().numpy())
                total_loss.append(val_loss)
                mses.append(mse)
                maes.append(mae)

        avg_loss = float(np.average(total_loss))
        mse = float(np.average(mses))
        mae = float(np.average(maes))
        print(
            "-----------start to {} {}-----------\n|  Normed  | mse:{:5.4f} | mae:{:5.4f} |".format(
                self.args.rank, avg_loss, mse, mae
            )
        )
        return avg_loss

    def _select_criterion(self, losstype):
        if losstype == "mse":
            criterion = nn.MSELoss()
        elif losstype == "mae":
            criterion = nn.L1Loss()
        else:
            criterion = nn.MSELoss()
        return criterion

    def train(self):
        """Minimal, sane training loop with optional AMP and rank-0 checkpointing."""
        t0 = tic()
        # Data
        train_loader = self._get_data("train")
        val_loader = self._get_data("val")
        test_loader = self._get_data("test")
        time_now = time.time()
        # Configure bands plot output dir to match OUTDIR/WANDB_NAME convention
        # Export envs used by plotting helper to avoid 'None' paths
        batch = next(iter(train_loader))[0]
        C = batch.size(2)
        self.model.module.C = C  # Get C from first batch
        optimizer = torch.optim.Adam(
            self.model.parameters(), lr=self.args.train.learning_rate
        )
        early_stopping = EarlyStopping(
            patience=self.args.train.patience, verbose=True, delta=2e-6
        )

        # Early stopping tracking
        self.early_stopping_counter = 0
        # amp_enabled = use_fp16 or use_bf16
        if self.args.train.use_amp:
            scaler = torch.cuda.amp.GradScaler()  # bf16 doesn't need scaling
        # amp_dtype   = torch.float16 if use_fp16 else (torch.bfloat16 if use_bf16 else None)
        best_val = float("inf")
        max_epochs = getattr(
            self.args.train, "max_epochs", getattr(self.args, "train_epochs", 1)
        )
        train_steps = len(train_loader)
        criterion = self._select_criterion(self.args.train.loss)
        # Handle case when save_dir is not yet set
        if hasattr(self, "save_dir") and self.save_dir is not None:
            ckpt_dir = self.save_dir / "checkpoint_dir"
        else:
            # Create a temporary directory if save_dir is not available
            ckpt_dir = Path("./temp/logs/temp_checkpoint")
            ckpt_dir.mkdir(parents=True, exist_ok=True)
        for epoch in range(int(max_epochs)):
            iter_count = 0
            self.model.module.set_current_epoch(epoch)  # For plotting purposes

            self.model.train()
            epoch_time = time.time()
            train_loss = []
            t0 = toc(t0, "train_time")
            for step, (bx, by) in enumerate(train_loader):
                iter_count += 1
                optimizer.zero_grad()

                # forward() now always returns tensor output (not loss)
                output, batch_y = self._process_one_batch_god(bx, by)
                loss = criterion(output, batch_y)

                train_loss.append(loss.item())
                # backward
                if self.args.train.use_amp:
                    scaler.scale(loss).backward()
                    # t0 = toc(t0, "backward")
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    loss.backward()
                    optimizer.step()
                # t0 = toc(t0, "opt.step")
                train_loss.append(loss.item())
                if step % 100 == 0:
                    print(
                        "\titers: {0}, epoch: {1} | loss: {2:.7f}".format(
                            step + 1, epoch + 1, loss.item()
                        )
                    )
                    speed = (time.time() - time_now) / iter_count
                    left_time = speed * ((max_epochs - epoch) * train_steps - step)
                    print(
                        "\tspeed: {:.4f}s/iter; left time: {:.4f}s".format(
                            speed, left_time
                        )
                    )
                    iter_count = 0
                    time_now = time.time()
                    print(f"epoch {epoch} step {step} loss {loss.item():.4f}")

            # lr = adjust_learning_rate(optimizer, epoch+1, self.args)
            print("Epoch: {} cost time: {}".format(epoch + 1, time.time() - epoch_time))
            avg_train = np.average(train_loss)
            print(f"[epoch {epoch}] train_loss={avg_train:.4f}")
            avg_val = self.vali(val_loader)
            print(f"[epoch {epoch}] val_loss={avg_val:.4f}")
            if avg_val < best_val:
                best_val = avg_val
            test_mse, test_mae = self.test(test_loader)
            print(f"[epoch {epoch}] test_mse={test_mse:.4f}, test_mae={test_mae:.4f}")

            # Log to WandB if available and on rank 0
            if WANDB_AVAILABLE:
                # Only log if WandB is initialized (rank 0)
                if wandb.run is not None:
                    wandb.log(
                        {
                            "epoch": epoch,
                            "train_loss": avg_train,
                            "val_loss": avg_val,
                            "test_mse": test_mse,
                            "test_mae": test_mae,
                            "best_val": best_val,
                        }
                    )
            t0 = toc(t0, "validation_time")
            # Use pre-created checkpoint directory
            # Build a unique run_name consistent with run_prism
            loss_name = f"loss_{self.args.train.loss}"
            tree_depth = getattr(self.args.model.params, "tree_depth", 2)

            run_name = f"{self.args.data.dataset}_pred{self.args.data.prediction_length}_cont{self.args.data.context_length}_s{self.args.seed}_{self.args.data.features}_w{self.args.num_components}_L{tree_depth}_d{self.args.model.params.dropout}_{loss_name}_p{self.args.train.patience}"
            early_stopping(avg_val, self.model, ckpt_dir / f"{run_name}_checkpoint.pth")
            # early_stopping(test_loss, self.model, path)
            if early_stopping.early_stop:
                print("Early stopping")

                break
            # lr = adjust_learning_rate(optimizer, epoch+1, self.args)

        # Name files by full run name to avoid many nested folders
        if hasattr(self, "save_dir") and self.save_dir is not None:
            run_name = (
                os.path.basename(str(self.save_dir))
                if "_" in os.path.basename(str(self.save_dir))
                else None
            )
        else:
            run_name = None
        # If save_dir is dataset folder, construct run_name like run_prism run_name
        if run_name is None:
            loss_name = f"loss_{self.args.train.loss}"

            run_name = f"{self.args.data.dataset}_pred{self.args.data.prediction_length}_cont{self.args.data.context_length}_s{self.args.seed}_{self.args.data.features}_w{self.args.num_components}_d{self.args.model.params.dropout}_{loss_name}_p{self.args.train.patience}"
        # Save best into checkpoint_dir with unique name as well
        best = ckpt_dir / f"{run_name}_best_model.pth"
        torch.save(
            {
                "epoch": epoch,
                "model_state_dict": self.model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "train_loss": avg_train,
                "val_loss": avg_val,
            },
            best,
        )
        print(f"🏆 saved best {best} (val {best_val:.4f})")

    def test(self, test_loader=None):
        """Test method - based on original exp_dynamight implementation."""
        self.model.eval()

        # Set test_mode = True at the start of test
        self.model.module.backbone.tree.test_mode = True

        if test_loader is None:
            test_loader = self._get_data("test")
        mse_loss = 0.0
        mae_loss = 0.0
        num_batches = 0
        # dtype = torch.float16 if self.args.train.precision_mode == "mixed16" else torch.float32
        with torch.no_grad():
            for batch in test_loader:
                # Move batch to device and convert to float
                batch = [b.cuda().float() if torch.is_tensor(b) else b for b in batch]
                bx, by = batch[0], batch[1]
                preds = self.model.module.forecast(bx, y=by)  # Forecast future
                # Compute metrics using the future part only
                mse = F.mse_loss(
                    preds.detach().cpu(), by.detach().cpu(), reduction="mean"
                )
                mae = F.l1_loss(
                    preds.detach().cpu(), by.detach().cpu(), reduction="mean"
                )

                mse_loss += mse.item()
                mae_loss += mae.item()
                num_batches += 1

        avg_mse = mse_loss / num_batches if num_batches > 0 else 0.0
        avg_mae = mae_loss / num_batches if num_batches > 0 else 0.0

        # Set test_mode = False at the end of test
        self.model.module.backbone.tree.test_mode = False

        self.model.train()
        return avg_mse, avg_mae

    def _process_one_batch_god(self, batch_x, batch_y):
        batch_x = batch_x.float().to(self.args.rank)
        batch_y = batch_y.float().to(self.args.rank)

        # Get predictions (output), not loss
        output = self.model(batch_x, batch_y)

        f_dim = -1 if self.args.data.features == "MS" else 0
        # batch_y
        batch_y = batch_y[:, -self.args.data.prediction_length :, f_dim:]
        return output, batch_y
