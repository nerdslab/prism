#!/usr/bin/env python3
import os, random, json
import argparse
import torch.multiprocessing as mp
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from pathlib import Path
import numpy as np
import torch

# WandB import (optional)
try:
    import wandb
    WANDB_AVAILABLE = True
except ImportError:
    WANDB_AVAILABLE = False
    print("Warning: WandB not available. Install with 'pip install wandb' for experiment tracking.")

# All experiment features are now unified in exp_long_term_forecasting

# Set environment variables for multiprocessing
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
os.environ['TORCH_DISTRIBUTED_DEBUG'] = 'INFO'

def seed_all(seed: int) -> None:
    """Set all random seeds properly."""
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

def is_dist() -> bool:
    return dist.is_available() and dist.is_initialized()

def is_rank0() -> bool:
    return (not is_dist()) or dist.get_rank() == 0

def run_name_from_args(args) -> str:
    """Generate run name from arguments - much faster than OmegaConf"""
    # Ensure loss has underscore: loss_{name}
    return f"{args.data}_pred{args.pred_len}_cont{args.seq_len}_s{args.seed}_{args.features}_w{args.num_components}_d{args.dropout}_loss_{args.loss}_p{args.patience}"

def run_dir_from_args(args) -> Path:
    # Save into a compact dataset-level folder: <base>/<DATASET>_cont<seq_len>_<features>
    base = Path("./temp/logs/exp_prism")
    return base / f"{args.data}_cont{args.seq_len}_{args.features}"

@torch.no_grad()
def broadcast_model_state(model, src: int = 0):
    """Broadcast model state (params + buffers) from src to all ranks."""
    if not is_dist():
        return
    for tensor in model.state_dict().values():
        dist.broadcast(tensor, src)

def load_state_dict_compatible(model: torch.nn.Module, ckpt_path: Path) -> None:
    """
    Load checkpoint but ignore keys that don't exist or whose shapes don't match.
    Also drop obsolete discriminator FiLM t_proj weights if present.
    """
    raw = torch.load(str(ckpt_path), map_location="cpu")
    sd  = raw.get("state_dict", raw.get("model_state_dict", raw))
    # strip DDP prefix
    sd  = { (k[7:] if k.startswith("module.") else k): v for k, v in sd.items() }
    # drop known-bad keys (legacy discriminator FiLM time proj)
    sd  = { k: v for k, v in sd.items() if ".film.t_proj." not in k }

    cur = model.state_dict()
    kept = { k: v for k, v in sd.items() if (k in cur and v.shape == cur[k].shape) }

    missing = [k for k in cur.keys() if k not in kept]
    unexpected = [k for k in sd.keys() if k not in cur]

    if is_rank0():
        print(f"[load] kept={len(kept)} missing={len(missing)} unexpected={len(unexpected)}")
        if unexpected[:10]: print("  unexpected (first 10):", unexpected[:10])
        if missing[:10]:    print("  missing (first 10):   ", missing[:10])

    cur.update(kept)
    model.load_state_dict(cur, strict=False)

class SimpleConfig:
    """Simple config class that mimics OmegaConf structure but much faster"""
    def __init__(self, **kwargs):
        for key, value in kwargs.items():
            if isinstance(value, dict):
                setattr(self, key, SimpleConfig(**value))
            else:
                setattr(self, key, value)

def create_config_from_args(args, rank=0):
    """Convert argparse args to a simple config object - much faster than OmegaConf"""
    # Determine input/output channels based on feature mode
    if args.features == 'M':
        enc_in_val = args.num_channels
        c_out_val = args.num_channels
    else:  # 'S'
        enc_in_val = 1
        c_out_val = 1
    return SimpleConfig(
        data=SimpleConfig(
            dataset=args.data,
            prediction_length=args.pred_len,
            context_length=args.seq_len,
            features=args.features,
            target=args.target,
            root_path=args.root_path,
            data_path=args.data_path,
            batch_size=args.batch_size,
            drop_last=bool(getattr(args, "drop_last", 1)),
            label_length=0,  # Required by data loader
            embed='timeF',   # Required by data loader
            num_workers=1,   # Required by data loader
        ),
        model=SimpleConfig(
            enc_hidden=args.enc_hidden,
            dec_hidden=args.dec_hidden,
            dropout=args.dropout,
            type='exp_prism',
            params=SimpleConfig(
                hidden_size=args.enc_hidden,
                dropout=args.dropout,
                init_skip=True,
                enc_in=enc_in_val,
                c_out=c_out_val,
                use_last_layer_only=args.use_last_layer_only,
                tree_depth=args.tree_depth,
                d_model=32,
                use_features=True,
                use_warmup_padding=False,
                diffusion_mode='pretrain',
                # pruning and branch-cutting removed
            ),
            encoder_type='dpvae',
        ),
        train=SimpleConfig(
            train_epochs=args.train_epochs,
            batch_size=args.batch_size,
            patience=args.patience,
            lr=args.lr,
            loss=args.loss,
            lradj=args.lradj,
            use_amp=args.use_amp,
            max_epochs=args.train_epochs,
            learning_rate=args.lr,
            precision_mode='mixed16',
            check_val_every_n_epoch=1,
            eval_every=1,
            inverse=False,
            use_validation_set=True,
        ),
        device_config=SimpleConfig(
            use_multi_gpu=args.use_multi_gpu,
            devices=args.devices,
            device='cuda:0',
            gpu_type='cuda',
            use_gpu=True,
        ),
        diffusion=SimpleConfig(
            epochs=args.train_epochs,
            load_epochs=args.train_epochs,
            sampler='DDPM',
            sampler_params=SimpleConfig(
                num_samples=100,
            ),
        ),
        pretrain=SimpleConfig(
            epochs=args.train_epochs,
            load_pretrain_epochs=None,
        ),
        seed=args.seed,
        rank=rank,
        log_dir=str(run_dir_from_args(args)),
        lr=args.lr,
        num_features=4,
        num_channels=args.num_channels,
        decomp_kind=args.decomp_kind,
        overlap=args.overlap,
        num_components=args.num_components,  # Add num_components to config
    )

def main(rank: int, world_size: int, args):
    """Main function with multiprocessing support - optimized version"""
    # Initialize distributed training
    torch.cuda.set_device(rank)
    dist.init_process_group("nccl", rank=rank, world_size=world_size)
    seed_all(args.seed * (rank + 1))

    # Initialize WandB if available and on rank 0
    if WANDB_AVAILABLE and rank == 0:
        wandb.init(
            project="prism-wavelet-sweep",
            name=f"{args.data}_pred{args.pred_len}_wavelets{args.num_components}_L{args.tree_depth}_seed{args.seed}_patience{args.patience}_dropout{args.dropout}_loss{args.loss}",
            config={
                "dataset": args.data,
                "pred_length": args.pred_len,
                "num_components": args.num_components,
                "seed": args.seed,
                "dropout": args.dropout,
                "context_length": args.seq_len,
                "batch_size": args.batch_size,
                "learning_rate": args.lr,
                "max_epochs": args.train_epochs,
            }
        )
    
    # Create config from args - much faster than OmegaConf
    cfg = create_config_from_args(args, rank)

    # Use unified experiment class (all regularization features are now in exp_long_term_forecasting)
    from src.exp.exp_long_term_forecasting import Exp_Long_Term_Forecast as _Exp
    exp = _Exp(cfg)
    run_dir = run_dir_from_args(args)
    if is_rank0():
        Path(run_dir).mkdir(parents=True, exist_ok=True)
    # Save directory is the compact dataset folder; files named by full run name
    exp.save_dir = run_dir
    if hasattr(exp.model, "save_dir"):
        exp.model.save_dir = str(run_dir)
    
    exp.model = DDP(exp.model, device_ids=[rank], find_unused_parameters=True)        
    
    # Set modes on model
    exp.train()
    print(f'>>>>>>>testing<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<')
    mae, mse = exp.test()
    print(f"test mae:{mae}, mse:{mse}")
    
def parse_args():
    """Parse command line arguments - much faster than OmegaConf"""
    parser = argparse.ArgumentParser(description='DPVAE Training - Optimized Version')
    
    # Dataset settings
    parser.add_argument('--data', type=str, default='ETTh1', 
                       choices=['ETTh1', 'ETTh2', 'ETTm1', 'ETTm2', 'ECL', 'traffic', 'weather', 'electricity', 'exchange'],
                       help='name of dataset')
    parser.add_argument('--root_path', type=str, default='./datasets/long/', 
                       help='root path of the data file')
    parser.add_argument('--data_path', type=str, default='ETTh1.csv', 
                       help='location of the data file')
    parser.add_argument('--features', type=str, default='M', choices=['S', 'M'], 
                       help='features S is univariate, M is multivariate')
    parser.add_argument('--target', type=str, default='OT', help='target feature')
    
    # Model settings
    parser.add_argument('--seq_len', type=int, default=336, help='look back window')
    parser.add_argument('--pred_len', type=int, default=336, help='prediction sequence length')
    parser.add_argument('--enc_hidden', type=int, default=336, help='hidden size of encoder')
    parser.add_argument('--dec_hidden', type=int, default=336, help='hidden size of decoder')
    parser.add_argument('--dropout', type=float, default=0.5, help='dropout')
    parser.add_argument('--tree_depth', type=int, default=2, help='depth of the decomposition tree')
    # Kept for backward compatibility; ignored by the model
    parser.add_argument('--num_heads', type=int, default=1, help='(legacy) attention heads - unused')
    parser.add_argument('--use_last_layer_only', action='store_true', default=False, help='use only the last layer for forecasting')
    
    # Training settings
    parser.add_argument('--train_epochs', type=int, default=100, help='train epochs')
    parser.add_argument('--batch_size', type=int, default=512, help='batch size')
    parser.add_argument('--patience', type=int, default=5, help='early stopping patience')
    parser.add_argument('--lr', type=float, default=0.0001, help='learning rate')
    parser.add_argument('--loss', type=str, default='mse', help='loss function')
    parser.add_argument('--lradj', type=int, default=1, help='adjust learning rate')
    parser.add_argument('--use_amp', action='store_true', default=False, 
                       help='use automatic mixed precision training')
    
    # Device settings
    parser.add_argument('--use_multi_gpu', action='store_true', default=True, 
                       help='use multi-GPU training')
    parser.add_argument('--devices', type=str, default='0,1,2,3', 
                       help='comma-separated list of GPU devices')
    
    # Other settings
    parser.add_argument('--seed', type=int, default=0, help='random seed')
    parser.add_argument('--overlap', type=int, default=24, help='overlap size')
    parser.add_argument('--num_channels', type=int, default=7, help='number of variables when features=M')
    parser.add_argument('--decomp_kind', type=str, default='haar', help='decomposition kind')
    parser.add_argument('--num_components', type=int, default=6, help='number of components for decomposition')
    parser.add_argument('--drop_last', type=int, default=1,
                        help='Whether to drop last incomplete batch (1=true, 0=false)')
    return parser.parse_args()

if __name__ == '__main__':
    args = parse_args()
    
    # Check if we should use multiprocessing
    if args.use_multi_gpu and len(args.devices.split(",")) > 1:
        world_size = len(args.devices.split(","))
        print(f"Starting multi-GPU training with {world_size} GPUs")
        mp.spawn(main, args=(world_size, args), nprocs=world_size, join=True)
    else:
        # Single GPU or CPU - run directly
        try:
            print("Starting single GPU training")
            main(0, 1, args)
        except Exception as e:
            print(f"❌ Error in single GPU mode: {e}")
            raise
