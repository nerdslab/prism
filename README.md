# PRISM
Hierarchical multiscale forecasting for long, multivariate time series.

PRISM builds a learnable multiresolution tree that decomposes signals into band-limited fragments, routes them with soft gates, mixes sibling bands, and fuses forecasts at the root. It is end-to-end, interpretable, and strong on long horizons with lower memory.

## Highlights
- Multiscale decomposition with learnable filters
- Temperature-controlled routing for band assignment
- Band-mixing block to capture cross-scale dependencies
- Squeeze-and-gate aggregation at the root
- Fast inference on long horizons

## Quick Start
1) Create your environment and install deps.
2) Put datasets under `./temp/datasets` as in the scripts.
3) Run a quick script.

```bash
# From repo root
cd PRISM
bash scripts/ETTh1_quick.sh
```

## Installation
This setup is tested for Python 3.10 and recent NVIDIA GPUs (e.g., 3090/4090).

```bash
git clone git@github.com:nerdslab/PRISM.git
cd PRISM

python3.10 -m venv tsdiff310
source tsdiff310/bin/activate  # Linux/Mac
# or: tsdiff310\Scripts\activate  # Windows

pip install gluonts==0.12.8
pip install aeon torch einops omegaconf lightning hydra-core matplotlib seaborn
pip install wandb==0.18.7 torchvision==0.20.1
pip install dtaidistance PyWavelets opt_einsum
pip install -U rich
pip install pytorch-lightning==1.9.5
pip install imageio patool sktime pykeops orjson torchtyping
```

## Data Layout
Each script expects a dataset folder and a CSV file under `./temp/datasets`.
Match the script paths exactly.

Examples:
- `./temp/datasets/ETT-small/ETTh1.csv`
- `./temp/datasets/electricity/electricity.csv`
- `./temp/datasets/traffic/traffic.csv`

Tip: open a script in `scripts/` to see the exact `root_path` and `data_path`.

## Run Training
Pick one of the quick scripts:

```bash
bash scripts/ETTh1_quick.sh
bash scripts/ETTh2_quick.sh
bash scripts/ETTm1_quick.sh
bash scripts/ETTm2_quick.sh
bash scripts/electricity_quick.sh
bash scripts/exchange_quick.sh
bash scripts/traffic_quick.sh
bash scripts/weather_quick.sh
```

## Run with the CLI
Use `run_prism.py` directly if you want full control:

```bash
python run_prism.py \
  --data ETTh1 \
  --root_path ./temp/datasets/ETT-small \
  --data_path ETTh1.csv \
  --target OT \
  --features M \
  --seq_len 336 \
  --pred_len 96 \
  --enc_hidden 336 \
  --dec_hidden 336 \
  --dropout 0.25 \
  --tree_depth 2 \
  --num_components 5 \
  --batch_size 512 \
  --train_epochs 120 \
  --lr 1e-4 \
  --patience 15 \
  --loss mae \
  --lradj 1 \
  --use_amp \
  --decomp_kind haar \
  --use_multi_gpu \
  --devices 0,1,2,3 \
  --seed 14 \
  --overlap 8
```

## Key Hyperparameters
- `--tree_depth`: depth of the multiresolution tree (typ. 2-3)
- `--num_components`: number of band fragments K (typ. 5-8)
- `--overlap`: overlap between fragments (typ. 8-24)

## Outputs
Logs are saved here:
```
./temp/logs/exp_prism/{dataset}_cont{context_length}_{features}/
```
Each log filename includes hyperparameters for easy tracking.

## Benchmarks
PRISM is evaluated on:
- ETT (ETTh1, ETTh2, ETTm1, ETTm2)
- Electricity
- Exchange Rate
- Traffic
- Weather

## Citation
If you use PRISM, please cite:

```bibtex
@article{prism2025,
  title={PRISM: A Hierarchical Multiscale Approach for Time Series Forecasting},
  author={Chen, Zihao and Andre, Alexandre and Ma, Wenrui and Knight, Ian and Shuvaev, Sergey and Dyer, Eva},
  journal={...},
  year={2025}
}
```

## Acknowledgements
- [Time Series Library (TSLib)](https://github.com/thuml/Time-Series-Library)
- [GluonTS](https://github.com/awslabs/gluonts)
- [PyWavelets](https://github.com/PyWavelets/pywt)
- [D-PAD](https://github.com/XYBbo5/D-PAD) training pipeline inspiration
