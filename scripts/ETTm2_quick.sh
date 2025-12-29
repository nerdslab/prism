#!/bin/bash

#SBATCH --job-name=ETTm2_quick
#SBATCH --nodes=1
#SBATCH --gpus=2
#SBATCH --ntasks-per-node=1
#SBATCH --mem-per-gpu=128G
#SBATCH --cpus-per-gpu=16
#SBATCH --time=01:00:00
#SBATCH --output=slurm_logs/ETTm2-nips-%j.out
#SBATCH --error=slurm_logs/ETTm2-nips-%j.err

# Resolve repo root relative to this script and activate env without hardcoded user path
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
source "$REPO_ROOT/../tsdiff310/bin/activate"

# Fixed parameters
context_length=336
device_id=0  # CUDA device IDs
dataset="ETTm2"
root_path="./temp/datasets/ETT-small"
data_path="ETTm2.csv"
target="OT"
features="M"
num_channels=7  

# Hyperparameters
batch_size=512
hidden_size=336
dropout=0.25
num_components=5
max_epochs=100
learning_rate=1e-4
overlap=8
loss="mae"
# Set random port for multiprocessing
export MASTER_ADDR=localhost
export MASTER_PORT=$((12000 + RANDOM % 1000))
# PRISM-specific parameters
tree_depth=2
decomp_kind=haar

# pred_len = 96
pred_length=96
for seed in 17; do
for dropout in 0.25; do
  OUTDIR=./temp/logs/exp_prism/${dataset}_cont${context_length}_${features}
  mkdir -p "$OUTDIR"
  echo "Logs directory: $OUTDIR"
  LOGFILE="pred${pred_length}_cont${context_length}_s${seed}_${features}_w${num_components}_L${tree_depth}_ov${overlap}_d${dropout}_loss${loss}_p${patience}_${decomp_kind}.log"
  CUDA_VISIBLE_DEVICES=${device_id} python run_prism.py \
    --data ${dataset} \
    --root_path ${root_path} \
    --data_path ${data_path} \
    --target ${target} \
    --features ${features} \
    --num_channels ${num_channels} \
    --seq_len ${context_length} \
    --pred_len ${pred_length} \
    --enc_hidden ${hidden_size} \
    --dec_hidden ${hidden_size} \
    --dropout ${dropout} \
    --num_components ${num_components} \
    --train_epochs ${max_epochs} \
    --batch_size ${batch_size} \
    --patience 15 \
    --lr ${learning_rate} \
    --loss mae \
    --lradj 1 \
    --use_amp \
    --decomp_kind ${decomp_kind} \
    --drop_last 0 \
    --use_multi_gpu \
    --devices ${device_id} \
    --seed ${seed} \
    --tree_depth ${tree_depth} \
    --overlap ${overlap} \
    > "$OUTDIR/${LOGFILE}" 2>&1
  echo "Log saved to: $OUTDIR/${LOGFILE}"
done
done

# pred_len = 192
pred_length=192
for seed in 17; do
for dropout in 0.25; do
  OUTDIR=./temp/logs/exp_prism/${dataset}_cont${context_length}_${features}
  mkdir -p "$OUTDIR"
  echo "Logs directory: $OUTDIR"
  LOGFILE="pred${pred_length}_cont${context_length}_s${seed}_${features}_w${num_components}_L${tree_depth}_ov${overlap}_d${dropout}_loss${loss}_p${patience}_${decomp_kind}.log"
  CUDA_VISIBLE_DEVICES=${device_id} python run_prism.py \
    --data ${dataset} \
    --root_path ${root_path} \
    --data_path ${data_path} \
    --target ${target} \
    --features ${features} \
    --num_channels ${num_channels} \
    --seq_len ${context_length} \
    --pred_len ${pred_length} \
    --enc_hidden ${hidden_size} \
    --dec_hidden ${hidden_size} \
    --dropout ${dropout} \
    --num_components ${num_components} \
    --train_epochs ${max_epochs} \
    --batch_size ${batch_size} \
    --patience 15 \
    --lr ${learning_rate} \
    --drop_last 0 \
    --decomp_kind ${decomp_kind} \
    --loss mae \
    --lradj 1 \
    --use_amp \
    --use_multi_gpu \
    --devices ${device_id} \
    --seed ${seed} \
    --tree_depth ${tree_depth} \
    --overlap ${overlap} \
    > "$OUTDIR/${LOGFILE}" 2>&1
  echo "Log saved to: $OUTDIR/${LOGFILE}"
done
done

# pred_len = 336
pred_length=336
for seed in 17; do
for dropout in 0.25; do
  OUTDIR=./temp/logs/exp_prism/${dataset}_cont${context_length}_${features}
  mkdir -p "$OUTDIR"
  echo "Logs directory: $OUTDIR"
  LOGFILE="pred${pred_length}_cont${context_length}_s${seed}_${features}_w${num_components}_L${tree_depth}_ov${overlap}_d${dropout}_loss${loss}_p${patience}_${decomp_kind}.log"
  CUDA_VISIBLE_DEVICES=${device_id} python run_prism.py \
    --data ${dataset} \
    --root_path ${root_path} \
    --data_path ${data_path} \
    --target ${target} \
    --features ${features} \
    --num_channels ${num_channels} \
    --seq_len ${context_length} \
    --pred_len ${pred_length} \
    --enc_hidden ${hidden_size} \
    --dec_hidden ${hidden_size} \
    --dropout ${dropout} \
    --num_components ${num_components} \
    --train_epochs ${max_epochs} \
    --batch_size ${batch_size} \
    --patience 15 \
    --lr ${learning_rate} \
    --loss mae \
    --decomp_kind ${decomp_kind} \
    --lradj 1 \
    --drop_last 0 \
    --use_amp \
    --use_multi_gpu \
    --devices ${device_id} \
    --seed ${seed} \
    --tree_depth ${tree_depth} \
    --overlap ${overlap} \
    > "$OUTDIR/${LOGFILE}" 2>&1
  echo "Log saved to: $OUTDIR/${LOGFILE}"
done
done

# pred_len = 720
pred_length=720
for seed in 17; do
for dropout in 0.25; do
  OUTDIR=./temp/logs/exp_prism/${dataset}_cont${context_length}_${features}
  mkdir -p "$OUTDIR"
  echo "Logs directory: $OUTDIR"
  LOGFILE="pred${pred_length}_cont${context_length}_s${seed}_${features}_w${num_components}_L${tree_depth}_ov${overlap}_d${dropout}_loss${loss}_p${patience}_${decomp_kind}.log"
  CUDA_VISIBLE_DEVICES=${device_id} python run_prism.py \
    --data ${dataset} \
    --root_path ${root_path} \
    --data_path ${data_path} \
    --target ${target} \
    --features ${features} \
    --num_channels ${num_channels} \
    --seq_len ${context_length} \
    --pred_len ${pred_length} \
    --enc_hidden ${hidden_size} \
    --dec_hidden ${hidden_size} \
    --dropout ${dropout} \
    --num_components ${num_components} \
    --train_epochs ${max_epochs} \
    --batch_size ${batch_size} \
    --patience 15 \
    --lr ${learning_rate} \
    --loss mae \
    --lradj 1 \
    --decomp_kind ${decomp_kind} \
    --use_amp \
    --use_multi_gpu \
    --devices ${device_id} \
    --seed ${seed} \
    --tree_depth ${tree_depth} \
    --overlap ${overlap} \
    > "$OUTDIR/${LOGFILE}" 2>&1
  echo "Log saved to: $OUTDIR/${LOGFILE}"
done
done

echo "Results saved in: ./temp/logs/exp_prism/"
