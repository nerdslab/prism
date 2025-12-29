#!/bin/bash

# Electricity Quick Script for Optimized DPVAE
# Uses run_prism.py with direct argument passing (faster startup)

# Fixed parameters
context_length=336
device_id=0  # CUDA device IDs
dataset="electricity"
root_path="./temp/datasets/electricity"
data_path="electricity.csv"
target="OT"
features="M"

# Hyperparameters
batch_size=128
hidden_size=256
dropout=0.4
num_components=5
max_epochs=300
learning_rate=1e-4


loss="mae"
patience=15

# PRISM-specific parameters
tree_depth=2
overlap=8

# Set random port for multiprocessing
export MASTER_ADDR=localhost
export MASTER_PORT=$((12000 + RANDOM % 1000))


# pred_len = 96
pred_length=96
for seed in 15; do
for dropout in ${dropout}; do
    OUTDIR=./temp/logs/exp_prism/${dataset}_cont${context_length}_${features}
    mkdir -p "$OUTDIR"
    echo "Logs directory: $OUTDIR"
    LOGFILE="pred${pred_length}_cont${context_length}_s${seed}_${features}_w${num_components}_L${tree_depth}_ov${overlap}_d${dropout}_loss${loss}_p${patience}.log"
    CUDA_VISIBLE_DEVICES=${device_id} python run_prism.py \
      --data ${dataset} \
      --root_path ${root_path} \
      --data_path ${data_path} \
      --target ${target} \
      --features ${features} \
      --seq_len ${context_length} \
      --pred_len ${pred_length} \
      --enc_hidden ${hidden_size} \
      --dec_hidden ${hidden_size} \
      --dropout ${dropout} \
      --tree_depth ${tree_depth} \
      --num_components ${num_components} \
      --train_epochs ${max_epochs} \
      --batch_size ${batch_size} \
      --patience 15 \
      --lr ${learning_rate} \
      --loss mae \
      --lradj 1 \
      --use_amp \
      --use_multi_gpu \
      --devices ${device_id} \
      --seed ${seed} \
      --overlap ${overlap} \
      > "$OUTDIR/${LOGFILE}" 2>&1
    echo "Log saved to: $OUTDIR/${LOGFILE}"
done
done

# pred_len = 192
pred_length=192
for seed in 15; do
for dropout in ${dropout}; do
    OUTDIR=./temp/logs/exp_prism/${dataset}_cont${context_length}_${features}
    mkdir -p "$OUTDIR"
    echo "Logs directory: $OUTDIR"
    LOGFILE="pred${pred_length}_cont${context_length}_s${seed}_${features}_w${num_components}_L${tree_depth}_ov${overlap}_d${dropout}_loss${loss}_p${patience}.log"
    CUDA_VISIBLE_DEVICES=${device_id} python run_prism.py \
      --data ${dataset} \
      --root_path ${root_path} \
      --data_path ${data_path} \
      --target ${target} \
      --features ${features} \
      --seq_len ${context_length} \
      --pred_len ${pred_length} \
      --enc_hidden ${hidden_size} \
      --dec_hidden ${hidden_size} \
      --dropout ${dropout} \
      --tree_depth ${tree_depth} \
      --num_components ${num_components} \
      --train_epochs ${max_epochs} \
      --batch_size ${batch_size} \
      --patience 15 \
      --lr ${learning_rate} \
      --loss mae \
      --lradj 1 \
      --use_amp \
      --use_multi_gpu \
      --devices ${device_id} \
      --seed ${seed} \
      --overlap ${overlap} \
      > "$OUTDIR/${LOGFILE}" 2>&1
    echo "Log saved to: $OUTDIR/${LOGFILE}"
done
done

# pred_len = 336
pred_length=336
for seed in 15; do
for dropout in ${dropout}; do
    OUTDIR=./temp/logs/exp_prism/${dataset}_cont${context_length}_${features}
    mkdir -p "$OUTDIR"
    echo "Logs directory: $OUTDIR"
    LOGFILE="pred${pred_length}_cont${context_length}_s${seed}_${features}_w${num_components}_L${tree_depth}_ov${overlap}_d${dropout}_loss${loss}_p${patience}.log"
    CUDA_VISIBLE_DEVICES=${device_id} python run_prism.py \
      --data ${dataset} \
      --root_path ${root_path} \
      --data_path ${data_path} \
      --target ${target} \
      --features ${features} \
      --seq_len ${context_length} \
      --pred_len ${pred_length} \
      --enc_hidden ${hidden_size} \
      --dec_hidden ${hidden_size} \
      --dropout ${dropout} \
      --tree_depth ${tree_depth} \
      --num_components ${num_components} \
      --train_epochs ${max_epochs} \
      --batch_size ${batch_size} \
      --patience 15 \
      --lr ${learning_rate} \
      --loss mae \
      --lradj 1 \
      --use_amp \
      --use_multi_gpu \
      --devices ${device_id} \
      --seed ${seed} \
      --overlap ${overlap} \
      > "$OUTDIR/${LOGFILE}" 2>&1
    echo "Log saved to: $OUTDIR/${LOGFILE}"
done
done

# pred_len = 720
pred_length=720
for seed in 15; do
for dropout in ${dropout}; do
    OUTDIR=./temp/logs/exp_prism/${dataset}_cont${context_length}_${features}
    mkdir -p "$OUTDIR"
    echo "Logs directory: $OUTDIR"
    LOGFILE="pred${pred_length}_cont${context_length}_s${seed}_${features}_w${num_components}_L${tree_depth}_ov${overlap}_d${dropout}_loss${loss}_p${patience}.log"
    CUDA_VISIBLE_DEVICES=${device_id} python run_prism.py \
      --data ${dataset} \
      --root_path ${root_path} \
      --data_path ${data_path} \
      --target ${target} \
      --features ${features} \
      --seq_len ${context_length} \
      --pred_len ${pred_length} \
      --enc_hidden ${hidden_size} \
      --dec_hidden ${hidden_size} \
      --dropout ${dropout} \
      --tree_depth ${tree_depth} \
      --num_components ${num_components} \
      --train_epochs ${max_epochs} \
      --batch_size ${batch_size} \
      --patience 15 \
      --lr ${learning_rate} \
      --loss mae \
      --lradj 1 \
      --use_amp \
      --use_multi_gpu \
      --devices ${device_id} \
      --seed ${seed} \
      --overlap ${overlap} \
      > "$OUTDIR/${LOGFILE}" 2>&1
    echo "Log saved to: $OUTDIR/${LOGFILE}"
done
done

echo "Results saved in: ./temp/logs/exp_prism/"
