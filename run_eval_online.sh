#!/bin/bash

# Online evaluation script for bandit environments
# Usage: ./run_eval_online.sh [environment_type] [epoch]

ENV_TYPE=${1:-bandit}
EPOCH=${2:--1}

echo "Running online evaluation for $ENV_TYPE (epoch $EPOCH)"

python eval_online.py \
    --env $ENV_TYPE \
    --envs 1000 \
    --hists 1 \
    --samples 10000 \
    --H 100 \
    --dim 5 \
    --lin_d 2 \
    --var 0.3 \
    --cov 0.0 \
    --embd 32 \
    --head 4 \
    --layer 4 \
    --lr 0.0001 \
    --dropout 0 \
    --shuffle \
    --epoch $EPOCH \
    --hor 100 \
    --n_eval 100 \
    --seed 1 \
    --online_suffix "_online"

echo "Online evaluation complete!"
