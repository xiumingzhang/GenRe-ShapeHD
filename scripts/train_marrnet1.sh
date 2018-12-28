#!/usr/bin/env bash

outdir=./output/marrnet1

if [ $# -lt 2 ]; then
    echo "Usage: $0 gpu class[ ...]"
    exit 1
fi
gpu="$1"
class="$2"
shift # shift the remaining arguments
shift

set -e

source activate shaperecon

python train.py \
    --net marrnet1 \
    --pred_depth_minmax \
    --dataset shapenet \
    --classes "$class" \
    --batch_size 4 \
    --epoch_batches 2500 \
    --eval_batches 5 \
    --log_time \
    --optim adam \
    --lr 1e-3 \
    --epoch 1000 \
    --vis_batches_vali 10 \
    --gpu "$gpu" \
    --save_net 10 \
    --workers 4 \
    --logdir "$outdir" \
    --suffix '{classes}' \
    --tensorboard \
    $*

source deactivate
