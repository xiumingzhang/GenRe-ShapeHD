#!/usr/bin/env bash

outdir=./output/wgangp

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
    --net wgangp \
    --canon_voxel \
    --dataset shapenet \
    --classes "$class" \
    --batch_size 4 \
    --epoch_batches 2500 \
    --eval_batches 5 \
    --log_time \
    --optim adam \
    --lr 1e-4 \
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
