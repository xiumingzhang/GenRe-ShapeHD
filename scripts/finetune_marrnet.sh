#!/usr/bin/env bash

# Finetune MarrNet-2 with MarrNet-1 predictions

outdir=./output/marrnet
class=drc
marrnet1=/path/to/marrnet1.pt
marrnet2=/path/to/marrnet2.pt

if [ $# -lt 1 ]; then
    echo "Usage: $0 gpu[ ...]"
    exit 1
fi
gpu="$1"
shift # shift the remaining arguments

set -e

source activate shaperecon

python train.py \
    --net marrnet \
    --marrnet1 "$marrnet1" \
    --marrnet2 "$marrnet2" \
    --dataset shapenet \
    --classes "$class" \
    --batch_size 4 \
    --epoch_batches 2500 \
    --eval_batches 5 \
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
