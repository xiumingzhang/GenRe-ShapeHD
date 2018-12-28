#!/usr/bin/env bash

# Finetune ShapeHD 3D estimator with GAN losses

outdir=./output/shapehd
class=drc
marrnet2=/path/to/marrnet2.pt
gan=/path/to/gan.pt

if [ $# -lt 2 ]; then
    echo "Usage: $0 gpu[ ...]"
    exit 1
fi
gpu="$1"
shift # shift the remaining arguments

set -e

source activate shaperecon

python train.py \
    --net shapehd \
    --marrnet2 "$marrnet2" \
    --gan "$gan" \
    --dataset shapenet \
    --classes "$class" \
    --canon_sup \
    --w_gan_loss 1e-3 \
    --batch_size 4 \
    --epoch_batches 1000 \
    --eval_batches 10 \
    --optim adam \
    --lr 1e-3 \
    --epoch 1000 \
    --vis_batches_vali 10 \
    --gpu "$gpu" \
    --save_net 1 \
    --workers 4 \
    --logdir "$outdir" \
    --suffix '{classes}_w_ganloss{w_gan_loss}' \
    --tensorboard \
    $*

source deactivate
