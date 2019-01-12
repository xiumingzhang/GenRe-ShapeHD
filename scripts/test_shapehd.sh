#!/usr/bin/env bash

# Test ShapeHD

out_dir="./output/test"
net1=./downloads/models/marrnet1_with_minmax.pt
net2=./downloads/models/shapehd.pt
rgb_pattern='./downloads/data/test/shapehd/*_rgb.*'
mask_pattern='./downloads/data/test/shapehd/*_mask.*'

if [ $# -lt 1 ]; then
    echo "Usage: $0 gpu[ ...]"
    exit 1
fi
gpu="$1"
shift # shift the remaining arguments

set -e


source activate shaperecon

python 'test.py' \
    --net shapehd \
    --net_file "$net2" \
    --marrnet1_file "$net1" \
    --input_rgb "$rgb_pattern" \
    --input_mask "$mask_pattern" \
    --output_dir "$out_dir" \
    --suffix '{net}' \
    --overwrite \
    --workers 1 \
    --batch_size 1 \
    --vis_workers 4 \
    --gpu "$gpu" \
    $*

source deactivate
