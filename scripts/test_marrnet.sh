#!/usr/bin/env bash

# Test MarrNet

out_dir="./output/test"
marrnet=/path/to/marrnet.pt
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
    --net marrnet \
    --net_file "$marrnet" \
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
