#!/usr/bin/env bash

## run the valing
python val.py \
--dataroot datasets/tooth_seg \
--name tooth_seg \
--arch meshunet \
--dataset_mode segmentation \
--ncf 32 64 128 256 512 \
--ninput_edges 7500 \
--pool_res 7000 6000 4500 3000 \
--resblocks 3 \
--batch_size 1 \
--export_folder meshes \
