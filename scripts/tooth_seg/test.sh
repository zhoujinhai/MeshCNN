#!/usr/bin/env bash

## run the test
python test.py \
--dataroot datasets/tooth_seg \
--name tooth_seg_20201210 \
--arch meshunet \
--dataset_mode segmentation \
--ncf 32 64 128 256 512 \
--ninput_edges 7500 \
--pool_res 7000 6000 4500 3000 \
--resblocks 3 \
--batch_size 1 \
--export_folder meshes \
--which_epoch 100 \
