#!/usr/bin/env bash

## run the test
python test.py \
--dataroot datasets/tooth_seg \
--name tooth_seg_20201224_512_3500 \
--arch meshunet \
--dataset_mode segmentation \
--ncf 64 128 256 512 \
--ninput_edges 7500 \
--pool_res 7000 5000 3500 \
--resblocks 3 \
--batch_size 1 \
--export_folder meshes \
--which_epoch 100 \
