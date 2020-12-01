#!/usr/bin/env bash

## run the training
python train.py \
--dataroot datasets/tooth_seg \
--name tooth_seg \
--arch meshunet \
--dataset_mode segmentation \
--ncf 32 64 128 256 \
--ninput_edges 7500 \
--pool_res 8000 5000 3000 \
--resblocks 3 \
--batch_size 2 \
--lr 0.001 \
--num_aug 20 \
--slide_verts 0.2 \
--save_epoch_freq 10 \
--niter 100 \
--niter_decay 200 \
