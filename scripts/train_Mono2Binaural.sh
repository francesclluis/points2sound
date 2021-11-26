#!/bin/bash

OPTS=""
OPTS+="--id default_Mono2Binaural_rgbd "
OPTS+="--list_train data/train.csv "
OPTS+="--list_val data/val.csv "

# Models
OPTS+="--arch_sound unet "
OPTS+="--arch_vision sparseresnet18 "
OPTS+="--visual_feature_size 512 "

# Loss
OPTS+="--loss l2 "

# Max Sources in Mixture
OPTS+="--num_mix 3 "

# pointcloud-related
OPTS+="--rgbs_feature 1 "

# audio-related
OPTS+="--audLen 27783 "
OPTS+="--audRate 44100 "

# learning params
OPTS+="--num_gpus 1 "
OPTS+="--workers 12 "
OPTS+="--batch_size_per_gpu 40 "
OPTS+="--lr_vision 1e-4 "
OPTS+="--lr_sound 1e-3 "
OPTS+="--num_epoch 166 "

# display, viz
OPTS+="--disp_iter 20 "
OPTS+="--num_val 256 "
OPTS+="--eval_epoch 1 "

python -u main.py $OPTS
