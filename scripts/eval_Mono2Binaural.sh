#!/bin/bash

OPTS=""
OPTS+="--mode eval "
OPTS+="--id Mono2Binaural "

# Models
OPTS+="--arch_sound unet "
OPTS+="--arch_vision sparseresnet18 "
OPTS+="--visual_feature_size 512 "

# Loss
OPTS+="--loss l2 "

# Sources in Mixture
OPTS+="--num_mix 3 "

# frames-related
OPTS+="--rgbs_feature 1 "

# audio-related
OPTS+="--audLen 27783 "
OPTS+="--audRate 44100 "

# learning params
OPTS+="--batch_size_per_gpu 1 "

python -u main.py $OPTS
