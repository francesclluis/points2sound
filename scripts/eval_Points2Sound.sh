#!/bin/bash

OPTS=""
OPTS+="--mode eval "
OPTS+="--id Points2Sound "

# Models
OPTS+="--arch_sound demucs "
OPTS+="--arch_vision sparseresnet18 "
OPTS+="--visual_feature_size 16 "

# Loss
OPTS+="--loss l1 "

# Sources in Mixture
OPTS+="--num_mix 3 "

# frames-related
OPTS+="--rgbs_feature 1 "

# audio-related
OPTS+="--audLen 441000 "
OPTS+="--audRate 44100 "

# learning params
OPTS+="--batch_size_per_gpu 1 "

python -u main.py $OPTS
