#!/bin/bash

OPTS=""
OPTS+="--mode inference "
OPTS+="--id Mono2Binaural "

# Models
OPTS+="--arch_sound unet "
OPTS+="--arch_vision sparseresnet18 "
OPTS+="--visual_feature_size 512 "

# Number of Sources Test
OPTS+="--num_mix 1 "

# frames-related
OPTS+="--rgbs_feature 1 "

# audio-related
OPTS+="--audLen 27783 "
OPTS+="--audRate 44100 "

# learning params
OPTS+="--batch_size_per_gpu 1 "

python -u inference_test_Mono2Binaural.py $OPTS
