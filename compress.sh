#!/bin/bash

python3 -W ignore compress.py \
    --test_dataset <YOUR PATH>/datasets/Pixel/test/ \
    --lmbda 0.0018 \
    --model_name ELIC \
    --pretrained_weight <YOUR PATH>/ELIC/pretrained/ELIC_00018.pth \
    --adapters_weight <YOUR PATH>/ELIC/adapters/ELIC_00018_Pixel.pth \
    --save_dir <YOUR PATH>