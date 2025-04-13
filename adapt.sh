#!/bin/bash

python3 -W ignore adapt.py \
    --train_dataset <YOUR PATH>/datasets/Pixel/train/ \
    --test_dataset <YOUR PATH>/datasets/Pixel/test/ \
    --lmbda 0.0018 \
    --model_name ELIC \
    --pretrained_weight <YOUR PATH>/ELIC/pretrained/ELIC_00018.pth \
    --save_dir <YOUR PATH>