#!/bin/bash
ROOT_PATH="./finetune_inception"
DATA_PATH="./datasets/ImageNet-100"
CKPT_PATH="./finetune_inception/torch_cache/hub/checkpoints/inception_v3_google-1a9a5a14.pth"

python main.py \
    --root_path $ROOT_PATH --data_path $DATA_PATH --inception_path $CKPT_PATH \
    --epochs 50 --resume_epoch 0 --save_freq 10 \
    --batch_size_train 128 --batch_size_test 128 \
    --lr_base 1e-4 --lr_decay_factor 0.1 --lr_decay_epochs 100_150 \
    --weight_dacay 1e-4 --seed 2021 --transform --num_classes 100 \
    2>&1 | tee output.txt