#!/bin/bash

ROOT_PATH="./"
DATA_PATH="./datasets/ImageNet-100"
GAN_G_CKPT_PATH="./eval_and_gan_ckpts/BigGAN_deep_96K/G_ema.pth"
GAN_D_CKPT_PATH="./eval_and_gan_ckpts/BigGAN_deep_96K/D.pth"
EVAL_CKPT_PATH="./eval_and_gan_ckpts/ckpt_PreCNNForEval_InceptionV3_epoch_50_SEED_2021_Transformation_True_finetuned.pth"

SEED=2021
GAN_NET="BigGANdeep"
SAMP_NROUNDS=1
SAMP_BS=125
SAMP_BURNIN=5000
SAMP_NFAKE_PER_CLASS=1000

CUDA_VISIBLE_DEVICES=0,1,2 python main.py \
    --root_path $ROOT_PATH --data_path $DATA_PATH --seed $SEED \
    --gan_net $GAN_NET --gan_gene_ckpt_path $GAN_G_CKPT_PATH --gan_disc_ckpt_path $GAN_D_CKPT_PATH \
    --ddls_n_steps 1000 --ddls_alpha 1 --ddls_step_lr 2e-4 --ddls_eps_std 2e-4 \
    --samp_round $SAMP_NROUNDS --samp_batch_size $SAMP_BS --samp_burnin_size $SAMP_BURNIN \
    --samp_nfake_per_class $SAMP_NFAKE_PER_CLASS --samp_dump_fake_data \
    --eval --eval_FID_batch_size 10 \
    --eval_ckpt_path $EVAL_CKPT_PATH --inception_from_scratch \
    2>&1 | tee output_${GAN_NET}_colla.txt
