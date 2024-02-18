#!/bin/zsh
set -ex

NAME='patch_ssim'

# Network configuration

BATCH_SIZE=16
NUM_THREADS=8
MLP_DIM='257 1024 512 256 128 1'
MLP_DIM_COLOR='513 1024 512 256 128 3'

CHECKPOINTS_NETG_PATH='./checkpoints/net_G'
CHECKPOINTS_NETC_PATH='./checkpoints/net_C'

DATAROOT='./data/datasets/sampled_2K2K/train'

# command
python ./apps/train_color.py \
	--name ${NAME} \
	--batch_size ${BATCH_SIZE} \
	--num_threads ${NUM_THREADS} \
	--mlp_dim ${MLP_DIM} \
	--mlp_dim_color ${MLP_DIM_COLOR} \
	--num_stack 4 \
	--num_hourglass 2 \
	--hg_down 'ave_pool' \
	--norm 'group' \
	--norm_color 'group' \
	--load_netG_checkpoint_path ${CHECKPOINTS_NETG_PATH} \
	--load_netC_checkpoint_path ${CHECKPOINTS_NETC_PATH} \
	--dataroot ${DATAROOT} \
	--num_sample_inout 0 \
	--num_sample_color 3000 \
  --use_3D_SSIM \
  --patch_sample \
  --learning_rate 0.00001
