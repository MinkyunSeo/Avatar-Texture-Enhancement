#!/bin/zsh

NAME='baseline'

NUM_THREADS=8
MLP_DIM='257 1024 512 256 128 1'
MLP_DIM_COLOR='513 1024 512 256 128 3'

VOL_RES=256

CHECKPOINTS_NETG_PATH='./checkpoints/net_G'
CHECKPOINTS_NETC_PATH='./checkpoints/net_C'

python ./apps/eval.py \
	--name ${NAME} \
	--mlp_dim ${MLP_DIM} \
	--mlp_dim_color ${MLP_DIM_COLOR} \
	--num_threads ${NUM_THREADS} \
	--num_stack 4 \
	--num_hourglass 2 \
	--resolution ${VOL_RES} \
	--hg_down 'ave_pool' \
	--norm 'group' \
	--norm_color 'group' \
	--load_netG_checkpoint_path ${CHECKPOINTS_NETG_PATH} \
	--load_netC_checkpoint_path ${CHECKPOINTS_NETC_PATH} \
	--dataroot ./data/datasets/sampled_2K2K/test
