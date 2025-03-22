#!/usr/bin/env bash

CONFIG=$1
GPUS=$2
NNODES=${NNODES:-1}
NODE_RANK=${NODE_RANK:-0}
PORT=${PORT:-29507}
MASTER_ADDR=${MASTER_ADDR:-"127.0.0.1"}

############################################
# 1) 这里显式指定 你的 conda env Python 路径
############################################
PY_EXE="/home/UNT/yz0370/anaconda3/envs/GiT/bin/python"

echo "Using python: $PY_EXE"
"$PY_EXE" --version

############################################
# 2) 继续保持 mmengine 等需要的 PYTHONPATH
############################################
PYTHONPATH="$(dirname $0)/..":$PYTHONPATH

############################################
# 3) 用 torch.distributed.launch 做分布式训练
############################################
"$PY_EXE" -m torch.distributed.launch \
    --nnodes=$NNODES \
    --node_rank=$NODE_RANK \
    --master_addr=$MASTER_ADDR \
    --nproc_per_node=$GPUS \
    --master_port=$PORT \
    $(dirname "$0")/train.py \
    $CONFIG \
    --launcher pytorch ${@:3}
