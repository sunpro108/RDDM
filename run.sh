export NCCL_IB_DISABLE=1
export NCCL_P2P_DISABLE=1
accelerate launch train.py
