# single-GPU
#CUDA_VISIBLE_DEVICES=0 python -m torch.distributed.launch --nproc_per_node=1 --master_port=4328 --use-env \
#basicsr/train.py -opt options/train/HDCAnet_train_stage1.yml --launcher pytorch

CUDA_VISIBLE_DEVICES=0 python -m torch.distributed.launch --nproc_per_node=1 --master_port=4328 --use-env \
basicsr/train.py -opt options/train/HDCAnet_train_stage2.yml --launcher pytorch

# multi-GPU
#CUDA_VISIBLE_DEVICES=0,1,2,3 python -m torch.distributed.launch --nproc_per_node=4 --master_port=4328 \
#basicsr/train.py -opt options/train/HDCAnet_train_stage2.yml --launcher pytorch