set -x

ln -s /modelblob/users/shujliu/ /home/shujliu

ln -s /home/shujliu/models/OpenSora/pretrained_models pretrained_models

cd /home/shujliu/code/Open-Sora

torchrun --nnodes=1 --nproc_per_node=8 scripts/train.py configs/opensora/train/64x512x512.py --data-path /home/shujliu/data/OpenSoraDataSet/MSRVTT-collated/train/annotations.csv
