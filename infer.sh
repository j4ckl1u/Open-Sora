torchrun --standalone --nproc_per_node 1 scripts/inference.py configs/opensora/inference/16x256x256.py --ckpt-path /home/shujliu/models/OpenSora/OpenSora-v1-16x256x256.pth --prompt-path ./assets/texts/t2v_samples.txt

