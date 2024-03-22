import os
from copy import deepcopy
import torch
from opensora.datasets import get_transforms_video
from opensora.registry import MODELS, SCHEDULERS, build_module
from opensora.utils.config_utils import  parse_configs
from opensora.utils.misc import  to_torch_dtype
from colossalai.utils import get_current_device
import torchvision
import csv
from opensora.datasets import save_sample
from torchvision.io import write_video

def get_filelist(file_path):
    Filelist = []
    for home, dirs, files in os.walk(file_path):
        for filename in files:
            Filelist.append(os.path.join(home, filename))
    return Filelist

if __name__ == "__main__":
    cfg = parse_configs(training=True)
    print(cfg)

    device = get_current_device()
    dtype = to_torch_dtype(cfg.dtype)

    transform=get_transforms_video(cfg.image_size[0])
    input_size = (cfg.num_frames, *cfg.image_size)
    vae = build_module(cfg.vae, MODELS)
    latent_size = vae.get_latent_size(input_size)

    vae = vae.to(device, dtype)
    torch.set_default_dtype(dtype)

    path = "/home/shujliu/data/OpenSoraDataSet/MSRVTT-collated/train/"
    files = get_filelist(path + "videos")
    i = 0
    for mp4file in files:
        file_name = os.path.basename(mp4file)
        output_name = file_name.split(".")[0]
        vframes, aframes, info = torchvision.io.read_video(filename=mp4file, pts_unit="sec", output_format="TCHW")
        total_frames = len(vframes)
        frames = transform(vframes)
        # TCHW -> CTHW
        frames = frames.permute(1, 0, 2, 3)
        frames = torch.unsqueeze(frames, 0)
        with torch.no_grad():
            x = frames.to(device, dtype)
            x = vae.encode(x)
            #y = vae.decode(x.to(dtype))[0]
            #save_path = "/home/shujliu/temp/reconstruct.mp4"
            #save_sample(y, fps = 24, save_path=save_path)
            x = x[0].permute(1, 0, 2, 3)
            torch.save(x, path + "features/" + output_name + ".pt")
            
        i = i+1
        print("processing " + str(i) + " line")
