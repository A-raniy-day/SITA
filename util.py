from __future__ import print_function
import torch
from torchvision import transforms
import os

def rgb2gary(x):
     x_gary = torch.mean(x, dim=1, keepdim=True)
     return x_gary


def get_normalizer(channel_mean, channel_std):
    MEAN = [-mean/std for mean, std in zip(channel_mean, channel_std)]
    STD = [1/std for std in channel_std]
    normalizer = transforms.Normalize(mean=channel_mean, std=channel_std)
    denormalizer = transforms.Normalize(mean=MEAN, std=STD)
    return normalizer, denormalizer


def save_tenor_image(images, dir, lables=None):
    os.makedirs(dir, exist_ok=True)
    bsz= images.shape[0]
    unloader = transforms.ToPILImage()

    for i in range(bsz):
        image = images[i].cpu().clone()
        image = image.squeeze(0) 
        image = unloader(image)

        if lables:
            image.save(f'{dir}/{lables[i].split(".")[0]}.png')
        else:
            image.save(f'{dir}/{i}.png')
