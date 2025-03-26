import torch.utils.data as data
import torch
from torchvision import transforms
from PIL import Image
import glob
import torch.nn as nn

class realnoise(nn.Module):
    def __init__(self, dataset='x2/', size=64):
        super(realnoise, self).__init__()

        base = dataset
        import os
        assert os.path.exists(base)

        # self.mat_files = sorted(glob.glob(base + '*.mat'))
        self.noise_imgs = sorted(glob.glob(base + '*.png'))
        self.pre_process = transforms.Compose([transforms.RandomCrop(size),
                                               transforms.ToTensor()])

    def __getitem__(self, index):
        noise = self.pre_process(Image.open(self.noise_imgs[index]))
        norm_noise = (noise - torch.mean(noise, dim=[1, 2], keepdim=True))
        return norm_noise

    def __len__(self):
        return len(self.noise_imgs)