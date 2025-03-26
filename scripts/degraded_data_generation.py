import numpy as np
import random
import torch
from basicsr.data.degradations import random_add_gaussian_noise_pt, random_add_poisson_noise_pt
from basicsr.data.transforms import paired_random_crop

from basicsr.utils import DiffJPEG, USMSharp
from basicsr.utils.img_process_util import filter2D
from basicsr.utils.registry import MODEL_REGISTRY
from collections import OrderedDict
from torch.nn import functional as F

from basicsr.data.degradations import circular_lowpass_kernel, random_mixed_kernels
from basicsr.data.transforms import augment
from basicsr.utils import FileClient, get_root_logger, imfrombytes, img2tensor
from basicsr.utils.registry import DATASET_REGISTRY
import math
import torch
import torch.nn as nn
import torch.nn.functional as F

import numpy as np
from matplotlib import pyplot as plt
import matplotlib.image as img
import os

from torchvision import transforms



class generate_kernel(nn.Module):
    def __init__(self):
        super(generate_kernel, self).__init__()

        # blur settings for the first degradation
        self.blur_kernel_size = 21
        self.kernel_list = ['iso', 'aniso', 'generalized_iso', 'generalized_aniso', 'plateau_iso', 'plateau_aniso', 'real']
        self.kernel_prob = [0.36, 0.2, 0.1, 0.02, 0.1, 0.02, 0.2]  # a list for each kernel probability
        self.blur_sigma = [0.2, 3]
        self.betag_range = [0.5, 4]  # betag used in generalized Gaussian blur kernels
        self.betap_range = [1, 2]  # betap used in plateau blur kernels
        self.sinc_prob = 0.1  # the probability for sinc filters

        # blur settings for the second degradation
        self.blur_kernel_size2 = 21
        self.kernel_list2 = ['iso', 'aniso', 'generalized_iso', 'generalized_aniso', 'plateau_iso', 'plateau_aniso', 'real']
        self.kernel_prob2 = [0.36, 0.2, 0.1, 0.02, 0.1, 0.02, 0.2]
        self.blur_sigma2 = [0.2, 1.5]
        self.betag_range2 = [0.5, 4]
        self.betap_range2 = [1, 2]
        self.sinc_prob2 = 0.1


        self.blur_kernel_size3 = 21
        self.kernel_list3 = ['iso', 'aniso', 'generalized_iso', 'generalized_aniso', 'plateau_iso', 'plateau_aniso',
                             'real']
        self.kernel_prob3 = [0.36, 0.2, 0.1, 0.02, 0.1, 0.02, 0.2]
        self.blur_sigma3 = [0.2, 1.5]
        self.betag_range3 = [0.5, 4]
        self.betap_range3 = [1, 2]
        self.sinc_prob3 = 0.1

        # a final sinc filter
        self.final_sinc_prob = 0.8

        self.kernel_range = [2 * v + 1 for v in range(3, 11)]  # kernel size ranges from 7 to 21
        # TODO: kernel range is now hard-coded, should be in the configure file
        self.pulse_tensor = torch.zeros(21, 21).float()  # convolving with pulse tensor brings no blurry effect
        self.pulse_tensor[10, 10] = 1

    def forward(self, x):
# ------------------------ Generate kernels (used in the first degradation) ------------------------ #
        kernel_size = random.choice(self.kernel_range)
        if np.random.uniform() < self.sinc_prob:
            # this sinc filter setting is for kernels ranging from [7, 21]
            if kernel_size < 13:
                omega_c = np.random.uniform(np.pi / 3, np.pi)
            else:
                omega_c = np.random.uniform(np.pi / 5, np.pi)
            kernel = circular_lowpass_kernel(omega_c, kernel_size, pad_to=False)
        else:
            kernel = random_mixed_kernels(
                self.kernel_list,
                self.kernel_prob,
                kernel_size,
                self.blur_sigma,
                self.blur_sigma, [-math.pi, math.pi],
                self.betag_range,
                self.betap_range,
                noise_range=None)
        # pad kernel
        pad_size = (21 - kernel_size) // 2
        if kernel.shape[0] == 21:
            kernel = kernel
        else:
            kernel = np.pad(kernel, ((pad_size, pad_size), (pad_size, pad_size)))
        # kernel = np.pad(kernel, ((pad_size, pad_size), (pad_size, pad_size)))

        # ------------------------ Generate kernels (used in the second degradation) ------------------------ #
        kernel_size = random.choice(self.kernel_range)
        if np.random.uniform() < self.sinc_prob2:
            if kernel_size < 13:
                omega_c = np.random.uniform(np.pi / 3, np.pi)
            else:
                omega_c = np.random.uniform(np.pi / 5, np.pi)
            kernel2 = circular_lowpass_kernel(omega_c, kernel_size, pad_to=False)
        else:
            kernel2 = random_mixed_kernels(
                self.kernel_list2,
                self.kernel_prob2,
                kernel_size,
                self.blur_sigma2,
                self.blur_sigma2, [-math.pi, math.pi],
                self.betag_range2,
                self.betap_range2,
                noise_range=None)

        # pad kernel
        pad_size = (21 - kernel_size) // 2
        if kernel2.shape[0] == 21:
            kernel2 = kernel2
        else:
            kernel2 = np.pad(kernel2, ((pad_size, pad_size), (pad_size, pad_size)))
        # kernel2 = np.pad(kernel2, ((pad_size, pad_size), (pad_size, pad_size)))

# ------------------------ Generate kernels (used in the third degradation) ------------------------ #
        kernel_size = random.choice(self.kernel_range)
        if np.random.uniform() < self.sinc_prob3:
            if kernel_size < 13:
                omega_c = np.random.uniform(np.pi / 3, np.pi)
            else:
                omega_c = np.random.uniform(np.pi / 5, np.pi)
            kernel3 = circular_lowpass_kernel(omega_c, kernel_size, pad_to=False)
        else:
            kernel3 = random_mixed_kernels(
                self.kernel_list3,
                self.kernel_prob3,
                kernel_size,
                self.blur_sigma3,
                self.blur_sigma3, [-math.pi, math.pi],
                self.betag_range3,
                self.betap_range3,
                noise_range=None)

        pad_size = (21 - kernel_size) // 2
        if kernel3.shape[0] == 21:
            kernel3 = kernel3
        else:
            kernel3 = np.pad(kernel3, ((pad_size, pad_size), (pad_size, pad_size)))

        # ------------------------------------- the final sinc kernel ------------------------------------- #
        if np.random.uniform() < self.final_sinc_prob:
            kernel_size = random.choice(self.kernel_range)
            omega_c = np.random.uniform(np.pi / 3, np.pi)
            sinc_kernel = circular_lowpass_kernel(omega_c, kernel_size, pad_to=21)
            sinc_kernel = torch.FloatTensor(sinc_kernel)
        else:
            sinc_kernel = self.pulse_tensor

        # # BGR to RGB, HWC to CHW, numpy to tensor
        img_gt = img2tensor([x], bgr2rgb=False, float32=True)[0].unsqueeze(0)
        # img_gt = x
        kernel = torch.FloatTensor(kernel)
        kernel2 = torch.FloatTensor(kernel2)
        kernel3 = torch.FloatTensor(kernel3)

        #return_d = {'gt': img_gt, 'kernel1': kernel, 'kernel2': kernel2, 'sinc_kernel': sinc_kernel}

        return img_gt, kernel, kernel2, kernel3, sinc_kernel

class degradation_function(nn.Module):
    def __init__(self, upscale_factor):
        super(degradation_function, self).__init__()

        # ----------------- options for synthesizing training data in RealESRNetModel ----------------- #
        #self.gt_usm = True  # USM the ground-truth


        self.scale = upscale_factor
        self.device = torch.device('cuda') ##baocuo cuda/cpu?
        # the first degradation process
        self.resize_prob = [0.2, 0.7, 0.1]  # up, down, keep
        self.resize_range = [0.15, 1.5]
        self.gaussian_noise_prob = 0.5
        self.noise_range = [1, 30]
        self.poisson_scale_range = [0.05, 3]
        self.gray_noise_prob = 0.4
        self.jpeg_range = [30, 95]

        # the second degradation process
        self.second_blur_prob = 0.5
        self.resize_prob2 = [0.3, 0.4, 0.3]  # up, down, keep
        self.resize_range2 = [0.3, 1.2]
        self.noise_prob2 = [0.4, 0.4, 0.2]
        self.gaussian_noise_prob2 = 0.5
        self.noise_range2 = [1, 25]
        self.poisson_scale_range2 = [0.05, 2.5]
        self.gray_noise_prob2 = 0.4
        self.jpeg_range2 = [30, 95]

        self.kernel = generate_kernel()#.to(self.device)
        self.jpeger = DiffJPEG(differentiable=False)#.to(self.device)  # simulate JPEG compression artifacts

    def forward(self, x):


        x, kernel1, kernel2, kernel3, sinc_kernel = self.kernel(x)

        ori_h, ori_w = x.size()[2:4]

        # ----------------------- The first degradation process ----------------------- #
        # blur
        if np.random.uniform() < 0.5:
            print('blur1 yes')
            out = filter2D(x, kernel1)
        else:
            out = x
        # random resize
        updown_type = random.choices(['up', 'down', 'keep'], self.resize_prob)[0]
        if updown_type == 'up':
            scale = np.random.uniform(1, self.resize_range[1])
        elif updown_type == 'down':
            scale = np.random.uniform(self.resize_range[0], 1)
        else:
            scale = 1
        mode = random.choice(['area', 'bilinear', 'bicubic'])
        out = F.interpolate(out, scale_factor=scale, mode=mode)
        # add noise
        if np.random.uniform() < 0.5:
            print('noise1 yes')
            gray_noise_prob = self.gray_noise_prob
            if np.random.uniform() < self.gaussian_noise_prob:
                out = random_add_gaussian_noise_pt(
                    out, sigma_range=self.noise_range, clip=True, rounds=False, gray_prob=gray_noise_prob)
            else:
                out = random_add_poisson_noise_pt(
                    out,
                    scale_range=self.poisson_scale_range,
                    gray_prob=gray_noise_prob,
                    clip=True,
                    rounds=False)

        # JPEG compression
        if np.random.uniform() < 0.5:
            print('jpeg1 yes')
            jpeg_p = out.new_zeros(out.size(0)).uniform_(*self.jpeg_range)
            out = torch.clamp(out, 0, 1)  # clamp to [0, 1], otherwise JPEGer will result in unpleasant artifacts
            out = self.jpeger(out, quality=jpeg_p)

        # ----------------------- The second degradation process ----------------------- #
        # blur
        if np.random.uniform() < self.second_blur_prob:
            print('blur2 yes')
            if np.random.uniform() < 0.5:  # left and right different degradation p=0.5?
                print('different blur2')
                lq_l, lq_r = out.chunk(2, dim=2)
                out_l = filter2D(lq_l, kernel2)
                out_r = filter2D(lq_r, kernel3)
                out = torch.cat([out_l, out_r], dim=2)
            else:
                out = filter2D(out, kernel2)

        # random resize
        updown_type = random.choices(['up', 'down', 'keep'], self.resize_prob2)[0]
        if updown_type == 'up':
            scale = np.random.uniform(1, self.resize_range2[1])
        elif updown_type == 'down':
            scale = np.random.uniform(self.resize_range2[0], 1)
        else:
            scale = 1
        mode = random.choice(['area', 'bilinear', 'bicubic'])
        out = F.interpolate(
            out, size=(int(ori_h / self.scale * scale), int(ori_w / self.scale * scale)), mode=mode)
        # add noise
        if np.random.uniform() < 0.5:
            print('noise2 yes')
            gray_noise_prob = self.gray_noise_prob2
            noise_type = random.choices(['g', 'p', 'real'], self.noise_prob2)[0]
            if np.random.uniform() < 0.5:  ### 左右不同噪声
                print('different noise2')
                lq_l, lq_r = out.chunk(2, dim=2)
                if noise_type == 'g':
                    lq_l = random_add_gaussian_noise_pt(lq_l, sigma_range=self.noise_range2, clip=True,
                                                        rounds=False,
                                                        gray_prob=gray_noise_prob)
                    lq_r = random_add_gaussian_noise_pt(lq_r, sigma_range=self.noise_range2, clip=True,
                                                        rounds=False,
                                                        gray_prob=gray_noise_prob)
                elif noise_type == 'p':
                    lq_l = random_add_poisson_noise_pt(
                        lq_l,
                        scale_range=self.poisson_scale_range2,
                        gray_prob=gray_noise_prob,
                        clip=True,
                        rounds=False)
                    lq_r = random_add_poisson_noise_pt(
                        lq_r,
                        scale_range=self.poisson_scale_range2,
                        gray_prob=gray_noise_prob,
                        clip=True,
                        rounds=False)
                else:
                    print('real noise!')
                    lq_l = add_real_noise(lq_l)
                    lq_r = add_real_noise(lq_r)
                out = torch.cat([lq_l, lq_r], dim=2)
            else:
                if noise_type == 'g':
                    out = random_add_gaussian_noise_pt(
                        out, sigma_range=self.noise_range2, clip=True, rounds=False, gray_prob=gray_noise_prob)
                elif noise_type == 'p':
                    out = random_add_poisson_noise_pt(
                        out,
                        scale_range=self.poisson_scale_range2,
                        gray_prob=gray_noise_prob,
                        clip=True,
                        rounds=False)
                else:
                    out = add_real_noise(out)
                    print('real noise!')
        # JPEG compression + the final sinc filter
        # We also need to resize images to desired sizes. We group [resize back + sinc filter] together
        # as one operation.
        # We consider two orders:
        #   1. [resize back + sinc filter] + JPEG compression
        #   2. JPEG compression + [resize back + sinc filter]
        # Empirically, we find other combinations (sinc + JPEG + Resize) will introduce twisted lines.
        if np.random.uniform() < 0.5:
            # resize back + the final sinc filter
            mode = random.choice(['area', 'bilinear', 'bicubic'])
            out = F.interpolate(out, size=(ori_h // self.scale, ori_w // self.scale), mode=mode)
            out = filter2D(out, sinc_kernel)
            # JPEG compression
            jpeg_p = out.new_zeros(out.size(0)).uniform_(*self.jpeg_range2)
            out = torch.clamp(out, 0, 1)
            out = self.jpeger(out, quality=jpeg_p)
        else:
            # JPEG compression
            jpeg_p = out.new_zeros(out.size(0)).uniform_(*self.jpeg_range2)
            out = torch.clamp(out, 0, 1)
            out = self.jpeger(out, quality=jpeg_p)
            # resize back + the final sinc filter
            mode = random.choice(['area', 'bilinear', 'bicubic'])
            out = F.interpolate(out, size=(ori_h // self.scale, ori_w // self.scale), mode=mode)
            out = filter2D(out, sinc_kernel)
        # clamp and round
        lq = torch.clamp((out * 255.0).round(), 0, 255) / 255.
        return lq


if __name__ == "__main__":
    degrade = degradation_function(upscale_factor=2)

    testset_dir = 'LinPeiMing/Dataset/Real_stereo_test/HR'
    dataset = 'Middlebury'
    file_list = os.listdir(testset_dir + dataset)
    output_path = '/LinPeiMing/Dataset/work2/Real_stereo_test/'

    for idx in range(len(file_list)):
        HR = img.imread(testset_dir + dataset + '/HR' + '/' + file_list[idx])
        scene_name = file_list[idx]
        with torch.no_grad():
            LR = degrade(HR)
        save_path_lr = output_path + dataset + '/' + 'LR'
        if not os.path.exists(save_path_lr):
            os.makedirs(save_path_lr)
        LR_img = transforms.ToPILImage()(torch.clamp(LR, 0, 1).squeeze())
        LR_img.save(save_path_lr + '/' + scene_name)
        print(idx, 'Done')


