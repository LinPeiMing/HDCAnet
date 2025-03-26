import lpips
import torch.nn as nn
from basicsr.utils.registry import METRIC_REGISTRY

@METRIC_REGISTRY.register()
class calculate_lpips:
    def __init__(self):
        self.lpips = lpips.LPIPS(net='vgg')

    def _lpips(self, img1, img2):
        return self.lpips(img1, img2, normalize=True)

    def cal_lpips(self, img1, img2):
        l1, r1 = img1[:, :3].cuda(), img1[:, 3:].cuda()
        l2, r2 = img2[:, :3].cuda(), img2[:, 3:].cuda()
        return (self._lpips(l1, l2) + self._lpips(r1, r2)) / 2