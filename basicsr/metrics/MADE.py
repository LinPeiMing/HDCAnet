import sys
sys.path.append('basicsr/losses/core')
import torch
import lpips
import argparse
import torch.nn as nn
import numpy as np
from raft_stereo import RAFTStereo
from basicsr.utils.registry import METRIC_REGISTRY

def parse_raft(fast):
    if fast:
        parser = argparse.ArgumentParser()
        parser.add_argument('--restore_ckpt', help="restore checkpoint", default='basicsr/losses/core/raftstereo-realtime.pth')
        parser.add_argument('--mixed_precision', default=True, help='use mixed precision')
        parser.add_argument('--valid_iters', type=int, default=7, help='number of flow-field updates during forward pass')
        # Architecture choices
        parser.add_argument('--hidden_dims', nargs='+', type=int, default=[128]*3, help="hidden state and context dimensions")
        parser.add_argument('--corr_implementation', choices=["reg", "alt", "reg_cuda", "alt_cuda"], default="reg", help="correlation volume implementation")
        parser.add_argument('--shared_backbone', default=True, help="use a single backbone for the context and feature encoders")
        parser.add_argument('--corr_levels', type=int, default=4, help="number of levels in the correlation pyramid")
        parser.add_argument('--corr_radius', type=int, default=4, help="width of the correlation pyramid")
        parser.add_argument('--n_downsample', type=int, default=3, help="resolution of the disparity field (1/2^K)")
        parser.add_argument('--slow_fast_gru', default=True, help="iterate the low-res GRUs more frequently")
        parser.add_argument('--n_gru_layers', type=int, default=2, help="number of hidden GRU levels")
        return parser.parse_args(args=[])
    parser = argparse.ArgumentParser()
    parser.add_argument('--restore_ckpt', help="restore checkpoint", default='basicsr/losses/core/raftstereo-middlebury.pth')
    parser.add_argument('--mixed_precision', default=False, help='use mixed precision')
    parser.add_argument('--valid_iters', type=int, default=32, help='number of flow-field updates during forward pass')
    # Architecture choices
    parser.add_argument('--hidden_dims', nargs='+', type=int, default=[128]*3, help="hidden state and context dimensions")
    parser.add_argument('--corr_implementation', choices=["reg", "alt", "reg_cuda", "alt_cuda"], default="alt", help="correlation volume implementation")
    parser.add_argument('--shared_backbone', default=False, help="use a single backbone for the context and feature encoders")
    parser.add_argument('--corr_levels', type=int, default=4, help="number of levels in the correlation pyramid")
    parser.add_argument('--corr_radius', type=int, default=4, help="width of the correlation pyramid")
    parser.add_argument('--n_downsample', type=int, default=2, help="resolution of the disparity field (1/2^K)")
    parser.add_argument('--slow_fast_gru', default=False, help="iterate the low-res GRUs more frequently")
    parser.add_argument('--n_gru_layers', type=int, default=3, help="number of hidden GRU levels")

    return parser.parse_args(args=[])


class calculate_MADE(nn.Module):
    def __init__(self, fast=False):
        super(calculate_MADE, self).__init__()
        self.args = parse_raft(fast)
        self.criterion = nn.L1Loss()
        self.model = torch.nn.DataParallel(RAFTStereo(self.args))
        self.model.load_state_dict(torch.load(self.args.restore_ckpt))
        self.model = self.model.module
        self.model.eval()
        for param in self.model.parameters():
            param.requires_grad = False
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.model.to(device)

    def forward(self, x, gt):
        x, gt = x.cuda(), gt.cuda()
        _, flow_up1 = self.model(x[:, :3], x[:, 3:], iters=self.args.valid_iters, test_mode=True)
        _, flow_up2 = self.model(gt[:, :3], gt[:, 3:], iters=self.args.valid_iters, test_mode=True)
        made = self.criterion(flow_up1, flow_up2)
        return made
