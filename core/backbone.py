import math 

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

from core.update import BasicUpdateBlock, DepthMaskHead
from core.extractor import BasicEncoder_Event, BasicEncoder_LiDAR
from core.corr import CorrBlock, AlternateCorrBlock
from core.utils import coords_grid, upflow8, feature_visualizer
from core.utils_point import invert_pose

import mathutils

try:
    autocast = torch.cuda.amp.autocast
except:
    class autocast:
        def __init__(self, enabled):
            pass

        def __enter__(self):
            pass

        def __exit__(self, *args):
            pass


class Backbone_Event(nn.Module):
    def __init__(self, args):
        super(Backbone_Event, self).__init__()
        self.args = args

        self.hidden_dim = hdim = 128
        self.context_dim = cdim = 128
        args.corr_levels = 4
        args.corr_radius = 4
        self.level = args.corr_levels
        self.radius = args.corr_radius

        if 'dropout' not in self.args:
            self.args.dropout = 0

        if 'alternate_corr' not in self.args:
            self.args.alternate_corr = False

        # feature network, context network, and update block
        self.fnet_event = BasicEncoder_Event(output_dim=256, norm_fn='instance', dropout=args.dropout)
        self.fnet_lidar = BasicEncoder_LiDAR(output_dim=256, norm_fn='instance', dropout=args.dropout)
        self.cnet = BasicEncoder_LiDAR(output_dim=hdim + cdim, norm_fn='batch', dropout=args.dropout)
        self.update_block = BasicUpdateBlock(self.args, hidden_dim=hdim)

    def freeze_bn(self):
        for m in self.modules():
            if isinstance(m, nn.BatchNorm2d):
                m.eval()

    def initialize_flow(self, img):
        """ Flow is represented as difference between two coordinate grids flow = coords1 - coords0"""
        N, C, H, W = img.shape
        coords0 = coords_grid(N, H // 8, W // 8).to(img.device)
        coords1 = coords_grid(N, H // 8, W // 8).to(img.device)

        # optical flow computed as difference: flow = coords1 - coords0
        return coords0, coords1

    def upsample_flow(self, flow, mask):
        """ Upsample flow field [H/8, W/8, 2] -> [H, W, 2] using convex combination """
        N, _, H, W = flow.shape
        mask = mask.view(N, 1, 9, 8, 8, H, W)
        mask = torch.softmax(mask, dim=2)

        up_flow = F.unfold(8 * flow, [3, 3], padding=1)
        up_flow = up_flow.view(N, 2, 9, 1, 1, H, W)

        up_flow = torch.sum(mask * up_flow, dim=2)
        up_flow = up_flow.permute(0, 1, 4, 2, 5, 3)
        return up_flow.reshape(N, 2, 8 * H, 8 * W)

    def warp(self, x, flo):
        """
        warp an image/tensor (im2) back to im1, according to the optical flow

        x: [B, C, H, W] (im2)
        flo: [B, 2, H, W] flow

        """
        B, C, H, W = x.size()
        # mesh grid 
        xx = torch.arange(0, W).view(1,-1).repeat(H,1)
        yy = torch.arange(0, H).view(-1,1).repeat(1,W)
        xx = xx.view(1,1,H,W).repeat(B,1,1,1)
        yy = yy.view(1,1,H,W).repeat(B,1,1,1)
        grid = torch.cat((xx,yy),1).float()

        if x.is_cuda:
            grid = grid.cuda()
        vgrid = Variable(grid) + flo

        # scale grid to [-1,1] 
        vgrid[:,0,:,:] = 2.0*vgrid[:,0,:,:].clone() / max(W-1,1)-1.0
        vgrid[:,1,:,:] = 2.0*vgrid[:,1,:,:].clone() / max(H-1,1)-1.0

        vgrid = vgrid.permute(0,2,3,1)        
        output = nn.functional.grid_sample(x, vgrid)
        mask = torch.autograd.Variable(torch.ones(x.size())).cuda()
        mask = nn.functional.grid_sample(mask, vgrid)

        mask[mask<0.9999] = 0
        mask[mask>0] = 1
        
        return output*mask

    def forward(self, image1, image2, iters=12, flow_init=None, test_mode=False, idx=0):
        """ 
            Estimate optical flow between pair of frames 
            image1: lidar_input
            image2: event_frame
        """
        image1 = 2 * image1 - 1.0
        image2 = 2 * image2 - 1.0

        # print(torch.min(image1), torch.max(image1))
        # print(torch.min(image2), torch.max(image2))

        image1 = image1.contiguous()
        image2 = image2.contiguous()

        hdim = self.hidden_dim
        cdim = self.context_dim

        # run the feature network
        with autocast(enabled=self.args.mixed_precision):
            fmap1 = self.fnet_lidar(image1)
            fmap2 = self.fnet_event(image2)

        fmap1 = fmap1.float()
        fmap2 = fmap2.float()

        # feature_visualizer(fmap1[0, ...].cpu().detach().numpy(), f"/home/eason/WorkSpace/EventbasedVisualLocalization/EVLoc_Reconstruction/visualization/feature/{idx}")

        if self.args.alternate_corr:
            corr_fn = AlternateCorrBlock(fmap1, fmap2, radius=self.args.corr_radius)
        else:
            corr_fn = CorrBlock(fmap1, fmap2, radius=self.args.corr_radius)

        # run the context network
        with autocast(enabled=self.args.mixed_precision):
            cnet = self.cnet(image1)
            net, inp = torch.split(cnet, [hdim, cdim], dim=1)
            net = torch.tanh(net)
            inp = torch.relu(inp)

        coords0, coords1 = self.initialize_flow(image1)

        if flow_init is not None:
            coords1 = coords1 + flow_init

        flow_predictions = []
        for itr in range(iters):
            coords1 = coords1.detach()

            corr = corr_fn(coords1)  # index correlation volume Bx(9x9x4)xH/8xW/8
            # feature_visualizer(corr[0, ...].cpu().detach().numpy(), f"/home/eason/WorkSpace/EventbasedVisualLocalization/EVLoc_Reconstruction/visualization/feature/{idx}_corr")

            flow = coords1 - coords0
            with autocast(enabled=self.args.mixed_precision):
                net, up_mask, delta_flow = self.update_block(net, inp, corr, flow)
            
            # F(t+1) = F(t) + \Delta(t)
            coords1 = coords1 + delta_flow

            # upsample predictions
            if up_mask is None:
                flow_up = upflow8(coords1 - coords0)
            else:
                flow_up = self.upsample_flow(coords1 - coords0, up_mask)

            flow_predictions.append(flow_up)

        if test_mode:
            return coords1 - coords0, flow_up
            
        return flow_predictions






class Backbone_Reconstruction(nn.Module):
    def __init__(self, args):
        super(Backbone_Reconstruction, self).__init__()
        self.args = args

        self.hidden_dim = hdim = 128
        self.context_dim = cdim = 128
        args.corr_levels = 4
        args.corr_radius = 4
        self.level = args.corr_levels
        self.radius = args.corr_radius

        if 'dropout' not in self.args:
            self.args.dropout = 0

        if 'alternate_corr' not in self.args:
            self.args.alternate_corr = False

        # feature network, context network, and update block
        self.fnet_lidar = BasicEncoder_LiDAR(output_dim=256, norm_fn='instance', dropout=args.dropout, return_all_layers=True)
        self.fnet_event = BasicEncoder_Event(output_dim=256, norm_fn='instance', dropout=args.dropout)
        self.cnet = BasicEncoder_LiDAR(output_dim=hdim + cdim, norm_fn='batch', dropout=args.dropout)
        self.update_block = BasicUpdateBlock(self.args, hidden_dim=hdim)
        self.depth_mask_head = DepthMaskHead(input_dim=324, output_dim=2)

    def freeze_bn(self):
        for m in self.modules():
            if isinstance(m, nn.BatchNorm2d):
                m.eval()

    def initialize_flow(self, img):
        """ Flow is represented as difference between two coordinate grids flow = coords1 - coords0"""
        N, C, H, W = img.shape
        coords0 = coords_grid(N, H // 8, W // 8).to(img.device)
        coords1 = coords_grid(N, H // 8, W // 8).to(img.device)

        # optical flow computed as difference: flow = coords1 - coords0
        return coords0, coords1

    def upsample_flow(self, flow, mask):
        """ Upsample flow field [H/8, W/8, 2] -> [H, W, 2] using convex combination """
        N, _, H, W = flow.shape
        mask = mask.view(N, 1, 9, 8, 8, H, W)
        mask = torch.softmax(mask, dim=2)

        up_flow = F.unfold(8 * flow, [3, 3], padding=1)
        up_flow = up_flow.view(N, 2, 9, 1, 1, H, W)

        up_flow = torch.sum(mask * up_flow, dim=2)
        up_flow = up_flow.permute(0, 1, 4, 2, 5, 3)
        return up_flow.reshape(N, 2, 8 * H, 8 * W)
    
    def upsample_depth(self, depth, mask):
        """ Upsample flow field [H/8, W/8, 2] -> [H, W, 2] using convex combination """
        N, _, H, W = depth.shape
        mask = mask.view(N, 1, 9, 8, 8, H, W)
        mask = torch.softmax(mask, dim=2)

        depth = F.unfold(8 * depth, [3, 3], padding=1)
        depth = depth.view(N, 1, 9, 1, 1, H, W)

        depth = torch.sum(mask * depth, dim=2)
        depth = depth.permute(0, 1, 4, 2, 5, 3)
        return depth.reshape(N, 1, 8 * H, 8 * W)

    def warp(self, x, flo):
        """
        warp an image/tensor (im2) back to im1, according to the optical flow

        x: [B, C, H, W] (im2)
        flo: [B, 2, H, W] flow

        """
        B, C, H, W = x.size()
        # mesh grid 
        xx = torch.arange(0, W).view(1,-1).repeat(H,1)
        yy = torch.arange(0, H).view(-1,1).repeat(1,W)
        xx = xx.view(1,1,H,W).repeat(B,1,1,1)
        yy = yy.view(1,1,H,W).repeat(B,1,1,1)
        grid = torch.cat((xx,yy),1).float()

        if x.is_cuda:
            grid = grid.cuda()
        vgrid = Variable(grid) + flo

        # scale grid to [-1,1] 
        vgrid[:,0,:,:] = 2.0*vgrid[:,0,:,:].clone() / max(W-1,1)-1.0
        vgrid[:,1,:,:] = 2.0*vgrid[:,1,:,:].clone() / max(H-1,1)-1.0

        vgrid = vgrid.permute(0,2,3,1)        
        output = nn.functional.grid_sample(x, vgrid)
        mask = torch.autograd.Variable(torch.ones(x.size())).cuda()
        mask = nn.functional.grid_sample(mask, vgrid)

        mask[mask<0.9999] = 0
        mask[mask>0] = 1
        
        return output*mask

    def forward(self, image1, image2, iters=12, flow_init=None, test_mode=False, idx=0):
        """ 
            Estimate optical flow between pair of frames 
            image1: lidar_input
            image2: event_frame
        """
        image1 = 2 * image1 - 1.0
        image2 = 2 * image2 - 1.0

        # print(torch.min(image1), torch.max(image1))
        # print(torch.min(image2), torch.max(image2))

        image1 = image1.contiguous()
        image2 = image2.contiguous()

        hdim = self.hidden_dim
        cdim = self.context_dim

        # run the feature network
        with autocast(enabled=self.args.mixed_precision):
            fmap1_one, fmap1_two, fmap1_three, fmap1_four, fmap1 = self.fnet_lidar(image1)
            fmap2 = self.fnet_event(image2)

        # feature_visualizer(fmap1[0, ...].cpu().detach().numpy(), f"/home/eason/WorkSpace/EventbasedVisualLocalization/EVLoc_Reconstruction/visualization/feature/{idx}")

        fmap1 = fmap1.float()
        fmap2 = fmap2.float()

        if self.args.alternate_corr:
            corr_fn = AlternateCorrBlock(fmap1, fmap2, radius=self.args.corr_radius)
        else:
            corr_fn = CorrBlock(fmap1, fmap2, radius=self.args.corr_radius)
        
        # run the context network
        with autocast(enabled=self.args.mixed_precision):
            cnet = self.cnet(image1)
            net, inp = torch.split(cnet, [hdim, cdim], dim=1)
            net = torch.tanh(net)
            inp = torch.relu(inp)

        coords0, coords1 = self.initialize_flow(image1)

        if flow_init is not None:
            coords1 = coords1 + flow_init

        flow_predictions = []
        for itr in range(iters):
            coords1 = coords1.detach()

            corr = corr_fn(coords1)  # index correlation volume Bx(9x9x4)xH/8xW/8

            flow = coords1 - coords0
            with autocast(enabled=self.args.mixed_precision):
                net, up_mask, delta_flow = self.update_block(net, inp, corr, flow)
            
            # F(t+1) = F(t) + \Delta(t)
            coords1 = coords1 + delta_flow

            # upsample predictions
            if up_mask is None:
                flow_up = upflow8(coords1 - coords0)
            else:
                flow_up = self.upsample_flow(coords1 - coords0, up_mask)

            flow_predictions.append(flow_up)

        # feature_visualizer(corr[0, ...].cpu().detach().numpy(), f"/home/eason/WorkSpace/EventbasedVisualLocalization/EVLoc_Reconstruction/visualization/feature/{idx}_corr")
        depth_mask = self.depth_mask_head(corr, fmap1_one, fmap1_two, fmap1_three, fmap1_four, fmap1)

        if test_mode:
            return flow_up, depth_mask
            
        return flow_predictions, depth_mask