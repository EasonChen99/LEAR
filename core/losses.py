import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import numpy as np
import visibility


def sequence_loss(flow_preds, flow_gt, gamma=0.8, MAX_FLOW=400):
    """ Loss function defined over sequence of flow predictions """

    mag = torch.sum(flow_gt ** 2, dim=1).sqrt()
    Mask = torch.zeros([flow_gt.shape[0], flow_gt.shape[1], flow_gt.shape[2],
                        flow_gt.shape[3]]).to(flow_gt.device)
    mask = (flow_gt[:, 0, :, :] != 0) + (flow_gt[:, 1, :, :] != 0)
    valid = mask & (mag < MAX_FLOW)
    Mask[:, 0, :, :] = valid
    Mask[:, 1, :, :] = valid
    Mask = Mask != 0
    mask_sum = torch.sum(mask, dim=[1, 2])

    n_predictions = len(flow_preds)
    flow_loss = 0.0

    for i in range(n_predictions):
        i_weight = gamma ** (n_predictions - i - 1)
        Loss_reg = (flow_preds[i] - flow_gt) * Mask
        Loss_reg = torch.norm(Loss_reg, dim=1)
        Loss_reg = torch.sum(Loss_reg, dim=[1, 2])
        Loss_reg = Loss_reg / (mask_sum + 1e-5)
        flow_loss += i_weight * Loss_reg.mean()

    epe = torch.sum((flow_preds[-1] - flow_gt) ** 2, dim=1).sqrt()
    epe = epe.view(-1)[valid.view(-1)]

    metrics = {
        'epe': epe.mean().item(),
    }

    return flow_loss, metrics


def DepthReconLoss(source_depth_maps, target_depth_map, gamma=0.8):
    n_predictions = len(source_depth_maps)
    loss = 0.0
    # target_depth_map = target_depth_map * 2 - 1
    for i in range(n_predictions):
        i_weight = gamma ** (n_predictions - i - 1)

        source_depth_map = source_depth_maps[i]
        mask = source_depth_map > 0
        # source_depth_map = source_depth_map * 2 - 1.

        # print("source_depth_map: ", torch.min(source_depth_map), torch.max(source_depth_map))
        # print("target_depth_map: ", torch.min(target_depth_map), torch.max(target_depth_map))

        loss_n = torch.sum(torch.abs((source_depth_map - target_depth_map) * mask), dim=[1,2,3]) / (torch.sum(mask, dim=[1,2,3]) + 1.)
        loss += i_weight * torch.mean(loss_n)
    
    return loss


def ChamferLossOneWay2D(source_depth_map, depth_mask, flow_preds, target_event_frame, gamma=0.8):
    mask = depth_mask>0.1
    source_depth_map = source_depth_map * mask

    n_predictions = len(flow_preds)
    loss = 0.0
    for i in range(n_predictions):
        i_weight = gamma ** (n_predictions - i - 1)

        source_depth_map = warp(source_depth_map, -1 * flow_preds[i])

        B, _, H, W = source_depth_map.shape

        # Get indices of valid depth points in the source depth map
        source_mask = source_depth_map > 0  # Non-zero points
        source_points = torch.nonzero(source_mask, as_tuple=False)  # [N_source, 3]

        # Get indices of valid depth points in the target depth map
        target_mask = (target_event_frame[:, 0, :, :] > 0) + (target_event_frame[:, 1, :, :] > 0)  # Non-zero points
        target_mask = target_mask.unsqueeze(1)
        target_points = torch.nonzero(target_mask, as_tuple=False)  # [N_target, 3]

        if source_points.shape[0] == 0 or target_points.shape[0] == 0:
            # If there are no valid points in either depth map, return zero loss
            return torch.tensor(0.0, device=source_depth_map.device)

        # Extract x, y coordinates of source and target points
        source_coords = source_points[:, 1:].float()  # Ignore batch index
        target_coords = target_points[:, 1:].float()  # Ignore batch index

        if target_coords.shape[0] < source_coords.shape[0]:
            if target_coords.shape[0] < 30000:
                indices = torch.randperm(source_coords.shape[0])[:target_coords.shape[0]]
                source_coords = source_coords[indices, :]
            else:
                indices = torch.randperm(source_coords.shape[0])[:30000]
                source_coords = source_coords[indices, :]
                indices = torch.randperm(target_coords.shape[0])[:30000]
                target_coords = target_coords[indices, :]
        else:
            if source_coords.shape[0] < 30000:
                indices = torch.randperm(target_coords.shape[0])[:source_coords.shape[0]]
                target_coords = target_coords[indices, :]
            else:
                indices = torch.randperm(source_coords.shape[0])[:30000]
                source_coords = source_coords[indices, :]
                indices = torch.randperm(target_coords.shape[0])[:30000]
                target_coords = target_coords[indices, :]  

        # Compute pairwise distances between each source point and each target point
        dist_matrix = torch.cdist(source_coords, target_coords, p=2)  # [N_source, N_target]

        # Get the minimum distance for each source point to the nearest target point
        min_dist, _ = torch.min(dist_matrix, dim=1)  # [N_source]

        # One-way Chamfer loss (sum of distances or mean depending on preference)
        chamfer_loss = torch.mean(min_dist)

        loss += i_weight * chamfer_loss.cuda()


    return loss


def ConsistencyLoss(source_depth_map, depth_mask, flow_preds, target_event_frame, gamma=0.8):
    mask = depth_mask!=0
    source_depth_map = source_depth_map * mask

    n_predictions = len(flow_preds)
    loss = 0.0
    for i in range(n_predictions):
        i_weight = gamma ** (n_predictions - i - 1)

        source_depth_map = warp(source_depth_map, -1 * flow_preds[i])

        source_mask = source_depth_map != 0  # Non-zero points
        source_mask = source_mask.float()

        target_mask = (target_event_frame[:, 0, :, :] != 0) + (target_event_frame[:, 1, :, :] != 0)  # Non-zero points
        target_mask = target_mask.unsqueeze(1).float()

        consist_loss = (source_mask - target_mask) ** 2
        consist_loss = torch.sum(consist_loss, dim=[1, 2, 3])

        consist_loss = consist_loss / (source_mask.sum() + 1e-5)

        loss += i_weight * consist_loss.mean()


    return loss


def warp(x, flo):
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

if __name__ == "__main__":
    flow_preds = torch.tensor([[[[2., 1., 1., 1.],
                                 [4., 1., 1., 1.],
                                 [6., 1., 1., 1.]],

                                [[1., 1., 1., 1.],
                                 [1., 1., 1., 1.],
                                 [1., 1., 1., 1.]]]])
    uncertainty_maps = torch.tensor([[[[1., 1., 1., 1.], 
                                       [1., 1., 1., 1.],
                                       [1., 1., 1., 1.]]]])
    flow_gt = torch.tensor([[[[1., 1., 1., 1.],
                              [1., 1., 1., 1.],
                              [1., 1., 1., 1.]],

                             [[1., 1., 1., 1.],
                              [1., 1., 1., 1.],
                              [1., 1., 1., 1.]]]])