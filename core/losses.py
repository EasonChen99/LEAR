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


def sequence_loss_plus(flow_preds, flow_gt, flow_ref, gamma=0.8, MAX_FLOW=400):
    """ Loss function defined over sequence of flow predictions """

    mag = torch.sum(flow_gt ** 2, dim=1).sqrt()
    Mask = torch.zeros([flow_gt.shape[0], flow_gt.shape[1], flow_gt.shape[2],
                        flow_gt.shape[3]]).to(flow_gt.device)
    mask = (flow_gt[:, 0, :, :] != 0) + (flow_gt[:, 1, :, :] != 0)
    valid = mask & (mag < MAX_FLOW)
    Mask[:, 0, :, :] = valid
    Mask[:, 1, :, :] = valid
    Mask = Mask != 0

    flow_gt_plus = flow_ref.clone()
    flow_gt_plus[Mask] = flow_gt[Mask]

    n_predictions = len(flow_preds)
    flow_loss = 0.0

    for i in range(n_predictions):
        i_weight = gamma ** (n_predictions - i - 1)
        Loss_reg = flow_preds[i] - flow_gt_plus
        Loss_reg = torch.norm(Loss_reg, dim=1)
        Loss_reg = torch.sum(Loss_reg, dim=[1, 2])
        Loss_reg = Loss_reg / (flow_gt.shape[2] * flow_gt.shape[3])
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


def ClassifyLoss(depth_mask, ground_truth, flow=None, lidar_input=None, loss_func="Cross_Entropy_Loss"):
    if loss_func == "Cross_Entropy_Loss":
        # cross-entropy loss
        criterion = nn.CrossEntropyLoss()
        loss = criterion(depth_mask, ground_truth)
    elif loss_func == "Focal_Loss":
        # focal loss
        alpha=0.25
        gamma=2.0
        criterion = nn.CrossEntropyLoss(reduction='none')
        ce_loss = criterion(depth_mask, ground_truth)
        pt = torch.exp(-ce_loss)
        loss = (alpha * (1 - pt) ** gamma * ce_loss).mean()
    elif loss_func == "Weighted_Cross_Entropy_Loss":
        # weighted cross-entropy loss
        count_neg = (ground_truth == 0).sum()
        count_pos = (ground_truth == 1).sum()
        beta = count_neg / (count_neg + count_pos)
        pos_weight = beta / (1 - beta)
        weights = torch.tensor([beta, pos_weight], device=ground_truth.device)
        criterion = nn.CrossEntropyLoss(weight=weights)
        loss = criterion(depth_mask, ground_truth)
    elif loss_func == "Masked_Weighted_Cross_Entropy_Loss":
        # masked weighted cross-entropy loss
        if lidar_input == None:
            raise "depth input doesn't exist"
        mask = lidar_input > 0
        count_neg = (ground_truth[mask[:, 0, ...]] == 0).sum()
        count_pos = (ground_truth[mask[:, 0, ...]] == 1).sum()
        beta = count_neg / (count_neg + count_pos)
        pos_weight = beta / (1 - beta)
        weights = torch.tensor([beta, pos_weight], device=ground_truth.device)
        criterion = nn.CrossEntropyLoss(weight=weights)
        loss = criterion(depth_mask[mask.repeat(1, 2, 1, 1)].view(1, 2, -1), ground_truth[mask[:, 0, ...]].view(1, -1))
    elif loss_func == "Warped_Weighted_Cross_Entropy_Loss":
         # warped weighted cross-entropy loss
        if flow == None:
            raise "Flow doesn't exist"
        count_neg = (ground_truth == 0).sum()
        count_pos = (ground_truth == 1).sum()
        beta = count_neg / (count_neg + count_pos)
        pos_weight = beta / (1 - beta)
        weights = torch.tensor([beta, pos_weight], device=ground_truth.device)
        criterion = nn.CrossEntropyLoss(weight=weights)
        depth_mask_warp = warp(depth_mask, flow)
        loss = criterion(depth_mask_warp, ground_truth)
    elif loss_func == "Inverse_Warped_Weighted_Cross_Entropy_Loss":
         # warped weighted cross-entropy loss
        if flow == None:
            raise "Flow doesn't exist"
        ground_truth_warp = warp(ground_truth, flow)
        ground_truth_warp = ground_truth_warp > 0
        ground_truth = ground_truth_warp[:, 0, ...].long()
        count_neg = (ground_truth == 0).sum()
        count_pos = (ground_truth == 1).sum()
        beta = count_neg / (count_neg + count_pos)
        pos_weight = beta / (1 - beta)
        weights = torch.tensor([beta, pos_weight], device=ground_truth.device)
        criterion = nn.CrossEntropyLoss(weight=weights)
        loss = criterion(depth_mask, ground_truth)  
    elif loss_func == "Masked_Inverse_Warped_Weighted_Cross_Entropy_Loss":
         # masked inversed warped weighted cross-entropy loss
        if flow == None:
            raise "Flow doesn't exist"
        if lidar_input == None:
            raise "depth input doesn't exist"
        mask = lidar_input > 0
        ground_truth_warp = warp(ground_truth, flow)
        ground_truth_warp = ground_truth_warp > 0
        ground_truth = ground_truth_warp[:, 0, ...].long()
        count_neg = (ground_truth[mask[:, 0, ...]] == 0).sum()
        count_pos = (ground_truth[mask[:, 0, ...]] == 1).sum()
        beta = count_neg / (count_neg + count_pos)
        pos_weight = beta / (1 - beta)
        weights = torch.tensor([beta, pos_weight], device=ground_truth.device)
        criterion = nn.CrossEntropyLoss(weight=weights)
        loss = criterion(depth_mask[mask.repeat(1, 2, 1, 1)].view(1, 2, -1), ground_truth[mask[:, 0, ...]].view(1, -1))   
    else:
        raise "Loss Function doesn't exist"
    return loss


class SigLoss(nn.Module):
    """SigLoss.

        We adopt the implementation in `Adabins <https://github.com/shariqfarooq123/AdaBins/blob/main/loss.py>`_.

    Args:
        valid_mask (bool): Whether filter invalid gt (gt > 0). Default: True.
        loss_weight (float): Weight of the loss. Default: 1.0.
        max_depth (int): When filtering invalid gt, set a max threshold. Default: None.
        warm_up (bool): A simple warm up stage to help convergence. Default: False.
        warm_iter (int): The number of warm up stage. Default: 100.
    """

    def __init__(self,
                 valid_mask=True,
                 loss_weight=1.0,
                 max_depth=1.,
                 warm_up=False,
                 warm_iter=100):
        super(SigLoss, self).__init__()
        self.valid_mask = valid_mask
        self.loss_weight = loss_weight
        self.max_depth = max_depth

        self.eps = 0.001 # avoid grad explode

        # HACK: a hack implementation for warmup sigloss
        self.warm_up = warm_up
        self.warm_iter = warm_iter
        self.warm_up_counter = 0

    def sigloss(self, input, target):
        if self.valid_mask:
            valid_mask = target > 0
            if self.max_depth is not None:
                valid_mask = torch.logical_and(target > 0, target <= self.max_depth)
            input = input[valid_mask]
            target = target[valid_mask]
        
        if self.warm_up:
            if self.warm_up_counter < self.warm_iter:
                g = torch.log(input + self.eps) - torch.log(target + self.eps)
                g = 0.15 * torch.pow(torch.mean(g), 2)
                self.warm_up_counter += 1
                return torch.sqrt(g)

        g = torch.log(input + self.eps) - torch.log(target + self.eps)
        Dg = torch.var(g) + 0.15 * torch.pow(torch.mean(g), 2)
        return torch.sqrt(Dg)

    def forward(self, depth_pred, depth_gt):
        """Forward function."""
        
        loss_depth = self.loss_weight * self.sigloss(depth_pred, depth_gt)
        return loss_depth


def FeatureTransferLoss(feature, feature_guide):

    return F.mse_loss(feature, feature_guide)



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