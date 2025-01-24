import os
import sys
import time 

import cv2
import numpy as np
import argparse
import random
import torch

from core.datasets_m3ed import DatasetM3ED as Dataset
from core.backbone import Backbone_Event, Backbone_Fuse, Backbone_Edge, Backbone_Edge_FF, Backbone_Edge_FF_Iter
from core.utils import (count_parameters, merge_inputs, fetch_optimizer, Logger)
from core.utils_point import overlay_imgs, to_rotation_matrix, quaternion_from_matrix
from core.data_preprocess import Data_preprocess
from core.flow2pose import Flow2Pose, err_Pose
from core.losses import warp, sequence_loss, ClassifyLoss, SigLoss, ProposedLoss
from core.flow_viz import flow_to_image

try:
    from torch.cuda.amp import GradScaler
except:
    class GradScaler:
        def __init__(self):
            pass

        def scale(self, loss):
            return loss

        def unscale_(self, optimizer):
            pass

        def step(self, optimizer): 
            optimizer.step()

        def update(self):
            pass
    

def _init_fn(worker_id, seed):
    seed = worker_id + seed
    print(f"Init worker {worker_id} with seed {seed}")
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def train(args, TrainImgLoader, model, optimizer, scheduler, scaler, logger, device, epoch, occlusion_kernel=5, occlusion_threshold=3):
    model.train()
    for i_batch, sample in enumerate(TrainImgLoader):
        event_frame = sample['event_frame']
        pc = sample['point_cloud']
        calib = sample['calib']
        T_err = sample['tr_error']
        R_err = sample['rot_error']

        data_generate = Data_preprocess(calib, occlusion_threshold, occlusion_kernel)
        if args.backbone == "baseline":
            event_input, depth_input, flow_gt = data_generate.push(event_frame, pc, T_err, R_err, device, MAX_DEPTH=args.max_depth, h=288, w=512)
        elif args.backbone == "fuse":
            event_input, depth_input, flow_gt, depth2edge_gt = data_generate.push_fuse(event_frame, pc, T_err, R_err, device, MAX_DEPTH=args.max_depth, h=288, w=512)
            event2depth_gt = depth_input[:, 2, :, :].unsqueeze(1)
            depth_input = depth_input[:, 0, :, :].unsqueeze(1)
        elif args.backbone == "edge":
            event_input, depth_input, flow_gt, depth2edge_gt = data_generate.push_fuse(event_frame, pc, T_err, R_err, device, MAX_DEPTH=args.max_depth, h=288, w=512)
            depth_input = depth_input[:, 0, :, :].unsqueeze(1)
        else:
            raise "Specified backbone doesn't exist"

        visualization_folder = f"./visualization/{args.backbone}"
        if not os.path.exists(visualization_folder):
            os.makedirs(f"{visualization_folder}/train")
            os.makedirs(f"{visualization_folder}/test")
        vis_event_time_image = event_input[0,...].permute(1, 2, 0).cpu().numpy()
        vis_event_time_image = np.concatenate((np.zeros([vis_event_time_image.shape[0], vis_event_time_image.shape[1], 1]), vis_event_time_image), axis=2)
        vis_event_time_image = vis_event_time_image[:, :, [2, 0, 1]]
        cv2.imwrite(f"./visualization/{args.backbone}/train/{i_batch:05d}_1_1_event_input.png", (vis_event_time_image / np.max(vis_event_time_image) * 255).astype(np.uint8))
        vis_depth_input = overlay_imgs(event_input[0, :3, :, :]*0, depth_input[0, 0, :, :])
        cv2.imwrite(f"./visualization/{args.backbone}/train/{i_batch:05d}_2_1_depth_input.png", (vis_depth_input / np.max(vis_depth_input) * 255).astype(np.uint8))
        flow_viz = flow_to_image(flow_gt[0, ...].permute(1,2,0).cpu().detach().numpy())
        cv2.imwrite(f"./visualization/{args.backbone}/train/{i_batch:05d}_3_1_flow_gt.png", flow_viz) 
        if args.backbone == "fuse":
            vis_event2depth_gt= overlay_imgs(event_input[0, :3, :, :]*0, event2depth_gt[0, 0, :, :])
            cv2.imwrite(f"./visualization/{args.backbone}/train/{i_batch:05d}_1_2_event2depth_gt.png", (vis_event2depth_gt / np.max(vis_event2depth_gt) * 255).astype(np.uint8))       
            ground_truth_depth2edge = depth2edge_gt[:, 0, :, :].long()
            cv2.imwrite(f'./visualization/{args.backbone}/train/{i_batch:05d}_2_2_depth2edge_gt.png', (ground_truth_depth2edge[0, ...].cpu().detach().numpy()* 255).astype(np.uint8))
        if args.backbone == "edge":
            ground_truth_depth2edge = depth2edge_gt[:, 0, :, :].long()
            cv2.imwrite(f'./visualization/{args.backbone}/train/{i_batch:05d}_2_2_depth2edge_gt.png', (ground_truth_depth2edge[0, ...].cpu().detach().numpy()* 255).astype(np.uint8))

        optimizer.zero_grad()
        if args.backbone == "baseline":
            flow_preds = model(depth_input, event_input, iters=args.iters)
            loss, metrics = sequence_loss(flow_preds, flow_gt, args.gamma, MAX_FLOW=400)
            flow_viz = flow_to_image(flow_preds[-1][0, ...].permute(1,2,0).cpu().detach().numpy())
            cv2.imwrite(f"./visualization/{args.backbone}/train/{i_batch:05d}_3_2_flow_pred.png", flow_viz)
        elif args.backbone == "fuse":
            flow_preds, depth2edge, event2depth = model(depth_input, event_input, iters=args.iters)
            ## flow loss
            loss_flow, metrics = sequence_loss(flow_preds, flow_gt, args.gamma, MAX_FLOW=400)
            flow_viz = flow_to_image(flow_preds[-1][0, ...].permute(1,2,0).cpu().detach().numpy())
            cv2.imwrite(f"./visualization/{args.backbone}/train/{i_batch:05d}_3_2_flow_pred.png", flow_viz)
            ## edge loss
            depth2edge_bi = torch.cat((1.-depth2edge, depth2edge), dim=1)
            loss_edge = ClassifyLoss(depth2edge_bi, ground_truth_depth2edge, loss_func="Weighted_Cross_Entropy_Loss")
            metrics['edge_loss'] = loss_edge.item()
            cv2.imwrite(f'./visualization/{args.backbone}/train/{i_batch:05d}_2_3_depth2edge_pred.png', (depth2edge[0, 0, ...].cpu().detach().numpy()* 255).astype(np.uint8))
            ## depth loss
            loss_depth_fn = SigLoss()
            loss_depth = loss_depth_fn(event2depth, event2depth_gt)
            metrics['depth_loss'] = loss_depth.item()
            original_depth = overlay_imgs(event_input[0, :, :, :]*0, event2depth[0, 0, :, :].detach())
            cv2.imwrite(f'./visualization/{args.backbone}/train/{i_batch:05d}_1_3_event2depth_pred.png', original_depth)
            alpha = 1
            beta = 100
            theta = 100
            loss = alpha * loss_flow + beta * loss_edge + theta * loss_depth
        elif args.backbone == "edge":
            if args.iteration_num == 1:
                # flow_preds, depth2edge = model(depth_input, event_input, iters=args.iters)
                flow_preds, depth2edge_preds = model(depth_input, event_input, iters=args.iters)
                ## flow loss
                loss_flow, metrics = sequence_loss(flow_preds, flow_gt, args.gamma, MAX_FLOW=400)
                flow_viz = flow_to_image(flow_preds[-1][0, ...].permute(1,2,0).cpu().detach().numpy())
                cv2.imwrite(f"./visualization/{args.backbone}/train/{i_batch:05d}_3_2_flow_pred.png", flow_viz)
                # loss_edge = ClassifyLoss(depth2edge, ground_truth_depth2edge, loss_func="Weighted_Cross_Entropy_Loss")
                loss_edge, loss_edge_last = ClassifyLoss(depth2edge_preds, ground_truth_depth2edge, loss_func="Sequence_Weighted_Cross_Entropy_Loss")
                # metrics['edge_loss'] = loss_edge.item()
                metrics['edge_loss'] = loss_edge_last.item()
                # cv2.imwrite(f'./visualization/{args.backbone}/train/{i_batch:05d}_2_3_depth2edge_pred.png', (depth2edge[0, 0, ...].cpu().detach().numpy()* 255).astype(np.uint8))
                cv2.imwrite(f'./visualization/{args.backbone}/train/{i_batch:05d}_2_3_depth2edge_pred.png', (depth2edge_preds[-1][0, 0, ...].cpu().detach().numpy()* 255).astype(np.uint8))
                alpha = 1
                beta = 100
                loss = alpha * loss_flow + beta * loss_edge
            else:
                loss = 0.
                depth2edge_input = torch.ones(depth_input.shape, dtype=depth_input.dtype, device=depth_input.device)
                for it in range(args.iteration_num):
                    flow_preds, depth2edge_preds = model(depth_input, event_input, depth2edge_input, iters=args.iters)
                    ## flow loss
                    loss_flow, metrics = sequence_loss(flow_preds, flow_gt, args.gamma, MAX_FLOW=400)
                    flow_viz = flow_to_image(flow_preds[-1][0, ...].permute(1,2,0).cpu().detach().numpy())
                    cv2.imwrite(f"./visualization/{args.backbone}/train/{i_batch:05d}_3_2_flow_pred.png", flow_viz)
                    # loss_edge = ClassifyLoss(depth2edge, ground_truth_depth2edge, loss_func="Weighted_Cross_Entropy_Loss")
                    loss_edge, loss_edge_last = ClassifyLoss(depth2edge_preds, ground_truth_depth2edge, loss_func="Sequence_Weighted_Cross_Entropy_Loss")
                    # metrics['edge_loss'] = loss_edge.item()
                    metrics['edge_loss'] = loss_edge_last.item()
                    # cv2.imwrite(f'./visualization/{args.backbone}/train/{i_batch:05d}_2_3_depth2edge_pred.png', (depth2edge[0, 0, ...].cpu().detach().numpy()* 255).astype(np.uint8))
                    cv2.imwrite(f'./visualization/{args.backbone}/train/{i_batch:05d}_2_3_depth2edge_pred_{it}.png', (depth2edge_input[0, 0, ...].cpu().detach().numpy()* 255).astype(np.uint8))
                    alpha = 1
                    beta = 100
                    it_weight = 0.8 ** (args.iteration_num - it - 1)
                    loss += (alpha * loss_flow + beta * loss_edge) * it_weight

                    depth2edge_input = depth2edge_preds[-1]
        else:
            raise "Specified backbone doesn't exist"

        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), args.clip)
        scaler.step(optimizer)
        scheduler.step()
        scaler.update()
        logger.push(metrics)


def test(args, TestImgLoader, model, device, occlusion_kernel=5, occlusion_threshold=3, is_test=False):
    model.eval()

    out_list, epe_list = [], []
    Time = 0.
    outliers, err_r_list, err_t_list = [], [], []
    pose_loss = []
    success_rate = 0.
    valid_rate = []

    pose_loss_fn = ProposedLoss(1., 1.)
    
    for i_batch, sample in enumerate(TestImgLoader):
        event_frame = sample['event_frame']
        pc = sample['point_cloud']
        calib = sample['calib']
        T_err = sample['tr_error']
        R_err = sample['rot_error']

        data_generate = Data_preprocess(calib, occlusion_threshold, occlusion_kernel)
        if args.backbone == "baseline":
            event_input, depth_input, flow_gt = data_generate.push(event_frame, pc, T_err, R_err, device, MAX_DEPTH=args.max_depth, split='test', h=288, w=512)
        elif args.backbone == "fuse":
            event_input, depth_input, flow_gt, depth2edge_gt = data_generate.push_fuse(event_frame, pc, T_err, R_err, device, MAX_DEPTH=args.max_depth, split='test', h=288, w=512)
            event2depth_gt = depth_input[:, 2, :, :].unsqueeze(1)
            depth_input = depth_input[:, 0, :, :].unsqueeze(1)
        elif args.backbone == "edge":
            event_input, depth_input, flow_gt, depth2edge_gt = data_generate.push_fuse(event_frame, pc, T_err, R_err, device, MAX_DEPTH=args.max_depth, split='test', h=288, w=512)
            depth_input = depth_input[:, 0, :, :].unsqueeze(1)
        else:
            raise "Specified backbone doesn't exist"

        end = time.time()
        if args.backbone == "baseline":
            _, flow_up = model(depth_input, event_input, iters=24, test_mode=True, idx=i_batch)
        elif args.backbone == "fuse":
            _, flow_up, depth2edge, event2depth = model(depth_input, event_input, iters=24, test_mode=True, idx=i_batch)
        elif args.backbone == "edge":
            if args.iteration_num > 1:
                depth2edge_input = torch.ones(depth_input.shape, dtype=depth_input.dtype, device=depth_input.device)
                for it in range(args.iteration_num):
                    _, flow_up, depth2edge = model(depth_input, event_input, depth2edge_input, iters=24, test_mode=True, idx=i_batch)
                    cv2.imwrite(f'./visualization/{args.backbone}/test/{i_batch:05d}_2_3_depth2edge_pred_{it}.png', (depth2edge_input[0, 0, ...].cpu().detach().numpy()* 255).astype(np.uint8)) 
                    depth2edge_input = depth2edge[-1].detach()
            else:
                _, flow_up, depth2edge = model(depth_input, event_input, iters=24, test_mode=True, idx=i_batch)

        visualization_folder = f"./visualization/{args.backbone}"
        if not os.path.exists(visualization_folder):
            os.makedirs(f"{visualization_folder}/train")
            os.makedirs(f"{visualization_folder}/test")
        vis_event_time_image = event_input[0,...].permute(1, 2, 0).cpu().numpy()
        vis_event_time_image = np.concatenate((np.zeros([vis_event_time_image.shape[0], vis_event_time_image.shape[1], 1]), vis_event_time_image), axis=2)
        vis_event_time_image = vis_event_time_image[:, :, [2, 0, 1]]
        cv2.imwrite(f"./visualization/{args.backbone}/test/{i_batch:05d}_1_1_event_input.png", (vis_event_time_image / np.max(vis_event_time_image) * 255).astype(np.uint8))
        vis_depth_input = overlay_imgs(event_input[0, :3, :, :]*0, depth_input[0, 0, :, :])
        cv2.imwrite(f"./visualization/{args.backbone}/test/{i_batch:05d}_2_1_depth_input.png", (vis_depth_input / np.max(vis_depth_input) * 255).astype(np.uint8))
        flow_viz = flow_to_image(flow_gt[0, ...].permute(1,2,0).cpu().detach().numpy())
        cv2.imwrite(f"./visualization/{args.backbone}/test/{i_batch:05d}_3_1_flow_gt.png", flow_viz) 
        flow_viz = flow_to_image(flow_up[0, ...].permute(1,2,0).cpu().detach().numpy())
        cv2.imwrite(f"./visualization/{args.backbone}/test/{i_batch:05d}_3_2_flow_pred.png", flow_viz)
        warp_vis_event_time_image = warp(event_input, flow_up)
        warp_vis_event_time_image = warp_vis_event_time_image[0,...].permute(1, 2, 0).cpu().detach().numpy()
        warp_vis_event_time_image = np.concatenate((np.zeros([warp_vis_event_time_image.shape[0], warp_vis_event_time_image.shape[1], 1]), warp_vis_event_time_image), axis=2)
        warp_vis_event_time_image = warp_vis_event_time_image[:, :, [2, 0, 1]]
        cv2.imwrite(f"./visualization/{args.backbone}/test/{i_batch:05d}_4_1_warp_event_input.png", (warp_vis_event_time_image / np.max(warp_vis_event_time_image) * 255).astype(np.uint8))
        if args.backbone == "fuse":
            vis_event2depth_gt= overlay_imgs(event_input[0, :3, :, :]*0, event2depth_gt[0, 0, :, :])
            cv2.imwrite(f"./visualization/{args.backbone}/test/{i_batch:05d}_1_2_event2depth_gt.png", (vis_event2depth_gt / np.max(vis_event2depth_gt) * 255).astype(np.uint8))       
            ground_truth_depth2edge = depth2edge_gt[:, 0, :, :].long()
            cv2.imwrite(f'./visualization/{args.backbone}/test/{i_batch:05d}_2_2_depth2edge_gt.png', (ground_truth_depth2edge[0, ...].cpu().detach().numpy()* 255).astype(np.uint8))
            cv2.imwrite(f'./visualization/{args.backbone}/test/{i_batch:05d}_2_3_depth2edge_pred.png', (depth2edge[0, 0, ...].cpu().detach().numpy()* 255).astype(np.uint8))
            original_depth = overlay_imgs(event_input[0, :, :, :]*0, event2depth[0, 0, :, :].detach())
            cv2.imwrite(f'./visualization/{args.backbone}/test/{i_batch:05d}_1_3_event2depth_pred.png', original_depth)
        if args.backbone == "edge":
            ground_truth_depth2edge = depth2edge_gt[:, 0, :, :].long()
            cv2.imwrite(f'./visualization/{args.backbone}/test/{i_batch:05d}_2_2_depth2edge_gt.png', (ground_truth_depth2edge[0, ...].cpu().detach().numpy()* 255).astype(np.uint8))
            cv2.imwrite(f'./visualization/{args.backbone}/test/{i_batch:05d}_2_3_depth2edge_pred_{args.iteration_num}.png', (depth2edge[-1][0, 0, ...].cpu().detach().numpy()* 255).astype(np.uint8))
            # for i in range(len(depth2edge)):
            #     cv2.imwrite(f'./visualization/{args.backbone}/test/{i_batch:05d}_2_3_depth2edge_pred_{i:02d}.png', (depth2edge[i][0, 0, ...].cpu().detach().numpy()* 255).astype(np.uint8))


        epe = torch.sum((flow_up - flow_gt) ** 2, dim=1).sqrt()
        mag = torch.sum(flow_gt ** 2, dim=1).sqrt()
        epe = epe.view(-1)
        mag = mag.view(-1)
        valid_gt = (flow_gt[:, 0, :, :] != 0) + (flow_gt[:, 1, :, :] != 0)

        val = valid_gt.view(-1) >= 0.5

        out = ((epe > 3.0) & ((epe / mag) > 0.05)).float()
        if np.isnan(epe[val].mean().item()):
            outliers.append(i_batch)
            continue
        epe_list.append(epe[val].mean().item())
        out_list.append(out[val].cpu().numpy())

        valid_rate.append(val.sum().item() / (depth_input > 0).sum().item())

        R_pred, T_pred, inliers, flag = Flow2Pose(flow_up, depth_input, calib, MAX_DEPTH=args.max_depth, x=36, y=64, h=288, w=512)
        Time += time.time() - end
        if flag:
            outliers.append(i_batch)
            continue
        else:
            pose_loss_i = pose_loss_fn(T_err, R_err, T_pred.unsqueeze(0), R_pred.unsqueeze(0))
            pose_loss.append(pose_loss_i.item())
            if is_test:
                err_r, err_t = err_Pose(R_pred, T_pred, R_err[0], T_err[0])
                err_r_list.append(err_r.item())
                err_t_list.append(err_t.item())
                if err_r < 1. and err_t < 10.:
                    success_rate += 1
                print(f"{i_batch:05d}: {np.mean(err_t_list):.5f} {np.mean(err_r_list):.5f} | {np.median(err_t_list):.5f} "
                        f"{np.median(err_r_list):.5f} | {np.mean(pose_loss):.5f} | {np.mean(valid_rate):.5f} | "
                        f"{len(outliers)} | {Time / (i_batch+1):.5f}")
                # # Define text properties
                # org = (flow_viz.shape[1]-400, flow_viz.shape[0]-70)
                # font = cv2.FONT_HERSHEY_SIMPLEX
                # fontScale = 1
                # color = (0, 0, 255)
                # thickness = 2
                # text = f"R={err_r.item():.3f} T={err_t.item():.3f}"
                # cv2.putText(flow_viz, text, org, font, fontScale, color, thickness, cv2.LINE_AA)
                # cv2.imwrite(f"./visualization/{args.backbone}/test/{i_batch:05d}_3_2_flow_pred.png", flow_viz)
              
    epe_list = np.array(epe_list)
    out_list = np.concatenate(out_list)

    epe = np.median(epe_list)
    f1 = 100 * np.mean(out_list)
    pose_loss = np.mean(pose_loss)

    if not is_test:
        return epe, f1, pose_loss
    else:
        return err_t_list, err_r_list, outliers, Time, epe, f1, pose_loss, success_rate   

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_path',
                        type=str,
                        metavar='DIR',
                        default='/media/eason/Backup/Datasets/M3ED/generated/Falcon',
                        help='path to dataset')
    parser.add_argument('--ev_input', 
                        '--event_representation',
                        type=str,
                        default='ours_denoise_stc_trail_pre_100000_half')
    parser.add_argument('--test_sequence',
                        type=str, 
                        default='falcon_indoor_flight_3')
    parser.add_argument('--load_checkpoints',
                        help="restore checkpoint")
    parser.add_argument('--epochs', 
                        default=100, 
                        type=int, 
                        metavar='N',
                        help='number of total epochs to run')
    parser.add_argument('--starting_epoch', 
                        default=0, 
                        type=int, 
                        metavar='N',
                        help='manual epoch number (useful on restarts)')
    parser.add_argument('--batch_size', 
                        default=2, 
                        type=int,
                        metavar='N', help='mini-batch size')
    parser.add_argument('--lr', 
                        '--learning_rate', 
                        default=4e-5, 
                        type=float,
                        metavar='LR', 
                        help='initial learning rate')
    parser.add_argument('--wdecay', 
                        type=float, 
                        default=.00005)
    parser.add_argument('--epsilon', 
                        type=float, 
                        default=1e-8)
    parser.add_argument('--clip', 
                        type=float, 
                        default=1.0)
    parser.add_argument('--gamma', 
                        type=float, 
                        default=0.8, 
                        help='exponential weighting')
    parser.add_argument('--iters', 
                        type=int, 
                        default=12)
    parser.add_argument('--gpus', 
                        type=int, 
                        nargs='+', 
                        default=[0])
    parser.add_argument('--max_r', 
                        type=float, 
                        default=5.)
    parser.add_argument('--max_t', 
                        type=float, 
                        default=0.5)
    parser.add_argument('--max_depth', 
                        type=float, 
                        default=10.)
    parser.add_argument('--num_workers', 
                        type=int, 
                        default=3)
    parser.add_argument('--mixed_precision', 
                        action='store_true', 
                        help='use mixed precision')
    parser.add_argument('--evaluate_interval', 
                        default=1, 
                        type=int, 
                        metavar='N',
                        help='Evaluate every \'evaluate interval\' epochs ')
    parser.add_argument('-e', 
                        '--evaluate', 
                        dest='evaluate', 
                        action='store_true',
                        help='evaluate model on validation set')
    parser.add_argument('--backbone',
                        type=str,
                        default='baseline')
    parser.add_argument('--use_feature_fusion', 
                        action='store_true')
    parser.add_argument('--iteration_num', 
                        default=1, 
                        type=int)
    args = parser.parse_args()    
    
    ## set key parameters
    occlusion_kernel = 5
    occlusion_threshold = 3
    seed = 1234
    device = torch.device(f"cuda:{args.gpus[0]}" if torch.cuda.is_available() else "cpu")
    os.environ["CUDA_LAUNCH_BLOCKING"] = "1"
    torch.cuda.set_device(args.gpus[0])
    batch_size = args.batch_size

    ## initialize using fixed seed
    _init_fn(0, seed)

    if args.backbone == "baseline":
        model = torch.nn.DataParallel(Backbone_Event(args), device_ids=args.gpus)
    elif args.backbone == "fuse":
        model = torch.nn.DataParallel(Backbone_Fuse(args), device_ids=args.gpus)
    elif args.backbone == "edge":
        if args.iteration_num > 1:
            model = torch.nn.DataParallel(Backbone_Edge_FF_Iter(args), device_ids=args.gpus)
        else:
            if args.use_feature_fusion:
                model = torch.nn.DataParallel(Backbone_Edge_FF(args), device_ids=args.gpus)
            else:
                model = torch.nn.DataParallel(Backbone_Edge(args), device_ids=args.gpus)
    else:
        raise "Specified backbone doesn't exist"
    print("Parameter Count: %d" % count_parameters(model))
    if args.load_checkpoints is not None:
        model.load_state_dict(torch.load(args.load_checkpoints))
    model.to(device)

    # # # freeze edge detector
    # for param in model.module.edge_detector.parameters():
    #     param.requires_grad = False
    # # # freeze optical flow estimator
    # for param in model.module.fnet_event.parameters():
    #     param.requires_grad = False
    # for param in model.module.fnet_lidar.parameters():
    #     param.requires_grad = False
    # for param in model.module.cnet.parameters():
    #     param.requires_grad = False
    # for param in model.module.update_block.parameters():
    #     param.requires_grad = False

    ## reinitialize using fixed seed
    _init_fn(0, seed)

    def init_fn(x):
        return _init_fn(x, seed)

    dataset_test = Dataset(args.data_path,
                           event_representation=args.ev_input,
                           max_r=args.max_r, 
                           max_t=args.max_t,
                           split='test', 
                           test_sequence=args.test_sequence)
    TestImgLoader = torch.utils.data.DataLoader(dataset=dataset_test,
                                                shuffle=False,
                                                batch_size=1,
                                                num_workers=args.num_workers,
                                                worker_init_fn=init_fn,
                                                collate_fn=merge_inputs,
                                                drop_last=False,
                                                pin_memory=True)
    if args.evaluate:
        with torch.no_grad():
            err_t_list, err_r_list, outliers, Time, epe, f1, pose_loss, success_rate = test(args, TestImgLoader, model, device, 
                                                                   occlusion_kernel=occlusion_kernel, occlusion_threshold=occlusion_threshold, 
                                                                   is_test=True)
            print(f"Mean trans error {np.mean(err_t_list):.5f}  Mean rotation error {np.mean(err_r_list):.5f}")
            print(f"Median trans error {np.median(err_t_list):.5f}  Median rotation error {np.median(err_r_list):.5f}")
            print(f"epe {epe:.5f} pose_loss {pose_loss:.5f} Mean {Time / len(TestImgLoader):.5f} per frame")
            print(f"success rate {success_rate/len(TestImgLoader):.5f}")
            print(f"Outliers number {len(outliers)}/{len(TestImgLoader)} {outliers}")
        sys.exit()

    dataset_train = Dataset(args.data_path,
                            event_representation=args.ev_input,
                            max_r=args.max_r, 
                            max_t=args.max_t,
                            split='train',
                            test_sequence=args.test_sequence)
    TrainImgLoader = torch.utils.data.DataLoader(dataset=dataset_train,
                                                 shuffle=True,
                                                 batch_size=batch_size,
                                                 num_workers=args.num_workers,
                                                 worker_init_fn=init_fn,
                                                 collate_fn=merge_inputs,
                                                 drop_last=False,
                                                 pin_memory=True)
    print("Train length: ", len(TrainImgLoader))
    print("Test length: ", len(TestImgLoader))

    optimizer, scheduler = fetch_optimizer(args, len(TrainImgLoader), model)
    scaler = GradScaler(enabled=args.mixed_precision)
    logger = Logger(model, scheduler, SUM_FREQ=100)

    datetime = time.strftime('%Y-%m-%d-%H-%M-%S',time.localtime(time.time()))
    if not os.path.exists(f'./checkpoints/{args.backbone}/{datetime}'):
        os.makedirs(f'./checkpoints/{args.backbone}/{datetime}')

    starting_epoch = args.starting_epoch
    if starting_epoch > 0:
        for i in range(starting_epoch * len(TrainImgLoader)):
            scaler.unscale_(optimizer)
            scaler.step(optimizer)
            scheduler.step()
            scaler.update()
        logger.total_steps = starting_epoch * len(TrainImgLoader)

    min_val_err = 9999.
    # max_epochs = args.epochs // args.iteration_num
    max_epochs = args.epochs
    for epoch in range(starting_epoch, max_epochs):
        train(args, TrainImgLoader, model, optimizer, scheduler, scaler, logger, device, epoch, occlusion_kernel=occlusion_kernel, occlusion_threshold=occlusion_threshold, )

        torch.cuda.empty_cache()

        if epoch % args.evaluate_interval == 0:
            epe, f1, pose_loss = test(args, TestImgLoader, model, device, occlusion_kernel=occlusion_kernel, occlusion_threshold=occlusion_threshold)
            print("Validation M3ED: %f, %f, %f" % (epe, f1, pose_loss))

            results = {'m3ed-epe': epe, 'm3ed-f1': f1, 'm3ed-poseloss': pose_loss}
            logger.write_dict(results)

            torch.save(model.state_dict(), f"./checkpoints/{args.backbone}/{datetime}/checkpoint.pth")

            if pose_loss < min_val_err:
                min_val_err = pose_loss
                torch.save(model.state_dict(), f'./checkpoints/{args.backbone}/{datetime}/best_model.pth')
            
            torch.cuda.empty_cache()