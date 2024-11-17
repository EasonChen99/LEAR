import os
import sys
import time 

import cv2
import numpy as np
import argparse
import random
import torch

from core.datasets_m3ed import DatasetM3ED as Dataset
from core.backbone import Backbone_Transfer as Backbone
from core.extractor import BasicEncoder_LiDAR
from core.utils import (count_parameters, merge_inputs, fetch_optimizer, Logger)
from core.utils_point import overlay_imgs, to_rotation_matrix, quaternion_from_matrix
from core.data_preprocess import Data_preprocess
from core.flow2pose import Flow2Pose, err_Pose
from core.losses import sequence_loss, FeatureTransferLoss, warp
from core.flow_viz import flow_to_image

occlusion_kernel = 5
occlusion_threshold = 3
seed = 1234

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
    seed = seed
    print(f"Init worker {worker_id} with seed {seed}")
    os.environ['PYTHONHASHSEED'] = str(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def train(args, TrainImgLoader, model, depth_encoder, optimizer, scheduler, scaler, logger, device, epoch):
    global occlusion_threshold, occlusion_kernel
    model.train()
    for i_batch, sample in enumerate(TrainImgLoader):
        event_frame = sample['event_frame']
        pc = sample['point_cloud']
        calib = sample['calib']
        T_err = sample['tr_error']
        R_err = sample['rot_error']

        data_generate = Data_preprocess(calib, occlusion_threshold, occlusion_kernel)
        # event_input, lidar_input, flow_gt = data_generate.push(event_frame, pc, T_err, R_err, device, MAX_DEPTH=args.max_depth, h=600, w=960)
        event_input, lidar_input, lidar_mask_input, flow_gt = data_generate.push_use_mask_plus(event_frame, pc, T_err, R_err, device, MAX_DEPTH=args.max_depth, h=296, w=512)


        vis_event_time_image = event_input[0,...].permute(1, 2, 0).cpu().numpy()
        if vis_event_time_image.shape[2] == 1:
            vis_event_time_image = event_input[0,...].permute(1, 2, 0).repeat(1, 1, 3).cpu().numpy()
        else:
            vis_event_time_image = np.concatenate((np.zeros([vis_event_time_image.shape[0], vis_event_time_image.shape[1], 1]), vis_event_time_image), axis=2)
        vis_event_time_image = vis_event_time_image[:, :, :3]
        cv2.imwrite(f"./visualization/input/{i_batch:05d}_event.png", (vis_event_time_image / np.max(vis_event_time_image) * 255).astype(np.uint8))
        if event_input.shape[1] == 1:
            vis_lidar_input = overlay_imgs(event_input[0, :, :, :].repeat(3, 1, 1)*0, lidar_input[0, 0, :, :])
        else:
            vis_lidar_input = overlay_imgs(event_input[0, :3, :, :]*0, lidar_input[0, 0, :, :])
        lidar_input[lidar_input==1000.] = 0.
        cv2.imwrite(f"./visualization/input/{i_batch:05d}_projection.png", (vis_lidar_input / np.max(vis_lidar_input) * 255).astype(np.uint8))
        if event_input.shape[1] == 1:
            vis_lidar_mask_input = overlay_imgs(event_input[0, :, :, :].repeat(3, 1, 1)*0, lidar_mask_input[0, 0, :, :])
        else:
            vis_lidar_mask_input = overlay_imgs(event_input[0, :3, :, :]*0, lidar_mask_input[0, 0, :, :])
        lidar_mask_input[lidar_mask_input==1000.] = 0.
        cv2.imwrite(f"./visualization/input/{i_batch:05d}_projection_mask.png", (vis_lidar_mask_input / np.max(vis_lidar_mask_input) * 255).astype(np.uint8))


        optimizer.zero_grad()
        flow_preds, fmap1, cnet = model(lidar_input, event_input, iters=args.iters)
        with torch.no_grad():  # Guide encoder is fixed
            fmap1_guide, cnet_guide = depth_encoder(lidar_mask_input)

        loss, metrics = sequence_loss(flow_preds, flow_gt, args.gamma, MAX_FLOW=400)

        loss_feature_fmap = FeatureTransferLoss(fmap1, fmap1_guide)
        loss_feature_cnet = FeatureTransferLoss(cnet, cnet_guide)
        metrics["feature_loss"] = loss_feature_fmap.item()
        metrics["cnet_loss"] = loss_feature_cnet.item()

        loss = loss + loss_feature_fmap + loss_feature_cnet

        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), args.clip)

        scaler.step(optimizer)
        scheduler.step()
        scaler.update()

        logger.push(metrics)

def test(args, TestImgLoader, model, device, cal_pose=False):
    global occlusion_threshold, occlusion_kernel
    model.eval()
    out_list, epe_list = [], []
    Time = 0.
    outliers, err_r_list, err_t_list = [], [], []
    
    for i_batch, sample in enumerate(TestImgLoader):
        event_frame = sample['event_frame']
        pc = sample['point_cloud']
        calib = sample['calib']
        T_err = sample['tr_error']
        R_err = sample['rot_error']

        data_generate = Data_preprocess(calib, occlusion_threshold, occlusion_kernel)
        # event_input, lidar_input, flow_gt = data_generate.push(event_frame, pc, T_err, R_err, device, MAX_DEPTH=args.max_depth, split='test', h=600, w=960)
        event_input, lidar_input, flow_gt = data_generate.push(event_frame, pc, T_err, R_err, device, MAX_DEPTH=args.max_depth, split='test', h=296, w=512)

        end = time.time()
        _, flow_up = model(lidar_input, event_input, iters=24, test_mode=True, idx=i_batch)

        epe = torch.sum((flow_up - flow_gt) ** 2, dim=1).sqrt()
        mag = torch.sum(flow_gt ** 2, dim=1).sqrt()
        epe = epe.view(-1)
        mag = mag.view(-1)
        valid_gt = (flow_gt[:, 0, :, :] != 0) + (flow_gt[:, 1, :, :] != 0)

        val = valid_gt.view(-1) >= 0.5

        out = ((epe > 3.0) & ((epe / mag) > 0.05)).float()
        epe_list.append(epe[val].mean().item())
        out_list.append(out[val].cpu().numpy())

        if cal_pose:
            # R_pred, T_pred, inliers, flag = Flow2Pose(flow_up, lidar_input, calib, MAX_DEPTH=args.max_depth, x=60, y=160, h=600, w=960)
            R_pred, T_pred, inliers, flag = Flow2Pose(flow_up, lidar_input, calib, MAX_DEPTH=args.max_depth, x=32, y=64, h=296, w=512)

            Time += time.time() - end
            if flag:
                outliers.append(i_batch)
                continue
            else:
                err_r, err_t = err_Pose(R_pred, T_pred, R_err[0], T_err[0])
                err_r_list.append(err_r.item())
                err_t_list.append(err_t.item())
            print(f"{i_batch:05d}: {np.mean(err_t_list):.5f} {np.mean(err_r_list):.5f} {np.median(err_t_list):.5f} "
                  f"{np.median(err_r_list):.5f} {len(outliers)} {Time / (i_batch+1):.5f}")
        

        # original_overlay = overlay_imgs(event_input[0, :, :, :]*0, lidar_input[0, 0, :, :])
        # cv2.imwrite(f'./visualization/output/{i_batch:05d}_1_depth_ori.png', original_overlay)
        # original_overlay = overlay_imgs(event_input[0, :, :, :], lidar_input[0, 0, :, :]*0)
        # cv2.imwrite(f'./visualization/output/{i_batch:05d}_1_event_ori.png', original_overlay)
        # # _, lidar_input_gt, _ = data_generate.push_use_mask(event_frame, pc, [torch.tensor([0.,0.,0.])], [torch.tensor([1., 0., 0., 0.])], device, MAX_DEPTH=args.max_depth, split='test', h=296, w=512) 
        # # gt_overlay = overlay_imgs(event_input[0, :, :, :]*0, lidar_input_gt[0, 0, :, :])
        # # cv2.imwrite(f'./visualization/output/{i_batch:05d}_2_depth_gt.png', gt_overlay)
        # RT_inv = to_rotation_matrix(R_err[0], T_err[0])
        # RT_inv = RT_inv.to(device)
        # RT = RT_inv.clone().inverse()
        # RT_pred = to_rotation_matrix(R_pred, T_pred)
        # RT_pred = RT_pred.to(device)
        # RT_new = torch.mm(RT, RT_pred)
        # T_composed = RT_new[:3, 3]
        # R_composed = quaternion_from_matrix(RT_new)
        # _, lidar_input_pred, _ = data_generate.push_use_mask(event_frame, pc, [T_composed], [R_composed], device, MAX_DEPTH=args.max_depth, split='test', h=296, w=512) 
        # pred_overlay = overlay_imgs(event_input[0, :, :, :]*0, lidar_input_pred[0, 0, :, :])
        # cv2.imwrite(f'./visualization/output/{i_batch:05d}_3_depth_pred.png', pred_overlay)

        flow_viz = flow_to_image(flow_up[0, ...].permute(1,2,0).cpu().detach().numpy())
        cv2.imwrite(f"./visualization/flow/{i_batch:05d}_flow.png", flow_viz)

        
    epe_list = np.array(epe_list)
    out_list = np.concatenate(out_list)

    epe = np.median(epe_list)
    f1 = 100 * np.mean(out_list)
    if not cal_pose:
        return epe, f1
    else:
        return err_t_list, err_r_list, outliers, Time, epe, f1   


class Depth_Encoder_Guide(torch.nn.Module):
    def __init__(self, args):
        super(Depth_Encoder_Guide, self).__init__()
        self.fnet_lidar = BasicEncoder_LiDAR(output_dim=256, norm_fn='instance', dropout=args.dropout)
        self.cnet = BasicEncoder_LiDAR(output_dim=256, norm_fn='batch', dropout=args.dropout)
    
    def forward(self, depth):
        depth = 2 * depth - 1.0
        depth = depth.contiguous()

        fmap1 = self.fnet_lidar(depth)
        cnet = self.cnet(depth)

        return fmap1, cnet


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
                        default='ours_denoise_pre_100000')
    parser.add_argument('--test_sequence',
                        type=str, 
                        default='falcon_indoor_flight_3')
    parser.add_argument('--load_checkpoints',
                        help="restore checkpoint")
    parser.add_argument('--load_pretrained_checkpoints',
                        help="load checkpoint")
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
    args = parser.parse_args()    


    device = torch.device(f"cuda:{args.gpus[0]}" if torch.cuda.is_available() else "cpu")
    os.environ["CUDA_LAUNCH_BLOCKING"] = "1"
    torch.cuda.set_device(args.gpus[0])

    batch_size = args.batch_size

    model = torch.nn.DataParallel(Backbone(args), device_ids=args.gpus)
    print("Parameter Count: %d" % count_parameters(model))
    if args.load_checkpoints is not None:
        model.load_state_dict(torch.load(args.load_checkpoints))
    model.to(device)

    depth_encoder = torch.nn.DataParallel(Depth_Encoder_Guide(args), device_ids=args.gpus)
    if args.load_pretrained_checkpoints is not None:
        checkpoint = torch.load(args.load_pretrained_checkpoints)
        state_dict = {}
        for key in checkpoint:
            if key in depth_encoder.state_dict().keys():
                print("checkpoint: ", key)
                state_dict[key] = checkpoint[key]
        depth_encoder.load_state_dict(state_dict)
        depth_encoder.to(device)
        depth_encoder.eval()
    else:
        if not args.evaluate:
            raise "pretrained model unfound"
    
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
            err_t_list, err_r_list, outliers, Time, epe, f1 = test(args, TestImgLoader, model, device, cal_pose=True)
            print(f"Mean trans error {np.mean(err_t_list):.5f}  Mean rotation error {np.mean(err_r_list):.5f}")
            print(f"Median trans error {np.median(err_t_list):.5f}  Median rotation error {np.median(err_r_list):.5f}")
            print(f"epe {epe:.5f}  Mean {Time / len(TestImgLoader):.5f} per frame")
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
    if not os.path.exists(f'./checkpoints/{datetime}'):
        os.mkdir(f'./checkpoints/{datetime}')

    starting_epoch = args.starting_epoch
    if starting_epoch > 0:
        for i in range(starting_epoch * len(TrainImgLoader)):
            scaler.unscale_(optimizer)
            scaler.step(optimizer)
            scheduler.step()
            scaler.update()
        logger.total_steps = starting_epoch * len(TrainImgLoader)

    min_val_err = 9999.
    for epoch in range(starting_epoch, args.epochs):
        train(args, TrainImgLoader, model, depth_encoder, optimizer, scheduler, scaler, logger, device, epoch)

        if epoch % args.evaluate_interval == 0:
            epe, f1 = test(args, TestImgLoader, model, device)
            print("Validation M3ED: %f, %f" % (epe, f1))

            results = {'m3ed-epe': epe, 'm3ed-f1': f1}
            logger.write_dict(results)

            torch.save(model.state_dict(), f"./checkpoints/{datetime}/checkpoint.pth")

            if epe < min_val_err:
                min_val_err = epe
                torch.save(model.state_dict(), f'./checkpoints/{datetime}/best_model.pth')