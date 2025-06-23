import csv
import open3d as o3
import os
import time
import math
import numpy as np
import torch
import visibility as visibility
import argparse
import cv2
import sys
import h5py

from core.backbone import Backbone_Event, Backbone_Edge_FF
from core.utils_point import quaternion_from_matrix, to_rotation_matrix, overlay_imgs
from core.utils import count_parameters
from core.flow2pose import Flow2Pose
from core.quaternion_distances import quaternion_distance
from core.data_preprocess import Data_preprocess

def get_calib_m3ed(sequence, camera="left"):
    if camera == "left":
        if sequence == "falcon_indoor_flight_1":
            return torch.tensor([1034.86278431, 1033.47800271, 629.70125104, 357.60071019])
        elif sequence in ["falcon_indoor_flight_2", "falcon_indoor_flight_3"]:
            return torch.tensor([1034.39696079, 1034.73607278, 636.27410756, 364.73952748])
        elif sequence in ['falcon_outdoor_day_penno_parking_1', 'falcon_outdoor_day_penno_parking_2']:
            return torch.tensor([1033.22781771, 1032.05548869, 631.84536312, 360.7175681])
        elif sequence in ['falcon_forest_into_forest_1', 'falcon_forest_into_forest_2', 'falcon_forest_into_forest_4']:
            return torch.tensor([1033.22946467, 1032.37940162 , 637.14899526,  359.16366857])        
        elif sequence in ['spot_forest_road_1', 'spot_forest_road_3']:
            return torch.tensor([1032.96138833, 1032.8263147,   632.38290823, 368.57212974])
        elif sequence in ['car_urban_day_penno_big_loop', 'car_urban_day_penno_small_loop']:
            return torch.tensor([1031.88399229, 1031.48192315 , 634.29808475 , 366.39105342])
        elif sequence in ['car_forest_into_ponds_long', 'car_forest_into_ponds_short']:
            return torch.tensor([1031.28303548, 1031.36293954,  635.73771622,  366.81925393])
        else:
            raise TypeError("Sequence Not Available")
    else:
        if sequence == "falcon_indoor_flight_1":
            return torch.tensor([1035.07712803, 1034.76733944,  632.04513322,  359.48878546])
        elif sequence in ["falcon_indoor_flight_2", "falcon_indoor_flight_3"]:
            return torch.tensor([1034.61302587, 1034.83604567,  638.12992827,  366.88002829])
        elif sequence in ['falcon_outdoor_day_penno_parking_1', 'falcon_outdoor_day_penno_parking_2']:
            return torch.tensor([1033.22781771, 1032.05548869, 631.84536312, 360.7175681])
        elif sequence in ['falcon_forest_into_forest_1', 'falcon_forest_into_forest_2', 'falcon_forest_into_forest_4']:
            return torch.tensor([1032.7719854,  1031.93236836,  639.10742672,  362.60248038]) 
        elif sequence in ['spot_forest_road_1', 'spot_forest_road_3']:
            return torch.tensor([1032.26867081, 1032.57448492,  632.85489859,  370.0490076 ])
        elif sequence in ['car_urban_day_penno_big_loop', 'car_urban_day_penno_small_loop']:
            return torch.tensor([1031.36879978 ,1031.06491961 , 634.87768084 , 367.62546105])
        elif sequence in ['car_forest_into_ponds_long', 'car_forest_into_ponds_short']:
            return torch.tensor([1030.62646968, 1030.98305576,  636.68872098,  368.70448786])
        else:
            raise TypeError("Sequence Not Available")

def get_prophesee_left_T_lidar_m3ed(sequence):
    if sequence == "falcon_indoor_flight_1":
        return torch.tensor([[ 0.0076, -0.9996, -0.0259,  0.0482],
                             [-0.2454,  0.0233, -0.9692, -0.2190],
                             [ 0.9694,  0.0137, -0.2451, -0.2298],
                             [ 0.0000,  0.0000,  0.0000,  1.0000]])
    elif sequence in ["falcon_indoor_flight_2", "falcon_indoor_flight_3"]:
        return torch.tensor([[-1.0447e-04, -9.9970e-01, -2.4339e-02,  5.0678e-02],
                             [-2.4484e-01,  2.3624e-02, -9.6928e-01, -2.1931e-01],
                             [ 9.6956e-01,  5.8579e-03, -2.4477e-01, -2.2846e-01],
                             [ 0.0000e+00,  0.0000e+00,  0.0000e+00,  1.0000e+00]])
    elif sequence in ['falcon_outdoor_day_penno_parking_1', 'falcon_outdoor_day_penno_parking_2']:
        return torch.tensor([[ 0.0045, -0.9996, -0.0265,  0.0488],
                             [-0.2653,  0.0243, -0.9638, -0.2194],
                             [ 0.9641,  0.0114, -0.2651, -0.2299],
                             [ 0.0000,  0.0000,  0.0000,  1.0000]])
    elif sequence in ['spot_forest_road_1', 'spot_forest_road_3']:
        return torch.tensor([[ 0.0050, -1.0000, -0.0020,  0.0596],
                             [ 0.0061,  0.0020, -1.0000, -0.1898],
                             [ 1.0000,  0.0050,  0.0061, -0.1581],
                             [ 0.0000,  0.0000,  0.0000,  1.0000]])
    elif sequence in ['car_urban_day_penno_big_loop', 'car_urban_day_penno_small_loop']:
        return torch.tensor([[ 4.3559e-03, -9.9987e-01, -1.5325e-02,  5.9558e-02],
                             [ 3.5460e-04,  1.5326e-02, -9.9988e-01, -1.8978e-01],
                             [ 9.9999e-01,  4.3499e-03,  4.2131e-04, -1.5842e-01],
                             [ 0.0000e+00,  0.0000e+00,  0.0000e+00,  1.0000e+00]])
    elif sequence in ['car_forest_into_ponds_long', 'car_forest_into_ponds_short']:
        return torch.tensor([[ 3.1243e-03, -9.9988e-01, -1.5221e-02,  5.9864e-02],
                             [-1.6407e-04,  1.5221e-02, -9.9988e-01, -1.8967e-01],
                             [ 1.0000e+00,  3.1264e-03, -1.1650e-04, -1.5859e-01],
                             [ 0.0000e+00,  0.0000e+00,  0.0000e+00,  1.0000e+00]])
    else:
        raise TypeError("Sequence Not Available")

def load_map(map_file, device):
    downpcd = o3.io.read_point_cloud(map_file)
    voxelized = torch.tensor(downpcd.points, dtype=torch.float)
    voxelized = torch.cat((voxelized, torch.ones([voxelized.shape[0], 1], dtype=torch.float)), 1)
    voxelized = voxelized.t()
    voxelized = voxelized.to(device)
    return voxelized

def crop_local_map(PC_map, pose, prophesee_left_T_lidar, max_depth):
    local_map = PC_map.clone()
    local_map = torch.mm(pose, local_map)
    indexes = local_map[0, :] > -1.
    if max_depth == 10.:
        indexes = indexes & (local_map[0, :] < 10.)
        indexes = indexes & (local_map[1, :] > -5.)
        indexes = indexes & (local_map[1, :] < 5.)
    elif max_depth == 100.:
        indexes = indexes & (local_map[0, :] < 100.)
        indexes = indexes & (local_map[1, :] > -25.)
        indexes = indexes & (local_map[1, :] < 25.)
    local_map = local_map[:, indexes]

    local_map = torch.mm(prophesee_left_T_lidar, local_map)

    return local_map

def main(args):
    print(args)

    device = torch.device(f"cuda:{args.gpus[0]}" if torch.cuda.is_available() else "cpu")
    calib = get_calib_m3ed(args.test_sequence)
    calib = calib.to(device)
    calib /= 2.
    prophesee_left_T_lidar = get_prophesee_left_T_lidar_m3ed(args.test_sequence)
    prophesee_left_T_lidar = prophesee_left_T_lidar.float().to(device)

    maps_file = os.path.join(args.data_folder, "original", args.platform, args.test_sequence, args.test_sequence+"_global.pcd")
    if os.path.exists(maps_file):
        print(f'load pointclouds from {maps_file}')
        vox_map = load_map(maps_file, device)
        print(f'load pointclouds finished! {vox_map.shape[1]} points')
    else:
        raise "LiDAR map does not exist"

    # load GT poses
    print('load ground truth poses')
    pose_path = os.path.join(args.data_folder, "original", args.platform, args.test_sequence, args.test_sequence+"_pose_gt.h5")
    poses = h5py.File(pose_path,'r')
    Ln_T_L0 = poses['Ln_T_L0']
    Ln_T_L0 = torch.tensor(Ln_T_L0)
    print('sequence length = ', len(Ln_T_L0))

    if args.backbone == "baseline":
        model = torch.nn.DataParallel(Backbone_Event(args), device_ids=args.gpus)
    elif args.backbone == "edge":
        model = torch.nn.DataParallel(Backbone_Edge_FF(args), device_ids=args.gpus)
    else:
        raise "Specified backbone doesn't exist"
    print("Parameter Count: %d" % count_parameters(model))
    model.load_state_dict(torch.load(args.load_checkpoints))
    model.to(device)
    model.eval()

    if args.save_log:
        if not os.path.exists('./logs'):
            os.makedirs('./logs')
        log_file = f'./logs/{args.backbone}_M3ED_{args.test_sequence}.csv'
        log_file_f = open(log_file, 'w')
        log_file = csv.writer(log_file_f)
        header = [f'timestamp', f'x', f'y', f'z',
                  f'qx', f'qy', f'qz', f'qw']
        log_file.writerow(header)
    
    est_rot = []
    est_trans = []
    err_t_list = []
    err_r_list = []
    print('Start tracking using EVLoc...')
    if args.test_sequence in ['falcon_indoor_flight_1', 'falcon_indoor_flight_2', 'falcon_indoor_flight_3']:
        left = 70
        right = 70
    elif args.test_sequence in ['falcon_outdoor_day_penno_parking_1', 'falcon_outdoor_day_penno_parking_2']:
        left = 100
        right = 100
    elif args.test_sequence in ['car_urban_day_penno_big_loop']:
        left = 90
        right = 48
    elif args.test_sequence in ['car_urban_day_penno_small_loop']:
        left = 70
        right = 20
    elif args.test_sequence in ['falcon_forest_into_forest_1']:
        left = 440
        right = 480
    elif args.test_sequence in ['falcon_forest_into_forest_2']:
        left = 240
        right = 230
    elif args.test_sequence in ['falcon_forest_into_forest_4']:
        left =  100
        right = 380
    elif args.test_sequence in ['spot_forest_road_1']:
        left =  35
        right = 360
    elif args.test_sequence in ['spot_forest_road_3']:
        left = 115
        right=1277
    elif args.test_sequence in ['car_forest_into_ponds_long']:
        left = 200
        right = 30
    elif args.test_sequence in ['car_forest_into_ponds_short']:
        left = 40
        right = 180
    else:
        raise "Sequence doesn't exist in the list"
    initial_T = Ln_T_L0[left+1][:3, 3]
    initial_R = quaternion_from_matrix(Ln_T_L0[left+1])
    est_rot.append(initial_R.to(device))
    est_trans.append(initial_T.to(device))
    data_generate = Data_preprocess([calib], args.occlusion_threshold, args.occlusion_kernel)
    end = time.time()
    if args.velocity:
        velocity = torch.tensor([[1., 0., 0., 0.],                                                                                                                                         
                                 [0., 1., 0., 0.],
                                 [0., 0., 1., 0.],                                                                                                                                         
                                 [0., 0., 0., 1.]], device=device)
    for idx in range(left, len(Ln_T_L0)-right):
        initial_R = est_rot[idx-left].to(device)
        initial_T = est_trans[idx-left].to(device)

        initial_RT = to_rotation_matrix(initial_R, initial_T)
        initial_RT = initial_RT.to(device)
        if args.velocity:
            # # add the velocity
            initial_RT = torch.mm(velocity, initial_RT)

        event_frame = np.load(os.path.join(args.data_folder, "generated", args.platform, args.test_sequence, f"event_frames_{args.method}", "left", f"event_frame_{idx:05d}.npy"))
        event_frame = torch.tensor(event_frame).permute(2, 0, 1)
        event_frame[event_frame<0] = 0
        event_frame /= torch.max(event_frame)       

        local_map = crop_local_map(vox_map, initial_RT, prophesee_left_T_lidar, max_depth=args.max_depth)   # 4xN
        # # M3ED Half Resolution
        crop_h, crop_w, crop_x, crop_y = 288, 512, 36, 64
        event_input, lidar_input, _, _ = data_generate.push_input([event_frame], [local_map], None, None, device, split='test', MAX_DEPTH=args.max_depth, h=crop_h, w=crop_w)

        original_overlay = overlay_imgs(event_input[0, :, :, :], lidar_input[0, 0, :, :].clone()*0)
        cv2.imwrite(f"./visualization/pose_tracking/initial/{idx:05d}_img.png", original_overlay)
        original_overlay = overlay_imgs(event_input[0, :, :, :]*0, lidar_input[0, 0, :, :].clone())
        cv2.imwrite(f"./visualization/pose_tracking/initial/{idx:05d}_depth.png", original_overlay)

        # run model
        if args.backbone == "baseline":
            _, flow_up = model(lidar_input, event_input, iters=args.iters, test_mode=True)
        elif args.backbone == "edge":
            _, flow_up, _ = model(lidar_input, event_input, iters=args.iters, test_mode=True)

        # update current pose
        R_pred, T_pred, _, _ = Flow2Pose(flow_up, lidar_input, calib.unsqueeze(0), MAX_DEPTH=args.max_depth, x=crop_x, y=crop_y, h=crop_h, w=crop_w, flag=False)
        RT_pred = to_rotation_matrix(R_pred, T_pred)
        RT_pred = RT_pred.to(device)

        RT_new = torch.mm(prophesee_left_T_lidar.inverse(), torch.mm(RT_pred, torch.mm(prophesee_left_T_lidar, initial_RT)))

        # calculate RTE RRE
        predicted_R = quaternion_from_matrix(RT_new)
        predicted_T = RT_new[:3, 3]
        GT_T = Ln_T_L0[idx+1][:3, 3]
        GT_R = quaternion_from_matrix(Ln_T_L0[idx+1])
        err_r = quaternion_distance(predicted_R.unsqueeze(0).to(device),
                                    GT_R.unsqueeze(0).to(device), device=device)
        err_r = err_r * 180. / math.pi
        err_t = torch.norm(predicted_T.to(device) - GT_T.to(device)) * 100.
        err_r_list.append(err_r.item())
        err_t_list.append(err_t.item())

        print(f"{idx:05d}: {np.mean(err_t_list):.5f} {np.mean(err_r_list):.5f} {np.std(err_t_list):.5f} "
              f"{np.std(err_r_list):.5f} {(time.time()-end)/(idx+1):.5f}")
        
        # update pose list
        est_rot[idx-left] = predicted_R
        est_trans[idx-left] = predicted_T
        if args.velocity:
            if idx > 0:
                cur_R = est_rot[idx-left].to(device)
                cur_T = est_trans[idx-left].to(device)
                cur_RT = to_rotation_matrix(cur_R, cur_T)
                cur_RT = cur_RT.to(device)
                pre_R = est_rot[idx-left-1].to(device)
                pre_T = est_trans[idx-left-1].to(device)
                pre_RT = to_rotation_matrix(pre_R, pre_T)
                pre_RT = pre_RT.to(device)
                new_velocity = torch.mm(pre_RT.inverse(), cur_RT)
                new_velocity_r = quaternion_from_matrix(new_velocity)
                new_velocity_t = new_velocity[:3, 3]
                velocity_r = quaternion_distance(new_velocity_r.unsqueeze(0).to(device), torch.tensor([1., 0., 0., 0.]).unsqueeze(0).to(device), device=device)
                velocity_r = velocity_r * 180. / math.pi
                velocity_t = torch.norm(new_velocity_t.to(device) - torch.tensor([0., 0., 0.]).to(device)) * 100.
                if velocity_r.item() < 3. and velocity_t.item() < 30.:
                    velocity = new_velocity
        est_rot.append(predicted_R)
        est_trans.append(predicted_T)

        if args.save_log:
            # predicted_T = predicted_T.cpu().numpy()
            # predicted_R = predicted_R.cpu().numpy()
            # save gt poses
            predicted_T = GT_T.cpu().numpy()
            predicted_R = GT_R.cpu().numpy()

            log_string = [f"{idx:05d}", str(predicted_T[0]), str(predicted_T[1]), str(predicted_T[2]),
                          str(predicted_R[1]), str(predicted_R[2]), str(predicted_R[3]), str(predicted_R[0])]
            log_file.writerow(log_string)

        _, lidar_input, _, _ = data_generate.push_input([event_frame], [local_map], [T_pred], [R_pred], device, MAX_DEPTH=args.max_depth, split='test', h=crop_h, w=crop_w)
        original_overlay = overlay_imgs(event_input[0, :, :, :], lidar_input[0, 0, :, :])
        cv2.imwrite(f"./visualization/pose_tracking/update/{idx:05d}.png", original_overlay)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_folder', type=str, metavar='DIR',
                        default='/media/eason/Backup/Datasets/Event_Datasets/M3ED',
                        help='path to dataset')
    parser.add_argument('--platform', type=str, default='Falcon')
    parser.add_argument('--test_sequence', type=str, default='falcon_indoor_flight_3')
    parser.add_argument('--occlusion_kernel', type=float, default=5.)
    parser.add_argument('--occlusion_threshold', type=float, default=3.)
    parser.add_argument('--iters', type=int, default=24)
    parser.add_argument('--backbone', type=str, default='baseline')
    parser.add_argument('--method', type=str, default='ours_denoise_stc_trail_pre_100000_half')
    parser.add_argument('--load_checkpoints', type=str)
    parser.add_argument('--max_depth', type=float, default=10.)
    parser.add_argument('--mixed_precision', action='store_true', help='use mixed precision')
    parser.add_argument('--gpus', type=int, nargs='+', default=[0])
    parser.add_argument('--save_log', action='store_true')
    parser.add_argument('--velocity', action='store_true')
    args = parser.parse_args()

    main(args)