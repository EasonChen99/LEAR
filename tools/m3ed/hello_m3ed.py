import os
import sys

sys.path.append("/usr/lib/python3/dist-packages/")
from metavision_sdk_cv import ActivityNoiseFilterAlgorithm, TrailFilterAlgorithm, SpatioTemporalContrastAlgorithm

import h5py
import numpy as np
import cv2
import open3d as o3
import torch
from tqdm import tqdm
from utils import load_data, load_map, pc_visualize, depth_generation, find_near_index
from utils_point import invert_pose, to_rotation_matrix, quaternion_from_matrix, overlay_imgs
from skimage.draw import line as line_function
import matplotlib.pyplot as plt
from matplotlib import colormaps

import argparse

if __name__ == '__main__':
    args = argparse.ArgumentParser()
    args.add_argument("--dataset", 
                    default="/media/eason/Backup/Datasets/M3ED/original/Falcon",
                    help="Root path to the dataset", 
                    type=str)
    args.add_argument("--sequence",
                    default="falcon_outdoor_day_fast_flight_2",
                    help="Sequence name for processing",
                    type=str)
    args.add_argument("--method",
                    default="TS",
                    help="Event representation method",
                    type=str)
    args.add_argument("--time_window",
                    default=100000,
                    type=int)    
    args.add_argument("--idx",
                    default=200,
                    type=int)
    args.add_argument("--half_resolution",
                      action='store_true')
    args = args.parse_args()

    device = torch.device(f"cuda:0" if torch.cuda.is_available() else "cpu")
    
    def calculate_patch_variance(image, patch_size_h, patch_size_w):
        # Get the dimensions of the image
        h, w = image.shape
        
        # Check that the image dimensions are divisible by the patch size
        if h % patch_size_h != 0 or w % patch_size_w != 0:
            raise ValueError("Image dimensions must be divisible by patch size")

        # Reshape the image to have patches of size patch_size x patch_size
        patches = image.reshape(h // patch_size_h, patch_size_h, w // patch_size_w, patch_size_w)
        patches = patches.swapaxes(1, 2).reshape(-1, patch_size_h, patch_size_w)
        
        # Calculate the variance for each patch
        variances = np.var(patches, axis=(1, 2))
        
        # Reshape the variances back to the grid of patches
        variance_map = variances.reshape(h // patch_size_h, w // patch_size_w)
        
        return variance_map
    
    def patch_median_blur(image, variance, patch_size_h, patch_size_w, blur_kernel_size=3):

        for i in range(image.shape[0] // patch_size_h):
            for j in range(image.shape[1] // patch_size_w):
                # Extract patch
                y_start, y_end = i * patch_size_h, (i + 1) * patch_size_h
                x_start, x_end = j * patch_size_w, (j + 1) * patch_size_w
                patch = image[y_start:y_end, x_start:x_end, :]
                
                # Apply median blur if variance is above the threshold
                if variance[i, j] < np.median(variance):
                    blurred_patch = cv2.medianBlur(patch, blur_kernel_size)
                    image[y_start:y_end, x_start:x_end, :] = blurred_patch

        return image        


    root = os.path.join(args.dataset, args.sequence)
    name = args.sequence
    h5_file = f"{name}_data.h5"
    pose_file = f"{name}_pose_gt.h5"
    pc_file = f"{name}_global.pcd"

    data_path = os.path.join(root, h5_file)
    pose_path = os.path.join(root, pose_file)
    pc_path = os.path.join(root, pc_file)

    data = h5py.File(data_path,'r')
    rgb_data = load_data(data, sensor='ovc', camera='left') 
    rgbs = rgb_data['img'][:]
    rgb_ts = rgb_data['ts'][:]
    event_data = load_data(data, sensor='prophesee', camera='left')
    # # T_left_to_right
    # print(np.linalg.inv(np.asarray(data["prophesee/right/calib/T_to_prophesee_left"])))
    # print(np.asarray(data["prophesee/left/calib/intrinsics"]))
    # print(np.asarray(data["prophesee/right/calib/intrinsics"]))
    # sys.exit()
    T_to_prophesee_left = torch.tensor(event_data['T_to_prophesee_left'])
    # # pose load
    poses = h5py.File(pose_path,'r')
    Cn_T_C0 = poses['Cn_T_C0']                                                  # 570 4 4
    Ln_T_L0 = poses['Ln_T_L0']                                                  # 570 4 4
    pose_ts = poses['ts']                                                       # 570
    ts_map_prophesee_left = poses['ts_map_prophesee_left']                      # 570 index

    if not args.half_resolution:
        k = 1
    else:
        k = 2
    
    if not args.half_resolution:
        save_file = f'./event_frame_{args.idx:05d}_{args.method}_{args.time_window}'
    else:
        save_file = f'./event_frame_{args.idx:05d}_{args.method}_{args.time_window}_half'
    

    idx_cur = int(ts_map_prophesee_left[args.idx+1])
    t_ref = event_data['t'][idx_cur]
    idx_start, idx_cur, idx_end = find_near_index(event_data['t'][idx_cur], event_data['t'], time_window=args.time_window * 2)
    rows, cols = event_data['resolution'][1]//k, event_data['resolution'][0]//k
    event_time_image = np.zeros((rows, cols, 2), dtype=np.float32)

    print(idx_start, idx_cur, idx_end)
    print(event_data['t'][idx_start], event_data['t'][idx_cur], event_data['t'][idx_end])
    
    if args.method == "TS":
        for id in tqdm(range(idx_cur-idx_start)):
            idx = int(id + idx_start)
            y, x = event_data['y'][idx]//k, event_data['x'][idx]//k
            if event_data['p'][idx] > 0:
                event_time_image[y, x, 0] = event_data['t'][idx]
            else:
                event_time_image[y, x, 1] = event_data['t'][idx]
        event_time_image[event_time_image > 0] -= event_data['t'][idx_start]
    # elif args.method == "Event_Frame":
    #     for id in tqdm(range(idx_cur-idx_start)):
    #         idx = int(id + idx_start)
    #         y, x = event_data['y'][idx]//k, event_data['x'][idx]//k
    #         if event_data['p'][idx] > 0:
    #             event_time_image[y, x, 0] += 1 
    #         else:
    #             event_time_image[y, x, 1] += 1
    # elif args.method == "TS_Denoise_Activity":
    #     activity_time_ths = args.time_window  # Length of the time window for activity filtering (in us)
    #     activity_filter = ActivityNoiseFilterAlgorithm(1280, 720, activity_time_ths)
    #     events_buf = ActivityNoiseFilterAlgorithm.get_empty_output_buffer()

    #     new_dtype = np.dtype({
    #                     'names': ['x', 'y', 'p', 't'],                # Field names
    #                     'formats': ['<u2', '<u2', '<i2', '<i8'],      # Data types for each field
    #                     'offsets': [0, 2, 4, 8],                      # Byte offsets for each field
    #                     'itemsize': 16                                # Total size of each entry
    #                     })
    #     events = np.zeros(idx_cur-idx_start, dtype=new_dtype)
    #     events['x'] = event_data['x'][idx_start:idx_cur]
    #     events['y'] = event_data['y'][idx_start:idx_cur]
    #     events['p'] = event_data['p'][idx_start:idx_cur]
    #     events['t'] = event_data['t'][idx_start:idx_cur]
    #     activity_filter.process_events(events, events_buf)
    #     events_buf = np.asarray(events_buf)

    #     for idx in tqdm(range(len(events_buf))):
    #         y, x = events_buf['y'][idx]//k, events_buf['x'][idx]//k
    #         if events_buf['p'][idx] > 0:
    #             event_time_image[y, x, 0] = events_buf['t'][idx]
    #         else:
    #             event_time_image[y, x, 1] = events_buf['t'][idx]
    #     event_time_image[event_time_image > 0] -= events_buf['t'][0]
    # elif args.method == "TS_Denoise_Trail":
    #     trail_filter_ths = args.time_window  # Length of the time window for activity filtering (in us)
    #     trail_filter = TrailFilterAlgorithm(1280, 720, trail_filter_ths)
    #     events_buf = ActivityNoiseFilterAlgorithm.get_empty_output_buffer()

    #     new_dtype = np.dtype({
    #                     'names': ['x', 'y', 'p', 't'],                # Field names
    #                     'formats': ['<u2', '<u2', '<i2', '<i8'],      # Data types for each field
    #                     'offsets': [0, 2, 4, 8],                      # Byte offsets for each field
    #                     'itemsize': 16                                # Total size of each entry
    #                     })
    #     events = np.zeros(idx_cur-idx_start, dtype=new_dtype)
    #     events['x'] = event_data['x'][idx_start:idx_cur]
    #     events['y'] = event_data['y'][idx_start:idx_cur]
    #     events['p'] = event_data['p'][idx_start:idx_cur]
    #     events['t'] = event_data['t'][idx_start:idx_cur]
    #     trail_filter.process_events(events, events_buf)
    #     events_buf = np.asarray(events_buf)

    #     for idx in tqdm(range(len(events_buf))):
    #         y, x = events_buf['y'][idx]//k, events_buf['x'][idx]//k
    #         if events_buf['p'][idx] > 0:
    #             event_time_image[y, x, 0] = events_buf['t'][idx]
    #         else:
    #             event_time_image[y, x, 1] = events_buf['t'][idx]
    #     event_time_image[event_time_image > 0] -= events_buf['t'][0]
    # elif args.method == "TS_Denoise_STC":
    #     stc_filter_ths = args.time_window  # Length of the time window for filtering (in us)
    #     stc_cut_trail = False  # If true, after an event goes through, it removes all events until change of polarity
    #     stc_filter = SpatioTemporalContrastAlgorithm(1280, 720, stc_filter_ths, stc_cut_trail)
    #     events_buf = ActivityNoiseFilterAlgorithm.get_empty_output_buffer()

    #     new_dtype = np.dtype({
    #                     'names': ['x', 'y', 'p', 't'],                # Field names
    #                     'formats': ['<u2', '<u2', '<i2', '<i8'],      # Data types for each field
    #                     'offsets': [0, 2, 4, 8],                      # Byte offsets for each field
    #                     'itemsize': 16                                # Total size of each entry
    #                     })
    #     events = np.zeros(idx_cur-idx_start, dtype=new_dtype)
    #     events['x'] = event_data['x'][idx_start:idx_cur]
    #     events['y'] = event_data['y'][idx_start:idx_cur]
    #     events['p'] = event_data['p'][idx_start:idx_cur]
    #     events['t'] = event_data['t'][idx_start:idx_cur]
    #     stc_filter.process_events(events, events_buf)
    #     events_buf = np.asarray(events_buf)

    #     for idx in tqdm(range(len(events_buf))):
    #         y, x = events_buf['y'][idx]//k, events_buf['x'][idx]//k
    #         if events_buf['p'][idx] > 0:
    #             event_time_image[y, x, 0] = events_buf['t'][idx]
    #         else:
    #             event_time_image[y, x, 1] = events_buf['t'][idx]
    #     event_time_image[event_time_image > 0] -= events_buf['t'][0]
    # elif args.method == "TS_Denoise_STC_Trail":
    #     trail_filter_ths = args.time_window // 10  # Length of the time window for activity filtering (in us)
    #     trail_filter = TrailFilterAlgorithm(1280, 720, trail_filter_ths)

    #     stc_filter_ths = args.time_window // 10  # Length of the time window for filtering (in us)
    #     stc_cut_trail = True  # If true, after an event goes through, it removes all events until change of polarity
    #     stc_filter = SpatioTemporalContrastAlgorithm(1280, 720, stc_filter_ths, stc_cut_trail)

    #     events_buf = ActivityNoiseFilterAlgorithm.get_empty_output_buffer()

    #     new_dtype = np.dtype({
    #                     'names': ['x', 'y', 'p', 't'],                # Field names
    #                     'formats': ['<u2', '<u2', '<i2', '<i8'],      # Data types for each field
    #                     'offsets': [0, 2, 4, 8],                      # Byte offsets for each field
    #                     'itemsize': 16                                # Total size of each entry
    #                     })
    #     events = np.zeros(idx_cur-idx_start, dtype=new_dtype)
    #     events['x'] = event_data['x'][idx_start:idx_cur]
    #     events['y'] = event_data['y'][idx_start:idx_cur]
    #     events['p'] = event_data['p'][idx_start:idx_cur]
    #     events['t'] = event_data['t'][idx_start:idx_cur]


    #     stc_filter.process_events(events, events_buf)
    #     events_buf = np.asarray(events_buf)
    #     events_buf_ = ActivityNoiseFilterAlgorithm.get_empty_output_buffer()
    #     trail_filter.process_events(events_buf, events_buf_)
    #     events_buf = np.asarray(events_buf_)

    #     for idx in tqdm(range(len(events_buf))):
    #         y, x = events_buf['y'][idx]//k, events_buf['x'][idx]//k
    #         if events_buf['p'][idx] > 0:
    #             event_time_image[y, x, 0] = events_buf['t'][idx]
    #         else:
    #             event_time_image[y, x, 1] = events_buf['t'][idx]
    #     event_time_image[event_time_image > 0] -= events_buf['t'][0]
    # elif args.method == "SITS":
    #     r=6
    #     for id in tqdm(range(idx_cur-idx_start)):
    #         idx = int(id + idx_start)
    #         y, x = event_data['y'][idx]//k, event_data['x'][idx]//k
    #         if event_data['p'][idx] > 0:
    #             patch = event_time_image[max(0, y-r):y+r+1, max(0, x-r):x+r+1, 0]
    #             patch = np.where(patch>=event_time_image[y,x,0], patch-1, patch)
    #             event_time_image[max(0, y-r):y+r+1, max(0, x-r):x+r+1, 0] = patch
    #             event_time_image[y, x, 0] = (2 * r + 1)**2
    #         else:
    #             patch = event_time_image[max(0, y-r):y+r+1, max(0, x-r):x+r+1, 1]
    #             patch = np.where(patch>=event_time_image[y,x,1], patch-1, patch)
    #             event_time_image[max(0, y-r):y+r+1, max(0, x-r):x+r+1, 1] = patch
    #             event_time_image[y, x, 1] = (2 * r + 1)**2
    #     event_time_image[event_time_image < 0] = 0
    # elif args.method == "CM":
    #     r = 6
    #     threshold = 0.32
    #     for id in tqdm(range(idx_cur-idx_start)):
    #         idx = int(id + idx_start)
    #         y, x = event_data['y'][idx]//k, event_data['x'][idx]//k
    #         if event_data['p'][idx] > 0:
    #             patch = event_time_image[max(0, y-r):y+r+1, max(0, x-r):x+r+1, 0]
    #             patch_valid = patch[patch>0]
    #             if len(patch_valid) <= 1:
    #                 contrast = 0.
    #             else:
    #                 patch_valid = (patch_valid - np.min(patch_valid))/(np.max(patch_valid) - np.min(patch_valid))
    #                 contrast = np.sqrt(np.var(patch_valid))
    #             if contrast>threshold:
    #                 continue
    #             patch = np.where(patch>0, patch-(event_data['t'][idx]-patch)/10., patch)
    #             patch[patch<0] = 0
    #             event_time_image[max(0, y-r):y+r+1, max(0, x-r):x+r+1, 0] = patch
    #             event_time_image[y, x, 0] = event_data['t'][idx]
    #         else:
    #             patch = event_time_image[max(0, y-r):y+r+1, max(0, x-r):x+r+1, 1]
    #             patch_valid = patch[patch>0]
    #             if len(patch_valid) <= 1:
    #                 contrast = 0.
    #             else:
    #                 patch_valid = (patch_valid - np.min(patch_valid))/(np.max(patch_valid) - np.min(patch_valid))
    #                 contrast = np.sqrt(np.var(patch_valid))
    #             if contrast>threshold:
    #                 continue
    #             patch = np.where(patch>0, patch-(event_data['t'][idx]-patch)/10., patch)
    #             patch[patch<0] = 0
    #             event_time_image[max(0, y-r):y+r+1, max(0, x-r):x+r+1, 1] = patch
    #             event_time_image[y, x, 1] = event_data['t'][idx]
    #     event_time_image[event_time_image > 0] -= event_data['t'][idx_start]
    #     event_time_image[event_time_image < 0] = 0
    # elif args.method == "Ours":
    #     r = 6
    #     for id in tqdm(range(idx_cur-idx_start)):
    #         idx = int(id + idx_start)
    #         y, x = event_data['y'][idx]//k, event_data['x'][idx]//k
    #         if event_data['p'][idx] > 0:
    #             patch = event_time_image[max(0, y-r):y+r+1, max(0, x-r):x+r+1, 0]
    #             patch = np.where(patch>0, patch-(event_data['t'][idx]-patch)/15., patch)
    #             patch[patch<0] = 0
    #             event_time_image[max(0, y-r):y+r+1, max(0, x-r):x+r+1, 0] = patch
    #             event_time_image[y, x, 0] = event_data['t'][idx]
    #         else:
    #             patch = event_time_image[max(0, y-r):y+r+1, max(0, x-r):x+r+1, 1]
    #             patch = np.where(patch>0, patch-(event_data['t'][idx]-patch)/15., patch)
    #             patch[patch<0] = 0
    #             event_time_image[max(0, y-r):y+r+1, max(0, x-r):x+r+1, 1] = patch
    #             event_time_image[y, x, 1] = event_data['t'][idx]
    #     event_time_image[event_time_image > 0] -= event_data['t'][idx_start]
    #     event_time_image[event_time_image < 0] = 0
    # elif args.method == "Ours_Denoise":
    #     r = 6
    #     B = 1 # 15
    #     R = 1
    #     threshold = 0.7 # 0.3
    #     total_range = np.arange(idx_start, idx_cur)
    #     subsequences = np.array_split(total_range, B)
    #     for subseq in subsequences:
    #         mask = np.zeros((rows, cols), dtype=bool)
    #         for idx in tqdm(subseq):
    #             y, x = event_data['y'][idx]//k, event_data['x'][idx]//k
    #             if event_data['p'][idx] > 0:
    #                 patch = event_time_image[max(0, y-r):y+r+1, max(0, x-r):x+r+1, 0]
    #                 patch = np.where(patch>0, patch-(event_data['t'][idx]-patch)/15., patch)
    #                 patch[patch<0] = 0
    #                 event_time_image[max(0, y-r):y+r+1, max(0, x-r):x+r+1, 0] = patch
    #                 event_time_image[y, x, 0] = event_data['t'][idx]
    #             else:
    #                 patch = event_time_image[max(0, y-r):y+r+1, max(0, x-r):x+r+1, 1]
    #                 patch = np.where(patch>0, patch-(event_data['t'][idx]-patch)/15., patch)
    #                 patch[patch<0] = 0
    #                 event_time_image[max(0, y-r):y+r+1, max(0, x-r):x+r+1, 1] = patch
    #                 event_time_image[y, x, 1] = event_data['t'][idx]
    #             patch = event_time_image[max(0, y-R):y+R+1, max(0, x-R):x+R+1, :]
    #             valid_count = ((patch[:, :, 0] > 0) | (patch[:, :, 1] > 0)).sum()
    #             if valid_count / (patch.shape[0]*patch.shape[1]) < threshold:
    #                 mask[y, x] = True
    #             else:
    #                 mask[y, x] = False
    #         event_time_image[mask] *= 0
    #     event_time_image[event_time_image > 0] -= event_data['t'][idx_start]
    #     event_time_image[event_time_image < 0] = 0
    # elif args.method == "Ours_Denoise_Activity":
    #     activity_time_ths = args.time_window // 100 # Length of the time window for activity filtering (in us)
    #     activity_filter = ActivityNoiseFilterAlgorithm(1280, 720, activity_time_ths)
    #     events_buf = ActivityNoiseFilterAlgorithm.get_empty_output_buffer()

    #     new_dtype = np.dtype({
    #                     'names': ['x', 'y', 'p', 't'],                # Field names
    #                     'formats': ['<u2', '<u2', '<i2', '<i8'],      # Data types for each field
    #                     'offsets': [0, 2, 4, 8],                      # Byte offsets for each field
    #                     'itemsize': 16                                # Total size of each entry
    #                     })
    #     events = np.zeros(idx_cur-idx_start, dtype=new_dtype)
    #     events['x'] = event_data['x'][idx_start:idx_cur]
    #     events['y'] = event_data['y'][idx_start:idx_cur]
    #     events['p'] = event_data['p'][idx_start:idx_cur]
    #     events['t'] = event_data['t'][idx_start:idx_cur]
    #     activity_filter.process_events(events, events_buf)
    #     events_buf = np.asarray(events_buf)

    #     r = 6
    #     for idx in tqdm(range(len(events_buf))):
    #         y, x = events_buf['y'][idx]//k, events_buf['x'][idx]//k
    #         if events_buf['p'][idx] > 0:
    #             patch = event_time_image[max(0, y-r):y+r+1, max(0, x-r):x+r+1, 0]
    #             patch = np.where(patch>0, patch-(events_buf['t'][idx]-patch)/15., patch)
    #             patch[patch<0] = 0
    #             event_time_image[max(0, y-r):y+r+1, max(0, x-r):x+r+1, 0] = patch
    #             event_time_image[y, x, 0] = events_buf['t'][idx]
    #         else:
    #             patch = event_time_image[max(0, y-r):y+r+1, max(0, x-r):x+r+1, 1]
    #             patch = np.where(patch>0, patch-(events_buf['t'][idx]-patch)/15., patch)
    #             patch[patch<0] = 0
    #             event_time_image[max(0, y-r):y+r+1, max(0, x-r):x+r+1, 1] = patch
    #             event_time_image[y, x, 1] = events_buf['t'][idx]
    #     event_time_image[event_time_image > 0] -= events_buf['t'][0]
    #     event_time_image[event_time_image < 0] = 0        
    # elif args.method == "Ours_Denoise_Activity_Variance":
        activity_time_ths = args.time_window // 100 # Length of the time window for activity filtering (in us)
        activity_filter = ActivityNoiseFilterAlgorithm(1280, 720, activity_time_ths)
        events_buf = ActivityNoiseFilterAlgorithm.get_empty_output_buffer()

        new_dtype = np.dtype({
                        'names': ['x', 'y', 'p', 't'],                # Field names
                        'formats': ['<u2', '<u2', '<i2', '<i8'],      # Data types for each field
                        'offsets': [0, 2, 4, 8],                      # Byte offsets for each field
                        'itemsize': 16                                # Total size of each entry
                        })
        events = np.zeros(idx_cur-idx_start, dtype=new_dtype)
        events['x'] = event_data['x'][idx_start:idx_cur]
        events['y'] = event_data['y'][idx_start:idx_cur]
        events['p'] = event_data['p'][idx_start:idx_cur]
        events['t'] = event_data['t'][idx_start:idx_cur]
        activity_filter.process_events(events, events_buf)
        events_buf = np.asarray(events_buf)

        r = 6
        B = 20
        patch_size_h, patch_size_w = 36, 64
        total_range = np.arange(0, events_buf.shape[0])
        subsequences = np.array_split(total_range, B)
        noise_mask = np.zeros((rows // patch_size_h, cols // patch_size_w), dtype=bool)
        for subseq_idx in tqdm(range(len(subsequences))):
            subseq = subsequences[subseq_idx]
            for idx in subseq:
                y, x = events_buf['y'][idx]//k, events_buf['x'][idx]//k
                if events_buf['p'][idx] > 0:
                    patch = event_time_image[max(0, y-r):y+r+1, max(0, x-r):x+r+1, 0]
                    patch = np.where(patch>0, patch-(events_buf['t'][idx]-patch)/15., patch)
                    patch[patch<0] = 0
                    event_time_image[max(0, y-r):y+r+1, max(0, x-r):x+r+1, 0] = patch
                    event_time_image[y, x, 0] = events_buf['t'][idx]
                else:
                    patch = event_time_image[max(0, y-r):y+r+1, max(0, x-r):x+r+1, 1]
                    patch = np.where(patch>0, patch-(events_buf['t'][idx]-patch)/15., patch)
                    patch[patch<0] = 0
                    event_time_image[max(0, y-r):y+r+1, max(0, x-r):x+r+1, 1] = patch
                    event_time_image[y, x, 1] = events_buf['t'][idx]
            valid_event = (np.sum(event_time_image, axis=2) > 0).astype(np.float32)
            variance = calculate_patch_variance(valid_event, patch_size_h=patch_size_h, patch_size_w=patch_size_w)
            event_time_image = patch_median_blur(event_time_image, variance, patch_size_h, patch_size_w, blur_kernel_size=3)
        event_time_image[event_time_image > 0] -= event_data['t'][idx_start]
        event_time_image[event_time_image < 0] = 0
    # elif args.method == "Ours_Denoise_Trail":
    #     trail_filter_ths = args.time_window  # Length of the time window for activity filtering (in us)
    #     trail_filter = TrailFilterAlgorithm(1280, 720, trail_filter_ths)
    #     events_buf = ActivityNoiseFilterAlgorithm.get_empty_output_buffer()

    #     new_dtype = np.dtype({
    #                     'names': ['x', 'y', 'p', 't'],                # Field names
    #                     'formats': ['<u2', '<u2', '<i2', '<i8'],      # Data types for each field
    #                     'offsets': [0, 2, 4, 8],                      # Byte offsets for each field
    #                     'itemsize': 16                                # Total size of each entry
    #                     })
    #     events = np.zeros(idx_cur-idx_start, dtype=new_dtype)
    #     events['x'] = event_data['x'][idx_start:idx_cur]
    #     events['y'] = event_data['y'][idx_start:idx_cur]
    #     events['p'] = event_data['p'][idx_start:idx_cur]
    #     events['t'] = event_data['t'][idx_start:idx_cur]
    #     trail_filter.process_events(events, events_buf)
    #     events_buf = np.asarray(events_buf)

    #     r = 6
    #     for idx in tqdm(range(len(events_buf))):
    #         y, x = events_buf['y'][idx]//k, events_buf['x'][idx]//k
    #         if events_buf['p'][idx] > 0:
    #             patch = event_time_image[max(0, y-r):y+r+1, max(0, x-r):x+r+1, 0]
    #             patch = np.where(patch>0, patch-(events_buf['t'][idx]-patch)/15., patch)
    #             patch[patch<0] = 0
    #             event_time_image[max(0, y-r):y+r+1, max(0, x-r):x+r+1, 0] = patch
    #             event_time_image[y, x, 0] = events_buf['t'][idx]
    #         else:
    #             patch = event_time_image[max(0, y-r):y+r+1, max(0, x-r):x+r+1, 1]
    #             patch = np.where(patch>0, patch-(events_buf['t'][idx]-patch)/15., patch)
    #             patch[patch<0] = 0
    #             event_time_image[max(0, y-r):y+r+1, max(0, x-r):x+r+1, 1] = patch
    #             event_time_image[y, x, 1] = events_buf['t'][idx]
    #     event_time_image[event_time_image > 0] -= events_buf['t'][0]
    #     event_time_image[event_time_image < 0] = 0        
    # elif args.method == "Ours_Denoise_STC":
    #     stc_filter_ths = args.time_window  # Length of the time window for filtering (in us)
    #     stc_cut_trail = False  # If true, after an event goes through, it removes all events until change of polarity
    #     stc_filter = SpatioTemporalContrastAlgorithm(1280, 720, stc_filter_ths, stc_cut_trail)
    #     events_buf = ActivityNoiseFilterAlgorithm.get_empty_output_buffer()

    #     new_dtype = np.dtype({
    #                     'names': ['x', 'y', 'p', 't'],                # Field names
    #                     'formats': ['<u2', '<u2', '<i2', '<i8'],      # Data types for each field
    #                     'offsets': [0, 2, 4, 8],                      # Byte offsets for each field
    #                     'itemsize': 16                                # Total size of each entry
    #                     })
    #     events = np.zeros(idx_cur-idx_start, dtype=new_dtype)
    #     events['x'] = event_data['x'][idx_start:idx_cur]
    #     events['y'] = event_data['y'][idx_start:idx_cur]
    #     events['p'] = event_data['p'][idx_start:idx_cur]
    #     events['t'] = event_data['t'][idx_start:idx_cur]
    #     stc_filter.process_events(events, events_buf)
    #     events_buf = np.asarray(events_buf)

    #     r = 6
    #     for idx in tqdm(range(len(events_buf))):
    #         y, x = events_buf['y'][idx]//k, events_buf['x'][idx]//k
    #         if events_buf['p'][idx] > 0:
    #             patch = event_time_image[max(0, y-r):y+r+1, max(0, x-r):x+r+1, 0]
    #             patch = np.where(patch>0, patch-(events_buf['t'][idx]-patch)/15., patch)
    #             patch[patch<0] = 0
    #             event_time_image[max(0, y-r):y+r+1, max(0, x-r):x+r+1, 0] = patch
    #             event_time_image[y, x, 0] = events_buf['t'][idx]
    #         else:
    #             patch = event_time_image[max(0, y-r):y+r+1, max(0, x-r):x+r+1, 1]
    #             patch = np.where(patch>0, patch-(events_buf['t'][idx]-patch)/15., patch)
    #             patch[patch<0] = 0
    #             event_time_image[max(0, y-r):y+r+1, max(0, x-r):x+r+1, 1] = patch
    #             event_time_image[y, x, 1] = events_buf['t'][idx]
    #     event_time_image[event_time_image > 0] -= events_buf['t'][0]
    #     event_time_image[event_time_image < 0] = 0
    elif args.method == "Ours_Denoise_STC_Trail":
        trail_filter_ths = args.time_window // 10  # Length of the time window for activity filtering (in us)
        trail_filter = TrailFilterAlgorithm(1280, 720, trail_filter_ths)

        stc_filter_ths = args.time_window  // 10   # Length of the time window for filtering (in us)
        stc_cut_trail = True  # If true, after an event goes through, it removes all events until change of polarity
        stc_filter = SpatioTemporalContrastAlgorithm(1280, 720, stc_filter_ths, stc_cut_trail)

        events_buf = ActivityNoiseFilterAlgorithm.get_empty_output_buffer()

        new_dtype = np.dtype({
                        'names': ['x', 'y', 'p', 't'],                # Field names
                        'formats': ['<u2', '<u2', '<i2', '<i8'],      # Data types for each field
                        'offsets': [0, 2, 4, 8],                      # Byte offsets for each field
                        'itemsize': 16                                # Total size of each entry
                        })
        events = np.zeros(idx_cur-idx_start, dtype=new_dtype)
        events['x'] = event_data['x'][idx_start:idx_cur]
        events['y'] = event_data['y'][idx_start:idx_cur]
        events['p'] = event_data['p'][idx_start:idx_cur]
        events['t'] = event_data['t'][idx_start:idx_cur]


        stc_filter.process_events(events, events_buf)
        events_buf = np.asarray(events_buf)
        events_buf_ = ActivityNoiseFilterAlgorithm.get_empty_output_buffer()
        trail_filter.process_events(events_buf, events_buf_)
        events_buf = np.asarray(events_buf_)

        r = 6
        for idx in tqdm(range(len(events_buf))):
            y, x = events_buf['y'][idx]//k, events_buf['x'][idx]//k
            if events_buf['p'][idx] > 0:
                patch = event_time_image[max(0, y-r):y+r+1, max(0, x-r):x+r+1, 0]
                patch = np.where(patch>0, patch-(events_buf['t'][idx]-patch)/15., patch)
                patch[patch<0] = 0
                event_time_image[max(0, y-r):y+r+1, max(0, x-r):x+r+1, 0] = patch
                event_time_image[y, x, 0] = events_buf['t'][idx]
            else:
                patch = event_time_image[max(0, y-r):y+r+1, max(0, x-r):x+r+1, 1]
                patch = np.where(patch>0, patch-(events_buf['t'][idx]-patch)/15., patch)
                patch[patch<0] = 0
                event_time_image[max(0, y-r):y+r+1, max(0, x-r):x+r+1, 1] = patch
                event_time_image[y, x, 1] = events_buf['t'][idx]
        event_time_image[event_time_image > 0] -= events_buf['t'][0]
        event_time_image[event_time_image < 0] = 0
    # elif args.method == "Ours_Denoise_STC_Trail_Activity":
    #     stc_filter_ths = args.time_window // 10 # Length of the time window for filtering (in us)
    #     stc_cut_trail = True  # If true, after an event goes through, it removes all events until change of polarity
    #     stc_filter = SpatioTemporalContrastAlgorithm(1280, 720, stc_filter_ths, stc_cut_trail)

    #     trail_filter_ths = args.time_window // 10 # Length of the time window for activity filtering (in us)
    #     trail_filter = TrailFilterAlgorithm(1280, 720, trail_filter_ths)

    #     activity_time_ths = args.time_window# Length of the time window for activity filtering (in us)
    #     activity_filter = ActivityNoiseFilterAlgorithm(1280, 720, activity_time_ths)

    #     events_buf = ActivityNoiseFilterAlgorithm.get_empty_output_buffer()

    #     new_dtype = np.dtype({
    #                     'names': ['x', 'y', 'p', 't'],                # Field names
    #                     'formats': ['<u2', '<u2', '<i2', '<i8'],      # Data types for each field
    #                     'offsets': [0, 2, 4, 8],                      # Byte offsets for each field
    #                     'itemsize': 16                                # Total size of each entry
    #                     })
    #     events = np.zeros(idx_cur-idx_start, dtype=new_dtype)
    #     events['x'] = event_data['x'][idx_start:idx_cur]
    #     events['y'] = event_data['y'][idx_start:idx_cur]
    #     events['p'] = event_data['p'][idx_start:idx_cur]
    #     events['t'] = event_data['t'][idx_start:idx_cur]


    #     stc_filter.process_events(events, events_buf)
    #     events_buf = np.asarray(events_buf)
    #     events_buf_ = ActivityNoiseFilterAlgorithm.get_empty_output_buffer()
    #     trail_filter.process_events(events_buf, events_buf_)
    #     events_buf = np.asarray(events_buf_)
    #     events_buf_ = ActivityNoiseFilterAlgorithm.get_empty_output_buffer()
    #     activity_filter.process_events(events_buf, events_buf_)
    #     events_buf = np.asarray(events_buf_)

    #     r = 6
    #     for idx in tqdm(range(len(events_buf))):
    #         y, x = events_buf['y'][idx]//k, events_buf['x'][idx]//k
    #         if events_buf['p'][idx] > 0:
    #             patch = event_time_image[max(0, y-r):y+r+1, max(0, x-r):x+r+1, 0]
    #             patch = np.where(patch>0, patch-(events_buf['t'][idx]-patch)/15., patch)
    #             patch[patch<0] = 0
    #             event_time_image[max(0, y-r):y+r+1, max(0, x-r):x+r+1, 0] = patch
    #             event_time_image[y, x, 0] = events_buf['t'][idx]
    #         else:
    #             patch = event_time_image[max(0, y-r):y+r+1, max(0, x-r):x+r+1, 1]
    #             patch = np.where(patch>0, patch-(events_buf['t'][idx]-patch)/15., patch)
    #             patch[patch<0] = 0
    #             event_time_image[max(0, y-r):y+r+1, max(0, x-r):x+r+1, 1] = patch
    #             event_time_image[y, x, 1] = events_buf['t'][idx]
    #     event_time_image[event_time_image > 0] -= events_buf['t'][0]
    #     event_time_image[event_time_image < 0] = 0            
    # elif args.method == "Ours_Denoise_Denoise_Activity":
    #     activity_time_ths = args.time_window // 100 # Length of the time window for activity filtering (in us)
    #     activity_filter = ActivityNoiseFilterAlgorithm(1280, 720, activity_time_ths)
    #     events_buf = ActivityNoiseFilterAlgorithm.get_empty_output_buffer()

    #     new_dtype = np.dtype({
    #                     'names': ['x', 'y', 'p', 't'],                # Field names
    #                     'formats': ['<u2', '<u2', '<i2', '<i8'],      # Data types for each field
    #                     'offsets': [0, 2, 4, 8],                      # Byte offsets for each field
    #                     'itemsize': 16                                # Total size of each entry
    #                     })
    #     events = np.zeros(idx_cur-idx_start, dtype=new_dtype)
    #     events['x'] = event_data['x'][idx_start:idx_cur]
    #     events['y'] = event_data['y'][idx_start:idx_cur]
    #     events['p'] = event_data['p'][idx_start:idx_cur]
    #     events['t'] = event_data['t'][idx_start:idx_cur]

    #     activity_filter.process_events(events, events_buf)
    #     events_buf = np.asarray(events_buf)

    #     r = 6
    #     B = 1 # 15
    #     R = 1
    #     threshold = 0.7 # 0.3
    #     total_range = np.arange(0, len(events_buf))
    #     subsequences = np.array_split(total_range, B)
    #     for subseq in subsequences:
    #         mask = np.zeros((rows, cols), dtype=bool)
    #         for idx in tqdm(subseq):
    #             y, x = events_buf['y'][idx]//k, events_buf['x'][idx]//k
    #             if events_buf['p'][idx] > 0:
    #                 patch = event_time_image[max(0, y-r):y+r+1, max(0, x-r):x+r+1, 0]
    #                 patch = np.where(patch>0, patch-(events_buf['t'][idx]-patch)/15., patch)
    #                 patch[patch<0] = 0
    #                 event_time_image[max(0, y-r):y+r+1, max(0, x-r):x+r+1, 0] = patch
    #                 event_time_image[y, x, 0] = events_buf['t'][idx]
    #             else:
    #                 patch = event_time_image[max(0, y-r):y+r+1, max(0, x-r):x+r+1, 1]
    #                 patch = np.where(patch>0, patch-(events_buf['t'][idx]-patch)/15., patch)
    #                 patch[patch<0] = 0
    #                 event_time_image[max(0, y-r):y+r+1, max(0, x-r):x+r+1, 1] = patch
    #                 event_time_image[y, x, 1] = events_buf['t'][idx]
    #             patch = event_time_image[max(0, y-R):y+R+1, max(0, x-R):x+R+1, :]
    #             valid_count = ((patch[:, :, 0] > 0) | (patch[:, :, 1] > 0)).sum()
    #             if valid_count / (patch.shape[0]*patch.shape[1]) < threshold:
    #                 mask[y, x] = True
    #             else:
    #                 mask[y, x] = False
    #         event_time_image[mask] *= 0
    #     event_time_image[event_time_image > 0] -= events_buf['t'][0]
    #     event_time_image[event_time_image < 0] = 0
    # elif args.method == "Ours_Denoise_Denoise_STC_Trail":
    #     trail_filter_ths = args.time_window  # Length of the time window for activity filtering (in us)
    #     trail_filter = TrailFilterAlgorithm(1280, 720, trail_filter_ths)

    #     stc_filter_ths = args.time_window  # Length of the time window for filtering (in us)
    #     stc_cut_trail = True  # If true, after an event goes through, it removes all events until change of polarity
    #     stc_filter = SpatioTemporalContrastAlgorithm(1280, 720, stc_filter_ths, stc_cut_trail)

    #     events_buf = ActivityNoiseFilterAlgorithm.get_empty_output_buffer()

    #     new_dtype = np.dtype({
    #                     'names': ['x', 'y', 'p', 't'],                # Field names
    #                     'formats': ['<u2', '<u2', '<i2', '<i8'],      # Data types for each field
    #                     'offsets': [0, 2, 4, 8],                      # Byte offsets for each field
    #                     'itemsize': 16                                # Total size of each entry
    #                     })
    #     events = np.zeros(idx_cur-idx_start, dtype=new_dtype)
    #     events['x'] = event_data['x'][idx_start:idx_cur]
    #     events['y'] = event_data['y'][idx_start:idx_cur]
    #     events['p'] = event_data['p'][idx_start:idx_cur]
    #     events['t'] = event_data['t'][idx_start:idx_cur]


    #     stc_filter.process_events(events, events_buf)
    #     events_buf = np.asarray(events_buf)
    #     events_buf_ = ActivityNoiseFilterAlgorithm.get_empty_output_buffer()
    #     trail_filter.process_events(events_buf, events_buf_)
    #     events_buf = np.asarray(events_buf_)

    #     r = 6
    #     B = 1 # 15
    #     R = 1
    #     threshold = 0.7 # 0.3
    #     total_range = np.arange(0, len(events_buf))
    #     subsequences = np.array_split(total_range, B)
    #     for subseq in subsequences:
    #         mask = np.zeros((rows, cols), dtype=bool)
    #         for idx in tqdm(subseq):
    #             y, x = events_buf['y'][idx]//k, events_buf['x'][idx]//k
    #             if events_buf['p'][idx] > 0:
    #                 patch = event_time_image[max(0, y-r):y+r+1, max(0, x-r):x+r+1, 0]
    #                 patch = np.where(patch>0, patch-(events_buf['t'][idx]-patch)/15., patch)
    #                 patch[patch<0] = 0
    #                 event_time_image[max(0, y-r):y+r+1, max(0, x-r):x+r+1, 0] = patch
    #                 event_time_image[y, x, 0] = events_buf['t'][idx]
    #             else:
    #                 patch = event_time_image[max(0, y-r):y+r+1, max(0, x-r):x+r+1, 1]
    #                 patch = np.where(patch>0, patch-(events_buf['t'][idx]-patch)/15., patch)
    #                 patch[patch<0] = 0
    #                 event_time_image[max(0, y-r):y+r+1, max(0, x-r):x+r+1, 1] = patch
    #                 event_time_image[y, x, 1] = events_buf['t'][idx]
    #             patch = event_time_image[max(0, y-R):y+R+1, max(0, x-R):x+R+1, :]
    #             valid_count = ((patch[:, :, 0] > 0) | (patch[:, :, 1] > 0)).sum()
    #             if valid_count / (patch.shape[0]*patch.shape[1]) < threshold:
    #                 mask[y, x] = True
    #             else:
    #                 mask[y, x] = False
    #         event_time_image[mask] *= 0
    #     event_time_image[event_time_image > 0] -= events_buf['t'][0]
    #     event_time_image[event_time_image < 0] = 0
    # elif args.method == "Ours_Line":
    #     r = 6
    #     for id in tqdm(range(idx_cur-idx_start)):
    #         idx = int(id + idx_start)
    #         y, x = event_data['y'][idx]//k, event_data['x'][idx]//k
    #         if event_data['p'][idx] > 0:
    #             patch = event_time_image[max(0, y-r):y+r+1, max(0, x-r):x+r+1, 0]
    #             patch = np.where(patch>0, patch-(event_data['t'][idx]-patch)/10., patch)
                
    #             valid_pixel_y, valid_pixel_x = np.where(patch>0)
    #             line_valid_pixel_list = [list(zip(line_function(valid_pixel_y[i], valid_pixel_x[i], 6, 6))) for i in range(len(valid_pixel_x))]
    #             # _ = [print(line_index_yx[0][0]) for line_index_yx in line_valid_pixel_list]
    #             line_count = [len(line_index_yx[0][0]) for line_index_yx in line_valid_pixel_list]
    #             line_valid_count = [(patch[line_index_yx[0][0], line_index_yx[1][0]]>0).sum() for line_index_yx in line_valid_pixel_list]
    #             weight = np.asarray(line_count) / np.asarray(line_valid_count)
    #             patch[valid_pixel_y, valid_pixel_x] -=  weight * (event_data['t'][idx]-patch[valid_pixel_y, valid_pixel_x])/20.
                
    #             patch[patch<0] = 0
    #             event_time_image[max(0, y-r):y+r+1, max(0, x-r):x+r+1, 0] = patch
    #             event_time_image[y, x, 0] = event_data['t'][idx]
    #         else:
    #             patch = event_time_image[max(0, y-r):y+r+1, max(0, x-r):x+r+1, 1]
    #             patch = np.where(patch>0, patch-(event_data['t'][idx]-patch)/10., patch)
                
    #             valid_pixel_y, valid_pixel_x = np.where(patch>0)
    #             line_valid_pixel_list = [list(zip(line_function(valid_pixel_y[i], valid_pixel_x[i], 6, 6))) for i in range(len(valid_pixel_x))]
    #             # _ = [print(line_index_yx[0][0]) for line_index_yx in line_valid_pixel_list]
    #             line_count = [len(line_index_yx[0][0]) for line_index_yx in line_valid_pixel_list]
    #             line_valid_count = [(patch[line_index_yx[0][0], line_index_yx[1][0]]>0).sum() for line_index_yx in line_valid_pixel_list]
    #             weight = np.asarray(line_count) / np.asarray(line_valid_count)
    #             patch[valid_pixel_y, valid_pixel_x] -=  weight * (event_data['t'][idx]-patch[valid_pixel_y, valid_pixel_x])/20.

    #             patch[patch<0] = 0
    #             event_time_image[max(0, y-r):y+r+1, max(0, x-r):x+r+1, 1] = patch
    #             event_time_image[y, x, 1] = event_data['t'][idx]
    #     event_time_image[event_time_image > 0] -= event_data['t'][idx_start]
    #     event_time_image[event_time_image < 0] = 0
    # elif args.method == "Bins":
    #     # # Voxel
    #     from event_utils import events_to_voxel_timesync_torch
    #     event_time_image = events_to_voxel_timesync_torch(torch.tensor(np.array(event_data['x'][idx_start:idx_end]//k, dtype=np.float32)), 
    #                                                     torch.tensor(np.array(event_data['y'][idx_start:idx_end]//k, dtype=np.float32)), 
    #                                                     torch.tensor(np.array(event_data['t'][idx_start:idx_end], dtype=np.float32)), 
    #                                                     torch.tensor(np.array(event_data['p'][idx_start:idx_end], dtype=np.float32)), 
    #                                                     B=10,
    #                                                     t0=event_data['t'][idx_start],
    #                                                     t1=event_data['t'][idx_end],
    #                                                     np_ts=np.array(event_data['t'][idx_start:idx_end], dtype=np.float32),
    #                                                     sensor_size=[rows, cols], temporal_bilinear=True) # 10 720 1280
    # else:
    #     raise "Method doesn't exist."
    
    event_time_image = np.array(event_time_image)
    event_time_image_disp = np.concatenate((np.zeros([event_time_image.shape[0], event_time_image.shape[1], 1]), event_time_image), axis=2)
    event_time_image_disp = (event_time_image_disp / np.max(event_time_image_disp) * 255).astype(np.uint8)
    if not args.half_resolution:
        event_time_image_disp = event_time_image_disp[60:660, 160:1060, :]
    else:
        event_time_image_disp = event_time_image_disp[32:32+296, 64:64+512, :]
    cv2.imwrite(f'{save_file}.png', event_time_image_disp)



    # # pc map load
    vox_map = load_map(pc_path, device)
    print(f'load pointclouds finished! {vox_map.shape[1]} points')
    pose = torch.tensor(Ln_T_L0[args.idx+1], device=device, dtype=torch.float32)
    local_map = vox_map.clone()
    local_map = torch.matmul(pose, local_map)
    indexes = local_map[0, :] > -1.
    # indexes = indexes & (local_map[0, :] < 10.)
    # indexes = indexes & (local_map[1, :] > -5.)
    # indexes = indexes & (local_map[1, :] < 5.)
    indexes = indexes & (local_map[0, :] < 100.)
    indexes = indexes & (local_map[1, :] > -25.)
    indexes = indexes & (local_map[1, :] < 25.)
    local_map = local_map[:, indexes]
    prophesee_left_T_lidar = torch.tensor(data["/ouster/calib/T_to_prophesee_left"], device=device, dtype=torch.float32)
    local_map = torch.matmul(prophesee_left_T_lidar, local_map)
    # pc_visualize(local_map.t().cpu().numpy())
    image_size = tuple(event_data['resolution'][[1, 0]])
    calib = torch.tensor(event_data['intrinsics'], device=device, dtype=torch.float)
    sparse_depth = depth_generation(local_map, image_size,
                                    calib, 3., 5, device=device)
    sparse_depth_disp = (sparse_depth / torch.max(sparse_depth) * 255).cpu().numpy().astype(np.uint8)
    sparse_depth_disp = sparse_depth_disp[0, ...]
    if not args.half_resolution:
        sparse_depth_disp = sparse_depth_disp[60:660, 160:1060]
    else:
        sparse_depth_disp = sparse_depth_disp[32:32+296, 64:64+512]
    cv2.imwrite('./LiDAR_projection.png', sparse_depth_disp)