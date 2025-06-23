import os
import glob
import h5py
import argparse
import numpy as np
import cv2
import torch
from utils import depth_generation
from utils_point import overlay_imgs

def get_calib_m3ed(sequence, camera):
    if camera == "left":
        if sequence == "falcon_indoor_flight_1":
            return torch.tensor([1034.86278431, 1033.47800271, 629.70125104, 357.60071019])
        elif sequence in ["falcon_indoor_flight_2", "falcon_indoor_flight_3"]:
            return torch.tensor([1034.39696079, 1034.73607278, 636.27410756, 364.73952748])
        elif sequence in ['falcon_outdoor_day_penno_parking_1', 'falcon_outdoor_day_penno_parking_2']:
            return torch.tensor([1033.22781771, 1032.05548869, 631.84536312, 360.7175681])
        # elif sequence in ['falcon_outdoor_night_penno_parking_1', 'falcon_outdoor_night_penno_parking_2']:
        #     return torch.tensor([1034.39696079, 1034.73607278,  636.27410756,  364.73952748])
        elif sequence in ['falcon_forest_into_forest_1', 'falcon_forest_into_forest_2', 'falcon_forest_into_forest_4']:
            return torch.tensor([1033.22946467, 1032.37940162 , 637.14899526,  359.16366857])        
        # elif sequence in ['spot_outdoor_day_srt_under_bridge_1', 'spot_outdoor_day_srt_under_bridge_1']:
        #     return torch.tensor([1031.52186613, 1031.70276624,  633.53886647,  364.34446137])
        elif sequence in ['spot_forest_road_1', 'spot_forest_road_3']:
            return torch.tensor([1032.96138833, 1032.8263147,   632.38290823, 368.57212974])
        elif sequence in ['car_urban_day_penno_big_loop', 'car_urban_day_penno_small_loop']:
            return torch.tensor([1031.88399229, 1031.48192315 , 634.29808475 , 366.39105342])
        # elif sequence in ['car_urban_night_penno_big_loop', 'car_urban_night_penno_small_loop']:
        #     return torch.tensor([1031.84262859, 1030.10381777 , 635.71409589 , 365.59991372])
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
        # elif sequence in ['falcon_outdoor_night_penno_parking_1', 'falcon_outdoor_night_penno_parking_2']:
        #     return torch.tensor([1034.61302587, 1034.83604567,  638.12992827,  366.88002829])
        elif sequence in ['falcon_forest_into_forest_1', 'falcon_forest_into_forest_2', 'falcon_forest_into_forest_4']:
            return torch.tensor([1032.7719854,  1031.93236836,  639.10742672,  362.60248038]) 
        # elif sequence in ['spot_outdoor_day_srt_under_bridge_1', 'spot_outdoor_day_srt_under_bridge_1']:
        #     return torch.tensor([1030.29161359, 1030.9024083, 634.79835424, 368.11576903])
        elif sequence in ['spot_forest_road_1', 'spot_forest_road_3']:
            return torch.tensor([1032.26867081, 1032.57448492,  632.85489859,  370.0490076 ])
        elif sequence in ['car_urban_day_penno_big_loop', 'car_urban_day_penno_small_loop']:
            return torch.tensor([1031.36879978 ,1031.06491961 , 634.87768084 , 367.62546105])
        # elif sequence in ['car_urban_night_penno_big_loop', 'car_urban_night_penno_small_loop']:
        #     return torch.tensor([1030.46186128, 1029.51180204,  635.69022466 , 364.32444857])
        elif sequence in ['car_forest_into_ponds_long', 'car_forest_into_ponds_short']:
            return torch.tensor([1030.62646968, 1030.98305576,  636.68872098,  368.70448786])
        else:
            raise TypeError("Sequence Not Available")

def get_left_right_T(sequence):
    if sequence == "falcon_indoor_flight_1":
        return torch.tensor([[ 9.9999e-01, -1.2158e-04, -4.6864e-03, -1.2023e-01],
                             [ 1.2021e-04,  1.0000e+00, -2.9335e-04,  8.3630e-04],
                             [ 4.6865e-03,  2.9279e-04,  9.9999e-01,  1.0119e-03],
                             [ 0.0000e+00,  0.0000e+00,  0.0000e+00,  1.0000e+00]])
    elif sequence in ["falcon_indoor_flight_2", "falcon_indoor_flight_3"]:
        return torch.tensor([[ 9.9999e-01, -6.6613e-04, -3.5103e-03, -1.2018e-01],
                             [ 6.6661e-04,  1.0000e+00,  1.3561e-04,  9.1033e-04],
                             [ 3.5102e-03, -1.3795e-04,  9.9999e-01, -4.3059e-04],
                             [ 0.0000e+00,  0.0000e+00,  0.0000e+00,  1.0000e+00]])
    elif sequence in ['falcon_outdoor_day_penno_parking_1', 'falcon_outdoor_day_penno_parking_2']:
        return torch.tensor([[ 9.9999e-01, -3.8986e-04, -3.2093e-03, -1.2005e-01],
                             [ 3.8672e-04,  1.0000e+00, -9.8024e-04,  9.3651e-04],
                             [ 3.2097e-03,  9.7899e-04,  9.9999e-01,  3.4862e-04],
                             [ 0.0000e+00,  0.0000e+00,  0.0000e+00,  1.0000e+00]])
    # elif sequence in ['falcon_outdoor_night_penno_parking_1', 'falcon_outdoor_night_penno_parking_2']:
    #     return torch.tensor([[ 9.99993617e-01, -6.66130268e-04, -3.51029598e-03, -1.20179395e-01],
    #                          [ 6.66610390e-04,  9.99999769e-01,  1.35607254e-04,  9.10326972e-04],
    #                          [ 3.51020483e-03, -1.37946389e-04,  9.99993830e-01, -4.30589681e-04],
    #                          [ 0.00000000e+00,  0.00000000e+00,  0.00000000e+00,  1.00000000e+00]])
    elif sequence in ['falcon_forest_into_forest_1', 'falcon_forest_into_forest_2', 'falcon_forest_into_forest_4']:
        return torch.tensor([[ 9.99989586e-01, -4.60304775e-04, -4.54058647e-03 ,-1.20070481e-01],
                             [ 4.55317167e-04,  9.99999292e-01, -1.09942280e-03 , 9.50882179e-04],
                             [ 4.54108932e-03,  1.09734395e-03,  9.99989087e-01 ,-5.19321703e-04],
                             [ 0.00000000e+00,  0.00000000e+00 , 0.00000000e+00,  1.00000000e+00]])         
    # elif sequence in ['spot_outdoor_day_srt_under_bridge_1', 'spot_outdoor_day_srt_under_bridge_1']:
    #     return torch.tensor([[ 9.9999e-01, -6.4771e-04, -3.6507e-03, -1.2011e-01],
    #                          [ 6.3675e-04,  1.0000e+00, -3.0024e-03,  8.7946e-04],
    #                          [ 3.6526e-03,  3.0000e-03,  9.9999e-01, -2.8266e-04],
    #                          [ 0.0000e+00,  0.0000e+00,  0.0000e+00,  1.0000e+00]])
    elif sequence in ['spot_forest_road_1', 'spot_forest_road_3']:
        return torch.tensor([[ 9.99992821e-01, -7.03218854e-04, -3.72345760e-03, -1.20137685e-01],
                             [ 6.95582202e-04,  9.99997653e-01, -2.05185517e-03,  1.03034216e-03],
                             [ 3.72489176e-03,  2.04925047e-03, 9.99990963e-01 , 4.66687603e-04],
                             [ 0.00000000e+00 , 0.00000000e+00 , 0.00000000e+00  ,1.00000000e+00]])
    elif sequence in ['car_urban_day_penno_big_loop', 'car_urban_day_penno_small_loop']:
        return torch.tensor([[ 9.99993926e-01 ,-7.53276578e-04, -3.40299857e-03, -1.20220978e-01],
                             [ 7.47230114e-04,  9.99998141e-01 ,-1.77772679e-03 , 9.05842201e-04],
                             [ 3.40433136e-03 , 1.77517317e-03,  9.99992630e-01, -1.13660539e-04],
                             [ 0.00000000e+00 , 0.00000000e+00 , 0.00000000e+00 , 1.00000000e+00]])
    # elif sequence in ['car_urban_night_penno_big_loop', 'car_urban_night_penno_small_loop']:
    #     return torch.tensor([[ 9.99996013e-01 ,-6.08160070e-04, -2.75745341e-03, -1.20188402e-01],
    #                          [ 6.09451349e-04,  9.99999705e-01,  4.67470872e-04 , 1.06054268e-03],
    #                          [ 2.75716830e-03, -4.69149542e-04,  9.99996089e-01,  1.07289756e-04],
    #                          [ 0.00000000e+00,  0.00000000e+00,  0.00000000e+00,  1.00000000e+00]])       
    elif sequence in ['car_forest_into_ponds_long', 'car_forest_into_ponds_short']:
        return torch.tensor([[ 9.99992228e-01, -6.45914237e-04, -3.88937476e-03, -1.20192478e-01],
                             [ 6.37427554e-04,  9.99997414e-01, -2.18286148e-03 , 9.52772120e-04],
                             [ 3.89077465e-03,  2.18036532e-03,  9.99990054e-01,  5.21780332e-05],
                             [ 0.00000000e+00,  0.00000000e+00,  0.00000000e+00 , 1.00000000e+00]]) 
    else:
        raise TypeError("Sequence Not Available")

if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", 
                    default="/media/eason/Backup/Datasets/M3ED/generated/Falcon",
                    help="Root path to the dataset", 
                    type=str)
    ap.add_argument("--sequence",
                    default="falcon_indoor_flight_1",
                    help="Sequence name for processing",
                    type=str)
    ap.add_argument("--method",
                    default="ours_denoise_stc_trail",
                    help="Event representation method",
                    type=str)
    ap.add_argument("--camera",
                    default="left",
                    help="which camera to use",
                    type=str)    
    args = ap.parse_args()    

    data_path = os.path.join(args.dataset, args.sequence)

    event_paths = os.path.join(data_path, f"event_frames_{args.method}", args.camera)
    pc_paths = os.path.join(data_path, 'local_maps')

    save_dir = f"./visualization/{args.sequence}/{args.method}/{args.camera}"
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    sequences = sorted(glob.glob(f"{event_paths}/*.npy"))
    for idx in range(len(sequences)):
        if idx < 100:
            print(idx)
            continue
        event_path = sequences[idx]
        print(idx, event_path)
        sequence = event_path.split('/')[-1].split('_')[-1].split('.')[0]
        pc_path = os.path.join(pc_paths, f'point_cloud_{int(sequence)+1:05d}.h5')

        events = np.load(event_path)
        event_time_image = events
        event_time_image[event_time_image<0]=0
        event_time_image = np.concatenate((np.zeros([event_time_image.shape[0], event_time_image.shape[1], 1]), event_time_image), axis=2)
        event_time_image = (event_time_image / np.max(event_time_image) * 255).astype(np.uint8)
        if args.method[-4:] == "half":
            event_time_image = event_time_image[36:288+36, 64:512+64, :]
        else:
            event_time_image = event_time_image[60:660, 160:960+160, :]

        # event_time_image = cv2.medianBlur(event_time_image, 3)
        # event_time_image = cv2.bilateralFilter(event_time_image, 9, 150, 150)
        # event_time_image = cv2.GaussianBlur(event_time_image, (7,7), 0)
        # event_time_image = cv2.GaussianBlur(event_time_image, (5,5), 0)

        cv2.imwrite(f"{save_dir}/{idx:05d}_event_frame.png", event_time_image)
        
        try:
            with h5py.File(pc_path, 'r') as hf:
                pc = hf['PC'][:]
        except Exception as e:
            print(f'File Broken: {pc_path}')
            raise e
        pc_in = torch.from_numpy(pc.astype(np.float32))
        if pc_in.shape[1] == 4 or pc_in.shape[1] == 3:
            pc_in = pc_in.t()
        if pc_in.shape[0] == 3:
            homogeneous = torch.ones(pc_in.shape[1]).unsqueeze(0)
            pc_in = torch.cat((pc_in, homogeneous), 0)
        elif pc_in.shape[0] == 4:
            if not torch.all(pc_in[3,:] == 1.):
                pc_in[3,:] = 1.

        if args.camera == 'right':
            T_to_prophesee_left = get_left_right_T(args.sequence)
            pc_in = torch.matmul(T_to_prophesee_left, pc_in)

        calib = get_calib_m3ed(args.sequence, args.camera)
        # calib = calib.cuda().float()
        # sparse_depth = depth_generation(pc_in.cuda(), (720, 1280), calib, 3., 5, device='cuda:0')
        calib = calib.cuda().float() / 2.
        sparse_depth = depth_generation(pc_in.cuda(), (360, 640), calib, 3., 5, device='cuda:0')
        sparse_depth = overlay_imgs(sparse_depth.repeat(3, 1, 1)*0, sparse_depth[0, ...])
        # sparse_depth = sparse_depth[60:660, 160:960+160, :]
        sparse_depth = sparse_depth[36:288+36, 64:512+64, :]
        cv2.imwrite(f"{save_dir}/{idx:05d}_depth.png", sparse_depth)

        # ## denoise
        # from Kinect_Smoothing.kinect_smoothing import HoleFilling_Filter, Denoising_Filter
        # noise_filter = Denoising_Filter(flag='anisotropic', 
        #                                 depth_min=0.01,
        #                                 depth_max=10., 
        #                                 niter=10,
        #                                 kappa=100,
        #                                 gamma=0.25,
        #                                 option=1)
        # image_frame = noise_filter.smooth_image(dense_depth)
        # cv2.imwrite(f"./visualization/{idx:05d}_4_dense_depth_denoise.png", (image_frame/10.*255).astype(np.uint8))


        # blur

        # import time
        # begin = time.time()
        # stretched_image, output_image, dense_edge = enhanced_depth_line_extract(dense_depth / 10. * 255)
        # print(time.time() - begin)
        # cv2.imwrite(f"./visualization/{idx:05d}_5_dense_stretched.png", stretched_image)
        # cv2.imwrite(f"./visualization/{idx:05d}_6_dense_output.png", output_image.cpu().numpy()*255)
        # cv2.imwrite(f"./visualization/{idx:05d}_7_dense_edge.png", dense_edge.cpu().numpy()*255)
        # # cv2.imwrite(f"./visualization/{idx:05d}_6_dense_output.png", output_image*255)
        # # cv2.imwrite(f"./visualization/{idx:05d}_7_dense_edge.png", dense_edge*255)

    

