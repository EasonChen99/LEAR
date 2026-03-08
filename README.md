# LEAR
This repository contains the source code for our paper:

[LEAR: Learning Edge-Aware Representations for Event-to-LiDAR Localization](https://arxiv.org/pdf/2603.01839)<br/>
Kuangyi Chen, Jun Zhang, Yuxi Hu, Yi Zhou, Friedrich Fraundorfer<br/>

## Requirements
The code has been trained and tested with PyTorch 2.2.2 and Cuda 12.2.
```Shell
conda create -n LEAR python=3.10 -y
conda activate LEAR
pip install -r requirements.txt
cd blender-mathutils
python setup.py install
cd ..
cd core/correlation_package
python setup.py install
cd ..
cd visibility_package
python setup.py install
cd ../..
```
Additionally, please compile [PoseLib](https://github.com/PoseLib/PoseLib) for use with pose solvers, and install the [Metavision SDK](https://docs.prophesee.ai/stable/installation/linux.html#chapter-installation-linux) to obtain the filters for data preprocessing.

## Required Data
To evaluate/train LEAR, you could download the M3ED dataset.
* [M3ED](https://m3ed.io/data_overview/)

To obtain the required data for direct use, we sample a pre-defined range of point cloud from the whole LiDAR maps for every provided pose.
```Shell
python tools/map2pc.py --dataset [PATH_TO_DATA] --platform [PLATFORM_NAME] --sequence [SPECIFIED_SEQUENCE] --save_dir [PATH_TO_SAVEDIR]
```
As for event data, we also need to generate event frames accordingly.
```Shell
python tools/event2frame.py --dataset [PATH_TO_DATA] --sequence [SPECIFIED_SEQUENCE] --save_dir [PATH_TO_SAVEDIR] --camera [WHICH_CAMERE] --method [EVENT_REPRESENTATION] --half_resolution
```

## Demos
Pretrained models can be downloaded from [google drive](https://drive.google.com/file/d/1NumLFxHTif-rJo9nRU0_DKwHg09KQwmG/view?usp=sharing)

You can demo a trained model on a sequence of frames
```Shell
python main.py --ev_input ours_denoise_stc_trail_pre_100000_half --backbone edge --load_checkpoints checkpoints/checkpoint.pth --use_feature_fusion -e
```

## Training
You can train a model using `main.py`. Training logs will be written to the `runs` which can be visualized using tensorboard.
```Shell
python main.py --data_path [PATH_TO_PREPROCESSED_DATA] --ev_input ours_denoise_stc_trail_pre_100000_half --test_sequence falcon_indoor_flight_3 --max_depth 10. --epochs 100 --batch_size 2 --lr 4e-5 --gpus 0 --max_r 5. --max_t 0.5 --backbone edge --use_feature_fusion
```