# YCB-Pose-Estimation
This code employs Grounded SAM and FFB6D with ICP to estimate object poses.

## Installation

* FFB6D

You can follow this link([ffb6d](https://github.com/IntelRealSense/librealsense)) to install it. The pretrained weights file shoule be placed in /FFB6D/ffb6d/train_log/ycb/checkpoints.

* Grounded SAM

You can follow this link([Grounded SAM](https://github.com/IDEA-Research/Grounded-Segment-Anything)) to install it.

First, you should go into the folder(Grounded-Segment-Anything).

```bash
cd Grounded-Segment-Anything
```

Second, you should set the environment variable manually as follows if you want to build a local GPU environment for Grounded-SAM:

```bash
export AM_I_DOCKER=False
export BUILD_WITH_CUDA=True
export CUDA_HOME=/path/to/cuda-11.3/
```

Install Segment Anything:

```bash
python -m pip install -e segment_anything
```

Install Grounding DINO:

```bash
python -m pip install -e GroundingDINO
```

Third, you can download the pretrained weights from the below way or refer to these two links([GroundingDINO](https://github.com/IDEA-Research/GroundingDINO),
[SAM](https://github.com/facebookresearch/segment-anything)) to download more pre-trained models.

```bash
cd Grounded-Segment-Anything

wget https://dl.fbaipublicfiles.com/segment_anything/sam_vit_h_4b8939.pth
wget https://github.com/IDEA-Research/GroundingDINO/releases/download/v0.1.0-alpha/groundingdino_swint_ogc.pth
```

Then you should place the pretrained weights in the folder(Grounded-Segment-Anything).

Finally, you must set global file path in /FFB6D/ffb6d/common.py, /FFB6D/ffb6d/datasets/ycb/ycb_dataset.py, /FFB6D/ffb6d/utils/basic_utils.py, /FFB6D/ffb6d/utils/pvn3d_eval_utils_kpls.py, /FFB6D/ffb6d/utils/basic_utils.py.

```bash
SYSTEM_PATH=/PATH/TO/FFB6D/ffb6d
sys.path.append(SYSTEM_PATH)
```