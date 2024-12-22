# RoboFireFuseNet

This is the official repository for our recent work: RoboFireFuseNet: Robust Fusion of Visible and Infrared Imaging for Real-Time Flame and Smoke Segmentation in Wildfire Scenarios

### Abstract
Concurrent segmentation of flames and smoke is a
challenging task, particularly when relying on a single spectral
band. Leveraging the combination of visible (RGB) and infrared
(thermal) modalities in wildfire imaging significantly enhances the
accuracy and robustness of fire segmentation systems. However,
fusing these modalities presents notable challenges due to the
high diversity in data representation, with certain features being
exclusive to specific spectra. This paper evaluates the effectiveness
of RGB and thermal data for flame and smoke segmentation,
exploring various fusion strategies. A novel intermediate fusion
architecture is proposed, built upon a real-time, state-of-the-
art segmentation model augmented with attention mechanisms
and specifically designed to address the complexities of modality
fusion. Practical challenges, such as robustness to unregistered
inputs and sensor failures are also addressed, resulting in one
of the first models to effectively tackle the issue of unregistered
fusion. The lightweight model achieves comparable accuracy to
state-of-the-art architectures on urban datasets and surpasses
them in wildfire scenarios, offering real-time capabilities and
enhanced robustness suitable for robotic applications in dynamic
and high-stakes environments.

<div align="center">
   <h4>MIOU vs FPS on MFNet dataset and RTX 4090</h4>
  <img src="figs/mioufps.png" alt="Model Architecture" width="400"/>
</div>

## Highlights

- 🔥 <b>Lightweight & Powerful Multimodal Segmentation</b>: Our real-time fusion model outperforms many SOTA models with fewer parameters in segmentation task with visible and thermal imaging.
- 🚁 <b>Smoke & Flame Segmentation</b>: Excels in dense smoke conditions, accurately segmenting concurently flames and smoke.
- 🛠️ <b>Robust Fusion</b>: Handles modality misalignment and sensor failures achieving semantic segmentation fusing unregistered inputs.
   
## Updates
- Paper is submitted to ...

## Demo


## Overview
Schematic overview of the proposed fusion model and the robust module.

### Fusion Architecture
Our model builds on PIDNet-Small by integrating SwinV2-T Transformer blocks to enhance capacity and capture long-range dependencies. We introduce dual modality paths to preserve and mine modality-specific features, and replace simple upscaling with a U-Net style decoder using shortcut connections. This design restores spatial dimensions while retaining essential small-scale features for improved segmentation.

<img src="figs/mymodel.png" alt="Model Architecture" width="700"/>

### Robust module
The optional, lightweight robustness module enhances modality alignment by iteratively estimating the optimal affine transformation to align Infrared and RGB modalities. It leverages cross-attention between the two modalities to generate a misalignment-aware map, ensuring more accurate feature fusion and improved robustness.

<img src="figs/robust.png" alt="Model Architecture" width="400"/>

### 📊 **Performance Comparison on Urban Scenes (MFNet)**
| **Method**                | **Avg Recall (%)** | **MIoU (%)** | **Params (M)** |
|--------------------------|-------------------|---------------|-----------------|
| [PIDNet-m RGB](https://github.com/XuJiacong/PIDNet)  | 65.59            | 51.52         | 34.4            |
| [PIDNet-m IR](https://github.com/XuJiacong/PIDNet)   | 65.27            | 50.70         | 34.4            |
| [PIDNet-m Early](https://github.com/XuJiacong/PIDNet) | 69.59            | 52.62         | 34.4            |
| [MFNet](https://github.com/haqishen/MFNet-pytorch)          | 59.1             | 39.7          | **0.73**         |
| [RTFNet](https://github.com/yuxiangsun/RTFNet)        | 63.08            | 53.2          | 185.24          |
| [GMNet](https://github.com/Jinfu0913/GMNet)        | **74.1**         | 57.3          | 153             |
| [EGFNet](https://github.com/ShaohuaDong2021/EGFNet)       | 72.7             | 54.8          | 62.5            |
| [CRM-T](https://github.com/UkcheolShin/CRM_RGBTSeg)        | -                | 59.7          | 59.1            |
| [Sigma-T](https://github.com/zifuwan/Sigma)       | 71.3             | 60.23         | 48.3            |
| **Ours**                                             | 71.1             | **60.6**      | 29.5            |
### 📊 **Performance Comparison on FLAME2**
| **Method**                | **Avg Recall (%)** | **MIoU (%)** | **Params (M)** |
|--------------------------|-------------------|---------------|-----------------|
| [PIDNet-RGB](https://github.com/XuJiacong/PIDNet)  | 75.66            | 61.21         | 34.4            |
| [PIDNet-IR](https://github.com/XuJiacong/PIDNet)   | 83.05            | 58.71         | 34.4            |
| [PIDNet-Early](https://github.com/XuJiacong/PIDNet) | 88.25            | 73.90         | 34.4            |
| [MFNet](https://github.com/haqishen/MFNet-pytorch)        | 93.53            | 80.26         | **0.73**         |
| [RTFNet](https://github.com/yuxiangsun/RTFNet)      | 73.87            | 65.42         | 185.24          |
| [GMNet](https://github.com/Jinfu0913/GMNet)      | 67.53            | 54.08         | 153             |
| [EGFNet](https://github.com/ShaohuaDong2021/EGFNet)     | 74.27            | 60.98         | 62.5            |
| [CRM-T](https://github.com/UkcheolShin/CRM_RGBTSeg)      | -                | -             | 59.1            |
| [Sigma-T](https://github.com/zifuwan/Sigma)     | 92.6             | 86.27         | 48.3            |
| **Ours**                            | **94.34**        | **88.39**     | 29.5            |

## Usage

### 0. Setup
- Install python requirements: `pip install -r requirements.txt` or recommender `Python 3.10.12`
- Download the [weights](https://drive.google.com/drive/folders/1wldeSDx5VVjynABJqm55RDREAnPonk5y?usp=sharing) inside the weights folder.
- Download the [data](https://drive.google.com/drive/folders/15bsStvQWBpMY1bXW3Wi-uliczz1-Zko8?usp=drive_link) inside the data folder.
  
### 2. Training
Customize configurations via the config/ folder or override them with inline arguments.
- train fusion model on wildfire: `python train.py --yaml_file wildfire.yaml --LR 0.001 --BATCHSIZE 5 --WD 0.00005 --SESSIONAME "train_simple" --EPOCHS 500 --DEVICE "cuda:0" --STOPCOUNTER 30 --ONLINELOG False --ROBUST_TRAIN False --PRETRAINED "weights/pretrained_480x640_w8_2_6.pth" --OPTIM "ADAM" --SCHED "COS"`
- train robust module on wildfire: `python train.py --yaml_file wildfire.yaml --LR 0.001 --BATCHSIZE 5 --WD 0.00005 --SESSIONAME "train_robust" --EPOCHS 500 --DEVICE "cuda:0" --STOPCOUNTER 30 --ONLINELOG False --ROBUST_TRAIN True --PRETRAINED "weights/robo_fire_aug.pth" --OPTIM "ADAM" --SCHED "COS"`
- train fusion model on urban dataset: `python train.py --yaml_file urban.yaml --LR 0.001 --BATCHSIZE 5 --WD 0.00001 --SESSIONAME "train_simple" --EPOCHS 500 --DEVICE "cuda:0" --STOPCOUNTER 30 --ONLINELOG False --ROBUST_TRAIN False --PRETRAINED "weights/pretrained_480x640_w8_2_6.pth" --OPTIM "ADAM" --SCHED "COS"`
  
### 3. Testing
Customize configurations via the config/ folder or override them with inline arguments.
- test fusion model on wildfire: `python test.py --yaml_file wildfire.yaml --ROBUST_TRAIN False --SESSIONAME "train_simple" --DEVICE "cuda:0" --PRETRAINED "weights/robo_fire_best.pth"`
- test robust module on wildfire: `python test.py --yaml_file wildfire.yaml --ROBUST_TRAIN True --SESSIONAME "train_simple" --DEVICE "cuda:0" --PRETRAINED "weights/robo_fire_robust.pth"`
- test fusion model on urban dataset: `python test.py --yaml_file urban.yaml --ROBUST_TRAIN False --SESSIONAME "train_simple" --DEVICE "cuda:0" --PRETRAINED "weights/robo_urban.pth"`
- To run the demo with custom images, place your files in the outputs/demo folder using the following naming conventions:     `<prefix>_rgb_<postfix>.png` for RGB images, `<prefix>_ir_<postfix>.png` for IR images, and a `.txt` file with rows formatted as `<prefix>_XXX_<postfix>.png`. Replace `<prefix>` and `<postfix>` with any values, ensuring `rgb` and `ir` indicate the modality. Optionally, include ground truth files named `<prefix>_gt_<postfix>.png` to calculate metrics. Run the demo using `python test.py --yaml_file wildfire_demo.yaml` for the wildfire demo or `python test.py --yaml_file urban_demo.yaml for the urban demo` for urban one. If you use custom `.txt` file instead of `demo_fire.txt` and `demo_urban.txt` adjust the YAML config files.

## Citation
(TODO: complete the information when paper accepted)

## Acknowledgment
<a id="1">[1]</a> : PIDnet: Xu, Jiacong et al. “PIDNet: A Real-time Semantic Segmentation Network Inspired by PID Controllers.” 2023 IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR) (2022): 19529-19539. </br>
<a id="2">[2]</a> : FLAME2: Bryce Hopkins, Leo O'Neill, Fatemeh Afghah, Abolfazl Razi, Eric Rowell, Adam Watts, Peter Fule, Janice Coen, August 29, 2022, "FLAME 2: Fire detection and modeLing: Aerial Multi-spectral imagE dataset", IEEE Dataport, doi: https://dx.doi.org/10.21227/swyw-6j78.
