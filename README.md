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

- 🔥 <b>Lightweight & Powerful multimodal semantic segmentation</b>: Our real-time fusion model outperforms many SOTA models with fewer parameters.
- 🚁 <b>Smoke & Flame Segmentation</b>: Excels in dense smoke conditions, accurately segmenting flames and smoke.
- 🛠️ <b>Robust Fusion</b>: Handles modality misalignment and sensor failures achieving semantic segmentation fusing unregistered inputs.
   
## Updates
- Paper is submitted to ...

## Demo


## Overview
Schematic overview of the proposed fusion model and the robust module.

### Fusion Architecture
<img src="figs/mymodel.png" alt="Model Architecture" width="700"/>

### Robust module
<img src="figs/robust.png" alt="Model Architecture" width="400"/>

### 📊 **Performance Comparison on Urban Scenes (MFNet)**
| **Method**                | **Avg Recall (%)** | **MIoU (%)** | **Params (M)** |
|--------------------------|-------------------|---------------|-----------------|
| [PIDNet-m RGB](https://example.com)  | 65.59            | 51.52         | 34.4            |
| [PIDNet-m IR](https://example.com)   | 65.27            | 50.70         | 34.4            |
| [PIDNet-m Early](https://example.com) | 69.59            | 52.62         | 34.4            |
| [MFNet](https://example.com)          | 59.1             | 39.7          | **0.73**         |
| [RTFNet](https://example.com)        | 63.08            | 53.2          | 185.24          |
| [GMNet](https://example.com)        | **74.1**         | 57.3          | 153             |
| [EGFNet](https://example.com)       | 72.7             | 54.8          | 62.5            |
| [CRM-T](https://example.com)        | -                | 59.7          | 59.1            |
| [Sigma-T](https://example.com)       | 71.3             | 60.23         | 48.3            |
| **Ours**                                             | 71.1             | **60.6**      | 29.5            |
### 📊 **Performance Comparison on FLAME2**
| **Method**                | **Avg Recall (%)** | **MIoU (%)** | **Params (M)** |
|--------------------------|-------------------|---------------|-----------------|
| [PIDNet-RGB](https://example.com)  | 75.66            | 61.21         | 34.4            |
| [PIDNet-IR](https://example.com)   | 83.05            | 58.71         | 34.4            |
| [PIDNet-Early](https://example.com) | 88.25            | 73.90         | 34.4            |
| [MFNet](https://example.com)        | 93.53            | 80.26         | **0.73**         |
| [RTFNet](https://example.com)      | 73.87            | 65.42         | 185.24          |
| [GMNet](https://example.com)      | 67.53            | 54.08         | 153             |
| [EGFNet](https://example.com)     | 74.27            | 60.98         | 62.5            |
| [CRM-T](https://example.com)      | -                | -             | 59.1            |
| [Sigma-T](https://example.com)     | 92.6             | 86.27         | 48.3            |
| **Ours**                            | **94.34**        | **88.39**     | 29.5            |

## Usage

### 0. Setup
- Run the `setup.sh` script to download pretrain models and datasets
- Run the `pip install -r requirements.txt` to install python requirements for the project

### 1. Dataset
- You download the desired dataset with pairs of RGB and IR images. For FLAME2 [[2]](#2) you can download our annotations from this link.
- The RGB, IR and ground truth labels should contain same name with difference a subword `rgb`, `ir`, `gt` respectively. For example `image_rgb_10201.png`, `image_ir_10201.png`, `image_gt_10201.png`
- The ground truth sould contain 3 annotated classed in RGB format. More over `background` pixels as `(0,0,0)`, `flame` pixels as `(255,255,255)` and `smoke` ones as `(125, 125, 125)`.
- Complete the desired paths in `config` file in `config` folder.
  
### 2. Training
- Complete the desired parameters in `config` file in `config` folder.
- Run the `traintrain_script.py`. For example:
  - RGB: `python train_script.py --BATCHSIZE=5 --DEVICE=cpu --LR=0.003351038521214215 --MODE=rgb --USE_OHEM=True --WD=3.4921444763980365e-05 --SESSIONAME=rgb --ONLINELOG=False --ONE_INP_AUG=True --OPTIM=SGD --MODEL=pidnet_s --ONE_INP_AUG=False`
  - IR: `python train_script.py --BATCHSIZE=29 --DEVICE=cpu --LR=0.009416483156345572 --MODE=ir --USE_OHEM=True --WD=0.004732551500334286 --SESSIONAME=ir --OPTIM=ADAM --MODEL=pidnet_s --ONE_INP_AUG=False --OPTIM=SGD --MODEL=pidnet_s --ONE_INP_AUG=False`
  - Early: `python train_script.py --BATCHSIZE=12 --DEVICE=cpu --LR=0.007831290330825162 --MODE=fusion --USE_OHEM=True --WD=1.3782621109462256e-05 --SESSIONAME=early --OPTIM=ADAM --MODEL=pidnet_s --ONE_INP_AUG=False`
  - MIFF: `python train_script.py --BATCHSIZE=12 --DEVICE=cpu --LR=0.0008314267980926564 --MODE=fusion --USE_OHEM=False --WD=3.058540738874065e-06 --STOPCOUNTER=12 --SESSIONAME=miff --ONLINELOG=False --ONE_INP_AUG=True --OPTIM=ADAM --MODEL=firesmoke_s`
  
### 3. Testing
- Complete the desired parameters in `config` file in `config` folder.
- Run the `test_script.py`. For example
  - RGB: `python test_script.py --MODE=rgb --MODEL=pidnet_s --PRETRAINED=weights/best_rgb_mode.pt`
  - IR: `python test_script.py --MODE=ir --MODEL=pidnet_s --PRETRAINED=weights/best_ir_mode.pt`
  - Early: `python test_script.py --MODE=fusion --MODEL=pidnet_s --PRETRAINED=weights/best_early_mode.pt`
  - MIFF: `python test_script.py --MODE=fusion --MODEL=firesmoke_s --PRETRAINED=weights/best_miff_mode.pt`
- The above commands will ouput several evaluaton metrics

## Citation
(TODO: complete the information when paper accepted)

## Acknowledgment
<a id="1">[1]</a> : PIDnet: Xu, Jiacong et al. “PIDNet: A Real-time Semantic Segmentation Network Inspired by PID Controllers.” 2023 IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR) (2022): 19529-19539. </br>
<a id="2">[2]</a> : FLAME2: Bryce Hopkins, Leo O'Neill, Fatemeh Afghah, Abolfazl Razi, Eric Rowell, Adam Watts, Peter Fule, Janice Coen, August 29, 2022, "FLAME 2: Fire detection and modeLing: Aerial Multi-spectral imagE dataset", IEEE Dataport, doi: https://dx.doi.org/10.21227/swyw-6j78.


train wildfire: python train.py --yaml_file wildfire.yaml --LR 0.001 --BATCHSIZE 5 --WD 0.00005 --SESSIONAME "train_simple" --EPOCHS 100 --DEVICE "cuda:0" --STOPCOUNTER 30 --ONLINELOG False --ROBUST_TRAIN False --PRETRAINED "weights/pretrained_480x640_w8_2_6.pth" --OPTIM "ADAM" --SCHED "COS"

train robust module wildfire: python train.py --yaml_file wildfire.yaml --LR 0.001 --BATCHSIZE 5 --WD 0.00005 --SESSIONAME "train_robust" --EPOCHS 100 --DEVICE "cuda:0" --STOPCOUNTER 30 --ONLINELOG False --ROBUST_TRAIN False --PRETRAINED "weights/robo_fire_aug.pth" --OPTIM "ADAM" --SCHED "COS"
