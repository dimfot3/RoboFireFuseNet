# async_fusion_fire_smoke_seg

train wildfire: python train.py --yaml_file wildfire.yaml --LR 0.001 --BATCHSIZE 5 --WD 0.00005 --SESSIONAME "train_simple" --EPOCHS 100 --DEVICE "cuda:0" --STOPCOUNTER 30 --ONLINELOG False --ROBUST_TRAIN False --PRETRAINED "weights/pretrained_480x640_w8_2_6.pth" --OPTIM "ADAM" --SCHED "COS"

train robust module wildfire: python train.py --yaml_file wildfire.yaml --LR 0.001 --BATCHSIZE 5 --WD 0.00005 --SESSIONAME "train_robust" --EPOCHS 100 --DEVICE "cuda:0" --STOPCOUNTER 30 --ONLINELOG False --ROBUST_TRAIN False --PRETRAINED "weights/robo_fire_aug.pth" --OPTIM "ADAM" --SCHED "COS"
