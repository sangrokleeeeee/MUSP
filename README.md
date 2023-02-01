# [MUSP]()
## Introduction
Official implementation of MUSP(Multi-attention-based soft partition network for vehicle re-identification)
This code is based on the [repository](https://github.com/Jakel21/vehicle-ReID).  

## Installation
Pytorch >= 1.6

## Datasets
+ [veri-776](https://github.com/VehicleReId/VeRidataset)
+ [VehicleID]()
+ [VERI-Wild]()

## Train

* baseline model (avg pooling)

```bash
python train_xent_tri.py -s veri -t veri \
--height 128 \
--width 256 \
--optim amsgrad \
--lr 0.0003 \
--max-epoch 60 \
--stepsize 20 40 \
--train-batch-size 64 \
--test-batch-size 100 \
-a resnet50 \
--save-dir log/resnet50-baseline-veri-t \
--gpu-devices 0 \
--train-sampler RandomIdentitySampler \
--random-erase \
--evaluate
```

* VehicleID MUSP(attention 5)

```bash
python train_xent_tri.py -s vehicleID -t vehicleID \
--height 256 \
--width 256 \
--optim adam \
--lr 0.00035 \
--max-epoch 90 \
--stepsize 30 60 \
--train-batch-size 64 \
--test-batch-size 10 \
-a resnet50 \
--save-dir log/resnet50-baseline-vehicleid-channel-global-5 \
--gpu-devices 1 \
--train-sampler RandomIdentitySampler \
--eval-freq 1 \
--random-erase \
--label-smooth \
--attn \
--num-prototypes 5 \
--discard 1 \
--stride3 2 \
--has-global \
--channel
```