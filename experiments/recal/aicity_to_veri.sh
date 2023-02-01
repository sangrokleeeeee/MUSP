python recal.py -s aicity -t veri \
--height 128 \
--width 256 \
--optim amsgrad \
--lr 0.0003 \
--max-epoch 60 \
--stepsize 20 40 \
--train-batch-size 64 \
--test-batch-size 100 \
-a resnet50 \
--save-dir log/resnet50-baseline-aicity-to-veri \
--gpu-devices 1 \
--train-sampler RandomIdentitySampler \
--random-erase
# --load-weights log/resnet50-baseline/model.pth.tar-60