python recal.py -s veri -t vehicleID \
--height 128 \
--width 256 \
--optim sgd \
--lr 0.0003 \
--max-epoch 60 \
--stepsize 20 40 \
--train-batch-size 32 \
--test-batch-size 100 \
-a resnet50 \
--save-dir log/resnet50-baseline-veri-to-vehicleID \
--gpu-devices 1 \
--train-sampler RandomIdentitySampler \
--evaluate \
--load-weights log/resnet50-baseline-veri-recal-recon/model.pth.tar-60