python recal.py -s vehicleID -t veri_wild \
--height 128 \
--width 256 \
--optim amsgrad \
--lr 0.0003 \
--max-epoch 60 \
--stepsize 20 40 \
--train-batch-size 64 \
--test-batch-size 100 \
-a resnet50 \
--save-dir log/resnet50-recal-vehicleID-to-veriwild \
--gpu-devices 0 \
--train-sampler RandomIdentitySampler \
--evaluate \
--load-weights log/resnet50-baseline-vehicleID-recal-recon/model.pth.tar-60