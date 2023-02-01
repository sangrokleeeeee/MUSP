python train_xent_tri.py -s veri_wild -t veri \
--height 128 \
--width 256 \
--optim amsgrad \
--lr 0.0003 \
--max-epoch 60 \
--stepsize 20 40 \
--train-batch-size 64 \
--test-batch-size 100 \
-a resnet50 \
--save-dir log/resnet50-baseline-veriwild-recal-recon-to-veri \
--gpu-devices 1 \
--train-sampler RandomIdentitySampler \
--random-erase \
--evaluate \
--load-weights log/resnet50-baseline-veriwild-recal-recon/model.pth.tar-60