python train_xent_tri_mem.py -s veri_wild -t veri_wild \
--height 128 \
--width 256 \
--optim amsgrad \
--lr 0.0003 \
--max-epoch 60 \
--stepsize 20 40 \
--train-batch-size 64 \
--test-batch-size 100 \
-a resnet50 \
--eval-freq 10 \
--save-dir log/resnet50-baseline-veriwild-mem \
--gpu-devices 1 \
--train-sampler RandomIdentitySampler \
--random-erase