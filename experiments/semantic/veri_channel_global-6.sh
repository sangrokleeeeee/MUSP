python train_xent_tri.py -s veri -t veri \
--height 256 \
--width 256 \
--optim adam \
--lr 0.00035 \
--max-epoch 150 \
--stepsize 30 60 90 120 150 \
--train-batch-size 64 \
--test-batch-size 100 \
-a resnet50 \
--save-dir log/resnet50-baseline-veri-single-channel-global-6 \
--gpu-devices 0 \
--train-sampler RandomIdentitySampler \
--eval-freq 1 \
--random-erase \
--label-smooth \
--attn \
--num-prototypes 6 \
--stride3 2 \
--has-global \
--discard 1 \
--channel