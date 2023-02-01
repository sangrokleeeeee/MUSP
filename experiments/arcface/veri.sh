python arcface.py -s veri -t veri \
--height 128 \
--width 256 \
--optim amsgrad \
--lr 0.0003 \
--max-epoch 60 \
--stepsize 20 40 \
--train-batch-size 192 \
--test-batch-size 100 \
--random-erase \
-a resnet50 \
--eval-freq 10 \
--save-dir log/resnet50-baseline-veri-arc \
--gpu-devices 0 \
--train-sampler None