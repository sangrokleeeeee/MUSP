python recal_two.py -s veri -t veri_wild \
--height 128 \
--width 256 \
--optim amsgrad \
--lr 0.0003 \
--max-epoch 60 \
--stepsize 20 40 \
--train-batch-size 32 \
--test-batch-size 100 \
-a resnet50 \
--save-dir log/resnet50-baseline-veri-recal-recon-two \
--gpu-devices 0 \
--train-sampler RandomIdentitySampler \
--random-erase \
--evaluate \
--load-weights log/resnet50-baseline-veri-recal-recon-two/model.pth.tar-60