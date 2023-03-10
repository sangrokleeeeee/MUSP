python train_xent_tri_mem_retrain.py -s veri_wild -t veri_wild \
--height 128 \
--width 256 \
--optim amsgrad \
--lr 0.000003 \
--max-epoch 10 \
--stepsize 60 80 \
--train-batch-size 64 \
--test-batch-size 100 \
-a resnet50 \
--eval-freq 1 \
--save-dir log/resnet50-baseline-vehicleid-to-veriwild-mem-retrain \
--gpu-devices 0 \
--train-sampler rgn \
--random-erase \
--load-weights log/resnet50-baseline-vehicleid-mem/model.pth.tar-60 \