python train_xent_tri_mem_retrain.py -s vehicleID -t vehicleID \
--height 128 \
--width 256 \
--optim amsgrad \
--lr 0.000003 \
--max-epoch 60 \
--stepsize 60 80 \
--train-batch-size 64 \
--test-batch-size 100 \
-a resnet50 \
--eval-freq 1 \
--save-dir log/resnet50-baseline-veriwild-to-vehicleid-mem-retrain \
--gpu-devices 1 \
--train-sampler RandomIdentitySampler \
--random-erase \
--load-weights log/resnet50-baseline-veriwild-mem/model.pth.tar-60 \