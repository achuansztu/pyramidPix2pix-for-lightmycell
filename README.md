# pyramidPix2pix/cyclegan-for-lightmycell
For pyramidPix2pix


train:python train.py --dataroot ./datasets/fold_data/Nucleus_fold_0 --name Nucleus_fold_0 --gpu_ids 3 --pattern L1_L2_L3_L4


test: python test.py --dataroot ./datasets/facades --name facades_pix2pix --model pix2pix

For cyclegan:

train:python train.py --dataroot ./datasets/ac_fold_0 --name ac_fold0 --model cycle_gan --gpu_ids 3

test:python test.py --dataroot ./datasets/facades --name facades_pix2pix --model cycle_gan

（refer to https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix and https://github.com/bupt-ai-cz/BCI）
