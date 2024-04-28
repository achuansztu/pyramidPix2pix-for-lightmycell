# cyclegan/pyramidPix2pix-for-lightmycell

This repository contains implementations for CycleGAN and pyramidPix2pix for the project "lightmycell".



## cyclegan

### Training

python train.py --dataroot ./datasets/ac_fold_0 --name ac_fold0 --model cycle_gan --gpu_ids 3


### Testing
python test.py --dataroot ./datasets/facades --name facades_pix2pix --model cycle_gan --no_dropout

## pyramidPix2pix

### Training

python train.py --dataroot ./datasets/fold_data/Nucleus_fold_0 --name Nucleus_fold_0 --gpu_ids 3 --pattern L1_L2_L3_L4

### Testing
python test.py --dataroot ./datasets/facades --name facades_pix2pix --model pix2pix




## Our weights

## cyclegan
https://pan.baidu.com/s/1Jcvl74IDmCaqxXLVQjkdXQ?pwd=lv7b 
## pyramidPix2pix
https://pan.baidu.com/s/1c-9pv9i_uus1fhQm_2-Nyw?pwd=jryj 


## References
Make sure to refer to the original repositories for more detailed instructions:

https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix and https://github.com/bupt-ai-cz/BCI

