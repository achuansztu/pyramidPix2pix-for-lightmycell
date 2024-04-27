"""General-purpose test script for image-to-image translation.

Once you have trained your model with train.py, you can use this script to test the model.
It will load a saved model from '--checkpoints_dir' and save the results to '--results_dir'.

It first creates model and dataset given the option. It will hard-code some parameters.
It then runs inference for '--num_test' images and save results to an HTML file.

Example (You need to train models first or download pre-trained models from our website):
    Test a CycleGAN model (both sides):
        python test.py --dataroot ./datasets/maps --name maps_cyclegan --model cycle_gan

    Test a CycleGAN model (one side only):
        python test.py --dataroot datasets/horse2zebra/testA --name horse2zebra_pretrained --model test --no_dropout

    The option '--model test' is used for generating CycleGAN results only for one side.
    This option will automatically set '--dataset_mode single', which only loads the images from one set.
    On the contrary, using '--model cycle_gan' requires loading and generating results in both directions,
    which is sometimes unnecessary. The results will be saved at ./results/.
    Use '--results_dir <directory_path_to_save_result>' to specify the results directory.

    Test a pix2pix model:
        python test.py --dataroot ./datasets/facades --name facades_pix2pix --model pix2pix --direction BtoA

See options/base_options.py and options/test_options.py for more test options.
See training and test tips at: https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix/blob/master/docs/tips.md
See frequently asked questions at: https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix/blob/master/docs/qa.md
"""
from pathlib import Path
from os.path import join, isdir, basename
from os import mkdir, listdir
import tifffile
import xmltodict
import os
from options.test_options import TestOptions
from data import create_dataset
from models import create_model
#from util.visualizer import save_images
from util import html
import torchvision.transforms as transforms
import random
import numpy as np
import torch.utils.data as data
from PIL import Image
import torchvision.transforms as transforms
from abc import ABC, abstractmethod
try:
    import wandb
except ImportError:
    print('Warning: wandb package cannot be found. The option "--use_wandb" will result in error.')


if __name__ == '__main__':
    
    def get_transform(opt, params=None, grayscale=False, method=transforms.InterpolationMode.BICUBIC, convert=True):
        transform_list = []
        if grayscale:
            transform_list.append(transforms.Grayscale(1))
        if 'resize' in opt.preprocess:
            osize = [opt.load_size, opt.load_size]
            transform_list.append(transforms.Resize(osize, method))
        elif 'scale_width' in opt.preprocess:
            transform_list.append(transforms.Lambda(lambda img: __scale_width(img, opt.load_size, opt.crop_size, method)))
    
        if 'crop' in opt.preprocess:
            if params is None:
                transform_list.append(transforms.RandomCrop(opt.crop_size))
            else:
                transform_list.append(transforms.Lambda(lambda img: __crop(img, params['crop_pos'], opt.crop_size)))
    
        if opt.preprocess == 'none':
            transform_list.append(transforms.Lambda(lambda img: __make_power_2(img, base=4, method=method)))
    
        if not opt.no_flip:
            if params is None:
                transform_list.append(transforms.RandomHorizontalFlip())
            elif params['flip']:
                transform_list.append(transforms.Lambda(lambda img: __flip(img, params['flip'])))
    
        if convert:
            #transform_list += [transforms.Lambda(lambda img: __normalize_image_to_01(img))]
            transform_list += [transforms.ToTensor()]
            if grayscale:
                transform_list += [transforms.Normalize((0.5,), (0.5,))]
            else:
                transform_list += [transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]
        return transforms.Compose(transform_list)
    
    def read_image(location):
        # Read the TIFF file and get the image and metadata
        with tifffile.TiffFile(location) as tif:
            image_data = tif.asarray()    # Extract image data
            metadata   = tif.ome_metadata # Get the existing metadata in a DICT
        return image_data, metadata

    def __normalize_image_to_01(img):

        # 将图像转换为浮点数numpy数组
        np_img = np.array(img, dtype=np.float32)
        
        # 获取数组的最大值和最小值
        min_val = np.min(np_img)
        max_val = np.max(np_img)
        
        # 归一化到0-1
        if max_val - min_val != 0:
            np_img = (np_img - min_val) / (max_val - min_val)
        else:
            # 避免除以0的情况，如果图像的所有像素值都相同，则直接设置为0
            np_img = np.zeros(np_img.shape, dtype=np.float32)
        np_img = Image.fromarray((np_img * 255).astype(np.uint8))
        
        return np_img
    
    def save_image(*, location, array, metadata, height, width):
        print(" --> save "+str(location))
        pixels = xmltodict.parse(metadata)["OME"]["Image"]["Pixels"]
        physical_size_x = float(pixels["@PhysicalSizeX"])
        physical_size_y = float(pixels["@PhysicalSizeY"])
        print('-----------physical_size_y------',physical_size_y)
    
        if array.is_cuda:
            array = array.cpu()
        array = array.detach()  # 如果张量需要梯度，先使用detach()
    
        # 转换为PIL图像
        array = array.squeeze(0)  # 移除批次维度
        array = (array + 1) / 2 * 255  # 将数据范围从[-1, 1]调整到[0, 255]
        array = array.numpy().astype(np.uint8)  # 转换为numpy数组并转换类型
        if array.ndim == 3 and array.shape[0] in {1, 3}:  # 对于单通道（灰度）或三通道（RGB）图像
            array = np.moveaxis(array, 0, -1)  # 将通道从(C,H,W)转换为(H,W,C)
        pil_image = Image.fromarray(array)
        
        # 调整图像大小
        pil_image = pil_image.resize((width,height), Image.LANCZOS)
    
        # 再次转换为numpy数组以保存为TIFF
        array = np.array(pil_image)
    
        # 保存为TIFF
        tifffile.imwrite(location,
                         array,
                         description=metadata,
                         resolution=(physical_size_x, physical_size_y),
                         metadata=pixels,
                         tile=(128, 128),
                         )
    
    opt = TestOptions().parse()  # get test options
    # hard-code some parameters for test
    opt.num_threads = 0   # test code only supports num_threads = 0
    opt.batch_size = 1    # test code only supports batch_size = 1
    opt.serial_batches = True  # disable data shuffling; comment this line if results on randomly chosen images are needed.
    opt.no_flip = True    # no flip; comment this line if results on flipped images are needed.
    opt.display_id = -1   # no visdom display; the test code saves the results to a HTML file.
    #dataset = create_dataset(opt)  # create a dataset given opt.dataset_mode and other options
    model = create_model(opt)      # create a model given opt.model and other options
    model.setup(opt)               # regular setup: load and print networks; create schedulers

    # initialize logger
    #if opt.use_wandb:
    #    wandb_run = wandb.init(project=opt.wandb_project_name, name=opt.name, config=opt) if not wandb.run else wandb.run
    #    wandb_run._label(repo='CycleGAN-and-pix2pix')

    # create a website
    #web_dir = os.path.join(opt.results_dir, opt.name, '{}_{}'.format(opt.phase, opt.epoch))  # define the website directory
    #if opt.load_iter > 0:  # load_iter is 0 by default
    #    web_dir = '{:s}_iter{:d}'.format(web_dir, opt.load_iter)
    #print('creating web directory', web_dir)
    #webpage = html.HTML(web_dir, 'Experiment = %s, Phase = %s, Epoch = %s' % (opt.name, opt.phase, opt.epoch))
    # test with eval mode. This only affects layers like batchnorm and dropout.
    # For [pix2pix]: we use batchnorm and dropout in the original pix2pix. You can experiment it with and without eval() mode.
    # For [CycleGAN]: It should not affect CycleGAN as CycleGAN uses instancenorm without dropout.
    if opt.eval:
        model.eval()
    INPUT_PATH = Path("/input")
    OUTPUT_PATH = Path("/output")
    transmitted_light_path = join(INPUT_PATH , "images","organelles-transmitted-light-ome-tiff")
    if not isdir(join(OUTPUT_PATH,"images")): mkdir(join(OUTPUT_PATH,"images"))
    for input_file_name in listdir(transmitted_light_path):
        if input_file_name.endswith(".tiff"):
            print(" --> Predict " + input_file_name)
            image_input,metadata=read_image(join(transmitted_light_path,input_file_name))
            height, width = image_input.shape
            print('--------image_input--------',image_input.shape)
            image_input = __normalize_image_to_01(image_input)
#B_transform =  get_transform(self.opt, transform_params, grayscale=(self.output_nc == 1))
            transform = get_transform(opt, grayscale='True')
            image_input = transform(image_input)
            description = xmltodict.parse(metadata)
            tl= description["OME"]['Image']["Pixels"]["Channel"]["@Name"]
            
            #for organelle in ["Nucleus", "Mitochondria", "Actin", "Tubulin"]:
            # Perform the prediction
            #image_predict = model.predict(image_input, tl, organelle)
            model.set_input(image_input)  # unpack data from data loader
            model.test()           # run inference
            image_predict = model.get_current_visuals() 
            #Save your new predicted images
            print(image_predict)
            output_organelle_path = join(OUTPUT_PATH, "images",  opt.name[6:].lower() + "-fluorescence-ome-tiff")
            if not isdir(output_organelle_path):  mkdir(output_organelle_path)
            save_image(location=join(output_organelle_path,basename(input_file_name)), array=image_predict['fake'],metadata=metadata,height = height, width = width)
    #for i, data in enumerate(dataset):
    #    if i >= opt.num_test:  # only apply our model to opt.num_test images.
    #        break
    #    model.set_input(data)  # unpack data from data loader
    #    model.test()           # run inference
    #    visuals = model.get_current_visuals()  # get image results
    #    img_path = model.get_image_paths()     # get image paths
    #    if i % 5 == 0:  # save images to an HTML file
    #        print('processing (%04d)-th image... %s' % (i, img_path))
    #    OUTPUT_PATH = Path("/output")
    #    output_organelle_path = join(OUTPUT_PATH, "images", organelle.lower() + "-fluorescence-ome-tiff")
    #    if not isdir(output_organelle_path):  mkdir(output_organelle_path)
    #    save_image(location=join(output_organelle_path,basename(input_file_name)), array=visuals ,metadata=metadata)
    #    #save_images(webpage, visuals, img_path, aspect_ratio=opt.aspect_ratio, width=opt.display_winsize, use_wandb=opt.use_wandb)
    ##webpage.save()  # save the HTML
