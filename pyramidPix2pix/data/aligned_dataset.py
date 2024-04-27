import os
from data.base_dataset import BaseDataset, get_params, get_transform
from data.image_folder import make_dataset
from PIL import Image
import random
import numpy as np
import torch 

class AlignedDataset(BaseDataset):
    """
    This dataset class can load unaligned/unpaired datasets.

    It requires two directories to host training images from domain A '/path/to/data/trainA'
    and from domain B '/path/to/data/trainB' respectively.
    You can train the model with the dataset flag '--dataroot /path/to/data'.
    Similarly, you need to prepare two directories:
    '/path/to/data/testA' and '/path/to/data/testB' during test time.
    """

    def __init__(self, opt):
        """Initialize this dataset class.

        Parameters:
            opt (Option class) -- stores all the experiment flags; needs to be a subclass of BaseOptions
        """
        BaseDataset.__init__(self, opt)
        self.dir_A = os.path.join(opt.dataroot, opt.phase + 'A')  # create a path '/path/to/data/trainA'
        self.dir_B = os.path.join(opt.dataroot, opt.phase + 'B')  # create a path '/path/to/data/trainB'

        self.A_paths = sorted(make_dataset(self.dir_A, opt.max_dataset_size))   # load images from '/path/to/data/trainA'
        self.B_paths = sorted(make_dataset(self.dir_B, opt.max_dataset_size))    # load images from '/path/to/data/trainB'
        self.A_size = len(self.A_paths)  # get the size of dataset A
        self.B_size = len(self.B_paths)  # get the size of dataset B
        btoA = self.opt.direction == 'BtoA'
        self.input_nc = self.opt.output_nc if btoA else self.opt.input_nc       # get the number of channels of input image
        self.output_nc = self.opt.input_nc if btoA else self.opt.output_nc      # get the number of channels of output image
        #self.transform_A = get_transform(self.opt ,grayscale=(input_nc == 1))
        #self.transform_B = get_transform(self.opt, grayscale=(output_nc == 1))


    def __getitem__(self, index):
        """Return a data point and its metadata information.

        Parameters:
            index (int)      -- a random integer for data indexing

        Returns a dictionary that contains A, B, A_paths and B_paths
            A (tensor)       -- an image in the input domain
            B (tensor)       -- its corresponding image in the target domain
            A_paths (str)    -- image paths
            B_paths (str)    -- image paths
        """
        A_path = self.A_paths[index % self.A_size]  # make sure index is within then range

        B_path = A_path.replace('trainA', 'trainB')

        A_img = Image.open(A_path)
        A_img = self.__normalize_image_to_01(A_img).convert('RGB')

        B_img = Image.open(B_path)
        B_img = self.__normalize_image_to_01(B_img).convert('RGB')
        # apply image transformation
        transform_params = get_params(self.opt, B_img.size)
        A_transform =  get_transform(self.opt, transform_params, grayscale=(self.input_nc == 1))
        B_transform =  get_transform(self.opt, transform_params, grayscale=(self.output_nc == 1))
        A = A_transform(A_img)
        B = B_transform(B_img)

        return {'A': A, 'B': B, 'A_paths': A_path, 'B_paths': B_path}

    def __len__(self):
        """Return the total number of images in the dataset.

        As we have two datasets with potentially different number of images,
        we take a maximum of
        """

        return max(self.A_size, self.B_size)
    
    def __normalize_image_to_01(self, img):

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
    def postprocess(self, image):
        image = ((image + 1) * 127.5).clamp(0, 255).to(torch.uint8)
        image = image.permute(1, 2, 0)
        image = image.contiguous().cpu().numpy()
        return Image.fromarray(image)