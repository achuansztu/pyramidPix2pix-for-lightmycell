from data.base_dataset import BaseDataset, get_transform
from data.image_folder import make_dataset
from PIL import Image
import numpy as np

class SingleDataset(BaseDataset):
    """This dataset class can load a set of images specified by the path --dataroot /path/to/data.

    It can be used for generating CycleGAN results only for one side with the model option '-model test'.
    """

    def __init__(self, opt):
        """Initialize this dataset class.

        Parameters:
            opt (Option class) -- stores all the experiment flags; needs to be a subclass of BaseOptions
        """
        BaseDataset.__init__(self, opt)
        self.A_paths = sorted(make_dataset(opt.dataroot, opt.max_dataset_size))
        input_nc = self.opt.output_nc if self.opt.direction == 'BtoA' else self.opt.input_nc
        self.transform = get_transform(opt, grayscale=(input_nc == 1))

    def __getitem__(self, index):
        """Return a data point and its metadata information.

        Parameters:
            index - - a random integer for data indexing

        Returns a dictionary that contains A and A_paths
            A(tensor) - - an image in one domain
            A_paths(str) - - the path of the image
        """
        #A_path = self.A_paths[index % self.A_size]  # make sure index is within then range
        ##print('-----',A_path)
        #if self.opt.serial_batches:   # make sure index is within then range
        #    index_B = index % self.B_size
        #else:   # randomize the index for domain B to avoid fixed pairs.
        #    index_B = random.randint(0, self.B_size - 1)
        #print('-----A_path----',A_path)
        #B_path = A_path.replace('trainA', 'trainB')
        #print('-----B_path----',B_path)
        A_path = self.A_paths[index]
        A_img = Image.open(A_path)
        A_img = self.__normalize_image_to_01(A_img)
#B_transform =  get_transform(self.opt, transform_params, grayscale=(self.output_nc == 1))
        A = self.transform(A_img)
        #B = B_transform(B_img)
        return {'A': A, 'A_paths': A_path}

    def __len__(self):
        """Return the total number of images in the dataset."""
        return len(self.A_paths)
    
    def postprocess(self, image):
        image = ((image + 1) * 127.5).clamp(0, 255).to(torch.uint8)
        image = image.permute(1, 2, 0)
        image = image.contiguous().cpu().numpy()
        return Image.fromarray(image)
    
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