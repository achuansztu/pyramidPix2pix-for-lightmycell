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
import os
from options.test_options import TestOptions
from data import create_dataset
from models import create_model
from util.visualizer import save_images
from util import html

if __name__ == '__main__':
    opt = TestOptions().parse()  # 获取测试选项
    # 硬编码一些测试参数
    opt.num_threads = 0   # 测试代码仅支持 num_threads = 0
    opt.batch_size = 1    # 测试代码仅支持 batch_size = 1
    opt.serial_batches = True  # 禁用数据混洗；如果需要在随机选择的图像上获取结果，请取消注释此行。
    opt.no_flip = True    # 不翻转；如果需要在翻转的图像上获取结果，请取消注释此行。
    opt.display_id = -1   # 不使用 visdom 显示；测试代码将结果保存到 HTML 文件中。
    dataset = create_dataset(opt)  # 根据 opt.dataset_mode 和其他选项创建数据集
    model = create_model(opt)      # 根据 opt.model 和其他选项创建模型
    model.setup(opt)               # 常规设置：加载和打印网络；创建调度器
    # 创建一个网站
    web_dir = os.path.join(opt.results_dir, opt.name, '{}_{}'.format(opt.phase, opt.epoch))  # 定义网站目录
    if opt.load_iter > 0:  # 默认情况下，load_iter 为 0
        web_dir = '{:s}_iter{:d}'.format(web_dir, opt.load_iter)
    print('creating web directory', web_dir)
    webpage = html.HTML(web_dir, 'Experiment = %s, Phase = %s, Epoch = %s' % (opt.name, opt.phase, opt.epoch))
    # 以评估模式进行测试。这仅影响像批标准化和 dropout 这样的层。
    # 对于 [pix2pix]：我们在原始的 pix2pix 中使用了批标准化和 dropout。您可以尝试使用和不使用 eval() 模式。
    # 对于 [CycleGAN]：这不应影响 CycleGAN，因为 CycleGAN 使用了没有 dropout 的实例标准化。
    if opt.eval:
        model.eval()
    for i, data in enumerate(dataset):
        if i >= opt.num_test:  # 仅将我们的模型应用于 opt.num_test 张图像。
            break
        model.set_input(data)  # 从数据加载器中解压数据
        model.test()           # 运行推断
        visuals = model.get_current_visuals()  # 获取图像结果
        img_path = model.get_image_paths()     # 获取图像路径
        if i % 5 == 0:  # 将图像保存到 HTML 文件
            print('processing (%04d)-th image... %s' % (i, img_path))
        save_images(webpage, visuals, img_path, aspect_ratio=opt.aspect_ratio, width=opt.display_winsize)
    webpage.save()  # 保存 HTML
