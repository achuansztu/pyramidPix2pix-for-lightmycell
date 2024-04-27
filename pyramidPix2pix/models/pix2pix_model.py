import torch
from .base_model import BaseModel  # 导入基础模型
from . import networks  # 导入网络定义模块
#import models.CoCosNetworks as CoCosNetworks
import torch.nn.functional as F
import torch.nn as nn
import numpy as np
import kornia

class Pix2PixModel(BaseModel):
    """Pix2Pix 模型的实现，用于学习从输入图像到输出图像的映射关系。

    模型训练需要使用 '--dataset_mode aligned' 数据集。
    默认情况下，使用 '--netG unet256' U-Net 生成器，
    使用 '--netD basic' 鉴别器（PatchGAN），
    使用 '--gan_mode' vanilla GAN 损失（原始 GAN 论文中使用的交叉熵目标）。

    Pix2Pix 论文：https://arxiv.org/pdf/1611.07004.pdf
    """

    @staticmethod
    def modify_commandline_options(parser, is_train=True):
        """添加新的数据集特定选项，并重写现有选项的默认值。

        参数:
            parser          -- 原始选项解析器
            is_train (bool) -- 是否为训练阶段。您可以使用此标志添加特定于训练或测试的选项。

        返回:
            修改后的解析器。

        对于 Pix2Pix，我们不使用图像缓冲区
        训练目标是：GAN 损失 + lambda_L1 * ||G(A)-B||_1
        默认情况下，我们使用 vanilla GAN 损失、带有批归一化的 UNet 和对齐的数据集。
        """
        # 将默认值更改为与 Pix2Pix 论文匹配（https://phillipi.github.io/pix2pix/）
        # parser.set_defaults(norm='batch', netG='unet_256', dataset_mode='aligned')
        if is_train:
            parser.set_defaults(pool_size=0, gan_mode='vanilla')
            parser.add_argument('--lambda_L1', type=float, default=25.0, help='L1 损失的权重')

        return parser

    def __init__(self, opt):
        """初始化 Pix2Pix 类。

        参数:
            opt (Option 类) -- 存储所有实验标志的对象；需要是 BaseOptions 的子类
        """
        BaseModel.__init__(self, opt)
        # 指定要打印的训练损失。训练/测试脚本将调用 <BaseModel.get_current_losses>
        if 'noGAN' in opt.pattern:
            self.loss_names = []
        else:
            if self.opt.netD == 'conv':
                self.loss_names = ['G_GAN', 'D']
                self.loss_D = 0
            else:
                self.loss_names = ['G_GAN', 'D_real', 'D_fake']
        if 'L1' in opt.pattern:
            self.loss_names += ['G_L1']
        if 'L2' in opt.pattern:
            self.loss_names += ['G_L2']
        if 'L3' in opt.pattern:
            self.loss_names += ['G_L3']
        if 'L4' in opt.pattern:
            self.loss_names += ['G_L4']
        if 'fft' in opt.pattern:
            self.loss_names += ['G_fft']
        if 'sobel' in opt.pattern:
            self.loss_names += ['G_sobel']
        if 'conv' in opt.pattern:
            self.loss_names += ['G_conv']
        if self.isTrain and ('perc' in opt.pattern or 'contextual' in opt.pattern):
            if 'perc' in opt.pattern:
                self.loss_names += ['G_perc']
                if self.opt.which_perceptual == '5_2':
                    self.perceptual_layer = -1
                elif self.opt.which_perceptual == '4_2':
                    self.perceptual_layer = -2
            if 'contextual' in opt.pattern:
                self.loss_names += ['G_contextual']
            self.vggnet_fix = CoCosNetworks.correspondence.VGG19_feature_color_torchversion(vgg_normal_correct=self.opt.vgg_normal_correct)
            self.vggnet_fix.load_state_dict(torch.load('models/vgg19_conv.pth'))
            self.vggnet_fix.eval()
            for param in self.vggnet_fix.parameters():
                param.requires_grad = False

            self.vggnet_fix.to(self.opt.gpu_ids[0])
            self.contextual_forward_loss = CoCosNetworks.ContextualLoss_forward(self.opt)

        # 指定要保存/显示的图像。训练/测试脚本将调用 <BaseModel.get_current_visuals>
        self.visual_names = ['real_A', 'fake_B', 'real_B']
        # 指定要保存到磁盘的模型。训练/测试脚本将调用 <BaseModel.save_networks> 和 <BaseModel.load_networks>
        if self.isTrain:
            self.model_names = ['G', 'D']
        else:  # 测试时，只加载 G
            self.model_names = ['G']
        # 定义网络（生成器和鉴别器）
        self.netG = networks.define_G(opt.input_nc, opt.output_nc, opt.ngf, opt.netG, opt.norm,
                                      not opt.no_dropout, opt.init_type, opt.init_gain, self.gpu_ids)

        if self.isTrain:  # 定义鉴别器；条件 GAN 需要同时考虑输入图像和输出图像；因此，D 的通道数为 input_nc + output_nc
            self.netD = networks.define_D(opt.input_nc + opt.output_nc, opt.ndf, opt.netD,
                                          opt.n_layers_D, opt.norm, opt.init_type, opt.init_gain, self.gpu_ids)
            if opt.netD == 'conv':
                prenet = torch.load('models/vgg19_conv.pth')
                netdict = {}
                netdict['module.conv1_1.weight'] = prenet['conv1_1.weight']
                netdict['module.conv1_1.bias'] = prenet['conv1_1.bias']
                self.netD.load_state_dict(netdict)

        if self.isTrain:
            # 定义损失函数
            self.criterionGAN = networks.GANLoss(opt.gan_mode).to(self.device)
            self.criterionL1 = torch.nn.L1Loss()
            # 初始化优化器；调度器将自动创建

def backward_D(self):
    """计算鉴别器的 GAN 损失"""
    if self.opt.netD == 'conv':
        fake_feature = self.netD(self.fake_B)
        real_feature = self.netD(self.real_B)
        self.loss_D = - self.criterionL1(fake_feature.detach(), real_feature) * self.opt.weight_conv
    else:
        # Fake; 停止反向传播到生成器，通过分离 fake_B
        fake_AB = torch.cat((self.real_A, self.fake_B), 1)  # 我们使用条件 GANs；我们需要将输入和输出都提供给鉴别器
        pred_fake = self.netD(fake_AB.detach())
        self.loss_D_fake = self.criterionGAN(pred_fake, False)
        # Real
        real_AB = torch.cat((self.real_A, self.real_B), 1)
        pred_real = self.netD(real_AB)
        self.loss_D_real = self.criterionGAN(pred_real, True)
        # 合并损失并计算梯度
        self.loss_D = (self.loss_D_fake + self.loss_D_real) * 0.5
    self.loss_D.backward()

def backward_G(self):
    """计算生成器的 GAN 和 L1 损失"""
    self.loss_G = 0
    if 'noGAN' not in self.opt.pattern:
        if self.opt.netD == 'conv':
            fake_feature = self.netD(self.fake_B)
            real_feature = self.netD(self.real_B)
            self.loss_G_GAN = self.criterionL1(fake_feature, real_feature) * self.opt.weight_conv
        else:
            fake_AB = torch.cat((self.real_A, self.fake_B), 1)
            pred_fake = self.netD(fake_AB)
            self.loss_G_GAN = self.criterionGAN(pred_fake, True)
        self.loss_G += self.loss_G_GAN

    if 'L1' in self.opt.pattern:
        self.loss_G_L1 = self.criterionL1(self.fake_B, self.real_B) * self.opt.lambda_L1
        self.loss_G += self.loss_G_L1
        if 'L2' in self.opt.pattern:
            # 计算 octave2_layer1 的 L2 损失
            # 更多的层级计算类似的 L2 损失
            if 'L3' in self.opt.pattern:
                if 'L4' in self.opt.pattern:
                    # 计算 octave4_layer1 的 L2 损失
                    pass
        if 'mask' in self.opt.pattern:
            self.loss_G_L1 = self.criterionL1(self.fake_B * self.mask, self.real_B * self.mask) * self.opt.lambda_L1

    if 'fft' in self.opt.pattern:
        # 计算频域分解损失
        pass

    if 'perc' in self.opt.pattern or 'contextual' in self.opt.pattern:
        # 计算感知损失或上下文损失
        pass

    if 'conv' in self.opt.pattern:
        # 计算卷积损失
        pass

    if 'sobel' in self.opt.pattern:
        # 计算 Sobel 梯度损失
        pass

    self.loss_G.backward()

def optimize_parameters(self, fixD=False):
    self.forward()  # 计算假图像：G(A)
    if 'noGAN' not in self.opt.pattern and not fixD:
        # 更新 D
        self.set_requires_grad(self.netD, True)  # 启用 D 的反向传播
        self.optimizer_D.zero_grad()  # 将 D 的梯度设置为零
        self.backward_D()  # 计算 D 的梯度
        self.optimizer_D.step()  # 更新 D 的权重
    # 更新 G
    self.set_requires_grad(self.netD, False)  # 优化 G 时，D 不需要梯度
    self.optimizer_G.zero_grad()  # 将 G 的梯度设置为零
    self.backward_G()  # 计算 G 的梯度
    self.optimizer_G.step()  # 更新 G 的权重
