from .base_model import BaseModel
from . import networks


class TestModel(BaseModel):
    """该TestModel用于仅生成CycleGAN结果的一个方向。
    该模型将自动设置'--dataset_mode single'，仅从一个集合加载图像。

    详细信息请参阅测试说明。
    """
    @staticmethod
    def modify_commandline_options(parser, is_train=True):
        """添加新的特定于数据集的选项，并重写现有选项的默认值。

        参数:
            parser          -- 原始选项解析器
            is_train (bool) -- 是否为训练阶段或测试阶段。您可以使用此标志添加特定于训练或特定于测试的选项。

        返回:
            修改后的解析器。

        该模型只能在测试时使用。它需要'--dataset_mode single'。
        您需要使用选项 '--model_suffix' 指定网络。
        """
        assert not is_train, 'TestModel不能在训练时使用'
        parser.set_defaults(dataset_mode='single')
        parser.add_argument('--model_suffix', type=str, default='', help='在checkpoints_dir中，[epoch]_net_G[model_suffix].pth将被加载为生成器。')

        return parser

    def __init__(self, opt):
        """初始化pix2pix类。

        参数:
            opt (Option类) -- 存储所有实验标志; 需要是BaseOptions的子类
        """
        assert(not opt.isTrain)
        BaseModel.__init__(self, opt)
        # 指定要打印的训练损失。训练/测试脚本将调用<BaseModel.get_current_losses>
        self.loss_names = []
        # 指定要保存/显示的图像。训练/测试脚本将调用<BaseModel.get_current_visuals>
        self.visual_names = ['real', 'fake']
        # 指定要保存到磁盘的模型。训练/测试脚本将调用<BaseModel.save_networks>和<BaseModel.load_networks>
        self.model_names = ['G' + opt.model_suffix]  # 只需要生成器。
        self.netG = networks.define_G(opt.input_nc, opt.output_nc, opt.ngf, opt.netG,
                                      opt.norm, not opt.no_dropout, opt.init_type, opt.init_gain, self.gpu_ids)

        # 将模型分配给self.netG_[suffix]，以便可以加载它
        # 请参阅<BaseModel.load_networks>
        setattr(self, 'netG' + opt.model_suffix, self.netG)  # 将netG存储在self中。

    def set_input(self, input):
        """从数据加载器中解压输入数据并执行必要的预处理步骤。

        参数:
            input: 一个包含数据本身及其元数据信息的字典。

        我们需要使用'single_dataset'数据集模式。它仅从一个域加载图像。
        """
        self.real = input['A'].to(self.device)
        self.image_paths = input['A_paths']

    def forward(self):
        """运行前向传播。"""
        self.fake = self.netG(self.real)  # G(real)

    def optimize_parameters(self):
        """测试模型没有优化。"""
        pass
