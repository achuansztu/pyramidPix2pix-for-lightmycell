"""
该脚本适用于各种模型（使用选项'--model'：例如，pix2pix、cyclegan、colorization）和不同的数据集（使用选项'--dataset_mode'：例如，aligned、unaligned、single、colorization）。
您需要指定数据集（'--dataroot'）、实验名称（'--name'）和模型（'--model'）。

它首先根据选项创建模型、数据集和可视化器。
然后进行标准网络训练。在训练过程中，它还可视化/保存图像，打印/保存损失图，并保存模型。
该脚本支持继续/恢复训练。使用'--continue_train'来恢复之前的训练。

示例:
    训练CycleGAN模型:
        python train.py --dataroot ./datasets/maps --name maps_cyclegan --model cycle_gan
    训练pix2pix模型:
        python train.py --dataroot ./datasets/facades --name facades_pix2pix --model pix2pix --direction BtoA

查看 options/base_options.py 和 options/train_options.py 了解更多训练选项。
在此查看训练和测试技巧: https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix/blob/master/docs/tips.md
在此查看常见问题: https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix/blob/master/docs/qa.md
"""

import time
from options.train_options import TrainOptions
from data import create_dataset
from models import create_model
from util.visualizer import Visualizer
# from tensorboardX import SummaryWriter
# import torch

if __name__ == '__main__':
    opt = TrainOptions().parse()   # 获取训练选项
    dataset = create_dataset(opt)  # 根据 opt.dataset_mode 和其他选项创建数据集
    dataset_size = len(dataset)    # 获取数据集中图像的数量
    print('训练图像数量 = %d' % dataset_size)

    model = create_model(opt)      # 根据 opt.model 和其他选项创建模型
    # tensor_input = torch.rand(8, 3, 256, 256)
    # with SummaryWriter('graph/perc') as w:
    #     w.add_graph(model, (tensor_input, ))
    model.setup(opt)               # 常规设置：加载并打印网络；创建调度器
    visualizer = Visualizer(opt)   # 创建一个可视化器来显示/保存图像和图表
    total_iters = 0                # 训练迭代总数

    for epoch in range(opt.epoch_count, opt.n_epochs + opt.n_epochs_decay + 1):    # 外部循环不同的 epochs；我们通过 <epoch_count>, <epoch_count>+<save_latest_freq> 保存模型
        epoch_start_time = time.time()  # 计时器，记录整个 epoch 的时间
        iter_data_time = time.time()    # 计时器，记录数据加载的时间
        epoch_iter = 0                  # 当前 epoch 中的训练迭代数，在每个 epoch 开始时重置为 0
        visualizer.reset()              # 重置可视化器：确保它至少每个 epoch 保存一次结果到 HTML
        model.update_learning_rate()    # 在每个 epoch 开始时更新学习率
        for i, data in enumerate(dataset):  # 内部循环每个 epoch 中的每个迭代
            iter_start_time = time.time()  # 计时器，记录每个迭代的计算时间
            if total_iters % opt.print_freq == 0:
                t_data = iter_start_time - iter_data_time

            total_iters += opt.batch_size
            epoch_iter += opt.batch_size
            model.set_input(data)         # 从数据集中解压数据并应用预处理
            model.optimize_parameters()   # 计算损失函数、获取梯度、更新网络权重

            if total_iters % opt.display_freq == 0:   # 在 visdom 上显示图像并保存图像到 HTML 文件
                save_result = total_iters % opt.update_html_freq == 0
                model.compute_visuals()
                visualizer.display_current_results(model.get_current_visuals(), epoch, save_result)

            if total_iters % opt.print_freq == 0:    # 打印训练损失并将日志信息保存到磁盘上
                losses = model.get_current_losses()
                t_comp = (time.time() - iter_start_time) / opt.batch_size
                visualizer.print_current_losses(epoch, epoch_iter, losses, t_comp, t_data)
                if opt.display_id > 0:
                    visualizer.plot_current_losses(epoch, float(epoch_iter) / dataset_size, losses)

            if total_iters % opt.save_latest_freq == 0:   # 每 <save_latest_freq> 迭代缓存最新模型
                print('保存最新模型 (epoch %d, total_iters %d)' % (epoch, total_iters))
                save_suffix = 'iter_%d' % total_iters if opt.save_by_iter else 'latest'
                model.save_networks(save_suffix)

            iter_data_time = time.time()
        if epoch % opt.save_epoch_freq == 0:              # 每 <save_epoch_freq> epochs 缓存我们的模型
            print('在 epoch %d 结束时保存模型，迭代数 %d' % (epoch, total_iters))
            model.save_networks('latest')
            model.save_networks(epoch)

        print('epoch %d / %d 结束 \t 用时: %d 秒' % (epoch, opt.n_epochs + opt.n_epochs_decay, time.time() - epoch_start_time))
