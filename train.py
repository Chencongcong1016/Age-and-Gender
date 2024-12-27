import torch.optim
import torch.utils.data
from tensorboardX import SummaryWriter
from torch import nn
from torch.optim.lr_scheduler import StepLR

from data_gen import AgeGenDataset
from focal_loss import FocalLoss
from models import resnet18, resnet34, resnet50, resnet101
from utils import *


def train_net(args):
    """
    定义训练函数，接受参数对象 `args`，该对象包含所有命令行传入的参数
    """
    torch.manual_seed(7)             # 设置 PyTorch 随机数生成器的种子为 7，以保证每次运行时模型初始化的一致性
    np.random.seed(7)                # 设置 NumPy 随机数生成器的种子为 7，以保证每次运行时数据处理中的随机操作一致性
    best_loss = 100000               # 初始化一个 `best_loss` 变量，用于跟踪训练过程中最小的损失值，初始值设置为一个较大的数（100000）
    torch.manual_seed(7)             # 再次设置 PyTorch 随机种子为 7（重复设置是冗余的，第一次已足够）
    np.random.seed(7)                # 再次设置 NumPy 随机种子为 7（同样，重复设置是冗余的）
    checkpoint = None                # 初始化 `checkpoint` 变量为 None，用于存储模型训练过程中的检查点（例如，保存模型的权重）
    start_epoch = 0                  # 设置训练的起始 epoch 为 0，通常在恢复训练时用来标记开始的 epoch
    writer = SummaryWriter()         # 创建一个 `SummaryWriter` 对象，用于将训练过程中的数据（如损失、精度等）写入 TensorBoard 进行可视化
    epochs_since_improvement = 0     # 初始化 `epochs_since_improvement` 变量为 0，用于跟踪自上次损失改善以来的 epoch 数

    # 初始化/加载检查点
    if checkpoint is None:                  # 如果没有提供检查点文件，说明是从头开始训练
        if args.network == 'r100':          # 根据命令行参数选择不同的网络架构
            model = resnet101(args)         # 如果选择的是 resnet101，则调用相应的函数来创建该网络模型
        elif args.network == 'r50':         # 如果选择的是 resnet50
            model = resnet50(args)          # 创建 resnet50 模型
        elif args.network == 'r34':         # 如果选择的是 resnet34
            model = resnet34(args)          # 创建 resnet34 模型
        elif args.network == 'r18':         # 如果选择的是 resnet18
            model = resnet18(args)          # 创建 resnet18 模型
        else:                               # 如果选择的网络架构不在上述四种中
            model = resnet50(args)          # 默认为 resnet50 架构

        # 初始化优化器
        optimizer = torch.optim.SGD(params=filter(lambda p: p.requires_grad, model.parameters()), lr=args.lr,
                                    momentum=args.mom, weight_decay=args.weight_decay)
        # 使用 SGD 优化器来优化网络的可训练参数。`filter(lambda p: p.requires_grad, model.parameters())` 会选出所有需要梯度更新的参数。
        # 学习率（lr）、动量（momentum）和权重衰减（weight_decay）均从命令行参数中传入。

    else:  # 如果提供了检查点，说明是恢复训练
        checkpoint = torch.load(checkpoint)                                 # 加载检查点文件
        start_epoch = checkpoint['epoch'] + 1                               # 从检查点中恢复起始 epoch，恢复后开始训练下一个 epoch
        epochs_since_improvement = checkpoint['epochs_since_improvement']   # 恢复自上次性能改善以来的 epoch 数
        model = checkpoint['model']                                         # 恢复模型的权重（包括网络结构和训练状态）
        optimizer = checkpoint['optimizer']                                 # 恢复优化器的状态（包括学习率、动量等信息）

    # 如果可用，移动到GPU
    model = model.to(device)                # 将模型移动到 GPU（如果 GPU 可用）。`device` 是由 PyTorch 自动判断设备类型的变量（CPU 或 GPU）

    # 检查是否选择了 focal loss 作为损失函数
    if args.focal_loss:             
        age_criterion = FocalLoss(gamma=args.gamma).to(device)              # 如果选择了 focal loss, 为年龄分类创建 FocalLoss 损失函数，并将其移动到设备（GPU 或 CPU）
        gender_criterion = FocalLoss(gamma=args.gamma).to(device)           # 为性别分类创建 FocalLoss 损失函数，并将其移动到设备
    else:
        age_criterion = nn.CrossEntropyLoss().to(device)                    # 如果没有选择 focal loss，则使用交叉熵损失函数（CrossEntropyLoss）进行年龄分类
        gender_criterion = nn.CrossEntropyLoss().to(device)                 # 使用交叉熵损失函数进行性别分类

    criterion_info = (age_criterion, gender_criterion, args.age_weight)     # 将年龄和性别的损失函数以及年龄的权重封装成元组

    # 自定义dataloaders
    train_dataset = AgeGenDataset('train')                               # 创建自定义的训练数据集，'train' 指定加载训练数据
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=workers,
                                               pin_memory=True)          # 使用 DataLoader 加载训练数据集，指定批大小、是否打乱数据、工作线程数等
    val_dataset = AgeGenDataset('valid')                                 # 创建自定义的验证数据集，'valid' 指定加载验证数据
    val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=workers,
                                             pin_memory=True)            # 使用 DataLoader 加载验证数据集，指定批大小等

    scheduler = StepLR(optimizer, step_size=args.lr_step, gamma=0.1)     # 创建学习率调度器 StepLR，每隔 step_size 个 epoch 更新学习率，gamma 表示学习率衰减的比例（每次衰减为 0.1）

    # 时代（Epoch）循环
    for epoch in range(start_epoch, epochs):    # 遍历每个训练周期（从 start_epoch 开始，直到 epochs）
        scheduler.step()                        # 更新学习率，根据调度器（scheduler）的策略进行学习率调整

        # 一个（Epoch）循环的训练
        train_loss, train_gen_accs, train_age_mae = train(train_loader=train_loader,     # 调用训练函数，传入训练数据加载器 train_loader
                                                          model=model,                   # 传入模型
                                                          criterion_info=criterion_info, # 传入损失函数信息（年龄和性别的损失函数，以及年龄损失的权重）
                                                          optimizer=optimizer,           # 传入优化器
                                                          epoch=epoch)                   # 当前的训练周期（epoch）
        writer.add_scalar('训练损失', train_loss, epoch)                   # 记录训练损失（Train Loss）
        writer.add_scalar('训练时性别分类的准确率', train_gen_accs, epoch)    # 记录训练时性别分类的准确率（Train Gender Accuracy）
        writer.add_scalar('训练时年龄预测的平均绝对误差', train_age_mae, epoch)             # 记录训练时年龄预测的平均绝对误差（Train Age MAE）

        # 一个时代的验证
        valid_loss, valid_gen_accs, valid_age_mae = validate(val_loader=val_loader,           # 调用验证函数，传入验证数据加载器 val_loader
                                                             model=model,                     # 传入模型
                                                             criterion_info=criterion_info)   # 传入损失函数信息

        # 将验证过程中的指标记录到 TensorBoard
        writer.add_scalar('验证损失', valid_loss, epoch)      # 记录验证损失（Valid Loss）
        writer.add_scalar('验证时性别分类的准确率', valid_gen_accs, epoch)   # 记录验证时性别分类的准确率（Valid Gender Accuracy）
        writer.add_scalar('验证时年龄预测的平均绝对误差', valid_age_mae, epoch)    # 记录验证时年龄预测的平均绝对误差（Valid Age MAE）

        # 检查是否有改进（即验证损失是否有所下降）
        is_best = valid_loss < best_loss        # 如果当前验证损失小于历史最佳损失，则说明有改进
        best_loss = min(valid_loss, best_loss)  # 更新历史最佳验证损失，保留最小的验证损失
        if not is_best:                         # 如果没有改进，则累计未改进的轮数
            epochs_since_improvement += 1       # 增加未改进的周期数
            print("\n自上次改进以来的时代: %d\n" % (epochs_since_improvement,))        # 打印自上次改进以来的周期数
        else:
            epochs_since_improvement = 0        # 如果有改进，则重置未改进的周期数

        # 保存模型检查点，记录当前的训练周期，未改进周期数，模型，优化器，最佳损失和是否为最佳模型
        save_checkpoint(epoch, epochs_since_improvement, model, optimizer, best_loss, is_best)


def train(train_loader, model, criterion_info, optimizer, epoch):
    """ 模型的训练

    即在给定的训练数据上通过反向传播优化模型的参数，使得模型能够更好地预测 年龄 和 性别。
    """
    model.train()                 # 设置模型为训练模式。在训练模式下，像 Dropout 和 BatchNorm 层会有不同的行为。

    losses = AverageMeter()       # 初始化跟踪损失和准确度的类（AverageMeter 是一个用来计算平均值的工具）
    gen_losses = AverageMeter()   # 性别分类的损失
    age_losses = AverageMeter()   # 年龄预测的损失
    gen_accs = AverageMeter()     # 性别分类的准确率
    age_mae = AverageMeter()      # 年龄预测的平均绝对误差（MAE）

    # 解包损失函数和权重
    age_criterion, gender_criterion, age_loss_weight = criterion_info

    # 批次循环，处理每个 batch 数据
    for i, (inputs, age_true, gen_true) in enumerate(train_loader):
        temp=len(train_loader.dataset.samples)  # 获取整个数据集的样本数量
        jindu=i*100/temp                        # 当前批次在整个训练集中的进度百分比
        print(f'时间：{datetime.now().strftime("%Y-%m-%d %H:%M:%S")}    总量：{temp}    当前：{i}    进度：{jindu}%')
        chunk_size = inputs.size()[0]           # 当前批次的样本数量（inputs.size() 返回 [batch_size, channels, height, width]，所以取 [0] 即为 batch_size）

         # 将数据移到 GPU（如果可用）
        inputs = inputs.to(device)
        age_true = age_true.to(device)  # 目标年龄
        gen_true = gen_true.to(device)  # 目标性别

        # 前向传播：模型计算预测的年龄和性别
        age_out, gen_out = model(inputs)  # 模型输出的年龄（age_out）和性别（gen_out）

        # 计算损失
        gen_loss = gender_criterion(gen_out, gen_true)  # 性别分类的损失
        age_loss = age_criterion(age_out, age_true)     # 年龄预测的损失
        age_loss *= age_loss_weight                     # 根据给定的权重调整年龄损失
        loss = gen_loss + age_loss                      # 总损失为性别损失和年龄损失之和

        # 反向传播
        optimizer.zero_grad()       # 清除之前的梯度
        loss.backward()             # 计算梯度

        # 梯度裁剪，防止梯度爆炸
        clip_gradient(optimizer, grad_clip)

        # 更新模型参数
        optimizer.step()

        # 计算准确度和 MAE
        gen_accuracy = accuracy(gen_out, gen_true)      # 计算性别分类的准确率
        _, ind = age_out.topk(1, 1, True, True)         # 取预测年龄的最大值（topk）作为预测结果
        l1_criterion = nn.L1Loss().to(device)           # 使用 L1 损失计算年龄的平均绝对误差（MAE）
        age_mae_loss = l1_criterion(ind.view(-1, 1).float(), age_true.view(-1, 1).float())  # 计算 MAE

        # 更新各个指标的平均值
        losses.update(loss.item(), chunk_size)          # 更新总损失
        gen_losses.update(gen_loss.item(), chunk_size)  # 更新性别损失
        age_losses.update(age_loss.item(), chunk_size)  # 更新年龄损失
        gen_accs.update(gen_accuracy, chunk_size)       # 更新性别分类准确率
        age_mae.update(age_mae_loss)                    # 更新年龄的 MAE

        # 打印状态
        if i % print_freq == 0:                 # 如果当前训练步数 i 是 print_freq 的倍数，即每经过一定数量的步数打印一次信息
            print('Epoch: [{0}][{1}/{2}]\t'     # 输出当前的训练 epoch 和当前步数 i 与总步数 len(train_loader) 的进度
                  'Loss {loss.val:.4f} ({loss.avg:.4f})\t'      # 输出总损失值：当前损失 (loss.val) 和平均损失 (loss.avg)
                  'Gender Loss {gen_loss.val:.4f} ({gen_loss.avg:.4f})\t'       # 输出性别分类损失值：当前性别损失 (gen_loss.val) 和平均性别损失 (gen_loss.avg)
                  'Age Loss {age_loss.val:.4f} ({age_loss.avg:.4f})\t'          # 输出年龄预测损失值：当前年龄损失 (age_loss.val) 和平均年龄损失 (age_loss.avg)
                  'Gender Accuracy {gen_accs.val:.3f} ({gen_accs.avg:.3f})\t'   # 输出性别分类准确率：当前准确率 (gen_accs.val) 和平均准确率 (gen_accs.avg)
                   # 输出年龄预测的平均绝对误差 (MAE)：当前误差 (age_mae.val) 和平均误差 (age_mae.avg)
                  'Age MAE {age_mae.val:.3f} ({age_mae.avg:.3f})'.format(epoch, i, len(train_loader),   # 格式化并输出当前的 epoch（训练轮次）、当前步骤 i 和总步骤数 len(train_loader)                                         
                                                                         loss=losses,   # 当前总损失的对象
                                                                         gen_loss=gen_losses,   # 当前性别损失的对象
                                                                         age_loss=age_losses,   # 当前年龄损失的对象
                                                                         gen_accs=gen_accs,     # 当前性别分类准确率的对象
                                                                         age_mae=age_mae))      # 当前年龄 MAE 的对象
        print('Epoch: [{0}][{1}/{2}]\t'
                  'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                  'Gender Loss {gen_loss.val:.4f} ({gen_loss.avg:.4f})\t'
                  'Age Loss {age_loss.val:.4f} ({age_loss.avg:.4f})\t'
                  'Gender Accuracy {gen_accs.val:.3f} ({gen_accs.avg:.3f})\t'
                  'Age MAE {age_mae.val:.3f} ({age_mae.avg:.3f})'.format(epoch, i, len(train_loader),
                                                                         loss=losses,
                                                                         gen_loss=gen_losses,
                                                                         age_loss=age_losses,
                                                                         gen_accs=gen_accs,
                                                                         age_mae=age_mae))

    return losses.avg, gen_accs.avg, age_mae.avg


def validate(val_loader, model, criterion_info):
    """作用是:

    评估训练好的模型在验证集上的性能，并计算与性别分类和年龄预测相关的多个指标。

    具体来说，它通过计算损失函数、准确率和预测误差来衡量模型的表现。
    """
    model.eval()  # 将模型设置为评估模式（不进行dropout或batch normalization）

    # 初始化用于记录平均值的对象（这些对象用于计算每个指标在验证集上的平均值）
    losses = AverageMeter()     # 用于记录总损失的平均值
    gen_losses = AverageMeter() # 用于记录性别分类损失的平均值
    age_losses = AverageMeter() # 用于记录年龄预测损失的平均值
    gen_accs = AverageMeter()   # 用于记录性别分类准确率的平均值
    age_mae = AverageMeter()    # 用于记录年龄预测的平均绝对误差（MAE）

    # 解包criterion_info，分别获取age_criterion, gender_criterion和age_loss_weight
    # 这些是计算损失所需的标准（比如交叉熵损失、均方误差等）和年龄损失的加权系数
    age_criterion, gender_criterion, age_loss_weight = criterion_info

    # 在验证过程中，不进行梯度更新
    with torch.no_grad():
        # 遍历验证集中的每一个批次
        for i, (inputs, age_true, gen_true) in enumerate(val_loader):
            chunk_size = inputs.size()[0]           # 获取当前批次的样本数量

            # 如果有GPU可用，将数据移到GPU上
            inputs = inputs.to(device)
            age_true = age_true.to(device)
            gen_true = gen_true.to(device)

            # 将输入数据通过模型进行前向传播，得到年龄和性别的预测结果
            age_out, gen_out = model(inputs)

            # 计算损失
            gen_loss = gender_criterion(gen_out, gen_true)  # 计算性别分类损失
            age_loss = age_criterion(age_out, age_true)     # 计算年龄预测损失
            age_loss *= age_loss_weight         # 将年龄损失乘以权重
            loss = gen_loss + age_loss      # 总损失是性别损失和年龄损失之和

            # 跟踪指标
            gender_accuracy = accuracy(gen_out, gen_true)   # 计算性别分类准确率

            # 计算年龄预测的MAE（平均绝对误差）
            _, ind = age_out.topk(1, 1, True, True)         # 获取年龄预测中每个样本的最可能预测值的索引
            l1_criterion = nn.L1Loss().to(device)           # 使用L1Loss来计算MAE
            age_mae_loss = l1_criterion(ind.view(-1, 1).float(), age_true.view(-1, 1).float())  # 计算每个样本的MAE

            # 更新各个指标的平均值
            losses.update(loss.item(), chunk_size)          # 更新总损失的平均值
            gen_losses.update(gen_loss.item(), chunk_size)  # 更新性别损失的平均值
            age_losses.update(age_loss.item(), chunk_size)  # 更新年龄损失的平均值
            gen_accs.update(gender_accuracy, chunk_size)    # 更新性别分类准确率的平均值
            age_mae.update(age_mae_loss, chunk_size)        # 更新年龄MAE的平均值

            # 每经过一定批次（print_freq），打印一次当前的验证结果
            if i % print_freq == 0:
                print('Validation: [{0}/{1}]\t'
                      'Loss {loss.val:.4f} ({loss.avg:.4f})\t'      # 打印当前批次的总损失和当前的平均总损失
                      'Gender Loss {gen_loss.val:.4f} ({gen_loss.avg:.4f})\t'   # 打印当前批次的性别损失和平均性别损失
                      'Age Loss {age_loss.val:.4f} ({age_loss.avg:.4f})\t'      # 打印当前批次的年龄损失和平均年龄损失
                      'Gender Accuracy {gen_accs.val:.3f} ({gen_accs.avg:.3f})\t'   # 打印当前批次的性别准确率和平均性别准确率
                      'Age MAE {age_mae.val:.3f} ({age_mae.avg:.3f})'.format(i, len(val_loader),
                                                                             loss=losses,
                                                                             gen_loss=gen_losses,
                                                                             age_loss=age_losses,
                                                                             gen_accs=gen_accs,
                                                                             age_mae=age_mae))
    # 返回整个验证集的总损失、性别分类准确率和年龄预测MAE的平均值
    return losses.avg, gen_accs.avg, age_mae.avg


def main():
    global args
    args = parse_args()
    train_net(args)


if __name__ == '__main__':
    main()
