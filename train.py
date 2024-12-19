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
    torch.manual_seed(7)
    np.random.seed(7)
    best_loss = 100000
    torch.manual_seed(7)
    np.random.seed(7)
    checkpoint = None
    start_epoch = 0
    writer = SummaryWriter()
    epochs_since_improvement = 0

    # 初始化/加载检查点
    if checkpoint is None:
        if args.network == 'r100':
            model = resnet101(args)
        elif args.network == 'r50':
            model = resnet50(args)
        elif args.network == 'r34':
            model = resnet34(args)
        elif args.network == 'r18':
            model = resnet18(args)
        else:  # “脸”
            model = resnet50(args)
        optimizer = torch.optim.SGD(params=filter(lambda p: p.requires_grad, model.parameters()), lr=args.lr,
                                    momentum=args.mom, weight_decay=args.weight_decay)
    else:
        checkpoint = torch.load(checkpoint)
        start_epoch = checkpoint['epoch'] + 1
        epochs_since_improvement = checkpoint['epochs_since_improvement']
        model = checkpoint['model']
        optimizer = checkpoint['optimizer']

    # 如果可用，移动到GPU
    model = model.to(device)

    # 损失函数
    if args.focal_loss:
        age_criterion = FocalLoss(gamma=args.gamma).to(device)
        gender_criterion = FocalLoss(gamma=args.gamma).to(device)
    else:
        age_criterion = nn.CrossEntropyLoss().to(device)
        gender_criterion = nn.CrossEntropyLoss().to(device)

    criterion_info = (age_criterion, gender_criterion, args.age_weight)

    # 自定义dataloaders
    train_dataset = AgeGenDataset('train')
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=workers,
                                               pin_memory=True)
    val_dataset = AgeGenDataset('valid')
    val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=workers,
                                             pin_memory=True)

    scheduler = StepLR(optimizer, step_size=args.lr_step, gamma=0.1)

    # 时代
    for epoch in range(start_epoch, epochs):
        scheduler.step()

        # 一个时代的训练
        train_loss, train_gen_accs, train_age_mae = train(train_loader=train_loader,
                                                          model=model,
                                                          criterion_info=criterion_info,
                                                          optimizer=optimizer,
                                                          epoch=epoch)
        writer.add_scalar('Train Loss', train_loss, epoch)
        writer.add_scalar('Train Gender Accuracy', train_gen_accs, epoch)
        writer.add_scalar('Train Age MAE', train_age_mae, epoch)

        # 一个时代的验证
        valid_loss, valid_gen_accs, valid_age_mae = validate(val_loader=val_loader,
                                                             model=model,
                                                             criterion_info=criterion_info)

        writer.add_scalar('Valid Loss', valid_loss, epoch)
        writer.add_scalar('Valid Gender Accuracy', valid_gen_accs, epoch)
        writer.add_scalar('Valid Age MAE', valid_age_mae, epoch)

        # 检查是否有改进
        is_best = valid_loss < best_loss
        best_loss = min(valid_loss, best_loss)
        if not is_best:
            epochs_since_improvement += 1
            print("\nEpochs since last improvement: %d\n" % (epochs_since_improvement,))
        else:
            epochs_since_improvement = 0

        # 保存检查点
        save_checkpoint(epoch, epochs_since_improvement, model, optimizer, best_loss, is_best)


def train(train_loader, model, criterion_info, optimizer, epoch):
    model.train()  # 训练模式（使用dropout和batchnorm）

    losses = AverageMeter()
    gen_losses = AverageMeter()
    age_losses = AverageMeter()
    gen_accs = AverageMeter()  # 性别的准确性
    age_mae = AverageMeter()  # 时代美

    age_criterion, gender_criterion, age_loss_weight = criterion_info

    # 批次
    for i, (inputs, age_true, gen_true) in enumerate(train_loader):
        temp=len(train_loader.dataset.samples)
        jindu=i*100/temp
        print(f'时间：{datetime.now().strftime("%Y-%m-%d %H:%M:%S")}    总量：{temp}    当前：{i}    进度：{jindu}%')
        chunk_size = inputs.size()[0]
        # 如果可用，移动到GPU
        inputs = inputs.to(device)
        age_true = age_true.to(device)  # [N, 1]
        gen_true = gen_true.to(device)  # [N, 1]

        # 道具。
        age_out, gen_out = model(inputs)  # age_out => [N, 1], gen_out => [N, 2]

        # 计算损失
        gen_loss = gender_criterion(gen_out, gen_true)
        age_loss = age_criterion(age_out, age_true)
        age_loss *= age_loss_weight
        loss = gen_loss + age_loss

        # Back prop.
        optimizer.zero_grad()
        loss.backward()

        # 剪辑梯度
        clip_gradient(optimizer, grad_clip)

        # 更新权重
        optimizer.step()

        # 跟踪指标
        gen_accuracy = accuracy(gen_out, gen_true)
        _, ind = age_out.topk(1, 1, True, True)
        l1_criterion = nn.L1Loss().to(device)
        age_mae_loss = l1_criterion(ind.view(-1, 1).float(), age_true.view(-1, 1).float())
        losses.update(loss.item(), chunk_size)
        gen_losses.update(gen_loss.item(), chunk_size)
        age_losses.update(age_loss.item(), chunk_size)
        gen_accs.update(gen_accuracy, chunk_size)
        age_mae.update(age_mae_loss)

        # 打印状态
        if i % print_freq == 0:
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
    model.eval()  # Eval模式（无dropout或batchnorm）

    losses = AverageMeter()
    gen_losses = AverageMeter()
    age_losses = AverageMeter()
    gen_accs = AverageMeter()  # 性别的准确性
    age_mae = AverageMeter()  # 时代美

    age_criterion, gender_criterion, age_loss_weight = criterion_info

    with torch.no_grad():
        # 批次
        for i, (inputs, age_true, gen_true) in enumerate(val_loader):
            chunk_size = inputs.size()[0]
            # 如果可用，移动到GPU
            inputs = inputs.to(device)
            age_true = age_true.to(device)
            gen_true = gen_true.to(device)

            # 道具。
            age_out, gen_out = model(inputs)

            # 计算损失
            gen_loss = gender_criterion(gen_out, gen_true)
            age_loss = age_criterion(age_out, age_true)
            age_loss *= age_loss_weight
            loss = gen_loss + age_loss

            # 跟踪指标
            gender_accuracy = accuracy(gen_out, gen_true)
            _, ind = age_out.topk(1, 1, True, True)
            l1_criterion = nn.L1Loss().to(device)
            age_mae_loss = l1_criterion(ind.view(-1, 1).float(), age_true.view(-1, 1).float())
            losses.update(loss.item(), chunk_size)
            gen_losses.update(gen_loss.item(), chunk_size)
            age_losses.update(age_loss.item(), chunk_size)
            gen_accs.update(gender_accuracy, chunk_size)
            age_mae.update(age_mae_loss, chunk_size)

            if i % print_freq == 0:
                print('Validation: [{0}/{1}]\t'
                      'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                      'Gender Loss {gen_loss.val:.4f} ({gen_loss.avg:.4f})\t'
                      'Age Loss {age_loss.val:.4f} ({age_loss.avg:.4f})\t'
                      'Gender Accuracy {gen_accs.val:.3f} ({gen_accs.avg:.3f})\t'
                      'Age MAE {age_mae.val:.3f} ({age_mae.avg:.3f})'.format(i, len(val_loader),
                                                                             loss=losses,
                                                                             gen_loss=gen_losses,
                                                                             age_loss=age_losses,
                                                                             gen_accs=gen_accs,
                                                                             age_mae=age_mae))

    return losses.avg, gen_accs.avg, age_mae.avg


def main():
    global args
    args = parse_args()
    train_net(args)


if __name__ == '__main__':
    main()
