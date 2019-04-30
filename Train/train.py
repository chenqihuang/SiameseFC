# -*- coding: utf-8 -*-
__author__ = "QiHuangChen"


from models.net import *
from config_train import *
import torchvision.transforms as transforms
from data.DataAugmentation import *
from data.VIDDataset import *
from torch.utils.data import DataLoader
from tqdm import tqdm
from models.net import *
from torch.autograd import Variable
from utils import *
from torch.optim.lr_scheduler import StepLR
import time
from tensorboardX import SummaryWriter

os.environ['CUDA_VISIABLE_DEVICES'] = '0'

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

config = Config()


def train(data_dir, train_imdb, val_imdb):

    writer = SummaryWriter(log_dir='')
    # 训练数据集
        # 数据增益
    center_crop_size = config.instance_size - config.stride
    random_crop_size = config.instance_size - 2 * config.stride
    train_z_transforms = transforms.Compose([
        RandomStretch(),
        CenterCrop((config.examplar_size, config.examplar_size)),
        ToTensor()
    ])
    train_x_transforms = transforms.Compose([
        RandomStretch(),
        CenterCrop((center_crop_size, center_crop_size)),
        RandomCrop((random_crop_size, random_crop_size)),
        ToTensor()
    ])
    valid_z_transforms = transforms.Compose([
        CenterCrop((config.examplar_size, config.examplar_size)),
        ToTensor(),
    ])
    valid_x_transforms = transforms.Compose([
        ToTensor()
    ])
        # 加载数据
    train_dataset = VIDDataset(train_imdb, data_dir, config, train_z_transforms, train_x_transforms)
    val_dataset = VIDDataset(val_imdb, data_dir, config, valid_z_transforms, valid_x_transforms, "Validation")
        # create dataloader
    train_loader = DataLoader(train_dataset, batch_size=config.batch_size,
                              shuffle=True, num_workers=config.train_num_workers, drop_last=True)
    val_loader = DataLoader(val_dataset, batch_size=config.batch_size,
                            shuffle=True, num_workers=config.val_num_workers, drop_last=True)


    # 模型
    net = SiamNet()
    if config.use_gpu:
        net.cuda()
    # 损失函数 net.weight_loss
    # 优化算法
    optimizer = torch.optim.SGD([
        {'params': net.feature.parameters()},
        {'params': net.adjust.bias},
        {'params': net.adjust.weight, 'lr': 0},
    ], config.lr, config.momentum, config.weight_decay)
    scheduler = StepLR(optimizer, config.step_size, config.gamma)
    # 训练
    train_response_flag = False
    valid_response_flag = False
    for i in range(config.start_epoch, config.end_epoch):
        scheduler.step()
        net.train()
        train_loss = []
        for j, data in enumerate(tqdm(train_loader)):
            exemplar_imgs, instance_imgs = data
            if config.use_gpu:
                exemplar_imgs = exemplar_imgs.cuda()
                instance_imgs = instance_imgs.cuda()
            output = net.forward(Variable(exemplar_imgs), Variable(instance_imgs))
            if not train_response_flag:
                train_response_flag = True
                response_size = output.shape[2:4]
                train_eltwise_label, train_instance_weight = create_label(response_size, config)


            loss = net.weight_loss(output, train_eltwise_label, train_instance_weight, config)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            loss_train = loss.to('cpu').squeeze().data.numpy()
            train_loss.append(loss_train)
        print np.mean(train_loss)
        writer.add_scalar('scalar/test', np.mean(train_loss), i+1)

        # 模型保存
        if not os.path.exists(config.model_save_path):
            os.makedirs(config.model_save_path)
        torch.save({
            'epoch': i + 1,
            'state_dict': net.state_dict(),
            'optimizer': optimizer.state_dict(),
        }, config.model_save_path + "SiamFC_dict_" + str(i + 1) + "_model.pth")
#
        net.eval()
        val_loss = []
        for j, data in enumerate(tqdm(val_loader)):

            exemplar_imgs, instance_imgs = data
            if config.use_gpu:
                exemplar_imgs = exemplar_imgs.cuda()
                instance_imgs = instance_imgs.cuda()

            output = net.forward(Variable(exemplar_imgs), Variable(instance_imgs))
            # 生成高斯标签
            if not valid_response_flag:
                valid_response_flag = True
                response_size = output.shape[2:4]
                valid_eltwise_label, valid_instance_weight = create_label(response_size, config)
            # loss
            loss = net.weight_loss(output, valid_eltwise_label, valid_instance_weight, config)
            # collect validation loss
            loss_val = loss.to('cpu').squeeze().data.numpy()
            val_loss.append(loss_val)

        # 可视化
        print np.mean(train_loss)
        print ("Epoch %d   training loss: %f, validation loss: %f" % (i + 1, np.mean(train_loss), np.mean(val_loss)))
    writer.close()


if __name__ == '__main__':
    data_dir = ""
    train_imdb = ""
    val_imdb = ""
    print time.strftime('start %Y-%m-%d %H:%M:%S', time.localtime(time.time()))
    train(data_dir, train_imdb, val_imdb)
    print time.strftime('end %Y-%m-%d %H:%M:%S', time.localtime(time.time()))