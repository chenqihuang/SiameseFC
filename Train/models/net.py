# -*- coding: utf-8 -*-
__author__ = "QiHuangChen"
import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy import io as sio

class Alex_like(nn.Module):

    def __init__(self):
        super(Alex_like, self).__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(3, 96, 11, 2),
            nn.BatchNorm2d(96),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(3, 2))
        self.conv2 = nn.Sequential(
            nn.Conv2d(96, 256, 5, 1, groups=2),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(3, 2))
        self.conv3 = nn.Sequential(
            nn.Conv2d(256, 384, 3, 1),
            nn.BatchNorm2d(384),
            nn.ReLU(inplace=True))
        self.conv4 = nn.Sequential(
            nn.Conv2d(384, 384, 3, 1, groups=2),
            nn.BatchNorm2d(384),
            nn.ReLU(inplace=True))
        self.conv5 = nn.Sequential(
            nn.Conv2d(384, 256, 3, 1, groups=2))

    def forward(self, x):
        conv1 = self.conv1(x)
        conv2 = self.conv2(conv1)
        conv3 = self.conv3(conv2)
        conv4 = self.conv4(conv3)
        conv5 = self.conv5(conv4)
        return conv5

    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight.data, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()


class SiamNet(nn.Module):

    def __init__(self):
        super(SiamNet, self).__init__()
        self.feature = Alex_like()
        self.adjust = nn.Conv2d(1, 1, 1, 1)

    def forward(self, z, x):
        z_feat = self.feature(z)
        x_feat = self.feature(x)
        xcorr_out = self.xcorr(z_feat, x_feat)
        score = self.adjust(xcorr_out)
        return score

    def xcorr(self, z, x):
        # correlation layer as in the original SiamFC
        batch_size_x, channel_x, w_x, h_x = x.shape
        x = torch.reshape(x, (1, batch_size_x * channel_x, w_x, h_x))
        # group convolution
        out = F.conv2d(x, z, groups = batch_size_x)
        batch_size_out, channel_out, w_out, h_out = out.shape
        xcorr_out = torch.reshape(out, (channel_out, batch_size_out, w_out, h_out))
        return xcorr_out

    def weight_loss(self, prediction, label, weight, config):
        return F.binary_cross_entropy_with_logits(prediction,
                                                  label,
                                                  weight,
                                                  size_average=False) / config.batch_size

    def init_weights(self):
        self.adjust.weight.data.fill_(1e-3)
        self.adjust.bias.data.zero_()

    def load_matconvnet(self, net_path):
        mat = sio.loadmat(net_path)
        net_dot_mat = mat.get('net')  # get net
        params = net_dot_mat['params']  # get net/params
        params = params[0][0]
        params_names = params['name'][0]  # get net/params/name
        params_names_list = [params_names[p][0] for p in range(params_names.size)]
        params_values = params['value'][0]  # get net/params/val
        params_values_list = [params_values[p] for p in range(params_values.size)]
        return params_names_list, params_values_list

    def load_params_from_mat(self, net_path):
        """
        initialize network parameters from pre-trained model from author( .mat)
        :param net_path:
        :return:
        """
        # so ugly!!!
        params_names_list, params_values_list = self.load_matconvnet(net_path)
        params_values_list = [torch.from_numpy(p) for p in params_values_list]  # values convert numpy to Tensor
        for index, param in enumerate(params_values_list):
            param_name = params_names_list[index]
            if 'conv' in param_name and param_name[-1] == 'f':
                param = param.permute(3, 2, 0, 1)
            param = torch.squeeze(param)
            params_values_list[index] = param

        self.conv1[0].weight.data[:] = params_values_list[params_names_list.index('conv%df' % 1)]
        self.conv1[0].bias.data[:] = params_values_list[params_names_list.index('conv%db' % 1)]

        self.conv1[1].weight.data[:] = params_values_list[params_names_list.index('bn%dm' % 1)]
        self.conv1[1].bias.data[:] = params_values_list[params_names_list.index('bn%db' % 1)]
        bn_moments = params_values_list[params_names_list.index('bn%dx' % 1)]
        self.conv1[1].running_mean[:] = bn_moments[:, 0]
        self.conv1[1].running_var[:] = bn_moments[:, 1] ** 2
        # --------------------------------------------------------------------------------------------------
        self.conv2[0].weight.data[:] = params_values_list[params_names_list.index('conv%df' % 2)]
        self.conv2[0].bias.data[:] = params_values_list[params_names_list.index('conv%db' % 2)]

        self.conv2[1].weight.data[:] = params_values_list[params_names_list.index('bn%dm' % 2)]
        self.conv2[1].bias.data[:] = params_values_list[params_names_list.index('bn%db' % 2)]
        bn_moments = params_values_list[params_names_list.index('bn%dx' % 2)]
        self.conv2[1].running_mean[:] = bn_moments[:, 0]
        self.conv2[1].running_var[:] = bn_moments[:, 1] ** 2
        # --------------------------------------------------------------------------------------------------
        self.conv3[0].weight.data[:] = params_values_list[params_names_list.index('conv%df' % 3)]
        self.conv3[0].bias.data[:] = params_values_list[params_names_list.index('conv%db' % 3)]

        self.conv3[1].weight.data[:] = params_values_list[params_names_list.index('bn%dm' % 3)]
        self.conv3[1].bias.data[:] = params_values_list[params_names_list.index('bn%db' % 3)]
        bn_moments = params_values_list[params_names_list.index('bn%dx' % 3)]
        self.conv3[1].running_mean[:] = bn_moments[:, 0]
        self.conv3[1].running_var[:] = bn_moments[:, 1] ** 2
        # --------------------------------------------------------------------------------------------------
        self.conv4[0].weight.data[:] = params_values_list[params_names_list.index('conv%df' % 4)]
        self.conv4[0].bias.data[:] = params_values_list[params_names_list.index('conv%db' % 4)]

        self.conv4[1].weight.data[:] = params_values_list[params_names_list.index('bn%dm' % 4)]
        self.conv4[1].bias.data[:] = params_values_list[params_names_list.index('bn%db' % 4)]
        bn_moments = params_values_list[params_names_list.index('bn%dx' % 4)]
        self.conv4[1].running_mean[:] = bn_moments[:, 0]
        self.conv4[1].running_var[:] = bn_moments[:, 1] ** 2
        # --------------------------------------------------------------------------------------------------
        self.conv5[0].weight.data[:] = params_values_list[params_names_list.index('conv%df' % 5)]
        self.conv5[0].bias.data[:] = params_values_list[params_names_list.index('conv%db' % 5)]
        # --------------------------------------------------------------------------------------------------


if __name__ == '__main__':
    x = torch.randn(10, 96, 10, 10)
    z = torch.randn(10, 3, 3, 3)
    # net = SiamNet()
    # net.xcorr(z, x)
    net1 = Alex_like()
    xx = net1.modules()
    net1.forward(x)
    print "test"