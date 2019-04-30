# -×- coding: utf-8 -*-
__author__ = 'QiHuangChen'

from config_track import *
from Train.models.net import *
import json
import time as time
from os.path import join, isdir
from os import makedirs
import cv2
import numpy as np
from utils import *
from torch.autograd import Variable

config = Config()

class Tracker():

    def __init__(self, im, init_rect):
        self.net = SiamNet()
        self.net.load_state_dict(torch.load(config.net_path))
        self.net.to(config.device)
        self.net.eval()

        self.target_position, self.target_size = rect1_2_cxy_wh(init_rect)
        img_uint8 = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)
        img_double = np.double(img_uint8)
        self.avg_chans = np.mean(img_double, axis=(0, 1))
        wc_z = self.target_size[1] + config.context_amount * sum(self.target_size)
        hc_z = self.target_size[0] + config.context_amount * sum(self.target_size)
        s_z = np.sqrt(wc_z * hc_z)
        scale_z = config.examplar_size / s_z
        z_crop = get_subwindow_tracking(img_double, self.target_position, config.examplar_size, round(s_z), self.avg_chans)
        z_crop = np.uint8(z_crop)
        z_crop_tensor = 255.0 * F.to_tensor(z_crop).unsqueeze(0)
        self.z_features = self.net.feature(Variable(z_crop_tensor).to(config.device))
        self.z_features = self.z_features.repeat(self.num_scale, 1, 1, 1)

        d_search = (config.instance_size - config.examplar_size) / 2
        pad = d_search / scale_z
        self.s_x = s_z + 2 * pad

        self.min_s_x = config.scale_min * self.s_x
        self.max_s_x = config.scale_max * self.s_x
        self.scales = config.scale_step ** np.linspace(-np.ceil(config.num_scale / 2), np.ceil(config.num_scale / 2),config.num_scale)

        if config.windowing == 'cosine':
            self.window = np.outer(np.hanning(config.score_size * config.response_UP), np.hanning(config.score_size * config.response_UP))
        elif config.windowing == 'uniform':
            self.window = np.ones((config.score_size * config.response_UP, config.score_size * config.response_UP))

    def track(self, im, frame):
        img_uint8 = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)
        img_double = np.double(img_uint8)

        scaled_instance = self.s_x * self.scales
        scaled_target = np.zeros((2, self.scales.size), dtype=np.double)
        scaled_target[0, :] = self.target_size[0] * self.scales
        scaled_target[1, :] = self.target_size[1] * self.scales
        x_crops = make_scale_pyramid(img_double, self.target_position, scaled_instance, p.instance_size, self.avg_chans, p)
        x_crops_tensor = torch.FloatTensor(x_crops.shape[3], x_crops.shape[2], x_crops.shape[1], x_crops.shape[0])
        for k in range(x_crops.shape[3]):
            tmp_x_crop = x_crops[:, :, :, k]
            tmp_x_crop = np.uint8(tmp_x_crop)
            x_crops_tensor[k, :, :, :] = 255.0 * F.to_tensor(tmp_x_crop).unsqueeze(0)

        # get features of search regions
        x_features = self.net.feature(Variable(x_crops_tensor).to(config.device))

        # evaluate the offline-trained network for exemplar x features
        self.target_position, new_scale = tracker_eval(self.net, round(self.s_x), self.z_features, x_features, self.target_position, self.window, config)

        # scale damping and saturation
        self.s_x = max(self.min_s_x, min(self.max_s_x, (1 - config.scale_LR) * self.s_x + config.scale_LR * scaled_instance[int(new_scale)]))
        self.target_size = (1 - config.scale_LR) * self.target_size + config.scale_LR * np.array(
            [scaled_target[0, int(new_scale)], scaled_target[1, int(new_scale)]])

        rect_position = np.array(
        [self.target_position[1] - self.target_size[1] / 2, self.target_position[0] - self.target_size[0] / 2, self.target_size[1],
         self.target_size[0]])
        if config.visualization:
            visualize_tracking_result(img_uint8, rect_position, 1)

        return cxy_wh_2_rect1(self.target_position, self.target_size)


if __name__ == '__main__':

    # 读取图片
    annos = json.load(open('data/OTB2015.json', 'r'))
    videos = sorted(annos.keys())  # 序列名
    base_path = '/home/esc/Experiment/DataSet/Benchmark/OTB100/'  # 序列所在路径
    # 结果保存
    #     res = []
    save_path = join('results/OTB2015/DCFNet','param_test')
    if not isdir(save_path):
        makedirs(save_path)

    speed = []
    # loop videos
    for video_id, video in enumerate(videos):  # run without resetting
        # 第一帧图像
        video_path_name = annos[video]['name']
        print annos[video]['name']
        init_rect = np.array(annos[video]['init_rect']).astype(np.float)
        gt = np.array(annos[video]['gt_rect']).astype(np.float)
        image_files = [join(base_path, video_path_name, 'img', im_f) for im_f in annos[video]['image_files']]
        n_images = len(image_files)

        tic = time.time()  # time start
        im = cv2.imread(image_files[0])

        tracker = Tracker(im, init_rect)

        res = [init_rect]

        # 第n帧跟踪
        for f in range(1, n_images):  # track
            im = cv2.imread(image_files[f])
            res.append(tracker.track(im, f))  # 1-index

        # 预测结果保存
            # 保存为txt文件
        toc = time.time() - tic
        fps = n_images / toc
        speed.append(fps)
        result_path = join(save_path, video_path_name + '.txt')
        with open(result_path, 'w') as f:
            for x in res:
                f.write(','.join(['{:.2f}'.format(i) for i in x]) + '\n')
            # 保存为.mat 文件
        res1 = {}
        res1['res'] = res
        res1['type'] = 'rect'
        res1['fps'] = fps
        result_path_mat = join(save_path, video_path_name + '.mat')
        sio.savemat(result_path_mat, res1)

        print('***{} {} is finished with Total Mean Speed: {:3.1f} (FPS)***'.format(video_id, video, np.mean(speed)))
