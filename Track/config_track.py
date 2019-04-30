# -×- coding: utf-8 -*-
__author__ = 'QiHuangChen'
import torch

class Config:
    def __init__(self):

        self.examplar_size = 127
        self.instance_size = 255
        self.sub_mean = 0
        self.stride = 8
        self.rPos = 16
        self.rNeg = 0

        self.num_scale = 3
        self.response_UP = 16
        self.windowing = "cosine"

        self.scale_step = 1.0375
        self.scale_penalty = 0.9745
        self.scale_LR = 0.59
        self.w_influence = 0.176

        self.context_amount = 0.5
        self.scale_min = 0.2
        self.scale_max = 5
        self.score_size = 17

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


        self.net_path = "/home/esc/Experiment/Code/SiameseFC/Train/checkpoints/SiamFC_dict_1_model.pth"
        self.save_path = ""

        self.visualization = True