# -*- coding: utf-8 -*-
import sys
import json
import os
import glob
from os.path import join as fullfile
import numpy as np
import argparse
import xlwt
import xlrd
from xlutils.copy import copy

def get_result_bb(arch, seq):
    result_path = fullfile(arch, seq + '.txt')
    if os.path.exists(result_path):
        temp = np.loadtxt(result_path, delimiter=',').astype(np.float)
    else:
        result_path = fullfile(arch, seq + '.json')
        temp = json.load(open(result_path, 'r'))
        temp = temp['res']
    return np.array(temp)


def get_result_bb_json(arch, seq):
    result_path = fullfile(arch, seq + '.json')
    temp = json.load(open(result_path, 'r'))
    temp = temp['res']
    return np.array(temp)


def convert_bb_to_center(bboxes):
    return np.array([(bboxes[:, 0] + (bboxes[:, 2] - 1) / 2),
                     (bboxes[:, 1] + (bboxes[:, 3] - 1) / 2)]).T


def overlap_ratio(rect1, rect2):

    if rect1.ndim==1:
        rect1 = rect1[None,:]
    if rect2.ndim==1:
        rect2 = rect2[None,:]

    left = np.maximum(rect1[:,0], rect2[:,0])
    right = np.minimum(rect1[:,0]+rect1[:,2], rect2[:,0]+rect2[:,2])
    top = np.maximum(rect1[:,1], rect2[:,1])
    bottom = np.minimum(rect1[:,1]+rect1[:,3], rect2[:,1]+rect2[:,3])

    intersect = np.maximum(0,right - left) * np.maximum(0,bottom - top)
    union = rect1[:,2]*rect1[:,3] + rect2[:,2]*rect2[:,3] - intersect
    iou = np.clip(intersect / union, 0, 1)
    return iou


# 重叠率
def compute_success_overlap(gt_bb, result_bb):
    thresholds_overlap = np.arange(0, 1.05, 0.05)
    n_frame = len(gt_bb)
    success = np.zeros(len(thresholds_overlap))
    iou = overlap_ratio(gt_bb, result_bb)
    for i in range(len(thresholds_overlap)):
        success[i] = sum(iou > thresholds_overlap[i]) / float(n_frame)
    return success


# 中心 20px
def compute_success_error(gt_center, result_center):
    thresholds_error = np.arange(0, 51, 1)
    n_frame = len(gt_center)
    success = np.zeros(len(thresholds_error))
    dist = np.sqrt(np.sum(np.power(gt_center - result_center, 2), axis=1))
    for i in range(len(thresholds_error)):
        success[i] = sum(dist <= thresholds_error[i]) / float(n_frame)
    return success



# xiugai
readbook = xlrd.open_workbook('')
workbooknew = copy(readbook)
sheet = workbooknew.get_sheet('')


##*** tongyong
col = 0

def eval_auc(result_path=''):

    list_path = os.path.join('OTB2015.json')
    annos = json.load(open(list_path, 'r'))
    seqs = list(sorted(annos.keys()))

    CVPR2013 = ['CarDark', 'Car4', 'David', 'David2', 'Sylvester', 'Trellis', 'Fish', 'Mhyang', 'Soccer', 'Matrix',
               'Ironman', 'Deer', 'Skating1', 'Shaking', 'Singer1', 'Singer2', 'Coke', 'Bolt', 'Boy', 'Dudek',
               'Crossing', 'Couple', 'Football1', 'Jogging_1', 'Jogging_2', 'Doll', 'Girl', 'Walking2', 'Walking',
               'FleetFace', 'Freeman1', 'Freeman3', 'Freeman4', 'David3', 'Jumping', 'CarScale', 'Skiing', 'Dog1',
               'Suv', 'MotorRolling', 'MountainBike', 'Lemming', 'Liquor', 'Woman', 'FaceOcc1', 'FaceOcc2',
               'Basketball', 'Football', 'Subway', 'Tiger1', 'Tiger2']

    n_seq = len(seqs)
    thresholds_overlap = np.arange(0, 1.05, 0.05)
    thresholds_error = np.arange(0, 51, 1)

    success_overlap = np.zeros((n_seq, len(thresholds_overlap)))
    success_error = np.zeros((n_seq, len(thresholds_error)))

    for i in range(n_seq):
        seq = seqs[i]
        gt_rect = np.array(annos[seq]['gt_rect']).astype(np.float)
        gt_center = convert_bb_to_center(gt_rect)

        # txt = annos[seq]['name'][annos[seq]['name'].index("."):]
        bb = get_result_bb(result_path, annos[seq]['name'])
        # bb = get_result_bb_json(result_path, annos[seq]['name'])
        center = convert_bb_to_center(bb)
        success_overlap[i] = compute_success_overlap(gt_rect, bb)
        success_error[i] = compute_success_error(gt_center, center)

        print(i)
        print(seq)
        print(success_overlap[i, :].mean())
        print(success_error[i, 20])
        sheet.write(i+1, col, seq)
        sheet.write(i+1, col+1, success_overlap[i, :].mean())
        sheet.write(i+1, col+2, success_error[i, 20])

    CVPR2013_id = []
    The_remain_id = []
    The_remain_name = []
    for j in range(n_seq):
        if annos[seqs[j]]['name'] in CVPR2013:
            CVPR2013_id.append(j)
        else:
            The_remain_id.append(j)
            The_remain_name.append(annos[seqs[j]]['name'])

    print('Success Overlap')
    print('CVPR2013 : %.4f' % (success_overlap[CVPR2013_id, :].mean()))
    print('OTB2015 : %.4f' % (success_overlap[:,:].mean()))
    print('The remain : %.4f' % (success_overlap[The_remain_id,:].mean()))

    print('Success Error')
    print('CVPR2013 : %.4f' % (success_error[CVPR2013_id, :].mean(0)[20]))
    print('OTB2015 : %.4f' % (success_error[:, :].mean(0)[20]))
    print('The remain : %.4f' % (success_error[The_remain_id, :].mean(0)[20]))
    print('Finished\n\n')
    # print(CVPR2013)
    # print(The_remain_id)
    sheet.write(i + 3, col+0, 'CVPR2013')
    sheet.write(i + 3, col+1, success_overlap[CVPR2013_id, :].mean())
    sheet.write(i + 3, col+2, success_error[CVPR2013_id, :].mean(0)[20])

    sheet.write(i + 4, col+0, 'OTB100')
    sheet.write(i + 4, col+1, success_overlap[:,:].mean())
    sheet.write(i + 4, col+2, success_error[:, :].mean(0)[20])

    sheet.write(i + 5, col+0, 'The remain')
    sheet.write(i + 5, col+1, success_overlap[The_remain_id,:].mean())
    sheet.write(i + 5, col+2, success_error[The_remain_id, :].mean(0)[20])
    # diyizhen
    # writebook.save('/home/esc/Experiment/Code/MyDCFNet/OTB_tookit/result.xls')
    # dierzhen
    workbooknew.save('')

if __name__ == "__main__":

    result_path = ''
    eval_auc(result_path)

