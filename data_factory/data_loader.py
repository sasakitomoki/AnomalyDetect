import torch
import os
import random
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from PIL import Image
import numpy as np
import collections
import numbers
import math
import pandas as pd
from sklearn.preprocessing import StandardScaler
import pickle


class SportDataLoader(object):
    def __init__(self, data_path, win_size, step, mode="train", body_part="lowerbody"):
        self.mode = mode
        self.step = step
        self.win_size = win_size
        self.scaler = StandardScaler()
        data = pd.read_csv(data_path + '/position_data.csv')
        #指定した部分によって異常検出する部位を分ける
        #異常検出部位を追加したいときはここを書き換え
        if body_part=="lowerbody":
            columns_li = [
            'LEFT_KNEE_x', 'LEFT_KNEE_y', 'RIGHT_KNEE_x', 'RIGHT_KNEE_y',
            'LEFT_ANKLE_x', 'LEFT_ANKLE_y', 'RIGHT_ANKLE_x', 'RIGHT_ANKLE_y',
            'LEFT_HEEL_x', 'LEFT_HEEL_y', 'RIGHT_HEEL_x', 'RIGHT_HEEL_y',
            'LEFT_FOOT_INDEX_x', 'LEFT_FOOT_INDEX_y', 'RIGHT_FOOT_INDEX_x', 'RIGHT_FOOT_INDEX_y'
            ]
        elif body_part=="righthand":
            columns_li = ['RIGHT_ELBOW_x', 'RIGHT_ELBOW_y', 'RIGHT_WRIST_x', 'RIGHT_WRIST_y']
        elif body_part=="lefthand":
            columns_li = ['LEFT_ELBOW_x', 'LEFT_ELBOW_y', 'LEFT_WRIST_x', 'LEFT_WRIST_y']
        elif body_part=="etc":
            columns_li=[
            "NOSE_x", "NOSE_y", "LEFT_EYE_INNER_x", "LEFT_EYE_INNER_y", "LEFT_EYE_x", "LEFT_EYE_y",
            "LEFT_EYE_OUTER_x", "LEFT_EYE_OUTER_y", "RIGHT_EYE_INNER_x", "RIGHT_EYE_INNER_y",
            "RIGHT_EYE_x", "RIGHT_EYE_y", "RIGHT_EYE_OUTER_x", "RIGHT_EYE_OUTER_y", "LEFT_EAR_x", "LEFT_EAR_y",
            "RIGHT_EAR_x", "RIGHT_EAR_y", "MOUTH_LEFT_x", "MOUTH_LEFT_y", "MOUTH_RIGHT_x", "MOUTH_RIGHT_y",
            "LEFT_SHOULDER_x", "LEFT_SHOULDER_y", "RIGHT_SHOULDER_x", "RIGHT_SHOULDER_y", "LEFT_PINKY_x",
            "LEFT_PINKY_y", "RIGHT_PINKY_x", "RIGHT_PINKY_y", "LEFT_INDEX_x", "LEFT_INDEX_y", "RIGHT_INDEX_x",
            "RIGHT_INDEX_y", "LEFT_THUMB_x", "LEFT_THUMB_y", "RIGHT_THUMB_x", "RIGHT_THUMB_y", "LEFT_HIP_x",
            "LEFT_HIP_y", "RIGHT_HIP_x", "RIGHT_HIP_y"
            ]

        data = data[columns_li].values
        data = np.nan_to_num(data)

        self.scaler.fit(data)
        data = self.scaler.transform(data)
        test_data = pd.read_csv(data_path + '/position_data.csv')

        test_data = test_data[columns_li].values
        test_data = np.nan_to_num(test_data)

        self.test = self.scaler.transform(test_data)

        self.train = data
        self.val = self.test

        self.test_labels = pd.read_csv(data_path + '/test_label.csv')[[body_part]].values

        print("test:", self.test.shape)
        print("train:", self.train.shape)

    def __len__(self):
        """
        Number of images in the object dataset.
        """
        if self.mode == "train":
            return (self.train.shape[0] - self.win_size) // self.step + 1
        elif (self.mode == 'val'):
            return (self.val.shape[0] - self.win_size) // self.step + 1
        elif (self.mode == 'test'):
            return (self.test.shape[0] - self.win_size) // self.step + 1
        else:
            return (self.test.shape[0] - self.win_size) // self.win_size + 1

    def __getitem__(self, index):
        index = index * self.step
        if self.mode == "train":
            return np.float32(self.train[index:index + self.win_size]), np.float32(self.test_labels[0:self.win_size])
        elif (self.mode == 'val'):
            return np.float32(self.val[index:index + self.win_size]), np.float32(self.test_labels[0:self.win_size])
        elif (self.mode == 'test'):
            return np.float32(self.test[index:index + self.win_size]), np.float32(
                self.test_labels[index:index + self.win_size])
        else:
            return np.float32(self.test[
                              index // self.step * self.win_size:index // self.step * self.win_size + self.win_size]), np.float32(
                self.test_labels[index // self.step * self.win_size:index // self.step * self.win_size + self.win_size])


def get_loader_segment(data_path, batch_size, win_size=100, step=100, mode='train', dataset='KDD', body_part='lowerbody'):

    dataset = SportDataLoader(data_path, win_size, 1, mode, body_part)
    shuffle = False
    if mode == 'train':
        shuffle = True

    data_loader = DataLoader(dataset=dataset,
                             batch_size=batch_size,
                             shuffle=shuffle,
                             num_workers=0)
    return data_loader
