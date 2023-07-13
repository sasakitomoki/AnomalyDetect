import os

from torch.backends import cudnn
from utils.utils import *

from solver import Solver


def str2bool(v):
    return v.lower() in ('true')


def Anomaly_detect(video_name):
    anomaly_frame_dic = {} #{"body_part" : [anomaly_frame]}　anomaly_frame は 0　正常　,　1　異常
    #各部位でどの程度異常を検出するかの設定
    #{"体の部分" : {"anomaly_ratio" : "検出割合, "channel : "検出部位の数(data_loader.pyで変更可)"}}
    body_part_param = {"lowerbody" : {"anomaly_ratio" : 18.0, "channel" : 16} , "righthand" : {"anomaly_ratio" : 5.0, "channel" : 4} , "lefthand" : {"anomaly_ratio" : 5.0, "channel" : 4}}
    #body_part_param = {"lowerbody" : {"anomaly_ratio" : 18.0, "channel" : 16} , "righthand" : {"anomaly_ratio" : 5.0, "channel" : 4} , "lefthand" : {"anomaly_ratio" : 5.0, "channel" : 4}, "etc" : {"anomaly_ratio" : 1.0, "channel" : 42}}   #←顔、肩、尻、指にも適用したいとき
    #各部位に対してモデル作成, 異常検出
    for body_part, param in body_part_param.items():
        config_train = {'lr': 0.0001, 'num_epochs': 10, 'k': 8, 'win_size': 1, 'input_c': param["channel"], 'output_c': param["channel"], 'batch_size': 6, 'dataset': 'SPORT', 'mode': 'train', 'data_path': 'dataset/'+video_name, 'model_save_path': 'checkpoints', 'anormly_ratio': param["anomaly_ratio"] , 'body_part' : body_part}
        config_test = {'lr': 0.0001, 'num_epochs': 10, 'k': 8, 'win_size': 1, 'input_c': param["channel"], 'output_c': param["channel"], 'batch_size': 6, 'pretrained_model': '20', 'dataset': 'SPORT', 'mode': 'test', 'data_path': 'dataset/'+video_name, 'model_save_path': 'checkpoints', 'anormly_ratio': param["anomaly_ratio"] ,'body_part' : body_part}
        cudnn.benchmark = True
        if (not os.path.exists(config_train['model_save_path'])):
            os.mkdir(config_train['model_save_path'])
        #学習/モデル作成
        solver = Solver(config_train)
        solver.train()
        #モデル適用/異常検出
        solver = Solver(config_test)
        pred_label = solver.test()
        anomaly_frame_dic[body_part] = pred_label

    return anomaly_frame_dic

if __name__ == '__main__':

    Anomaly_frame = Anomaly_detect('IMG_0893_1940_2060')

    print(Anomaly_frame)

