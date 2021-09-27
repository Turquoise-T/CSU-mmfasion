from __future__ import division
import argparse
import numpy as np
import os
import torch
import json
from mmcv import Config
from mmcv.runner import load_checkpoint
from mmfashion.core import AttrPredictor, CatePredictor
from mmfashion.models import build_predictor
from mmfashion.utils import get_img_tensor

def create_text(filename):
    path = 'D:\\PycharmProjects\\fashion\\result\\re\\' #需自定义路径
    file_path = path + filename + '.txt'
    file = open(file_path,'w')
    file.close()




def parse_args():
    parser = argparse.ArgumentParser(
        description='MMFashion Attribute Prediction Demo')
    parser.add_argument(
        '--input',
        type=str,
        help='input image path',
        default='D:/PycharmProjects/fashion/demo/imgs/attr_pred_demo2.jpg')
    parser.add_argument(
        '--checkpoint1',
        type=str,
        help='checkpoint file',
        default='D:/PycharmProjects/fashion/checkpoint/CateAttrPredict/vgg attrcate.pth')
    parser.add_argument(
        '--checkpoint2',
        type=str,
        help='checkpoint file',
        default='D:/PycharmProjects/fashion/checkpoint/Predict/vgg/global/vgg.pth')
    parser.add_argument(
        '--config1',
        help='test config file path',
        default='D:/PycharmProjects/fashion/demo/configs/global_predictor_vgg.py'
    )
    parser.add_argument(
        '--config2',
        help='test config file path',
        default='D:/PycharmProjects/fashion/demo/configs/global_predictor_vgg_attr.py')
    parser.add_argument(
        '--use_cuda', type=bool, default=True, help='use gpu or not')
    args = parser.parse_args()
    return args


def main():
    args = parse_args()
    cfg1 = Config.fromfile(args.config1)
    cfg2 = Config.fromfile(args.config2)
    img_tensor = get_img_tensor(args.input, args.use_cuda)
    # global attribute predictor will not use landmarks
    # just set a default value
    landmark_tensor = torch.zeros(8)
    create_text("result")  # 设置文件
    model1 = build_predictor(cfg1.model)
    model2 = build_predictor(cfg2.model)
    load_checkpoint(model1, args.checkpoint1, map_location='cpu')
    load_checkpoint(model2, args.checkpoint2, map_location='cpu')
    print('model loaded from {}'.format(args.checkpoint1))
    if args.use_cuda:
        #model.cuda()
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        model1.to(device)
        landmark_tensor = landmark_tensor.cuda()

    print('model loaded from {}'.format(args.checkpoint2))
    if args.use_cuda:
        # model.cuda()
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        model2.to(device)
        landmark_tensor = landmark_tensor.cuda()
    model1.eval()
    model2.eval()
    # predict probabilities for each attribute
    attr_prob = model2(
        img_tensor, attr=None, landmark=landmark_tensor, return_loss=False)
    attr_predictor = AttrPredictor(cfg2.data.test)
    attr_predictor.show_prediction(attr_prob)

    attr_prob,cate_prob = model1(
        img_tensor, attr=None, landmark=landmark_tensor, return_loss=False)
    cate_predictor = CatePredictor(cfg1.data.test)
    cate_predictor.show_prediction(cate_prob)
    # 要读取的文件夹路径
    readpath = r"D:/PycharmProjects/fashion/result/re"
    filejson, length = attr_predictor.txtToJson(readpath)

    # 保存的文件路径 1.json可以更换成其他的名字
    save_path = r"D:/PycharmProjects/fashion/result/re/1.json"
    attr_predictor.saveInJsonFile(filejson, save_path)
    print(filejson)

if __name__ == '__main__':
    main()
