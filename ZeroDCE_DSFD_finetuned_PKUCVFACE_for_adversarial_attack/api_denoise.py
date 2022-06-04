#-*- coding:utf-8 -*-

from __future__ import division
from __future__ import absolute_import
from __future__ import print_function

import os
import torch
import argparse
import torch.nn as nn
import torch.utils.data as data
import torch.backends.cudnn as cudnn
import torchvision.transforms as transforms

import cv2
import time
import numpy as np
from PIL import Image

import warnings

from data.config import cfg
from models.factory import build_net, build_net_zdce
from torch.autograd import Variable
from utils.augmentations import to_chw_bgr

warnings.filterwarnings("ignore")

class ApiClass:
    full_net = build_net_zdce('test', cfg.NUM_CLASSES)
    zdce_net = build_net_zdce('test', cfg.NUM_CLASSES)
    use_cuda = torch.cuda.is_available()
    h = 11

    @classmethod
    def init(cls):
        cls.full_net.load_weights_all('./weights/weights.pth')
        cls.zdce_net.load_weights_zdce('./weights/zdce-original.pth')
        if cls.use_cuda:
            cls.full_net = cls.full_net.cuda()
            cls.zdce_net = cls.zdce_net.cuda()
    
    @classmethod
    def get_confidence_by_path(cls, img_path): # img_path: 暗光图片; 返回一个[0,1]中的float表示置信度
        warnings.filterwarnings("ignore")

        if os.path.isfile('./tmp/tmp.png'):
            os.remove('./tmp/tmp.png')
        if os.path.isfile('./tmp/out.txt'):
            os.remove('./tmp/out.txt')
        os.system(f"ffmpeg -i {img_path} -filter_complex bm3d=sigma=7/255:block=4:bstep=2:group=1:hdthr=10000:estim=basic ./tmp/tmp.png 2> ./tmp/out.txt")
        img_path = './tmp/tmp.png'

        img = Image.open(img_path)
        if img.mode == 'L':
            img = img.convert('RGB')

        img = np.array(img)

        x = to_chw_bgr(img)
        x = x.astype('float32')
        x -= cfg.img_mean
        x = x[[2, 1, 0], :, :]

        x = Variable(torch.from_numpy(x).unsqueeze(0))
        if cls.use_cuda:
            x = x.cuda()
        t1 = time.time()
        with torch.no_grad():
            y = cls.full_net(x)
        detections = y.data
        max_conf = 0
        for i in range(detections.size(1)):
            max_conf = max(max_conf, detections[0, i, 0, 0].item())
        
        return max_conf**5
    
    @classmethod
    def get_confidence_by_ndarray(cls, imgs): # imgs: ndarray with shape (B,C,H,W) (not (B,C,W,H)!!!)
        B, C, H, W = imgs.shape

        for i in range(B):
            img = imgs[i, [2,1,0], ...].transpose((1,2,0))
            cv2.imwrite(f'./tmp/tmq.png', np.maximum(0, np.minimum(255, (img*255).astype(np.int16))))
            if os.path.isfile('./tmp/tmp.png'):
                os.remove('./tmp/tmp.png')
            if os.path.isfile('./tmp/out.txt'):
                os.remove('./tmp/out.txt')
            os.system(f"ffmpeg -i ./tmp/tmq.png -filter_complex bm3d=sigma=7/255:block=4:bstep=2:group=1:hdthr=10000:estim=basic ./tmp/tmp.png 2> ./tmp/out.txt")
            imgs[i] = cv2.imread('./tmp/tmp.png').astype('float32')[:, :, [2,1,0]].transpose((2,0,1)) # / 255
        
        # imgs *= 255
        x = imgs[:, [2, 1, 0], :, :]
        x = x.astype('float32')
        x -= cfg.img_mean[np.newaxis, :, :, :]
        x = x[:, [2, 1, 0], :, :]

        x = Variable(torch.from_numpy(x))
        if cls.use_cuda:
            x = x.cuda()
        
        with torch.no_grad():
            y = cls.full_net(x)
        detections = y.data
        max_conf = detections[:, :, 0, 0].cpu().numpy().max(axis=1)
        
        return max_conf**5
    
    @classmethod
    def lowlight_enhance_by_path(cls, src_img_path, target_img_path): # src_img_path: 暗光图片; target_img_path: 提亮后的图片保存到哪里
        warnings.filterwarnings("ignore")

        img = Image.open(src_img_path)
        if img.mode == 'L':
            img = img.convert('RGB')

        img = np.array(img)

        x = to_chw_bgr(img)
        x = x.astype('float32')
        x -= cfg.img_mean
        x = x[[2, 1, 0], :, :]

        x = Variable(torch.from_numpy(x).unsqueeze(0))
        if cls.use_cuda:
            x = x.cuda()
        
        with torch.no_grad():
            y = cls.zdce_net.enhance(x)[0]
            y = y[[2, 1, 0], :, :]
            y = y.permute([1,2,0]).cpu().numpy()
            cv2.imwrite(target_img_path, np.maximum(0, np.minimum(255, (y*255).astype(np.int16))))

if __name__ == '__main__': # 示例代码
    t1 = time.perf_counter()
    
    ApiClass.init()
    t2 = time.perf_counter()
    print(f'init taken {t2-t1}s')

    print(f"confidence = {ApiClass.get_confidence_by_path(img_path='./img/face2.png')}")
    t3 = time.perf_counter()
    print(f'get_confidence_by_path taken {t3-t2}s')

    ApiClass.lowlight_enhance_by_path(src_img_path='./img/face2.png', target_img_path='./enh/face2_enh.png') # 提亮后的图片会存储在target_img_path
    t4 = time.perf_counter()
    print(f'lowlight_enhance_by_path taken {t4-t3}s')

    img = [cv2.imread('./img/face1.png').astype('float32'),
           cv2.imread('./img/face2.png').astype('float32'),
           cv2.imread('./img/noface1.png').astype('float32'),
           cv2.imread('./img/noface2.png').astype('float32')]
    img = np.array(img)
    img = img[:, :, :, [2,1,0]].transpose((0,3,1,2)) / 255
    print(f"confidence = {ApiClass.get_confidence_by_ndarray(img)}") # 注意：传入的img的维度形如(B,C,H,W)，不是(B,C,W,H)!!!
    t5 = time.perf_counter()
    print(f'get_confidence_by_ndarray taken {t5-t4}s')