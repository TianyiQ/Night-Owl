
""""
python -W ignore darkface_val_zdce_debug.py --trained_model ./weights/val_0/2022.5.19/dsfd.pth --save_folder qy001 --val_file ./data/OLD_wider_train.txt

green is std answer
red is your answer

plz use --val_file filename
format:
picname facenumber (posX posY lenX lenY 1(confidence)) * facenumber

"""
from __future__ import print_function

import argparse
from logging.config import valid_ident
import math
import os
import pdb
import sys
import time

import cv2
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.backends.cudnn as cudnn
import torch.nn as nn
import torch.utils.data as data
import torchvision.transforms as transforms
from PIL import Image
from torch.autograd import Variable
from data.config import cfg

from _data import *
from _data import BaseTransform, TestBaseTransform
from _data import DARKFace_CLASSES as labelmap
from _data import (DARKFace_ROOT, DARKFaceAnnotationTransform,
                  DARKFaceDetection)
from models.factory import build_net_zdce


plt.switch_backend('agg')

parser = argparse.ArgumentParser(description='DSFD: Dual Shot Face Detector')
parser.add_argument('--trained_model', default='',
                    type=str, help='Trained state_dict file path to open')
# parser.add_argument('--save_folder', default='eval_tools/WIDERFace_DSFD_RES152_results/', type=str,
                    # help='Dir to save results')
parser.add_argument('--save_folder', default='', type=str,
                    help='Dir to save results')
parser.add_argument('--visual_threshold', default=0.01, type=float,
                    help='Final confidence threshold')
parser.add_argument('--cuda', default=torch.cuda.is_available(), type=bool,
                    help='Use cuda to train model')
parser.add_argument('--val_file', default='', help='a list of val file and answer')

args = parser.parse_args()
print('cuda?', args.cuda)

# output_dir = 'C:/Programming/git-projects/low-light-face-detection/outputs/'.replace('\\','/')
output_dir = './results/' + args.save_folder + '/'
print(output_dir)

if args.trained_model=='':
    print('specify a trained model using `trained_model`')
    exit()

if args.cuda and torch.cuda.is_available():
    torch.set_default_tensor_type('torch.cuda.FloatTensor')
else:
    torch.set_default_tensor_type('torch.FloatTensor')
# if not os.path.exists(args.save_folder):
#     os.mkdir(args.save_folder)

def detect_face(image, shrink):
    with torch.no_grad():
        x = image
        if shrink != 1:
            x = cv2.resize(image, None, None, fx=shrink, fy=shrink, interpolation=cv2.INTER_LINEAR)
        #print('shrink:{}'.format(shrink))
        width = x.shape[1]
        height = x.shape[0]
        x = x.astype(np.float32)
        x -= np.array([104, 117, 123],dtype=np.float32)
        
        x = torch.from_numpy(x).permute(2, 0, 1)
        x = x[[2, 1, 0], :, :]
        x = x.unsqueeze(0)
        x = Variable(x.cuda() if args.cuda else x)

        #net.priorbox = PriorBoxLayer(width,height)
        print('net start')
        y = net(x)
        print('net end')
        detections = y.data
        scale = torch.Tensor([width, height, width, height])

    boxes=[]
    scores = []
    for i in range(detections.size(1)):
        j = 0
        while detections[0,i,j,0] >= 0.01:
            score = detections[0,i,j,0].cpu().numpy()
            pt = (detections[0, i, j, 1:]*scale).cpu().numpy()
            boxes.append([pt[0],pt[1],pt[2],pt[3]])
            scores.append(score)
            j += 1
            if j >= detections.size(2):
                break

    det_conf = np.array(scores)
    boxes = np.array(boxes)

    if boxes.shape[0] == 0:
        return np.array([[0,0,0,0,0.001]])

    det_xmin = boxes[:,0] / shrink
    det_ymin = boxes[:,1] / shrink
    det_xmax = boxes[:,2] / shrink
    det_ymax = boxes[:,3] / shrink
    det = np.column_stack((det_xmin, det_ymin, det_xmax, det_ymax, det_conf))

    keep_index = np.where(det[:, 4] >= 0)[0]
    det = det[keep_index, :]
    return det


def multi_scale_test(image, max_im_shrink):
    # shrink detecting and shrink only detect big face
    st = 0.5 if max_im_shrink >= 0.75 else 0.5 * max_im_shrink
    det_s = detect_face(image, st)
    if max_im_shrink > 0.75:
        det_s = np.row_stack((det_s,detect_face(image,0.75)))
    index = np.where(np.maximum(det_s[:, 2] - det_s[:, 0] + 1, det_s[:, 3] - det_s[:, 1] + 1) > 30)[0]
    det_s = det_s[index, :]
    # enlarge one times
    bt = min(2, max_im_shrink) if max_im_shrink > 1 else (st + max_im_shrink) / 2
    det_b = detect_face(image, bt)

    # enlarge small iamge x times for small face
    if max_im_shrink > 1.5:
        det_b = np.row_stack((det_b,detect_face(image,1.5)))
    if max_im_shrink > 2:
        bt *= 2
        while bt < max_im_shrink: # and bt <= 2:
            det_b = np.row_stack((det_b, detect_face(image, bt)))
            bt *= 2

        det_b = np.row_stack((det_b, detect_face(image, max_im_shrink)))

    # enlarge only detect small face
    if bt > 1:
        index = np.where(np.minimum(det_b[:, 2] - det_b[:, 0] + 1, det_b[:, 3] - det_b[:, 1] + 1) < 100)[0]
        det_b = det_b[index, :]
    else:
        index = np.where(np.maximum(det_b[:, 2] - det_b[:, 0] + 1, det_b[:, 3] - det_b[:, 1] + 1) > 30)[0]
        det_b = det_b[index, :]

    return det_s, det_b

def multi_scale_test_pyramid(image, max_shrink):
    # shrink detecting and shrink only detect big face
    det_b = detect_face(image, 0.25)
    index = np.where(
        np.maximum(det_b[:, 2] - det_b[:, 0] + 1, det_b[:, 3] - det_b[:, 1] + 1)
        > 30)[0]
    det_b = det_b[index, :]

    st = [1.25, 1.75, 2.25]
    for i in range(len(st)):
        if (st[i] <= max_shrink):
            det_temp = detect_face(image, st[i])
            # enlarge only detect small face
            if st[i] > 1:
                index = np.where(
                    np.minimum(det_temp[:, 2] - det_temp[:, 0] + 1,
                               det_temp[:, 3] - det_temp[:, 1] + 1) < 100)[0]
                det_temp = det_temp[index, :]
            else:
                index = np.where(
                    np.maximum(det_temp[:, 2] - det_temp[:, 0] + 1,
                               det_temp[:, 3] - det_temp[:, 1] + 1) > 30)[0]
                det_temp = det_temp[index, :]
            det_b = np.row_stack((det_b, det_temp))
    return det_b



def flip_test(image, shrink):
    image_f = cv2.flip(image, 1)
    det_f = detect_face(image_f, shrink)

    det_t = np.zeros(det_f.shape)
    det_t[:, 0] = image.shape[1] - det_f[:, 2]
    det_t[:, 1] = det_f[:, 1]
    det_t[:, 2] = image.shape[1] - det_f[:, 0]
    det_t[:, 3] = det_f[:, 3]
    det_t[:, 4] = det_f[:, 4]
    return det_t


def bbox_vote(det):
    order = det[:, 4].ravel().argsort()[::-1]
    det = det[order, :]
    dets = np.zeros((0, 5),dtype=np.float32)
    while det.shape[0] > 0:
        # IOU
        area = (det[:, 2] - det[:, 0] + 1) * (det[:, 3] - det[:, 1] + 1)
        xx1 = np.maximum(det[0, 0], det[:, 0])
        yy1 = np.maximum(det[0, 1], det[:, 1])
        xx2 = np.minimum(det[0, 2], det[:, 2])
        yy2 = np.minimum(det[0, 3], det[:, 3])
        w = np.maximum(0.0, xx2 - xx1 + 1)
        h = np.maximum(0.0, yy2 - yy1 + 1)
        inter = w * h
        o = inter / (area[0] + area[:] - inter)

        # get needed merge det and delete these det
        merge_index = np.where(o >= 0.3)[0]
        det_accu = det[merge_index, :]
        det = np.delete(det, merge_index, 0)

        if merge_index.shape[0] <= 1:
            continue
        det_accu[:, 0:4] = det_accu[:, 0:4] * np.tile(det_accu[:, -1:], (1, 4))
        max_score = np.max(det_accu[:, 4])
        det_accu_sum = np.zeros((1, 5))
        det_accu_sum[:, 0:4] = np.sum(det_accu[:, 0:4], axis=0) / np.sum(det_accu[:, -1:])
        det_accu_sum[:, 4] = max_score
        try:
            dets = np.row_stack((dets, det_accu_sum))
        except:
            dets = det_accu_sum

    dets = dets[0:750, :]
    return dets


def write_to_txt(f, det, H, W):
    # f.write('{:s}\n'.format(event + '/' + im_name))
    # f.write('{:d}\n'.format(det.shape[0]))
    for i in range(det.shape[0]):
        xmin = max(0,min(H-1,int(det[i][0]+0.5)))
        ymin = max(0,min(W-1,int(det[i][1]+0.5)))
        xmax = max(0,min(H-1,int(det[i][2]+0.5)))
        ymax = max(0,min(W-1,int(det[i][3]+0.5)))
        assert xmin <= xmax
        assert ymin <= ymax
        score = det[i][4] 
        f.write('{:.1f} {:.1f} {:.1f} {:.1f} {:.3f}\n'.
                format(xmin, ymin, xmax, ymax, score))

def read_from_txt(f):
    # f.write('{:s}\n'.format(event + '/' + im_name))
    # f.write('{:d}\n'.format(det.shape[0]))
    li = []
    for i in f.readlines():
        li.append(list(map(float, i.split())))
    return np.array(li)

if __name__=='__main__':
    # load net
    # cfg = darkface_640
    # num_classes = len(DARKFace_CLASSES) + 1 # +1 background
    # net = build_ssd('test', cfg['min_dim'], num_classes) # initialize SSD
    net = build_net_zdce('test', cfg.NUM_CLASSES)
    if args.cuda:
        print(f'using {args.trained_model}')
        net.load_state_dict(torch.load(args.trained_model))
        net.cuda()
    else:
        net.load_state_dict(torch.load('.\\weights\\val_0\\2022.5.19\\dsfd.pth', map_location='cpu'))
    net.eval()
    print('Finished loading model!')


def area(s):
    return (s[2]-s[0])*(s[3]-s[1])
def vis_detections(imgid, im,  dets, dets_answer, H, W, thresh=0.5):
    '''Draw detected bounding boxes.'''
    class_name = 'face'
    inds = np.where(dets[:, -1] >= thresh)[0]
    # if len(inds) == 0:
        # return
    im = im[:, :, (2, 1, 0)]
    print (len(inds))
    fig, ax = plt.subplots(figsize=(12, 12))
    ax.imshow(im, aspect='equal')
    for i in inds:
        bbox = dets[i, :4]
        score = dets[i, -1]
        xmin = max(0,min(H-1,int(bbox[0]+0.5)))
        ymin = max(0,min(W-1,int(bbox[1]+0.5)))
        xmax = max(0,min(H-1,int(bbox[2]+0.5)))
        ymax = max(0,min(W-1,int(bbox[3]+0.5)))
        assert xmin <= xmax
        assert ymin <= ymax
        ax.add_patch(
            plt.Rectangle((xmin, ymin),
                          xmax-xmin,
                          ymax-ymin, fill=False,
                          edgecolor='red', linewidth=2.5)
            )
        '''
        ax.text(bbox[0], bbox[1] - 5,
                '{:s} {:.3f}'.format(class_name, score),
                bbox=dict(facecolor='blue', alpha=0.5),
                fontsize=10, color='white')
        '''
    for i in dets_answer:
        xmin = max(0,min(H-1,int(i[0]+0.5)))
        ymin = max(0,min(W-1,int(i[1]+0.5)))
        xmax = max(0,min(H-1,int(i[2]+0.5)))
        ymax = max(0,min(W-1,int(i[3]+0.5)))
        ax.add_patch(
            plt.Rectangle((xmin, ymin),
                          xmax-xmin,
                          ymax-ymin, fill=False,
                          edgecolor='green', linewidth=2.5)
            )
    #ax.set_title(('{} detections with '
    #              'p({} | box) >= {:.1f}').format(class_name, class_name,
    #                                              thresh),
    #              fontsize=10)
    
    for i in inds:
        bbox = dets[i, :4]
        for i in dets_answer:
            xmin = max(0,min(H-1,int(max(bbox[0],i[0])+0.5)))
            ymin = max(0,min(W-1,int(max(bbox[1],i[1])+0.5)))
            xmax = max(0,min(H-1,int(min(bbox[2],i[2])+0.5)))
            ymax = max(0,min(W-1,int(min(bbox[3],i[3])+0.5)))
            if (xmax-xmin)*(ymax-ymin)>0.5*(area(i)+area(bbox))/2:
                if (xmin <= xmax) and (ymin <= ymax):
                     ax.add_patch(plt.Rectangle((xmin, ymin),xmax-xmin,ymax-ymin, fill=False,edgecolor='orange', linewidth=2.5))
    plt.axis('off')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir.replace('\\','/') , 'pics/'+str(imgid)), dpi=fig.dpi)
    plt.close('all')

if __name__=='__main__':
    print('Finished loading data')    

from time import perf_counter

def test_darkface():
    # evaluation
    cuda = args.cuda
    transform = TestBaseTransform((104, 117, 123))
    thresh=cfg['CONF_THRESH']
    save_path = output_dir #args.save_folder

    pics = open(args.val_file, 'r')
    piclist = pics.readlines()

    # print(plt.savefig(os.path.join(output_dir.replace('\\','/') , 'pics/'+str(0)), dpi=fig.dpi)

    num_images = len(piclist)
    for i in range(0, num_images):
        ss = piclist[i].split()
        print('loading pic ', ss[0])

        image = cv2.imread(ss[0], cv2.IMREAD_COLOR)
        img_id = ss[0].split('\\')[-1].split('/')[-1]
        answer_cnt = int(ss[1])
        dets_answer = []
        for j in range(answer_cnt):
            px = float(ss[j * 5 + 2])
            py = float(ss[j * 5 + 3])
            lx = float(ss[j * 5 + 4])
            ly = float(ss[j * 5 + 5])
            dets_answer.append([px, py, px + lx, py + ly, 1])
        print('Testing image {:d}/{:d} {} {} ....'.format(i+1, num_images , img_id , image.shape))
        end_enhance_clock = perf_counter()

        if os.path.exists(os.path.join(save_path , img_id.split('.')[0] + '.txt').replace('\\','/')):
            print('skipped')
            continue
        
        # event = testset.pull_event(i)
        W = image.shape[0]
        H = image.shape[1]

        #max_im_shrink = ( (2000.0*2000.0) / (img.shape[0] * img.shape[1])) ** 0.5
        max_im_shrink = (0x7fffffff / 200.0 / (image.shape[0] * image.shape[1])) ** 0.5 # the max size of input image for caffe
#         max_im_shrink = np.sqrt(1500 * 1000 / (image.shape[0] * image.shape[1]))
        max_im_shrink = 3 if max_im_shrink > 3 else max_im_shrink
            
        shrink = max_im_shrink if max_im_shrink < 1 else 1

#        print('--Debug : state A:')
        det0 = detect_face(image, shrink)  # origin test
#        print('--Debug : state B:')
        det1 = flip_test(image, shrink)    # flip test
        [det2, det3] = multi_scale_test(image, max_im_shrink)#min(2,1400/min(image.shape[0],image.shape[1])))  #multi-scale test
#        print('--Debug : state C:')
        det4 = multi_scale_test_pyramid(image, max_im_shrink)
#        print('--Debug : state D:')
        det = np.row_stack((det0, det1, det2, det3, det4))
#        print('--Debug : state E:')

        dets = bbox_vote(det)
#        print('--Debug : state F:')
        #vis_detections(i ,image, dets , 0.8)   
        
        
#         det0 = detect_face(image, shrink)  # origin test
#         det = np.row_stack((det0, ))

#         dets = bbox_vote(det)
        
        if not os.path.exists(save_path):
            os.makedirs(save_path)
        f = open(os.path.join(save_path , img_id.split('.')[0] + '.txt').replace('\\','/'), 'w')
        write_to_txt(f, dets , H , W)
        end_inference_clock = perf_counter()
        print(f'inference taken {end_inference_clock-end_enhance_clock} sec')
        vis_detections(img_id.split('.')[0], image, dets, dets_answer, H, W)
#         print(f'visualization taken {perf_counter()-end_inference_clock} sec', end='\n\n')
        
        
if __name__=='__main__':
    if not os.path.exists(os.path.join(output_dir.replace('\\','/'),'txts')):
        os.mkdir(os.path.join(output_dir.replace('\\','/'),'txts'))
    if not os.path.exists(os.path.join(output_dir.replace('\\','/'),'pics')):
        os.mkdir(os.path.join(output_dir.replace('\\','/'),'pics'))
    test_darkface()
