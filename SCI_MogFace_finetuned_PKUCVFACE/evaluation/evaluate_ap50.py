"""
WiderFace evaluation code
author: wondervictor
mail: tianhengcheng@gmail.com
copyright@wondervictor
"""

from __future__ import absolute_import
import os
import tqdm
import pickle
import argparse
import numpy as np
from scipy.io import loadmat
import utils.bbox_utils as bbox
import torch


def np_around(array, num_decimals=0):
    return array
    #return np.around(array, decimals=num_decimals)
def compute_iou(box_a, box_b):
    x0 = np.maximum(box_a[:,0], box_b[0])
    y0 = np.maximum(box_a[:,1], box_b[1])
    x1 = np.minimum(box_a[:,2], box_b[2])
    y1 = np.minimum(box_a[:,3], box_b[3])
    #print ('x0', x0[0], x1[0], y0[0], y1[0], box_a[0], box_b[:])
    #w = np.maximum(x1 - x0 + 1, 0) 
    w = np_around(x1 - x0 + 1) 
    #h = np.maximum(y1 - y0 + 1, 0)
    h = np_around(y1 - y0 + 1)
    inter = np_around(w * h)
    area_a = (box_a[:,2] - box_a[:,0] + 1) * (box_a[:,3] - box_a[:,1] + 1)
    area_a = np_around(area_a)
    area_b = (box_b[2] - box_b[0] + 1) * (box_b[3] - box_b[1] + 1)
    area_b = np_around(area_b)
    iou = inter / (area_a + area_b - inter)
    iou[w <= 0] = 0
    iou[h <=0] = 0
    return iou

def np_round(val, decimals=4):
    return val

def get_gt_boxes(gt_dir):
    """ gt dir: (wider_face_val.mat, wider_easy_val.mat, wider_medium_val.mat, wider_hard_val.mat)"""

    gt_mat = loadmat(os.path.join(gt_dir, 'wider_face_val.mat'))
    hard_mat = loadmat(os.path.join(gt_dir, 'wider_hard_val.mat'))
    medium_mat = loadmat(os.path.join(gt_dir, 'wider_medium_val.mat'))
    easy_mat = loadmat(os.path.join(gt_dir, 'wider_easy_val.mat'))

    facebox_list = gt_mat['face_bbx_list']
    event_list = gt_mat['event_list']
    file_list = gt_mat['file_list']

    hard_gt_list = hard_mat['gt_list']
    medium_gt_list = medium_mat['gt_list']
    easy_gt_list = easy_mat['gt_list']

    return facebox_list, event_list, file_list, hard_gt_list, medium_gt_list, easy_gt_list


def get_gt_boxes_from_txt(gt_path, cache_dir):

    cache_file = os.path.join(cache_dir, 'gt_cache.pkl')
    if os.path.exists(cache_file):
        f = open(cache_file, 'rb')
        boxes = pickle.load(f)
        f.close()
        return boxes

    f = open(gt_path, 'r')
    state = 0
    lines = f.readlines()
    lines = list(map(lambda x: x.rstrip('\r\n'), lines))
    boxes = {}
    print(len(lines))
    f.close()
    current_boxes = []
    current_name = None
    for line in lines:
        if state == 0 and '--' in line:
            state = 1
            current_name = line
            continue
        if state == 1:
            state = 2
            continue

        if state == 2 and '--' in line:
            state = 1
            boxes[current_name] = np.array(current_boxes).astype('float32')
            current_name = line
            current_boxes = []
            continue

        if state == 2:
            box = [float(x) for x in line.split(' ')[:4]]
            current_boxes.append(box)
            continue

    f = open(cache_file, 'wb')
    pickle.dump(boxes, f)
    f.close()
    return boxes


def read_pred_file(filepath):

    with open(filepath, 'r') as f:
        lines = f.readlines()
        img_file = lines[0].rstrip('\n\r')
        lines = lines[2:]

    boxes = np.array(list(map(lambda x: [float(a) for a in x.rstrip('\r\n').split(' ')], lines))).astype('float')
    return img_file.split('/')[-1], boxes


def get_preds(pred_dir):
    events = os.listdir(pred_dir)
    boxes = dict()
    pbar = tqdm.tqdm(events)
    
    for event in pbar:
        pbar.set_description('Reading Predictions ')
        event_dir = os.path.join(pred_dir, event)
        event_images = os.listdir(event_dir)
        current_event = dict()
        for imgtxt in event_images:
            imgname, _boxes = read_pred_file(os.path.join(event_dir, imgtxt))
            current_event[imgname.rstrip('.png').rstrip('.png')] = _boxes
        boxes[event] = current_event
    return boxes


def norm_score(pred):
    """ norm score
    pred {key: [[x1,y1,x2,y2,s]]}
    """

    #max_score = -1
    #min_score = 2
    max_score = 0
    min_score = 1

    for _, k in pred.items():
        for _, v in k.items():
            if len(v) == 0:
                continue
            _min = np.min(v[:, -1])
            _max = np.max(v[:, -1])
            max_score = max(_max, max_score)
            min_score = min(_min, min_score)

    diff = max_score - min_score
    for _, k in pred.items():
        for _, v in k.items():
            if len(v) == 0:
                continue
            v[:, -1] = (v[:, -1] - min_score).astype(np.float64)/diff
    return pred


def image_eval(pred, gt, ignore, iou_thresh):
    """ single image evaluation
    pred: Nx5
    gt: Nx4
    ignore:
    """

    
    _pred = pred.copy()
    _gt = gt.copy()
    pred_recall = np.zeros(_pred.shape[0])
    recall_list = np.zeros(_gt.shape[0])
    proposal_list = np.ones(_pred.shape[0])

    _pred[:, 2] = _pred[:, 2] + _pred[:, 0]
    _pred[:, 3] = _pred[:, 3] + _pred[:, 1]
    _gt[:, 2] = _gt[:, 2] + _gt[:, 0]
    _gt[:, 3] = _gt[:, 3] + _gt[:, 1]

    #overlaps = (jaccard(torch.FloatTensor(_pred[:, :4]), torch.FloatTensor(_gt))).numpy()
    #overlaps = compute_iou((_pred[:, :4]), (_gt))
    overlaps = bbox.bbox_overlap(_pred[:, :4], _gt)
    # overlaps = bbox.bbox_overlaps(_pred[:, :4], _gt)

    for h in range(_pred.shape[0]):

        gt_overlap = overlaps[h]
        #max_overlap, max_idx = gt_overlap.max(), gt_overlap.argmax()
        #gt_overlap = compute_iou(_gt, _pred[h, :4])
        #exit()
        #exit()
        #print ('overlap', gt_overlap)
        max_overlap, max_idx = gt_overlap.max(), gt_overlap.argmax()

        if max_overlap >= iou_thresh:
            if ignore[max_idx] == 0:
                recall_list[max_idx] = -1
                proposal_list[h] = -1
            elif recall_list[max_idx] == 0:
                recall_list[max_idx] = 1

        r_keep_index = np.where(recall_list == 1)[0]
        pred_recall[h] = len(r_keep_index)

    return pred_recall, proposal_list


def img_pr_info(thresh_num, pred_info, proposal_list, pred_recall):
    pr_info = np.zeros((thresh_num, 2)).astype('float')
    for t in range(thresh_num):

        thresh = 1 - (t+1)/thresh_num
        r_index = np.where(pred_info[:, 4] >= thresh)[0]
        if len(r_index) == 0:
            pr_info[t, 0] = 0
            pr_info[t, 1] = 0
        else:
            r_index = r_index[-1]
            p_index = np.where(proposal_list[:r_index+1] == 1)[0]
            pr_info[t, 0] = len(p_index)
            pr_info[t, 1] = pred_recall[r_index]
    return pr_info


def dataset_pr_info(thresh_num, pr_curve, count_face):
    _pr_curve = np.zeros((thresh_num, 2))
    for i in range(thresh_num):
        _pr_curve[i, 0] = round(pr_curve[i, 1] / pr_curve[i, 0], 4)
        _pr_curve[i, 1] = round(pr_curve[i, 1] / count_face, 4)
    return _pr_curve


def voc_ap(rec, prec):

    # correct AP calculation
    # first append sentinel values at the end
    #print ('rec:', rec)
    #print ('pre:', prec)
    mrec = np.concatenate(([0.], rec, [1.]))
    mpre = np.concatenate(([0.], prec, [0.]))

    # compute the precision envelope
    for i in range(mpre.size - 1, 0, -1):
        mpre[i - 1] = np.maximum(mpre[i - 1], mpre[i])

    # to calculate area under PR curve, look for points
    # where X axis (recall) changes value
    i = np.where(mrec[1:] != mrec[:-1])[0]

    # and sum (\Delta recall) * prec
    ap = np_round(np.sum((mrec[i + 1] - mrec[i]) * mpre[i + 1]))
    return ap


def evaluation_ap50(pred, gt_path, iter, det_result_txt=None):
    if det_result_txt is None:
        det_result_txt = './result.txt'
    test_method = pred.split('/')[-1]
    pred = get_preds(pred)
    pred = norm_score(pred)
    facebox_list, event_list, file_list, hard_gt_list, medium_gt_list, easy_gt_list = get_gt_boxes(gt_path)
    event_num = len(event_list)
    thresh_num = 1000
    settings = ['easy', 'medium', 'hard']
    setting_gts = [easy_gt_list, medium_gt_list, hard_gt_list]
    aps = []
    setting_id = 2
    for setting_id in range(3):
        # different setting
        iou_th = 0.5 #+ 0.05 * idx
        # different setting
        gt_list = setting_gts[setting_id]
        count_face = 0
        pr_curve = np.zeros((thresh_num, 2)).astype('float')
        # [hard, medium, easy]
        pbar = tqdm.tqdm(range(event_num))
        for i in pbar:
            pbar.set_description('Processing {}'.format(settings[setting_id]))
            event_name = str(event_list[i][0][0]).replace(' ','')
            img_list = file_list[i][0]
            pred_list = pred[event_name]
            sub_gt_list = gt_list[i][0]
            # img_pr_info_list = np.zeros((len(img_list), thresh_num, 2))
            gt_bbx_list = facebox_list[i][0]

            for j in range(len(img_list)):
                pred_info = pred_list[str(img_list[j][0][0]).replace(' ','')]

                gt_boxes = gt_bbx_list[j][0].astype('float')
                keep_index = sub_gt_list[j][0]
                #print ('keep_index', keep_index)
                count_face += len(keep_index)
                

                if len(gt_boxes) == 0 or len(pred_info) == 0:
                    continue
                ignore = np.zeros(gt_boxes.shape[0])
                if len(keep_index) != 0:
                    ignore[keep_index-1] = 1
                pred_info = np_round(pred_info,1)
                pred_sort_idx= np.argsort(pred_info[:,4])
                pred_info = pred_info[pred_sort_idx][::-1]
                #print ('pred_info', pred_info[:20, 4])
                #exit()


                gt_boxes = np_round(gt_boxes)
                ignore = np_round(ignore)
                pred_recall, proposal_list = image_eval(pred_info, gt_boxes, ignore, iou_th)
                #print('1 stage', pred_recall, proposal_list)

                _img_pr_info = img_pr_info(thresh_num, pred_info, proposal_list, pred_recall)
                #print ('img_pr_info', _img_pr_info)
                pr_curve += _img_pr_info
        #print ('pr_curve', pr_curve, count_face)
        pr_curve = dataset_pr_info(thresh_num, pr_curve, count_face)

        propose = pr_curve[:, 0]
        recall = pr_curve[:, 1]

        #print ('propose, recall', propose, recall)
        #exit()
        ap = voc_ap(recall, propose)
        aps.append(ap)
        #print ('ap:{}, iou:{}:'.format(ap, iou_th))
    # print("Easy Val AP: ", aps[0])
    # print("Medium Val AP: ", aps[1])
    # print("Hard Val AP: ", aps[2])
    # fo_res = open(det_result_txt, 'a+')
    # fo_res.write("=================Iter : {} k  Results ====================".format(iter) + "\n")
    # fo_res.write("Test Method: {}".format(test_method) + "\n")
    # fo_res.write('Iter:' + ' ' + str(iter) + 'k' + ' Easy  Val AP: {}'.format(aps[0]) + '\n')
    # fo_res.write('Iter:' + ' ' + str(iter) + 'k' + ' Medium Val AP: {}'.format(aps[1]) + '\n')
    # fo_res.write('Iter:' + ' ' + str(iter) + 'k' + ' Hard Val AP: {}'.format(aps[2]) + '\n')
    # fo_res.write('\n')
    # fo_res.close()
    return aps
    
if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('-p', '--pred', default='')
    parser.add_argument('-g', '--gt', default='./dataset/ground_truth/')
    parser.add_argument('-i', '--iter', default='140')
    parser.add_argument('-d', '--det_result_txt', default=None)

    args = parser.parse_args()
    evaluation_ap50(args.pred, args.gt, args.iter, args.det_result_txt)












