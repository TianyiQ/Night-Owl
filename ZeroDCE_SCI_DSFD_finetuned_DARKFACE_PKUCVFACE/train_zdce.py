#-*- coding:utf-8 -*-

from __future__ import division
from __future__ import absolute_import
from __future__ import print_function

import os
import time
import torch
import argparse
import torch.nn as nn
import torch.optim as optim
import torch.nn.init as init
import torch.utils.data as data
import numpy as np
from torch.autograd import Variable
import torch.backends.cudnn as cudnn

from data.config import cfg
from layers.modules import MultiBoxLoss
from data.widerface import WIDERDetection, detection_collate
from models.factory import build_net, basenet_factory, build_net_zdce

parser = argparse.ArgumentParser(
    description='DSFD face Detector Training With Pytorch')
train_set = parser.add_mutually_exclusive_group()
parser.add_argument('--batch_size',
                    default=16, type=int,
                    help='Batch size for training')
parser.add_argument('--model',
                    default='vgg', type=str,
                    choices=['vgg', 'resnet50', 'resnet101', 'resnet152'],
                    help='model for training')
parser.add_argument('--resume',
                    default=None, type=str,
                    help='Checkpoint state_dict file to resume training from')
parser.add_argument('--num_workers',
                    default=4, type=int,
                    help='Number of workers used in dataloading')
parser.add_argument('--cuda',
                    default=True, type=bool,
                    help='Use CUDA to train model')
parser.add_argument('--lr', '--learning-rate',
                    default=1e-3, type=float,
                    help='initial learning rate')
parser.add_argument('--momentum',
                    default=0.9, type=float,
                    help='Momentum value for optim')
parser.add_argument('--weight_decay',
                    default=5e-4, type=float,
                    help='Weight decay for SGD')
parser.add_argument('--gamma',
                    default=0.1, type=float,
                    help='Gamma update for SGD')
parser.add_argument('--multigpu',
                    default=False, type=bool,
                    help='Use mutil Gpu training')
parser.add_argument('--save_folder',
                    default='NONE',
                    help='Directory for saving checkpoint models')
args = parser.parse_args()

if args.save_folder=='NONE':
    print('specify a folder name, e.g. val_0')
    exit()
    
if args.resume==None:
    print('specify a model, or use `pretrain` to start from scratch')
    exit()

if not args.multigpu:
    os.environ['CUDA_LAUNCH_BLOCKING'] = '1'

if torch.cuda.is_available():
    if args.cuda:
        torch.set_default_tensor_type('torch.cuda.FloatTensor')
    if not args.cuda:
        print("WARNING: It looks like you have a CUDA device, but aren't " +
              "using CUDA.\nRun with --cuda for optimal training speed.")
        torch.set_default_tensor_type('torch.FloatTensor')
else:
    torch.set_default_tensor_type('torch.FloatTensor')


save_folder = os.path.join('/code/weights', args.save_folder)
if not os.path.exists(save_folder):
    os.mkdir(save_folder)


train_dataset = WIDERDetection(cfg.FACE.TRAIN_FILE, mode='train')

val_dataset = WIDERDetection(cfg.FACE.VAL_FILE, mode='val')

train_loader = data.DataLoader(train_dataset, args.batch_size,
                               num_workers=args.num_workers,
                               shuffle=True,
                               collate_fn=detection_collate,
                               pin_memory=True,
                               generator=torch.Generator(device='cuda' if torch.cuda.is_available() else 'cpu'))
val_batchsize = args.batch_size // 2 # why //2 ?
val_loader = data.DataLoader(val_dataset, args.batch_size, # val_batchsize,
                             num_workers=args.num_workers,
                             shuffle=False,
                             collate_fn=detection_collate,
                             pin_memory=True)


min_loss = np.inf


def train():
    per_epoch_size = len(train_dataset) // args.batch_size
    start_epoch = 0
    iteration = 0
    step_index = 0

    # basenet = basenet_factory(args.model)
    dsfd_net = build_net_zdce('train', cfg.NUM_CLASSES)
    net = dsfd_net
    
    with torch.no_grad():
        if args.resume!='pretrain':
            print('Resuming training, loading {}...'.format(args.resume))
            start_epoch = net.load_weights_all(args.resume)
            iteration = start_epoch * per_epoch_size
        else:
            net.load_weights_dsfd('./weights/dsfd_vgg_0.880.pth')
#             net.load_weights_zdce('./weights/zdce_Epoch99.pth')
            net.load_weights_zdce('./ZeroDCE_finetune/snapshots/Epoch50.pth')
            # base_weights = torch.load(args.save_folder + basenet,map_location=torch.device('cpu'))
            # print('Load base network {}'.format(args.save_folder + basenet))
            # if args.model == 'vgg':
            #     net.vgg.load_state_dict(base_weights)
            # else:
            #     net.resnet.load_state_dict(base_weights)
    
    if not torch.cuda.is_available():
        args.cuda = False

    if args.cuda:
        if args.multigpu:
            net = torch.nn.DataParallel(dsfd_net)
        net = net.cuda()
        cudnn.benckmark = True

    # if not args.resume:
    #     print('Initializing weights...')
    #     dsfd_net.extras.apply(dsfd_net.weights_init)
    #     dsfd_net.fpn_topdown.apply(dsfd_net.weights_init)
    #     dsfd_net.fpn_latlayer.apply(dsfd_net.weights_init)
    #     dsfd_net.fpn_fem.apply(dsfd_net.weights_init)
    #     dsfd_net.loc_pal1.apply(dsfd_net.weights_init)
    #     dsfd_net.conf_pal1.apply(dsfd_net.weights_init)
    #     dsfd_net.loc_pal2.apply(dsfd_net.weights_init)
    #     dsfd_net.conf_pal2.apply(dsfd_net.weights_init)

    #optimizer = optim.SGD(net.parameters(), lr=args.lr, momentum=args.momentum,
    #                      weight_decay=args.weight_decay)
    optimizer = optim.Adam(net.parameters(), lr=args.lr, betas=(0.9,0.999),eps=1e-8,
                      weight_decay=args.weight_decay)
    criterion = MultiBoxLoss(cfg, args.cuda)
    print('Loading wider dataset...')
    print('Using the specified args:')
    print(args)

    for step in cfg.LR_STEPS:
        if iteration > step:
            step_index += 1
            adjust_learning_rate(optimizer, args.gamma, step_index)

    net.train()
#     with torch.no_grad():
#         val(0, net, dsfd_net, criterion)
    for epoch in range(start_epoch, cfg.EPOCHES):
        losses = 0
        for batch_idx, (images, targets) in enumerate(train_loader):
#             print(f'batch_idx: {batch_idx}, shape: {images.shape}')
            if args.cuda:
                images = Variable(images.cuda())
                targets = [Variable(ann.cuda(), volatile=True)
                           for ann in targets]
            else:
                images = Variable(images)
                targets = [Variable(ann, volatile=True) for ann in targets]

            if iteration in cfg.LR_STEPS:
                step_index += 1
                adjust_learning_rate(optimizer, args.gamma, step_index)

            t0 = time.time()
            out = net(images)
            # backprop
            optimizer.zero_grad()
            loss_l_pa1l, loss_c_pal1 = criterion(out[:3], targets)
            loss_l_pa12, loss_c_pal2 = criterion(out[3:], targets)

            loss = loss_l_pa1l + loss_c_pal1 + loss_l_pa12 + loss_c_pal2
            loss.backward()
            optimizer.step()
            t1 = time.time()
            losses += loss.item()

            if iteration % 50 == 0:
                tloss = losses / (batch_idx + 1)
                print('Timer: %.4f' % (t1 - t0))
                print('epoch:' + repr(epoch) + ' || iter:' +
                      repr(iteration) + ' || Loss:%.4f' % (tloss))
                print('->> pal1 conf loss:{:.4f} || pal1 loc loss:{:.4f}'.format(
                    loss_c_pal1.item(), loss_l_pa1l.item()))
                print('->> pal2 conf loss:{:.4f} || pal2 loc loss:{:.4f}'.format(
                    loss_c_pal2.item(), loss_l_pa12.item()))
                print('->>lr:{}'.format(optimizer.param_groups[0]['lr']))

            if iteration != 0 and iteration % 2000 == 0:
                print('Saving state, iter:', iteration)
                file = 'dsfd_zdce_' + repr(iteration) + '.pth'
                torch.save(dsfd_net.state_dict(),
                           os.path.join(save_folder, file))
            iteration += 1

        with torch.no_grad():
            val(epoch, net, dsfd_net, criterion)
        if iteration >= cfg.MAX_STEPS:
            break


def val(epoch, net, dsfd_net, criterion):
    f=open(f'{save_folder}/log.log','a')
    net.eval()
    step = 0
    losses = 0
    t1 = time.time()
    for batch_idx, (images, targets) in enumerate(val_loader):
#         print(f'batch_idx: {batch_idx}, shape: {images.shape}')
        if args.cuda:
            images = Variable(images.cuda())
            targets = [Variable(ann.cuda(), volatile=True)
                       for ann in targets]
        else:
            images = Variable(images)
            targets = [Variable(ann, volatile=True) for ann in targets]

        out = net(images)
        loss_l_pa1l, loss_c_pal1 = criterion(out[:3], targets)
        loss_l_pa12, loss_c_pal2 = criterion(out[3:], targets)
        loss = loss_l_pa12 + loss_c_pal2
        losses += loss.item()
        step += 1

    tloss = losses / step
    t2 = time.time()
    print('Timer: %.4f' % (t2 - t1))
    global min_loss
    print('test epoch:' + repr(epoch) + ' || Loss:%.4f' % (tloss) + ' || MinLoss:%.4f' % (min_loss))
    f.write('test epoch:' + repr(epoch) + ' || Loss:%.4f' % (tloss) + ' || MinLoss:%.4f' % (min_loss) + '\n')
    if tloss < min_loss:
        print('Saving best state,epoch', epoch)
        f.write(f'Saving best state,epoch {epoch}\n')
        torch.save(dsfd_net.state_dict(), os.path.join(
            save_folder, 'dsfd.pth'))
        min_loss = tloss

    states = {
        'epoch': epoch,
        'weight': dsfd_net.state_dict(),
    }
    torch.save(states, os.path.join(save_folder, 'dsfd_checkpoint.pth'))
    f.close()


def adjust_learning_rate(optimizer, gamma, step):
    """Sets the learning rate to the initial LR decayed by 10 at every
        specified step
    # Adapted from PyTorch Imagenet example:
    # https://github.com/pytorch/examples/blob/master/imagenet/main.py
    """
    lr = args.lr * (gamma ** (step))
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


if __name__ == '__main__':
#     torch.autograd.set_detect_anomaly(True)
    train()
