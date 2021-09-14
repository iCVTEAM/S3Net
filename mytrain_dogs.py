from __future__ import print_function
from __future__ import division
from torchFewShot.utils.torchtools import one_hot, adjust_learning_rate
from torchFewShot.utils.logger import Logger
from torchFewShot.utils.avgmeter import AverageMeter
from torchFewShot.utils.iotools import save_checkpoint, check_isfile
from torchFewShot.optimizers import init_optimizer
from torchFewShot.losses import CrossEntropyLoss
from torchFewShot.data_manager import DataManager
from torchFewShot.models.net_cos_conv64 import Model
from args_dogs import argument_parser
from center_loss import CenterLoss
import os
import sys
import time
import datetime
import argparse
import os.path as osp
import numpy as np
import random

import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
from torch.utils.data import DataLoader
from torch.optim import lr_scheduler
import torch.nn.functional as F
import Utilizes
sys.path.append('./torchFewShot')

# from args_tiered import argument_parser


parser = argument_parser()
args = parser.parse_args()

def main():
    torch.manual_seed(args.seed)
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu_devices
    use_gpu = torch.cuda.is_available()

    sys.stdout = Logger(osp.join(args.save_dir, 'log_train.txt'))
    print("==========\nArgs:{}\n==========".format(args))

    if use_gpu:
        print("Currently using GPU {}".format(args.gpu_devices))
        cudnn.benchmark = True
        torch.cuda.manual_seed_all(args.seed)
    else:
        print("Currently using CPU (GPU is highly recommended)")

    print('Initializing image data manager')
    dm = DataManager(args, use_gpu)
    trainloader, testloader = dm.return_dataloaders()

    model = Model(temperature=args.temperature, num_classes=args.num_classes)
    criterion = CrossEntropyLoss() 
    #criterion_cent = CenterLoss(num_classes=10240,feat_dim=36,use_gpu=use_gpu)
    criterion_cent = CenterLoss(num_classes=4320,feat_dim=64,use_gpu=use_gpu)
    optimizer = init_optimizer(args.optim, model.parameters(), args.lr, args.weight_decay)
    optimizer_centloss  = init_optimizer('cent', criterion_cent.parameters(), 0.5, args.weight_decay)
	
########=================loading best model================
    #checkpoint = torch.load(args.resume)
    #print(args.resume)
    #parameters =  model.parameters
    #model.load_state_dict(checkpoint)
    #print(parameters)
    #model.load_state_dict(checkpoint['state_dict'], False)
    #print("Loaded checkpoint from '{}'".format(args.resume))
    

    if use_gpu:
        model = model.cuda()

    start_time = time.time()
    train_time = 0
    best_acc = -np.inf
    best_epoch = 0
    print("==> Start training")

    for epoch in range(args.max_epoch):
        learning_rate = adjust_learning_rate(optimizer, epoch, args.LUT_lr)

        start_train_time = time.time()
        train(epoch, model,criterion,criterion_cent,optimizer,optimizer_centloss,trainloader, learning_rate, use_gpu)
        train_time += round(time.time() - start_train_time)

        if epoch == 0 or epoch > (args.stepsize[0]-1) or (epoch + 1) % 10 == 0:
        #if epoch == 0 or epoch > 0: 
            acc = test(model, testloader, use_gpu)
            is_best = acc > best_acc

            if is_best:
                best_acc = acc
                best_epoch = epoch + 1
            save_checkpoint({
                'state_dict': model.state_dict(), 
                'acc': acc,
                'epoch': epoch,
            }, is_best, osp.join(args.save_dir, 'checkpoint_ep' + str(epoch + 1) + '.pth.tar'))
            print("==> Test 5-way Best accuracy {:.2%}, achieved at epoch {}".format(best_acc, best_epoch))


def train(epoch, model, criterion,criterion_cent,optimizer,optimizer_centloss, trainloader, learning_rate, use_gpu):
    batch_time = AverageMeter()
    losses = AverageMeter()
    softmax_loss = AverageMeter()
    data_time = AverageMeter()
    cent_losses = AverageMeter()
    few_losses  = AverageMeter()
    group_losses = AverageMeter()
    loss_mse = nn.MSELoss(reduction=False) 
    model.train()

    end = time.time()
    for batch_idx, (images_train, labels_train, images_test, labels_test, pids, pids_train) in enumerate(trainloader):
        data_time.update(time.time() - end)

        if use_gpu:
            images_train, labels_train = images_train.cuda(), labels_train.cuda()
            images_test, labels_test = images_test.cuda(), labels_test.cuda()
            pids = pids.cuda()
            pids_train = pids_train.cuda()

        batch_size, num_train_examples, channels, height, width = images_train.size()
        num_test_examples = images_test.size(1)

        labels_train_1hot = one_hot(labels_train).cuda()
        labels_test_1hot = one_hot(labels_test).cuda()
        # ytest, cls_scores = model(images_train, images_test, labels_train_1hot, labels_test_1hot)
        ytest, ytest_avg, score = model(images_train, images_test, labels_train_1hot, labels_test_1hot)    #[120*36, 512]  [120, 64, 6,6]
        #print(ytest.shape, pids.shape, pids_train.shape ,  ytrain.shape)
        #labels_feat = ((pids.view(-1)).unsqueeze(1)).repeat(1, ytest.size(2))
        #labels_feat = labels_feat.reshape(labels_feat.size(0)*labels_feat.size(1))
        ########=====SVD=====##########
        #u, s, v = torch.svd(feat.t())
        #l1 = s.size(0)
        #BSS=0
        #for i in range(100):
        #    BSS = BSS + torch.pow(s[l1-1-i],2)
        #loss_bss = BSS



        #print(feat.shape, labels_feat.shape) 
        loss1 = criterion(ytest, pids.view(-1))
        #loss2 = criterion(feat_avg, pids.view(-1))
        #loss_cent = criterion_cent(feat,labels_feat)
        loss_few = criterion(score, labels_test.view(-1) )
        cent_weight = 0.5
        #loss =  loss1 + cent_weight*loss_cent + 0*0.01*loss_few
        #print(loss1, loss_cent, loss_few)
        #loss = loss1 + 0.01*loss_few
        #optimizer_centloss.zero_grad()
       	#temp_loss_ = loss_reg*0
       	#loss_reg = loss_mse(loss_reg, temp_loss_)*1e-4 
        loss = loss1 + cent_weight*loss_few	
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        #for param in criterion_cent.parameters():
        #    param.grad.data *= (1. / cent_weight)
        #for param in criterion_cent.parameters():
        #    param.grad.data* =(1. /0.5)
        #optimizer_centloss.step()

        losses.update(loss.item(), pids.size(0))
        #cent_losses.update(loss_cent.item(), labels_feat.size(0))
        softmax_loss.update(loss1.item(), pids.size(0))
        few_losses.update(loss_few.item(), pids.size(0))
        #group_losses.update(loss_reg.item(), pids.size(0))
        batch_time.update(time.time() - end)
        end = time.time()

        #print('loss1:',loss1, 'loss_cent:', loss_cent, 'loss_cent2:', loss_cent2)
    print('Epoch{0} '
          'lr: {1} '
          'Time:{batch_time.sum:.1f}s '
          'Data:{data_time.sum:.1f}s '
          'Loss_a:{loss.avg:.4f} '
          'Loss_s:{loss_softmax.avg:.4f} '
          'Loss_c:{loss_group.avg:.4f} '
	  'Loss_f:{loss_few.avg:.4f}' .format(
              epoch+1, learning_rate, batch_time=batch_time,
              data_time=data_time, loss=losses, loss_softmax=softmax_loss, loss_group=few_losses, loss_few = few_losses), 'weigh_cent:',cent_weight )


def test(model, testloader, use_gpu):
    accs = AverageMeter()
    test_accuracies = []
    model.eval()

    with torch.no_grad():
        for batch_idx, (images_train, labels_train, images_test, labels_test) in enumerate(testloader):
            if use_gpu:
                images_train = images_train.cuda()
                images_test = images_test.cuda()

            end = time.time()

            batch_size, num_train_examples, channels, height, width = images_train.size()
            num_test_examples = images_test.size(1)

            labels_train_1hot = one_hot(labels_train).cuda()
            labels_test_1hot = one_hot(labels_test).cuda()
            cls_scores = model(images_train, images_test,
                               labels_train_1hot, labels_test_1hot)
            cls_scores = cls_scores.view(batch_size * num_test_examples, -1)
            labels_test = labels_test.view(batch_size * num_test_examples)

            _, preds = torch.max(cls_scores.detach().cpu(), 1)
            acc = (torch.sum(preds == labels_test.detach().cpu()
                             ).float()) / labels_test.size(0)
            accs.update(acc.item(), labels_test.size(0))

            gt = (preds == labels_test.detach().cpu()).float()
            gt = gt.view(batch_size, num_test_examples).numpy()  # [b, n]
            acc = np.sum(gt, 1) / num_test_examples
            acc = np.reshape(acc, (batch_size))
            test_accuracies.append(acc)

    accuracy = accs.avg
    test_accuracies = np.array(test_accuracies)
    test_accuracies = np.reshape(test_accuracies, -1)
    stds = np.std(test_accuracies, 0)
    ci95 = 1.96 * stds / np.sqrt(args.epoch_size)
    print('Accuracy: {:.2%}, std: :{:.2%}'.format(accuracy, ci95))

    return accuracy


if __name__ == '__main__':
    main()
