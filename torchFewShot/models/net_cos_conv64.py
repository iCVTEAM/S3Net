import math
import torch
import torch.nn as nn
import torch.nn.functional as F

from .backbone import distLinear
from .conv4 import define_DN4Net
from .myspp import *
from .non_local import NLBlockND 
eps = torch.finfo().eps
#loss = nn.MSELoss(reduction=False)
class Model(nn.Module):
    def __init__(self, temperature, num_classes=64):
        super(Model, self).__init__()
        self.temperature = temperature
        self.base = define_DN4Net()
        self.nFeat = 64
        self.clasifier = nn.Conv2d(self.nFeat, num_classes, kernel_size=1)
        self.avg = nn.Sequential(nn.AdaptiveAvgPool2d((1, 1)))
        self.fcl = distLinear(64, num_classes)
        self.non_local = NLBlockND(in_channels=64, dimension = 2)


    def test(self, ftrain, ftest, batch_size, K, num_test):
        ftrain = self.avg(ftrain)
        ftrain = ftrain.view(batch_size, K, -1)
        ftrain = ftrain.unsqueeze(1).repeat(1, num_test, 1, 1)
        ftest = self.avg(ftest)
        ftest = ftest.view(batch_size, num_test, -1)
        ftest = ftest.unsqueeze(2).repeat(1, 1, K, 1)
        ftest = F.normalize(ftest, p=2, dim=3, eps=1e-12)
        ftrain = F.normalize(ftrain, p=2, dim=3, eps=1e-12)
        scores = torch.sum(ftest * ftrain, dim=3)  # [4 30 5]
        return scores

    def test_spp(self, ftrain, ftest, batch_size, K, num_test):
	#####============SPP===============
        
        out_pool_size = []
        scores_a  = 0
        ftrain1 = ftrain
        ftest1 = ftest 
        for i in range(4):
            if i==0:
                a=torch.tensor(1)
            else:
                a = torch.tensor(7*i)
            out_pool_size.append(a) 
            num_sample_test = ftest1.size(0)
            num_sample_train = ftrain1.size(0)
            previous_conv_size = torch.tensor([ftrain1.size(3), ftrain1.size(3)])
            ftrain = spatial_pyramid_pool(ftrain1, num_sample_train, previous_conv_size ,out_pool_size)
            ftest= spatial_pyramid_pool(ftest1, num_sample_test, previous_conv_size ,out_pool_size)
        ftrain = ftrain.view(batch_size, K, -1)
        ftrain = ftrain.unsqueeze(1).repeat(1, num_test, 1, 1)
        ftest = ftest.view(batch_size, num_test, -1)
        ftest = ftest.unsqueeze(2).repeat(1, 1, K, 1)
        ftest = F.normalize(ftest, p=2, dim=3, eps=1e-12)
        ftrain = F.normalize(ftrain, p=2, dim=3, eps=1e-12)
        scores = torch.sum(ftest * ftrain, dim=3)  # [4 30 5]
        return scores

    def test_part(self, ftrain, ftest, batch_size, K, num_test, nparts):
        ftrain = self.avg(ftrain)
        ftrain = ftrain.view(batch_size, K, -1)
        ftrain = ftrain.unsqueeze(1).repeat(1, num_test, 1, 1)
        ftest = self.avg(ftest)
        ftest = ftest.view(batch_size, num_test, -1)
        ftest = ftest.unsqueeze(2).repeat(1, 1, K, 1)
        inter_part = ftest.size(3)//nparts  
        ftest = ftest.reshape(ftest.size(0), ftest.size(1), ftest.size(2), nparts, inter_part)
        ftrain = ftrain.reshape(ftrain.size(0), ftrain.size(1), ftrain.size(2), nparts, inter_part)
        ftest = F.normalize(ftest, p=2, dim=4, eps=1e-12)
        ftrain = F.normalize(ftrain, p=2, dim=4, eps=1e-12)
        scores = torch.sum(ftest * ftrain, dim=4)  
        [scores, label] = torch.topk(scores,1 , 3)  
        scores = torch.sum(scores, dim=3)
        return scores

    def calScore2(self, ftrain, ftest, batch_size, K, num_test):
        ftrain = F.normalize(ftrain, dim=1)
        ftrain = self.clasifier(ftrain).view(
            batch_size, K, -1, *ftrain.size()[2:])
        ftrain = F.softmax(ftrain, dim=2)
        ftest = F.normalize(ftest, dim=1)
        ftest = self.clasifier(ftest).view(
            batch_size, num_test, -1, *ftest.size()[2:])
        ftest = F.softmax(ftest, dim=2)
        score = self.test(ftrain, ftest, batch_size, K, num_test)
        return score

    def forward(self, xtrain, xtest, ytrain, ytest):
        batch_size, num_train = xtrain.size(0), xtrain.size(1)
        num_test = xtest.size(1)
        K = ytrain.size(2)
        ytrain = ytrain.transpose(1, 2)

        xtrain = xtrain.view(-1, xtrain.size(2),
                             xtrain.size(3), xtrain.size(4))
        xtest = xtest.view(-1, xtest.size(2), xtest.size(3), xtest.size(4))
        x = torch.cat((xtrain, xtest), 0)
        f = self.base(x)
        f = self.non_local(f)
		
        ftrain = f[:batch_size * num_train]
        ftrain = ftrain.view(batch_size, num_train, -1)
        ftrain = torch.bmm(ytrain, ftrain)
        ftrain = ftrain.div(ytrain.sum(dim=2, keepdim=True).expand_as(ftrain))
        ftrain = ftrain.view(-1, *f.size()[1:])  # [4*5 512 6 6]
        ftest = f[batch_size * num_train:]
        ftest = ftest.view(-1, *f.size()[1:])  # [4*30 512 6 6]
        if not self.training:
            score = self.test_spp(ftrain, ftest, batch_size, K, num_test)
            score = score.view(batch_size * num_test, K)
            return score

        score = self.test_spp(ftrain, ftest, batch_size, K, num_test)
        score = score.view(batch_size * num_test, K)
        

        ytest = self.avg(ftest)
        ytest_avg = ytest.squeeze()
        ytest = self.fcl(ytest_avg) * self.temperature

        return ytest, ytest_avg, score
