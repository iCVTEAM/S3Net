import torch
import torch.nn.functional as F
import torch.nn as nn
import math
from .non_local import NLBlockND

        #####============SPP===============
        #out_pool_size = []
        #for i  in range(3):
        #    a = torch.tensor(6/(i+1))
        #    out_pool_size.append(a)
       
        
        #num_sample_test = ftest.size(0)
        #num_sample_train = ftrain.size(0)
        #previous_conv_size = torch.tensor([ftrain.size(3), ftrain.size(3)])
    
        #ftrain_spp = spatial_pyramid_pool(ftrain, num_sample_train, previous_conv_size ,out_pool_size)
        #ftest_spp = spatial_pyramid_pool(ftest, num_sample_test, previous_conv_size ,out_pool_size)

def spatial_pyramid_pool(previous_conv, num_sample, previous_conv_size, out_pool_size):
    '''
    previous_conv: a tensor vector of previous convolution layer
    num_sample: an int number of image in the batch
    previous_conv_size: an int vector [height, width] of the matrix features size of previous convolution layer
    out_pool_size: a int vector of expected output size of max pooling layer
    
    returns: a tensor vector with shape [1 x n] is the concentration of multi-level pooling
    '''    
    avg = nn.Sequential(nn.AdaptiveAvgPool2d((1, 1))) 
    # print(previous_conv.size())
    #non_local = NLBlockND(in_channels=64, dimension = 2)
    s=torch.tensor(0).cuda()
    for i in range(len(out_pool_size)):
        # print(previous_conv_size)
        #print(previous_conv_size[0].dtype, out_pool_size[i].dtype)
        h_wid = int(math.ceil(previous_conv_size[0].float() / out_pool_size[i]))
        w_wid = int(math.ceil(previous_conv_size[1].float() / out_pool_size[i]))
        h_pad = (h_wid*out_pool_size[i].float()  - previous_conv_size[0].float() + 1)//20
        w_pad = (w_wid*out_pool_size[i].float()  - previous_conv_size[1].float() + 1)//20
        avgpool = nn.AvgPool2d((h_wid, w_wid), stride=(h_wid, w_wid), padding=[int(h_pad), int(w_pad)])
        #print(previous_conv.shape)
        #print(previous_conv.is_cuda)
        x = avgpool(previous_conv)
        #print(i,x.shape)
        #x=non_local(x)
        #x = avg(x)
        #x = x.permute(0, 3, 2, 1)
        #x = x.reshape(x.size(0), x.size(2)*x.size(1), x.size(3))
        x = F.interpolate(x, size=[21,21], mode='bilinear')
        #x = x.sum(1)
        #x = x.squeeze()
        s = s+x 
        #if(i == 0):
        #    spp = x.view(num_sample,-1)
            # print("spp size:",spp.size())
        #else:
            # print("size:",spp.size())


    #print(s.shape)
        # spp = torch.cat((spp,x.view(num_sample,-1)), 1)
    return s
