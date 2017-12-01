import torch.nn as nn
import torch
import os
import numpy as np
import torchvision.utils as vutils
import math
import torch.nn.functional as Funct

class down_block(nn.Module):
    #using the input channels I specify the channels at for repeated use of this block
    def __init__(self, channels, num_of_convs = 2):
        super(down_block,self).__init__()

        self.num_of_convs = num_of_convs

        if(num_of_convs == 2):
            self.conv1 = nn.Conv2d(channels[0], channels[1], kernel_size=(3,3),stride=1,padding=1,dilation=1,bias=True)
            self.conv2 = nn.Conv2d(channels[1], channels[1], kernel_size=(3,3),stride=1,padding=1,dilation=1,bias=True)


        elif(num_of_convs == 3):
            self.conv1 = nn.Conv2d(channels[0], channels[1], kernel_size=(3,3),stride=1,padding=1,dilation=1,bias=True)
            self.conv2 = nn.Conv2d(channels[1], channels[1], kernel_size=(3,3),stride=1,padding=1,dilation=1,bias=True)
            self.conv3 = nn.Conv2d(channels[1], channels[1], kernel_size=(3,3),stride=1,padding=1,dilation=1,bias=True)

        self.batchnorm = nn.BatchNorm2d(channels[1])

        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=(2,2), stride=2, return_indices = True)

        
        # Initialize Kernel weights with a normal distribution of mean = 0 , stdev = sqrt(2. / n)
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
                
    #forward function through the block
    def forward(self, x):
        input_size = x.size()
        if(self.num_of_convs == 2):
            fwd_map = self.conv1(x)
            fwd_map = self.batchnorm(fwd_map)
            self.relu(fwd_map)

            fwd_map = self.conv2(fwd_map)
            fwd_map = self.batchnorm(fwd_map)
            self.relu(fwd_map)

        elif(self.num_of_convs == 3):
            fwd_map = self.conv1(x)
            fwd_map = self.batchnorm(fwd_map)
            self.relu(fwd_map)

            fwd_map = self.conv2(fwd_map)
            fwd_map = self.batchnorm(fwd_map)
            self.relu(fwd_map)

            fwd_map = self.conv3(fwd_map)
            fwd_map = self.batchnorm(fwd_map)
            self.relu(fwd_map)

        #Saving the tensor and for unpooling tensor size & indeces to map it to the layers deeper in the model
        output_size = fwd_map.size()
        fwd_map, indices = self.maxpool(fwd_map)
        
        size = {"input_size": input_size, "b4max": output_size}
        #size = fwd_map.size()
        #print("down block output_size: ", size)
        
        return (fwd_map, indices, size)
    



class up_block(nn.Module):

    def __init__(self,channels,num_of_convs = 2):
        super(up_block,self).__init__()
        
        self.num_of_convs = num_of_convs
        
        #Upsampling
        self.unpooled = nn.MaxUnpool2d(kernel_size=(2,2) , stride=2)
        #self.unconv = nn.Conv2d(channels[0], channels[1], kernel_size=(8,8), stride=1, padding=1, dilation=1, bias=True)
            
        if(num_of_convs== 2):
            self.conv1 = nn.Conv2d(channels[0], channels[0], kernel_size=(3,3), stride=1, padding=1, dilation=1, bias=True)
            self.conv2 = nn.Conv2d(channels[0], channels[1], kernel_size=(3,3), stride=1,padding=1, dilation=1, bias=True)

        elif(num_of_convs == 3):
            self.conv1 = nn.Conv2d(channels[0], channels[0], kernel_size=(3,3), stride=1, padding=1, dilation=1, bias=True)
            self.conv2 = nn.Conv2d(channels[0], channels[0], kernel_size=(3,3), stride=1, padding=1, dilation=1, bias=True)
            self.conv3 = nn.Conv2d(channels[0], channels[1], kernel_size=(3,3), stride=1, padding=1, dilation=1, bias=True)
        
        self.batchnorm = nn.BatchNorm2d(channels[0])
        self.batchnorm_for_last_conv = nn.BatchNorm2d(channels[1])
        
        self.relu = nn.ReLU(inplace=True)
        
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))

                
    #forward function through the block
    def forward(self, x, indices, size):

        #print("Before upsampling: ", x.size())
        fwd_map = self.unpooled(x, indices, output_size=size)
        #fwd_map = self.unpooled(x, indices)
        #print("After Unpooling: ", fwd_map.size())
        #fwd_map = self.unconv(fwd_map)

    
        if(self.num_of_convs == 2):
            fwd_map = self.conv1(fwd_map)
            fwd_map = self.batchnorm(fwd_map)
            self.relu(fwd_map)

            fwd_map = self.conv2(fwd_map)
            fwd_map = self.batchnorm_for_last_conv(fwd_map)
            self.relu(fwd_map)

        elif(self.num_of_convs == 3):
            fwd_map = self.conv1(fwd_map)
            fwd_map = self.batchnorm(fwd_map)
            self.relu(fwd_map)

            fwd_map = self.conv2(fwd_map)
            fwd_map = self.batchnorm(fwd_map)
            self.relu(fwd_map)

            fwd_map = self.conv3(fwd_map)
            fwd_map = self.batchnorm_for_last_conv(fwd_map)
            self.relu(fwd_map)

        #print("down block after convs: ", fwd_map.size())
        
        return fwd_map

class network(nn.Module):

    def __init__(self):
        super(network,self).__init__()
        self.layer1 = down_block((3,64), 2)
        self.layer2 = down_block((64,128), 2)
        self.layer3 = down_block((128,256), 3)
        self.layer4 = down_block((256,512), 3)
        self.layer5 = down_block((512,512), 3)
        
        #self.layer6 = up_block((inp,curr,next), 3)
        self.layer6 = up_block((512,512), 3)
        self.layer7 = up_block((512,256), 3)
        self.layer8 = up_block((256,128), 3)
        self.layer9 = up_block((128,64), 2)
        self.layer10 = up_block((64,1), 2)
        
        self.softmax = nn.Softmax()

    def forward(self,x):

        #print("\nLayer1...")
        out1, indices1, size1= self.layer1(x)
        #print("in forward ", Funct.softmax(out1).size())
        #print("\nLayer2...")
        out2, indices2, size2 = self.layer2(out1)
        #print("\nLayer3...")
        out3, indices3, size3= self.layer3(out2)
        #print("\nLayer4...")
        out4, indices4,size4 = self.layer4(out3)
        #print("\nLayer5...")
        out5, indices5, size5 = self.layer5(out4)

        #print("\nLayer6...")
        out6 = self.layer6(out5, indices5, size5['b4max'])
        #print("\nLayer7...")
        out7 = self.layer7(out6, indices4, size4['b4max'])
        #print("\nLayer8...")
        out8 = self.layer8(out7, indices3, size3['b4max'])
        #print("\nLayer9...")
        out9 = self.layer9(out8, indices2, size2['b4max'])
        #print("\nLayer10...")
        out10 = self.layer10(out9, indices1, size1['b4max'])
        
        #print(out10)
        #print("size of out10:", out10.size())
        #print("\nSoftmax Layer...")
        res = Funct.softmax(out10)

        return res



# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -

# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -

# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -


# import torch
# import torch.nn as nn
# import torch.nn.functional as F
# from collections import OrderedDict

# class SegNet(nn.Module):
#     def __init__(self,input_nbr=3,label_nbr=10):
#         super(SegNet, self).__init__()

#         batchNorm_momentum = 0.1

#         self.conv11 = nn.Conv2d(input_nbr, 64, kernel_size=3, padding=1)
#         self.bn11 = nn.BatchNorm2d(64, momentum= batchNorm_momentum)
#         self.conv12 = nn.Conv2d(64, 64, kernel_size=3, padding=1)
#         self.bn12 = nn.BatchNorm2d(64, momentum= batchNorm_momentum)

#         self.conv21 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
#         self.bn21 = nn.BatchNorm2d(128, momentum= batchNorm_momentum)
#         self.conv22 = nn.Conv2d(128, 128, kernel_size=3, padding=1)
#         self.bn22 = nn.BatchNorm2d(128, momentum= batchNorm_momentum)

#         self.conv31 = nn.Conv2d(128, 256, kernel_size=3, padding=1)
#         self.bn31 = nn.BatchNorm2d(256, momentum= batchNorm_momentum)
#         self.conv32 = nn.Conv2d(256, 256, kernel_size=3, padding=1)
#         self.bn32 = nn.BatchNorm2d(256, momentum= batchNorm_momentum)
#         self.conv33 = nn.Conv2d(256, 256, kernel_size=3, padding=1)
#         self.bn33 = nn.BatchNorm2d(256, momentum= batchNorm_momentum)

#         self.conv41 = nn.Conv2d(256, 512, kernel_size=3, padding=1)
#         self.bn41 = nn.BatchNorm2d(512, momentum= batchNorm_momentum)
#         self.conv42 = nn.Conv2d(512, 512, kernel_size=3, padding=1)
#         self.bn42 = nn.BatchNorm2d(512, momentum= batchNorm_momentum)
#         self.conv43 = nn.Conv2d(512, 512, kernel_size=3, padding=1)
#         self.bn43 = nn.BatchNorm2d(512, momentum= batchNorm_momentum)

#         self.conv51 = nn.Conv2d(512, 512, kernel_size=3, padding=1)
#         self.bn51 = nn.BatchNorm2d(512, momentum= batchNorm_momentum)
#         self.conv52 = nn.Conv2d(512, 512, kernel_size=3, padding=1)
#         self.bn52 = nn.BatchNorm2d(512, momentum= batchNorm_momentum)
#         self.conv53 = nn.Conv2d(512, 512, kernel_size=3, padding=1)
#         self.bn53 = nn.BatchNorm2d(512, momentum= batchNorm_momentum)

#         self.conv53d = nn.Conv2d(512, 512, kernel_size=3, padding=1)
#         self.bn53d = nn.BatchNorm2d(512, momentum= batchNorm_momentum)
#         self.conv52d = nn.Conv2d(512, 512, kernel_size=3, padding=1)
#         self.bn52d = nn.BatchNorm2d(512, momentum= batchNorm_momentum)
#         self.conv51d = nn.Conv2d(512, 512, kernel_size=3, padding=1)
#         self.bn51d = nn.BatchNorm2d(512, momentum= batchNorm_momentum)

#         self.conv43d = nn.Conv2d(512, 512, kernel_size=3, padding=1)
#         self.bn43d = nn.BatchNorm2d(512, momentum= batchNorm_momentum)
#         self.conv42d = nn.Conv2d(512, 512, kernel_size=3, padding=1)
#         self.bn42d = nn.BatchNorm2d(512, momentum= batchNorm_momentum)
#         self.conv41d = nn.Conv2d(512, 256, kernel_size=3, padding=1)
#         self.bn41d = nn.BatchNorm2d(256, momentum= batchNorm_momentum)

#         self.conv33d = nn.Conv2d(256, 256, kernel_size=3, padding=1)
#         self.bn33d = nn.BatchNorm2d(256, momentum= batchNorm_momentum)
#         self.conv32d = nn.Conv2d(256, 256, kernel_size=3, padding=1)
#         self.bn32d = nn.BatchNorm2d(256, momentum= batchNorm_momentum)
#         self.conv31d = nn.Conv2d(256,  128, kernel_size=3, padding=1)
#         self.bn31d = nn.BatchNorm2d(128, momentum= batchNorm_momentum)

#         self.conv22d = nn.Conv2d(128, 128, kernel_size=3, padding=1)
#         self.bn22d = nn.BatchNorm2d(128, momentum= batchNorm_momentum)
#         self.conv21d = nn.Conv2d(128, 64, kernel_size=3, padding=1)
#         self.bn21d = nn.BatchNorm2d(64, momentum= batchNorm_momentum)

#         self.conv12d = nn.Conv2d(64, 64, kernel_size=3, padding=1)
#         self.bn12d = nn.BatchNorm2d(64, momentum= batchNorm_momentum)
#         self.conv11d = nn.Conv2d(64, label_nbr, kernel_size=3, padding=1)


#     def forward(self, x):

#         # Stage 1
#         x11 = F.relu(self.bn11(self.conv11(x)))
#         x12 = F.relu(self.bn12(self.conv12(x11)))
#         x1p, id1 = F.max_pool2d(x12,kernel_size=2, stride=2,return_indices=True)

#         # Stage 2
#         x21 = F.relu(self.bn21(self.conv21(x1p)))
#         x22 = F.relu(self.bn22(self.conv22(x21)))
#         x2p, id2 = F.max_pool2d(x22,kernel_size=2, stride=2,return_indices=True)

#         # Stage 3
#         x31 = F.relu(self.bn31(self.conv31(x2p)))
#         x32 = F.relu(self.bn32(self.conv32(x31)))
#         x33 = F.relu(self.bn33(self.conv33(x32)))
#         x3p, id3 = F.max_pool2d(x33,kernel_size=2, stride=2,return_indices=True)

#         # Stage 4
#         x41 = F.relu(self.bn41(self.conv41(x3p)))
#         x42 = F.relu(self.bn42(self.conv42(x41)))
#         x43 = F.relu(self.bn43(self.conv43(x42)))
#         x4p, id4 = F.max_pool2d(x43,kernel_size=2, stride=2,return_indices=True)

#         # Stage 5
#         x51 = F.relu(self.bn51(self.conv51(x4p)))
#         x52 = F.relu(self.bn52(self.conv52(x51)))
#         x53 = F.relu(self.bn53(self.conv53(x52)))
#         x5p, id5 = F.max_pool2d(x53,kernel_size=2, stride=2,return_indices=True)


#         # Stage 5d = Stage 6
#         x5d = F.max_unpool2d(x5p, id5, kernel_size=2, stride=2)
#         x53d = F.relu(self.bn53d(self.conv53d(x5d)))
#         x52d = F.relu(self.bn52d(self.conv52d(x53d)))
#         x51d = F.relu(self.bn51d(self.conv51d(x52d)))

#         # Stage 4d = Stage 7
#         x4d = F.max_unpool2d(x51d, id4, kernel_size=2, stride=2)
#         x43d = F.relu(self.bn43d(self.conv43d(x4d)))
#         x42d = F.relu(self.bn42d(self.conv42d(x43d)))
#         x41d = F.relu(self.bn41d(self.conv41d(x42d)))

#         # Stage 3d = Stage 8
#         x3d = F.max_unpool2d(x41d, id3, kernel_size=2, stride=2)
#         x33d = F.relu(self.bn33d(self.conv33d(x3d)))
#         x32d = F.relu(self.bn32d(self.conv32d(x33d)))
#         x31d = F.relu(self.bn31d(self.conv31d(x32d)))

#         # Stage 2d = Stage 9
#         x2d = F.max_unpool2d(x31d, id2, kernel_size=2, stride=2)
#         x22d = F.relu(self.bn22d(self.conv22d(x2d)))
#         x21d = F.relu(self.bn21d(self.conv21d(x22d)))

#         # Stage 1d = Stage 10
#         x1d = F.max_unpool2d(x21d, id1, kernel_size=2, stride=2)
#         x12d = F.relu(self.bn12d(self.conv12d(x1d)))
#         x11d = self.conv11d(x12d)

#         return x11d

#     def load_from_segnet(self, model_path):
#         s_dict = self.state_dict()# create a copy of the state dict
#         th = torch.load(model_path).state_dict() # load the weigths
#         # for name in th:
#             # s_dict[corresp_name[name]] = th[name]
#         self.load_state_dict(th)
