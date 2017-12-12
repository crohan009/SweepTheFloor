
# %run 'segnet_model.ipynb'
# %run 'data_loader.ipynb'

import torch 
import numpy as np 
from torch.autograd import Variable
import torch.nn as nn
from torchvision import datasets,models,transforms
import torchvision
import torch.optim as optim
from segnet_model import network
#from segnet_model import SegNet
from data_loader import data_loader_seg
from matplotlib import pyplot as plt
from PIL import Image

model_ft = torch.load("/Users/crohan009/Documents/Stuff/CS/Deep_Learning/NYUAutonomous/Semantic_segmentation_implementations/SegNet/saved_model_weights/model_save.pth.tar")
    
if torch.cuda.is_available():
    model_ft = model_ft.cuda()

#APPLY TRANSFORM IF NEEDED
trans = transforms.Compose([transforms.ToTensor()])

dsets_test = data_loader_seg('images/test/',trans=trans)
dsets_enqueuer_test = torch.utils.data.DataLoader(dsets_test, batch_size=1, num_workers=0, drop_last=False)


print("\n\n\n......Testing......\n")

for idx,data in enumerate(dsets_enqueuer_test,1):
        print("\nTest image number = ", idx, end=".....\n")
        image,image_seg = data['image'], data['image_seg']
        print("imageSize = ", image.size())
        if torch.cuda.is_available():
            image, image_seg = Variable(image.cuda(), requires_grad = False), Variable(image_seg.cuda(), requires_grad = False)
        else:
            image, image_seg = Variable(image, requires_grad = False), Variable(image_seg, requires_grad = False)

        output = model_ft(image)



        print("\nImage", idx, "forward prop complete",end="\n")
        print("fd_prop_output size = ", output.size(),"\n")
        print(output)
        image_segmented = output.data.numpy()[0,0,:,:] 
        print(np.max(image_segmented))
        print(np.shape(image_segmented))
        print(image_segmented)
        plt.imshow(image_segmented, 'gray')

        # plt.show()
        # to_pil = torchvision.transforms.ToPILImage()
        # img = to_pil(output)
        # print(type(img))
        break


        # image_np = np.array(image)
        # image_seg_np = np.array(image_seg)

        # print("1")

        # plt.imshow(image_np)
        # plt.imshow(image_seg_np)

        # print("2")





