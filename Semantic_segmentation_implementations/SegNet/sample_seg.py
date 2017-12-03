# %run 'segnet_model.ipynb'
# %run 'data_loader.ipynb'

import torch 
import numpy as np 
from torch.autograd import Variable
import torch.nn as nn
from torchvision import datasets,models,transforms
import torch.optim as optim
from segnet_model import network
#from segnet_model import SegNet
from data_loader import data_loader_seg
import matplotlib as plt
from PIL import Image

model_ft = network()
    
if torch.cuda.is_available():
    model_ft = model_ft.cuda()

#APPLY TRANSFORM IF NEEDED
trans = transforms.Compose([transforms.ToTensor()])

dsets_train = data_loader_seg('images/training/',trans=trans)
dsets_enqueuer_training = torch.utils.data.DataLoader(dsets_train, batch_size=1, num_workers=0, drop_last=False)

dsets_test = data_loader_seg('images/test/',trans=trans)
dsets_enqueuer_test = torch.utils.data.DataLoader(dsets_test, batch_size=1, num_workers=0, drop_last=False)

criterion = nn.BCEWithLogitsLoss()
optimizer = optim.Adam(model_ft.parameters(),lr = 0.001, betas=(0.9, 0.999), eps=1e-08)

if torch.cuda.is_available():
    criterion = criterion.cuda()

loss_data = 0.0


print("\n\n\n......Training......\n\n\n")


for Epoch in range(10):

    for idx,data in enumerate(dsets_enqueuer_training,1):
        print("\nTrain image number = ", idx, ".....\n")
        image,image_seg = data['image'], data['image_seg']
        #print("imageSize = ", image.size())
        if torch.cuda.is_available():
            image, image_seg = Variable(image.cuda(), requires_grad = False), Variable(image_seg.cuda(), requires_grad = False)
        else:
            image, image_seg = Variable(image, requires_grad = False), Variable(image_seg, requires_grad = False)

        output = model_ft(image)
        #print("\nImage", idx, "forward prop complete",end="\n")
        #print("fd_prop_output size = ", output.size(),"\n")
        #print(output)
        #break
        loss = criterion(output,image_seg)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        loss_data += loss.data
        #print ("Epoch {0} /10, loss = {1}".format(Epoch,loss_data))
        with open('model_save.pth.tar', 'wb') as f: 
            torch.save(model_ft, f)
        


# print("\n\n\n......Testing......\n\n\n")

# for idx,data in enumerate(dsets_enqueuer_test,1):
#         print("\nTest image number = ", idx, end=".....\n")
#         image,image_seg = data['image'], data['image_seg']
#         print("imageSize = ", image.size())
#         if torch.cuda.is_available():
#             image, image_seg = Variable(image.cuda(), requires_grad = False), Variable(image_seg.cuda(), requires_grad = False)
#         else:
#             image, image_seg = Variable(image, requires_grad = False), Variable(image_seg, requires_grad = False)

#         output = model_ft(image)



#         print("\nImage", idx, "forward prop complete",end="\n")
#         print("fd_prop_output size = ", output.size(),"\n")

#         print(type(output))
#         break


        # image_np = np.array(image)
        # image_seg_np = np.array(image_seg)

        # print("1")

        # plt.imshow(image_np)
        # plt.imshow(image_seg_np)

        # print("2")





