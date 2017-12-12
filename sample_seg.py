# %run 'segnet_model.ipynb'
# %run 'data_loader.ipynb'

from segnet_model import network
from data_loader import data_loader_seg

import torch 
import numpy as np 
from torch.autograd import Variable
import torch.nn as nn
from torchvision import datasets,models,transforms
import torch.optim as optim
from PIL import Image
import pickle

model_ft = network()
    
if torch.cuda.is_available():
    model_ft = model_ft.cuda()

#APPLY TRANSFORM IF NEEDED
trans = transforms.Compose([ 
	transforms.CenterCrop((1200, 350)), 
	transforms.ToTensor(),
])

dsets_train = data_loader_seg('images/training/',trans=trans)
dsets_enqueuer_training = torch.utils.data.DataLoader(dsets_train, batch_size=100, num_workers=0, drop_last=False)

dsets_test = data_loader_seg('images/test/',trans=trans)
dsets_enqueuer_test = torch.utils.data.DataLoader(dsets_test, batch_size=100, num_workers=0, drop_last=False)

criterion = nn.BCEWithLogitsLoss()
optimizer = optim.Adam(model_ft.parameters(),lr = 0.001, betas=(0.9, 0.999), eps=1e-08)

if torch.cuda.is_available():
    criterion = criterion.cuda()

loss_data = 0.0
loss_data_testing = 0.0

loss_per_epoch_lst = []

print("\n\n\n......Training......")
loss_lst_train = []
loss_lst_test = []


for Epoch in range(100):
    
    for idx,data in enumerate(dsets_enqueuer_training,1):
        image,image_seg = data['image'], data['image_seg']
        #print("imageSize = ", image.size())
        if torch.cuda.is_available():
            image, image_seg = Variable(image.cuda(), requires_grad = False), Variable(image_seg.cuda(), requires_grad = False)
        else:
            image, image_seg = Variable(image, requires_grad = False), Variable(image_seg, requires_grad = False)

        model_ft.train(True)
        output = model_ft(image)
        optimizer.zero_grad()
        loss = criterion(output,image_seg)
        loss.backward()
        optimizer.step()

        loss_data += loss.data
          
        print ("Epoch {0} /10, loss = {1}".format(Epoch,loss_data))
    loss_lst_train.append(loss_data.cpu().numpy()/idx)
    

    for idx,data in enumerate(dsets_enqueuer_test,1):
        image,image_seg = data['image'], data['image_seg']
        #print("imageSize = ", image.size())
        if torch.cuda.is_available():
            image, image_seg = Variable(image.cuda(), volatile=True), Variable(image_seg.cuda(), volatile=True)
        else:
            image, image_seg = Variable(image, volatile=True), Variable(image_seg, volatile=True)

        model_ft.eval()
        output = model_ft(image)
        loss = criterion(output,image_seg)

        loss_data_testing += loss.data

    loss_lst_test.append(loss_data_testing.cpu().numpy()/idx)
    
    with open('saved_model_weights/model_save{0}.pth.tar'.format(Epoch), 'wb') as fModel: 
        torch.save(model_ft, fModel)   
    
    pickle.dump( (loss_lst_train, loss_lst_test), open( "saveSegnetValues.p", "wb" ) )
    


