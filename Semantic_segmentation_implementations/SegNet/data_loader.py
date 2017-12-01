import os
import torch
from torch.utils.data import Dataset 
from torchvision import transforms, utils
from PIL import Image
import random
from torchvision import datasets,models,transforms
import numpy as np

trans = transforms.ToTensor()#<------------WHY NEEDED???????????????????????????????

class data_loader_seg(Dataset):

    def __init__(self,root_dir,trans=None):
        self.root_dir = root_dir
        self.files = [fn for fn in os.listdir(root_dir + 'color') if fn.endswith('.png')]
        self.trans = trans 

    def __len__(self):
        return len(self.files)

    def __getitem__(self,idx):
        image = Image.open(self.root_dir + 'color/' + self.files[idx])
        image_seg = Image.open(self.root_dir + 'label/' + self.files[idx].split('_')[0] + '_lane_' + self.files[idx].split('_')[1])

        image_seg = image_seg.convert('L')
        #image_seg = image_seg.resize((388,388))
        image_seg = np.array(image_seg)
        image_seg[image_seg>=100] = 1
        image_seg[image_seg<100] = 0
        image_seg = Image.fromarray(image_seg.astype('uint8'))

        if self.trans:
            image = self.trans(image)
            image_seg = self.trans(image_seg)



        return {'image': image, 'image_seg': image_seg}
