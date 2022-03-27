#!/usr/bin/python
# -*- encoding: utf-8 -*-

from unittest import result
from logger import setup_logger
from model import BiSeNet
import torch
import os
import os.path as osp
import numpy as np
from PIL import Image
import torchvision.transforms as transforms
import cv2

'''
(0, 'background') 
(1, 'skin') 
(2, 'l_brow') 
(3, 'r_brow') 
(4, 'l_eye') 
(5, 'r_eye') 
(6, 'eye_g (eye glasses)') 
(7, 'l_ear') 
(8, 'r_ear') 
(9, 'ear_r (ear ring)') 
(10, 'nose') 
(11, 'mouth') 
(12, 'u_lip') 
(13, 'l_lip') 
(14, 'neck') 
(15, 'neck_l (necklace)') 
(16, 'cloth') 
(17, 'hair') 
(18, 'hat')
'''

def get_face(img):
    n_classes = 19
    net = BiSeNet(n_classes=n_classes)
    net.cuda()
    cp='79999_iter.pth'
    save_pth = osp.join('res/cp', cp)
    net.load_state_dict(torch.load(save_pth))
    net.eval()

    to_tensor = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
    ])
    with torch.no_grad():
        image = img
        img = to_tensor(image)
        img = torch.unsqueeze(img, 0)
        img = img.cuda()
        out = net(img)[0]
        parsing = out.squeeze(0).cpu().numpy().argmax(0)
        # print(parsing.shape)
        # print(parsing)
        lst1 = np.unique(parsing)
        # print("Output list : ", lst1)

        # vis_parsing_maps(image, parsing, stride=1, save_im=True, save_path=osp.join(respth, image_path))

        masks =[]
        for n in range (19):
            mask = np.zeros((parsing.shape[0], parsing.shape[1]), dtype=np.uint8).tolist()
            for i in range(parsing.shape[0]):
                for j in range(parsing.shape[1]):
                    if (parsing[i][j] == n):
                        mask[i][j] = 255
            masks.append(mask)
        masks = np.array(masks)
        mask1 = masks[1].astype(np.uint8) #skin
        mask2 = masks[2].astype(np.uint8) #lbrow
        mask3 = masks[3].astype(np.uint8) #rbrow
        mask4 = masks[4].astype(np.uint8) #l_eye
        mask5 = masks[5].astype(np.uint8) #r_eye
        mask10 = masks[10].astype(np.uint8) #nose
        mask11 = masks[11].astype(np.uint8) #mouth
        mask12 = masks[12].astype(np.uint8) #u_lip
        mask13 = masks[13].astype(np.uint8) #l_lip
        mask = np.zeros_like(image)
        mask[mask1 == 255] = 255
        mask[mask2 == 255] = 255
        mask[mask3 == 255] = 255
        mask[mask4 == 255] = 255
        mask[mask5 == 255] = 255
        mask[mask10 == 255] = 255
        mask[mask11 == 255] = 255
        mask[mask12 == 255] = 255
        mask[mask13 == 255] = 255

        image_to_mask = np.array(image)
        result1 = image_to_mask.copy()
        result1[mask == 0] = 0
        result1[mask != 0] = image_to_mask[mask != 0]
        
        return result1

def get_lips(img):
    n_classes = 19
    net = BiSeNet(n_classes=n_classes)
    net.cuda()
    cp='79999_iter.pth'
    save_pth = osp.join('res/cp', cp)
    net.load_state_dict(torch.load(save_pth))
    net.eval()

    to_tensor = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
    ])
    with torch.no_grad():
        image = img
        img = to_tensor(image)
        img = torch.unsqueeze(img, 0)
        img = img.cuda()
        out = net(img)[0]
        parsing = out.squeeze(0).cpu().numpy().argmax(0)
        # print(parsing.shape)
        # print(parsing)
        lst1 = np.unique(parsing)
        # print("Output list : ", lst1)

        # vis_parsing_maps(image, parsing, stride=1, save_im=True, save_path=osp.join(respth, image_path))

        masks =[]
        for n in range (19):
            mask = np.zeros((parsing.shape[0], parsing.shape[1]), dtype=np.uint8).tolist()
            for i in range(parsing.shape[0]):
                for j in range(parsing.shape[1]):
                    if (parsing[i][j] == n):
                        mask[i][j] = 255
            masks.append(mask)
        masks = np.array(masks)
        mask12 = masks[12].astype(np.uint8) #u_lip
        mask13 = masks[13].astype(np.uint8) #l_lip
        mask = np.zeros_like(image)
        mask[mask12 == 255] = 255
        mask[mask13 == 255] = 255

        image_to_mask = np.array(image)
        result1 = image_to_mask.copy()
        result1[mask == 0] = 0
        result1[mask != 0] = image_to_mask[mask != 0]
        
        return result1, mask