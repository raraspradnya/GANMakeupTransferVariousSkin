#!/usr/bin/python
# -*- encoding: utf-8 -*-

from unittest import result
from model import BiSeNet
import torch
import os
import os.path as osp
import numpy as np
from PIL import Image
import torchvision.transforms as transforms
import cv2
import matplotlib.pyplot as plt

'''
(0, 'background') 
(1, 'skin') 
(2, 'l_brow') 
(3, 'r_brow') 
(4, 'l_eye') 
(5, 'r_eye') 
(6, 'eye_g (eye glasses)') -- no
(7, 'l_ear') 
(8, 'r_ear') 
(9, 'ear_r (ear ring)') --
(10, 'nose')
(11, 'mouth') 
(12, 'u_lip') 
(13, 'l_lip') 
(14, 'neck') 
(15, 'neck_l (necklace)') 
(16, 'cloth') 
(17, 'hair') 
(18, 'hat')

0 background, 
1 face,
2 left / 
3 right eyebrow, 
4 left / 
5 right eye, 
6 nose, 
7 upper / 
8 mouth, 
9 lower lip,
10 hair, 
11 left /
12 right ear, 
13 neck,
'''
def vis_parsing_maps(im, parsing_anno, stride, save_im=False, save_path='faceparsing/res/test_res/seg.jpg'):
    # Colors for all 20 parts
    part_colors = [[255, 0, 0], [255, 85, 0], [255, 170, 0],
                   [255, 0, 85], [255, 0, 170],
                   [0, 255, 0], [85, 255, 0], [170, 255, 0],
                   [0, 255, 85], [0, 255, 170],
                   [0, 0, 255], [85, 0, 255], [170, 0, 255],
                   [0, 85, 255], [0, 170, 255],
                   [255, 255, 0], [255, 255, 85], [255, 255, 170],
                   [255, 0, 255], [255, 85, 255], [255, 170, 255],
                   [0, 255, 255], [85, 255, 255], [170, 255, 255]]

    im = np.array(im)
    parsing_anno = np.array(parsing_anno)
    
    im = cv2.resize(im, (256, 256), interpolation = cv2.INTER_AREA)
    parsing_anno= cv2.resize(parsing_anno, (256, 256), interpolation = cv2.INTER_AREA)
    print(im.shape, parsing_anno.shape)

    vis_im = im.copy().astype(np.uint8)
    vis_parsing_anno = parsing_anno.copy().astype(np.uint8)
    vis_parsing_anno = cv2.resize(vis_parsing_anno, None, fx=stride, fy=stride, interpolation=cv2.INTER_NEAREST)
    seg = Image.fromarray(vis_parsing_anno, 'L')

    vis_parsing_anno_color = np.zeros((vis_parsing_anno.shape[0], vis_parsing_anno.shape[1], 3)) + 255

    num_of_class = np.max(vis_parsing_anno)

    for pi in range(1, num_of_class + 1):
        index = np.where(vis_parsing_anno == pi)
        vis_parsing_anno_color[index[0], index[1], :] = part_colors[pi]

    vis_parsing_anno_color = vis_parsing_anno_color.astype(np.uint8)
    # print(vis_parsing_anno_color.shape, vis_im.shape)
    vis_im = cv2.addWeighted(cv2.cvtColor(vis_im, cv2.COLOR_RGB2BGR), 0.4, vis_parsing_anno_color, 0.6, 0)

    # Save result or not
    if save_im:
        cv2.imwrite(save_path[:-4] +'.png', vis_parsing_anno)
        cv2.imwrite(save_path, vis_im, [int(cv2.IMWRITE_JPEG_QUALITY), 100])
        print(save_path)

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
        # print(parsing)
        lst1 = np.unique(parsing)
        # print("Output list : ", lst1)

        # vis_parsing_maps(image, parsing, stride=1, save_im=True, save_path='C:/Users/RYZEN 9/Documents/GitHub/TA/faceparsing/res/test_res/seg.jpg')

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
    net.load_state_dict(torch.load(save_pth, map_location=torch.device('cpu')))
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

def create_mask(img):
    n_classes = 19
    net = BiSeNet(n_classes=n_classes)
    # net.cuda()
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
        # img = img.cuda()
        out = net(img)[0]
        parsing = out.squeeze(0).cpu().numpy().argmax(0)
        lst1 = np.unique(parsing)
        # print("Output list : ", lst1)

        # vis_parsing_maps(image, parsing, stride=1, save_im=True, save_path="./test-img/coba.jpg")

        mask = np.zeros((parsing.shape[0], parsing.shape[1]), dtype=np.uint8).tolist()
        reorder_array = [0,1,2,3,4,5,1,11,12,11,6,8,7,9,13,13,0,10,0]
        for n in range (19):
            for i in range(parsing.shape[0]):
                for j in range(parsing.shape[1]):
                    if (parsing[i][j] == n):
                        mask[i][j] = reorder_array[n]
        mask = np.array(mask)
        return mask

# test_set = []
# for (path, dirnames, filenames) in os.walk('D:/# Raras/src/data/edited/final'):
#     test_set.extend(os.path.join(path, name) for name in filenames)

# for instance in test_set:
#     if (os.path.exists(instance.strip())):
#         print("Parsing... ", instance)
#         img_name= instance[32:]
#         img_path = "D:/# Raras/src/makeup_dataset/final/makeup/scrape/" + str(img_name)[:-4] + ".png" 
#         img = cv2.imread(instance)

#         seg = create_mask(img)
#         im = np.array(img)
#         parsing_anno = np.array(seg)
#         vis_im = im.copy().astype(np.uint8)
#         vis_parsing_anno = parsing_anno.copy().astype(np.uint8)
#         vis_parsing_anno = cv2.resize(vis_parsing_anno, None, fx=1, fy=1, interpolation=cv2.INTER_NEAREST)
#         seg = Image.fromarray(vis_parsing_anno, 'L')
#         seg_path = "D:/# Raras/src/makeup_dataset/final/makeup_segs/scrape/" + str(img_name)[:-4] + ".png" 
#         seg.save(seg_path)


#         img = Image.fromarray(img[:,:,::-1], 'RGB')
#         img.save(img_path)