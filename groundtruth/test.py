from importlib.resources import path
from face_parsing import get_face, vis_parsing_maps
import cv2
import numpy as np


pathh = "D:/# Raras/src/makeup_dataset/all/images/makeup/1c950a7b02087bf4743ff8d4f9d3572c11dfcf38.png"
path_seg = "D:/# Raras/src/makeup_dataset/all/segs/makeup/1c950a7b02087bf4743ff8d4f9d3572c11dfcf38.png"
img = cv2.imread(pathh)
img2 = cv2.imread(path_seg)
gray = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)
dim = (img.shape[0], img.shape[1])
a = get_face(img)
lst1 = np.unique(gray)
gray = cv2.resize(gray, dim, fx=1, fy=1, interpolation=cv2.INTER_NEAREST)

print(img.shape)
print(gray.shape)
vis_parsing_maps(img, gray, stride=1, save_im=True, save_path='C:/Users/RYZEN 9/Documents/GitHub/TA/faceparsing/res/test_res/seg1.jpg')
