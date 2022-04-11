#! /usr/bin/env python
import os
import cv2
import argparse

from face_detection import select_face, select_all_faces
from face_swap import face_swap
from face_parsing import get_face


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='FaceSwapApp')
    # parser.add_argument('--src', required=True, help='Path for source image')
    # parser.add_argument('--dst', required=True, help='Path for target image')
    # parser.add_argument('--out', required=True, help='Path for storing output images')
    parser.add_argument('--warp_2d', default=False, action='store_true', help='2d or 3d warp')
    parser.add_argument('--correct_color', default=False, action='store_true', help='Correct color')
    parser.add_argument('--no_debug_window', default=False, action='store_true', help='Don\'t show debug window')
    args = parser.parse_args()
 
    makeup = []
    non_makeup = []

    makeup.append("D:/# Raras/GitHub/TA/dataset/RawData/images/makeup/12.png")
    makeup.append("D:/# Raras/GitHub/TA/dataset/RawData/images/makeup/vFG827.png")
    makeup.append("D:/# Raras/GitHub/TA/dataset/RawData/images/makeup/12.png")
    makeup.append("D:/# Raras/GitHub/TA/dataset/RawData/images/makeup/vFG827.png")

    non_makeup.append("D:/# Raras/GitHub/TA/dataset/RawData/images/non-makeup/vSYYZ663.png")
    non_makeup.append("D:/# Raras/GitHub/TA/dataset/RawData/images/non-makeup/xfsy_0478.png")
    non_makeup.append("D:/# Raras/GitHub/TA/dataset/RawData/images/non-makeup/xfsy_0478.png")
    non_makeup.append("D:/# Raras/GitHub/TA/dataset/RawData/images/non-makeup/vSYYZ663.png")

    for i in range (len(makeup)):
        print("warping ", makeup[i], non_makeup[i])

        # Read images
        src_img = cv2.imread(makeup[i])
        dst_img = cv2.imread(non_makeup[i])
        # cv2.imshow("src_img", src_img)
        # cv2.imshow("dst_img", dst_img)
        # cv2.waitKey(0)

        src_img_seg = get_face(src_img)
        src_points, src_shape, src_face, check = select_face(src_img_seg)
        
        src_faceBoxes = select_all_faces(src_img)
        dst_img_seg = get_face(dst_img)
        a, b, c, d = select_face(dst_img_seg)
        dst_faceBoxes = select_all_faces(dst_img)

        if dst_faceBoxes is None:
            print('Detect 0 Face !!!')
        else:
            output = dst_img
            for k, dst_face in dst_faceBoxes.items():
                output = face_swap(src_face, dst_face["face"], src_points,
                                dst_face["points"], dst_face["shape"],
                                output)
        
            # cv2.imshow("src_img",src_img)
            # cv2.imshow("dst_img",dst_img)
            # cv2.imshow("output",output)
            # cv2.waitKey(0)
            dim = (256,256)
            src_img = cv2.resize(src_img, dim, interpolation = cv2.INTER_AREA)
            dst_img = cv2.resize(dst_img, dim, interpolation = cv2.INTER_AREA)
            output = cv2.resize(output, dim, interpolation = cv2.INTER_AREA)
            images = [src_img, dst_img, output]
            new_im = cv2.hconcat(images)

            img_path = "D:/# Raras/GitHub/TA/groundtruth/coba/warping_{}.png".format(i)
            print(img_path)
            cv2.imwrite(img_path, new_im)