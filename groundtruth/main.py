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
 
    test_set = []
    for (path, dirnames, filenames) in os.walk("D:/# Raras/GitHub/TA/experiment/data/RawData/images/non-makeup"):
        test_set.extend(os.path.join(path, name) for name in filenames)

    # test_set1 = []
    # for (path, dirnames, filenames) in os.walk('D:/# Raras/GitHub/TA/experiment/data/RawData/images/non-makeup'):
    #     test_set1.extend(os.path.join(path, name) for name in filenames)

    makeup = test_set
    # non_makeup = test_set1 + test_set1
    # test_set = test_set[]
    for i in range (len(test_set)):
        print("checking... ", i)
        # print("warping ", makeup[i], non_makeup[i])

        # Read images
        src_img = cv2.imread(makeup[i], cv2.COLOR_BGR2RGB)
        # dst_img = cv2.imread(non_makeup[i], cv2.COLOR_BGR2RGB)
        # cv2.imshow("src", src_img)
        # cv2.imshow("dst", dst_img)
        # cv2.waitKey(0)
        source_error = []
        dist_error = []
        try:
            src_img_seg = get_face(src_img)
            
        except:
            source_error.append(src_img)
            print(str(test_set[i][62:]) + " gabisa")

        try:
            # cv2.imshow("src_img_seg", src_img_seg)
            # cv2.waitKey(0)
            src_points, src_shape, src_face = select_face(src_img_seg)
            # print(str(test_set[i][59:]) + " bisa")
        except:
            source_error.append(src_img)
            print(str(test_set[i][62:]) + " gabisaaaaaaaaaaaaaaaa==========")
        
        try:
            src_faceBoxes = select_all_faces(src_img)
        except:
            source_error.append(src_img)
            print(str(test_set[i][62:]) + "gabisa")


        # try:
        #     dst_img_seg = get_face(dst_img)
        #     a, b, c = select_face(dst_img_seg)
        #     dst_faceBoxes = select_all_faces(dst_img)
        # except:
        #     dist_error.append(dst_img)
        #     continue

        # if dst_faceBoxes is None:
        #     print('Detect 0 Face !!!')
        # else:
        #     output = dst_img
        #     for k, dst_face in dst_faceBoxes.items():
        #         output = face_swap(src_face, dst_face["face"], src_points,
        #                         dst_face["points"], dst_face["shape"],
        #                         output)
        
        #     # cv2.imshow("src_img",src_img)
        #     # cv2.imshow("dst_img",dst_img)
        #     # cv2.imshow("output",output)
        #     # cv2.waitKey(0)
        #     dim = (256,256)
        #     src_img = cv2.resize(src_img, dim, interpolation = cv2.INTER_AREA)
        #     dst_img = cv2.resize(dst_img, dim, interpolation = cv2.INTER_AREA)
        #     output = cv2.resize(output, dim, interpolation = cv2.INTER_AREA)
        #     images = [src_img, dst_img, output]
        #     new_im = cv2.hconcat(images)

        #     img_path = "D:/# Raras/src/data/groundtruth/coba/" + str(makeup[i][48:])
        #     print(img_path)
        #     cv2.imwrite(img_path, new_im)

    
    print(source_error)
    # print(dist_error)
    with open("output.txt", "w") as txt_file:
        txt_file.write("source\n")
        for line in source_error:
            txt_file.write(" ".join(line) + "\n")
        # txt_file.write("source/n")
        # for line in dist_error:
        #     txt_file.write(" ".join(line) + "/n")