from cgi import test
from operator import index
import cv2
from matplotlib.pyplot import new_figure_manager
import numpy as np
import dlib
import os
from test import get_face

# img2 = cv2.imread( "C:/Users/RYZEN 9/Downloads/tara1.jpg")
# # img = cv2.imread( "C:/Users/RYZEN 9/Documents/GitHub/TA/faceparsing/res/test_res//498_res.png") 
# img = cv2.imread( "D:/# Raras/src/data/edited/final//3 (337).jpg")

def extract_index_nparray(nparray):
    index = None
    for num in nparray[0]:
        index = num
        break
    return index

def show_image(item):
    cv2.imshow(str(item), item)
    cv2.waitKey(0)

def warping(path_img1, path_img2):
    img = cv2.imread(path_img1)
    img2 = cv2.imread(path_img2)

    img2_gray = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)

    detector = dlib.get_frontal_face_detector()
    predictor = dlib.shape_predictor("../model/shape_predictor_68_face_landmarks.dat")
    height, width, channels = img2.shape
    img2_new_face = np.zeros((height, width, channels), np.uint8)

    # Face 1
    img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = detector(img_gray)
    mask = np.zeros_like(img_gray)

    if (len(faces) == 0):
        print("no face found!")
    for face in faces:
        landmarks = predictor(img_gray, face)
        landmarks_points = []
        for n in range(0, 68):
            x = landmarks.part(n).x
            y = landmarks.part(n).y
            landmarks_points.append((x, y))
        

        points = np.array(landmarks_points, np.int32)
        convexhull = cv2.convexHull(points)
        cv2.fillConvexPoly(mask, convexhull, 255)

        face_image_1 = cv2.bitwise_and(img, img, mask=mask)

        # Delaunay triangulation
        rect = cv2.boundingRect(convexhull)
        subdiv = cv2.Subdiv2D(rect)
        subdiv.insert(landmarks_points)
        triangles = subdiv.getTriangleList()
        triangles = np.array(triangles, dtype=np.int32)
        # print(triangles)

        indexes_triangles = []
        for t in triangles:
            pt1 = (t[0], t[1])
            pt2 = (t[2], t[3])
            pt3 = (t[4], t[5])


            index_pt1 = np.where((points == pt1).all(axis=1))
            index_pt1 = extract_index_nparray(index_pt1)

            index_pt2 = np.where((points == pt2).all(axis=1))
            index_pt2 = extract_index_nparray(index_pt2)

            index_pt3 = np.where((points == pt3).all(axis=1))
            index_pt3 = extract_index_nparray(index_pt3)

            if index_pt1 is not None and index_pt2 is not None and index_pt3 is not None:
                triangle = [index_pt1, index_pt2, index_pt3]
                indexes_triangles.append(triangle)


    # Face 2
    faces2 = detector(img2_gray)
    if (len(faces2) == 0):
        print("no face found!")
    for face in faces2:

        landmarks = predictor(img2_gray, face)
        landmarks_points2 = []
        for n in range(0, 68):
            x = landmarks.part(n).x
            y = landmarks.part(n).y
            landmarks_points2.append((x, y))

        
        left_eye_points2 = np.array(landmarks_points2[36:42], np.int32)
        right_eye_points2 = np.array(landmarks_points2[42:48], np.int32)
        mouth_points2 = np.array(landmarks_points2[60:68], np.int32)
        points2 = np.array(landmarks_points2, np.int32)
        convexhull2 = cv2.convexHull(points2)


    lines_space_mask = np.zeros_like(img_gray)
    lines_space_new_face = np.zeros_like(img2)
    # Triangulation of both faces

    # indexes_triangles = [e for e in indexes_triangles if e not in indexes]
    for triangle_index in indexes_triangles:
        # Triangulation of the first face
        tr1_pt1 = landmarks_points[triangle_index[0]]
        tr1_pt2 = landmarks_points[triangle_index[1]]
        tr1_pt3 = landmarks_points[triangle_index[2]]
        triangle1 = np.array([tr1_pt1, tr1_pt2, tr1_pt3], np.int32)


        rect1 = cv2.boundingRect(triangle1)
        (x, y, w, h) = rect1
        cropped_triangle = img[y: y + h, x: x + w]
        cropped_tr1_mask = np.zeros((h, w), np.uint8)


        points = np.array([[tr1_pt1[0] - x, tr1_pt1[1] - y],
                        [tr1_pt2[0] - x, tr1_pt2[1] - y],
                        [tr1_pt3[0] - x, tr1_pt3[1] - y]], np.int32)

        cv2.fillConvexPoly(cropped_tr1_mask, points, 255)

        # Lines space
        cv2.line(lines_space_mask, tr1_pt1, tr1_pt2, 255)
        cv2.line(lines_space_mask, tr1_pt2, tr1_pt3, 255)
        cv2.line(lines_space_mask, tr1_pt1, tr1_pt3, 255)
        lines_space = cv2.bitwise_and(img, img, mask=lines_space_mask)

        # Triangulation of second face
        tr2_pt1 = landmarks_points2[triangle_index[0]]
        tr2_pt2 = landmarks_points2[triangle_index[1]]
        tr2_pt3 = landmarks_points2[triangle_index[2]]
        triangle2 = np.array([tr2_pt1, tr2_pt2, tr2_pt3], np.int32)


        rect2 = cv2.boundingRect(triangle2)
        (x, y, w, h) = rect2

        cropped_tr2_mask = np.zeros((h, w), np.uint8)

        points2 = np.array([[tr2_pt1[0] - x, tr2_pt1[1] - y],
                            [tr2_pt2[0] - x, tr2_pt2[1] - y],
                            [tr2_pt3[0] - x, tr2_pt3[1] - y]], np.int32)

        cv2.fillConvexPoly(cropped_tr2_mask, points2, 255)

        # Warp triangles
        points = np.float32(points)
        points2 = np.float32(points2)
        M = cv2.getAffineTransform(points, points2)
        warped_triangle = cv2.warpAffine(cropped_triangle, M, (w, h))
        warped_triangle = cv2.bitwise_and(warped_triangle, warped_triangle, mask=cropped_tr2_mask)

        # Reconstructing destination face
        img2_new_face_rect_area = img2_new_face[y: y + h, x: x + w]
        img2_new_face_rect_area_gray = cv2.cvtColor(img2_new_face_rect_area, cv2.COLOR_BGR2GRAY)
        _, mask_triangles_designed = cv2.threshold(img2_new_face_rect_area_gray, 1, 255, cv2.THRESH_BINARY_INV)
        warped_triangle = cv2.bitwise_and(warped_triangle, warped_triangle, mask=mask_triangles_designed)

        img2_new_face_rect_area = cv2.add(img2_new_face_rect_area, warped_triangle)
        img2_new_face[y: y + h, x: x + w] = img2_new_face_rect_area
    
    # cv2.imshow("img2_new_face", img2_new_face)
    # cv2.waitKey(0)

    # Face swapped (putting 1st face into 2nd face)
    img2_face_mask = np.zeros_like(img2_gray)
    img2_head_mask = cv2.fillConvexPoly(img2_face_mask, convexhull2, 255)
    img2_face_mask = cv2.bitwise_not(img2_head_mask)
    # cv2.imshow("img2_face_mask", img2_face_mask)
    # cv2.waitKey(0)


    mask_eye = np.zeros_like(img2_gray)
    convexhull_left_eye = cv2.convexHull(left_eye_points2)
    convexhull_right_eye = cv2.convexHull(right_eye_points2)
    convexhull_mouth = cv2.convexHull(mouth_points2)
    mask_eye = cv2.fillConvexPoly(mask_eye, convexhull_left_eye, 255)
    mask_eye = cv2.fillConvexPoly(mask_eye, convexhull_right_eye, 255)
    mask_eye = cv2.fillConvexPoly(mask_eye, convexhull_mouth, 255)
    cv2.imshow("img2_face_mask", img2_face_mask)
    cv2.imshow("mask_eye", mask_eye)
    cv2.waitKey(0)
    img2_face_mask = cv2.bitwise_xor(img2_face_mask, mask_eye)
    img2_new_face[mask_eye == 255] = 0

    for i in range(img2_face_mask.shape[0]):
        for j in range(img2_face_mask.shape[1]):
            if (img2_new_face[i][j].any() != 0):
                img2_face_mask[i][j] = 0
            else:
                img2_face_mask[i][j] = 255



    img2_head_noface = cv2.bitwise_and(img2, img2, mask=img2_face_mask)
    result = cv2.add(img2_head_noface, img2_new_face)
    # cv2.imshow("img2_head_noface", img2_head_noface)
    # cv2.imshow("img2_new_face", img2_new_face)
    # cv2.imshow("result", result)
    # cv2.waitKey(0)
    


    (x, y, w, h) = cv2.boundingRect(convexhull2)
    center_face2 = (int((x + x + w) / 2), int((y + y + h) / 2))

    seamlessclone = cv2.seamlessClone(result, img2, img2_head_mask, center_face2, cv2.MIXED_CLONE)
    # cv2.imshow("final", seamlessclone)
    # cv2.waitKey(0)

    dim = (256, 256)
    seamlessclone = cv2.resize(seamlessclone, dim, interpolation = cv2.INTER_AREA)
    result = cv2.resize(result, dim, interpolation = cv2.INTER_AREA)
    img = cv2.resize(img, dim, interpolation = cv2.INTER_AREA)
    img2 = cv2.resize(img2, dim, interpolation = cv2.INTER_AREA)

    images = [img, img2, result, seamlessclone]
    new_im = cv2.hconcat(images)

    return new_im

test_set = []
for (path, dirnames, filenames) in os.walk("D:/# Raras/src/data/edited/5"):
    test_set.extend(os.path.join(path, name) for name in filenames)

test_set1 = []
for (path, dirnames, filenames) in os.walk('D:/# Raras/src/makeup_dataset/all/images/non-makeup'):
    test_set1.extend(os.path.join(path, name) for name in filenames)

makeup = test_set[:50]
non_makeup = test_set1[50:101]

for i in range (50):
    # try:
    print("warping ", makeup[i], non_makeup[i])
    result = warping(makeup[i], non_makeup[i])
    img_path = "D:/# Raras/src/data/groundtruth/" + str(i) + ".jpg"
    cv2.imwrite(img_path, result)
    # except:
    #     pass

