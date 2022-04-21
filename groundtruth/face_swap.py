#! /usr/bin/env python
from unittest import result
import cv2
import numpy as np
import scipy.spatial as spatial
import logging
import matplotlib.pyplot as plt
import math

## 3D Transform
def bilinear_interpolate(img, coords):
    """ Interpolates over every image channel
    http://en.wikipedia.org/wiki/Bilinear_interpolation
    :param img: max 3 channel image
    :param coords: 2 x _m_ array. 1st row = xcoords, 2nd row = ycoords
    :returns: array of interpolated pixels with same shape as coords
    """
    int_coords = np.int32(coords)
    x0, y0 = int_coords
    dx, dy = coords - int_coords

    # 4 Neighour pixels
    # if (y0 < img.shape[0] and x0<img.shape[0]):
    q11 = img[y0, x0]
    q21 = img[y0, x0 + 1]
    q12 = img[y0 + 1, x0]
    q22 = img[y0 + 1, x0 + 1]

    btm = q21.T * dx + q11.T * (1 - dx)
    top = q22.T * dx + q12.T * (1 - dx)
    inter_pixel = top * dy + btm * (1 - dy)

    return inter_pixel.T

def grid_coordinates(points):
    """ x,y grid coordinates within the ROI of supplied points
    :param points: points to generate grid coordinates
    :returns: array of (x, y) coordinates
    """
    xmin = np.min(points[:, 0])
    xmax = np.max(points[:, 0]) + 1
    ymin = np.min(points[:, 1])
    ymax = np.max(points[:, 1]) + 1

    return np.asarray([(x, y) for y in range(ymin, ymax)
                       for x in range(xmin, xmax)], np.uint32)


def process_warp(src_img, result_img, tri_affines, dst_points, delaunay):
    """
    Warp each triangle from the src_image only within the
    ROI of the destination image (points in dst_points).
    """
    roi_coords = grid_coordinates(dst_points)
    # indices to vertices. -1 if pixel is not in any triangle
    roi_tri_indices = delaunay.find_simplex(roi_coords)

    for simplex_index in range(len(delaunay.simplices)):
        coords = roi_coords[roi_tri_indices == simplex_index]
        num_coords = len(coords)
        out_coords = np.dot(tri_affines[simplex_index],
                            np.vstack((coords.T, np.ones(num_coords))))
        x, y = coords.T
        x0, y0 = out_coords
        x_src = []
        y_src = []
        x_res = []
        y_res = []
        for i in range(len(x0)):
            if (x0[i] + 1 < src_img.shape[1]) and (y0[i] + 1 < src_img.shape[0] and x[i] < result_img.shape[1] and y[i] < result_img.shape[0]):
                x_src.append(x0[i])
                y_src.append(y0[i])
                x_res.append(x[i])
                y_res.append(y[i])
        out_coords = x_src, y_src
        x, y = x_res, y_res
        result_img[y, x] = bilinear_interpolate(src_img, out_coords)

    return None


def triangular_affine_matrices(vertices, src_points, dst_points):
    """
    Calculate the affine transformation matrix for each
    triangle (x,y) vertex from dst_points to src_points
    :param vertices: array of triplet indices to corners of triangle
    :param src_points: array of [x, y] points to landmarks for source image
    :param dst_points: array of [x, y] points to landmarks for destination image
    :returns: 2 x 3 affine matrix transformation for a triangle
    """
    ones = [1, 1, 1]
    for tri_indices in vertices:
        src_tri = np.vstack((src_points[tri_indices, :].T, ones))
        dst_tri = np.vstack((dst_points[tri_indices, :].T, ones))
        mat = np.dot(src_tri, np.linalg.inv(dst_tri))[:2, :]
        yield mat


def warp_image_3d(src_img, src_points, dst_points, dst_shape, dtype=np.uint8):
    rows, cols = dst_shape[:2]
    result_img = np.zeros((rows, cols, 3), dtype=dtype)

    delaunay = spatial.Delaunay(dst_points)
    tri_affines = np.asarray(list(triangular_affine_matrices(
        delaunay.simplices, src_points, dst_points)))

    process_warp(src_img, result_img, tri_affines, dst_points, delaunay)

    return result_img


## 2D Transform
def transformation_from_points(points1, points2):
    points1 = points1.astype(np.float64)
    points2 = points2.astype(np.float64)

    c1 = np.mean(points1, axis=0)
    c2 = np.mean(points2, axis=0)
    points1 -= c1
    points2 -= c2

    s1 = np.std(points1)
    s2 = np.std(points2)
    points1 /= s1
    points2 /= s2

    U, S, Vt = np.linalg.svd(np.dot(points1.T, points2))
    R = (np.dot(U, Vt)).T

    return np.vstack([np.hstack([s2 / s1 * R,
                                (c2.T - np.dot(s2 / s1 * R, c1.T))[:, np.newaxis]]),
                      np.array([[0., 0., 1.]])])


def warp_image_2d(im, M, dshape):
    output_im = np.zeros(dshape, dtype=im.dtype)
    cv2.warpAffine(im,
                   M[:2],
                   (dshape[1], dshape[0]),
                   dst=output_im,
                   borderMode=cv2.BORDER_TRANSPARENT,
                   flags=cv2.WARP_INVERSE_MAP)

    return output_im


## Generate Mask
def mask_from_points(size, points, erode_flag=0):
    radius = 10  # kernel size
    kernel = np.ones((radius, radius), np.uint8)

    mask = np.zeros(size, np.uint8)
    cv2.fillConvexPoly(mask, cv2.convexHull(points), 255)

    mask_eye = np.zeros_like(mask)
    left_eye_points2 = np.array(points[36:42], np.int32)
    right_eye_points2 = np.array(points[42:48], np.int32)
    mouth_points2 = np.array(points[60:68], np.int32)
    convexhull_left_eye = cv2.convexHull(left_eye_points2)
    convexhull_right_eye = cv2.convexHull(right_eye_points2)
    convexhull_mouth = cv2.convexHull(mouth_points2)
    mask_eye = cv2.fillConvexPoly(mask_eye, convexhull_left_eye, 255)
    mask_eye = cv2.fillConvexPoly(mask_eye, convexhull_right_eye, 255)
    mask_eye = cv2.fillConvexPoly(mask_eye, convexhull_mouth, 255)
    mask = cv2.bitwise_not(mask)
    mask = cv2.bitwise_xor(mask, mask_eye)
    mask = cv2.bitwise_not(mask)

    if erode_flag:
        mask = cv2.erode(mask, kernel,iterations=1)

    return mask

def get_lip_mask(size, points, erode_flag=0):
    radius = 10  # kernel size
    kernel = np.ones((radius, radius), np.uint8)

    mask = np.zeros(size, np.uint8)
    cv2.fillConvexPoly(mask, cv2.convexHull(points), 255)

    mask1 = np.zeros_like(mask)
    lips_points = np.array(points[48:60], np.int32)
    mouth_points = np.array(points[60:68], np.int32)
    convexhull_lips = cv2.convexHull(lips_points)
    convexhull_mouth = cv2.convexHull(mouth_points)
    
    mask_lips = cv2.fillConvexPoly(mask1, convexhull_lips, 255)
    mask_lips = cv2.bitwise_not(mask_lips)
    mask_lips_no_mouth = cv2.fillConvexPoly(mask_lips, convexhull_mouth, 255)
    mask_lips_no_mouth = cv2.bitwise_not(mask_lips_no_mouth)

    if erode_flag:
        mask = cv2.erode(mask, kernel, iterations=1)

    return mask_lips_no_mouth

def get_eye_region(size, mask_l, mask_r, points):
    mask = np.zeros(size, np.uint8)
    # eye_points_l = np.array(points[36:42], np.int32)
    # eye_points_r = np.array(points[42:48], np.int32)
    # radius = int((abs(points[36][0] - points[39][0]) // 2) * 1.75)
    # convexhull_eye_l = cv2.convexHull(eye_points_l)
    # convexhull_eye_r = cv2.convexHull(eye_points_r)

    # cv2.fillConvexPoly(mask, convexhull_eye_r, 255)
    # cv2.fillConvexPoly(mask, convexhull_eye_l, 255)
    # print(mask_l.dtype, mask_l.shape)

    ret, threshl = cv2.threshold(mask_l, 50, 255, cv2.THRESH_BINARY)
    ret, threshr = cv2.threshold(mask_r, 50, 255, cv2.THRESH_BINARY)
    contoursl, hierarchy = cv2.findContours(threshl, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    contoursr, hierarchy = cv2.findContours(threshr, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    hull_l = []
    hull_r = []
    for i in range(len(contoursl)):
        hull_l.append(cv2.convexHull(contoursl[i], False))
    for i in range(len(contoursr)):
        hull_r.append(cv2.convexHull(contoursr[i], False))
    
    r_l = cv2.boundingRect(hull_l[0])
    center_l = ((r_l[0] + int(r_l[2] / 2), r_l[1] + int(r_l[3] / 2)))
    r_r = cv2.boundingRect(hull_r[0])
    center_r = ((r_r[0] + int(r_r[2] / 2), r_r[1] + int(r_r[3] / 2)))
    
    corner_left_l = points[36]
    corner_left_r = points[39]
    corner_right_l = points[42]
    corner_right_r = points[45]

    r_left = int(math.sqrt((corner_left_l[0]- corner_left_r[0])**2 + (corner_left_l[1]- corner_left_r[1])**2) * 0.8)
    r_right = int(math.sqrt((corner_right_l[0]- corner_right_r[0])**2 + (corner_right_l[1]- corner_right_r[1])**2) *0.8)

    circle_l = cv2.circle(mask, center_l, r_left, (255, 255, 255), -1)
    circle_r = cv2.circle(mask, center_r, r_right, (255, 255, 255), -1)

    return mask

## Color Correction
def correct_colours(im1, im2, landmarks1):
    COLOUR_CORRECT_BLUR_FRAC = 0.75
    LEFT_EYE_POINTS = list(range(42, 48))
    RIGHT_EYE_POINTS = list(range(36, 42))

    blur_amount = COLOUR_CORRECT_BLUR_FRAC * np.linalg.norm(
                              np.mean(landmarks1[LEFT_EYE_POINTS], axis=0) -
                              np.mean(landmarks1[RIGHT_EYE_POINTS], axis=0))
    blur_amount = int(blur_amount)
    if blur_amount % 2 == 0:
        blur_amount += 1
    im1_blur = cv2.GaussianBlur(im1, (blur_amount, blur_amount), 0)
    im2_blur = cv2.GaussianBlur(im2, (blur_amount, blur_amount), 0)

    # Avoid divide-by-zero errors.
    im2_blur = im2_blur.astype(int)
    im2_blur += 128*(im2_blur <= 1)

    result = im2.astype(np.float64) * im1_blur.astype(np.float64) / im2_blur.astype(np.float64)
    result = np.clip(result, 0, 255).astype(np.uint8)

    return result


## Copy-and-paste
def apply_mask(img, mask):
    """ Apply mask to supplied image
    :param img: max 3 channel image
    :param mask: [0-255] values in mask
    :returns: new image with mask applied
    """
    masked_img=cv2.bitwise_and(img,img,mask=mask)

    return masked_img


## Alpha blending
def alpha_feathering(src_img, dest_img, img_mask, blur_radius=15):
    mask = cv2.blur(img_mask, (blur_radius, blur_radius))
    mask = mask / 255.0

    result_img = np.empty(src_img.shape, np.uint8)
    for i in range(3):
        result_img[..., i] = src_img[..., i] * mask + dest_img[..., i] * (1-mask)

    return result_img


def check_points(img,points):
    # Todo: I just consider one situation.
    if points[8,1]>img.shape[0]:
        logging.error("Jaw part out of image")
    else:
        return True
    return False


def face_swap(src_face, dst_face, src_points, dst_points, dst_shape, dst_img, seg_dst, end=68):
    h_face, w_face = dst_face.shape[:2]
    h_img, w_img = dst_img.shape[:2]

    id_brow = [2, 3]
    id_eye_l = [4]
    id_eye_r = [5]
    id_lip = [7, 9]

    ## 3d warp
    warped_src_face = warp_image_3d(src_face, src_points[:end], dst_points[:end], (h_face, w_face))
    
    ## Mask for blending
    mask = mask_from_points((h_face, w_face), dst_points)
    mask_src = np.mean(warped_src_face, axis=2) > 0
    mask = np.asarray(mask * mask_src, dtype=np.uint8)

    ## Poisson Blending
    r = cv2.boundingRect(mask)
    center = ((r[0] + int(r[2] / 2), r[1] + int(r[3] / 2)))
    output = cv2.seamlessClone(warped_src_face, dst_face, mask, center, cv2.MIXED_CLONE)

    brow_mask = get_mask((h_img, w_img), seg_dst, id_brow)
    lip_mask = get_mask((h_img, w_img), seg_dst, id_lip)
    eyeball_mask_l = get_mask((h_img, w_img), seg_dst, id_eye_l)
    eyeball_mask_r = get_mask((h_img, w_img), seg_dst, id_eye_r)
    eye_mask = get_eye_region((h_img, w_img), eyeball_mask_l, eyeball_mask_r, dst_points)
    mask_copy = brow_mask + lip_mask + eye_mask

    x, y, w, h = dst_shape
    warped_face = np.zeros_like(dst_img, dtype='uint8')
    warped_face[y:y + h, x:x + w] = warped_src_face
    seg_copy = cv2.bitwise_and(warped_face, warped_face, mask=mask_copy)

    dst_img_cp = dst_img.copy()
    dst_img_cp[y:y + h, x:x + w] = output
    dst_img_cp[seg_copy > 0] = seg_copy[seg_copy > 0]
    dst_img_cp[dst_img_cp == 0] = dst_img[dst_img_cp == 0]

    return dst_img_cp


def face_blend(src_face, dst_face, src_points, dst_points, dst_shape, dst_img, end=68):
    h_face, w_face = dst_face.shape[:2]

    ## 3d warp
    warped_src_face = warp_image_3d(src_face, src_points[:end], dst_points[:end], (h_face, w_face))
    
    ## Mask for blending
    mask = mask_from_points((h_face, w_face), dst_points)
    mask_src = np.mean(warped_src_face, axis=2) > 0
    mask = np.asarray(mask * mask_src, dtype=np.uint8)
    
    ## Poisson Blending
    r = cv2.boundingRect(mask)
    center = ((r[0] + int(r[2] / 2), r[1] + int(r[3] / 2)))
    output = cv2.seamlessClone(warped_src_face, dst_face, mask, center, cv2.MIXED_CLONE)

    x, y, w, h = dst_shape
    warped_face = np.zeros_like(dst_img, dtype='uint8')
    warped_face[y:y + h, x:x + w] = warped_src_face

    gt_blend = dst_img.copy()
    gt_blend[y:y + h, x:x + w] = output

    return gt_blend

def face_copy(src_face, dst_face, src_points, dst_points, dst_shape, dst_img, copy_mask, end=68):
    h_face, w_face = dst_face.shape[:2]

    ## 3d warp
    warped_src_face = warp_image_3d(src_face, src_points[:end], dst_points[:end], (h_face, w_face))
    mask_copy = copy_mask

    x, y, w, h = dst_shape
    warped_face = np.zeros_like(dst_img, dtype='uint8')
    warped_face[y:y + h, x:x + w] = warped_src_face
    seg_copy = cv2.bitwise_and(warped_face, warped_face, mask=mask_copy)

    gt_copy = dst_img.copy()
    gt_copy[seg_copy > 0] = seg_copy[seg_copy > 0]

    return gt_copy

def get_mask(size, seg, classes_list):
    mask = np.zeros(size, dtype=np.uint8)
    for i in classes_list:
        mask[seg == i] = 255
    return mask