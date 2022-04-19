#! /usr/bin/env python
import os
import cv2
import argparse
from matplotlib.pyplot import hist, table
import tensorflow as tf
import numpy as np
from skimage.exposure import match_histograms
import copy

from face_detection import select_face, select_all_faces
from face_swap import face_swap, getMakeupGroundTruth_warping
from face_parsing import get_face

id_face = [1, 6, 11, 12, 13]
id_brow = [2, 3]
id_eye = [2, 3, 4, 5]
id_lip = [7, 9]
id_hair = [10]

def eye_regions_func(mask):
    vertical = tf.clip_by_value(tf.reduce_sum(mask, axis = 0, keepdims = True), 0, 1)
    horizontal = tf.clip_by_value(tf.reduce_sum(mask, axis = 1, keepdims = True), 0, 1)
    vertical_n = tf.clip_by_value(tf.cumsum(vertical, axis = 1), 0, 1)
    vertical_r = tf.clip_by_value(tf.cumsum(vertical, axis = 1, reverse = True), 0, 1)
    horizontal_n = tf.clip_by_value(tf.cumsum(horizontal, axis = 0), 0, 1)
    horizontal_r = tf.clip_by_value(tf.cumsum(horizontal, axis = 0, reverse = True), 0, 1)
    vertical = vertical_n * vertical_r
    horizontal = horizontal_n * horizontal_r
    return (vertical * horizontal)

def masking_func(mask, classes_list):
    mask = tf.cast(mask, dtype = tf.int32)
    cum_mask = tf.zeros_like(mask, dtype = tf.int32)
    for i in classes_list:
        cum_mask +=  tf.cast(mask == i, dtype = tf.int32)
    cum_mask = tf.cast(tf.clip_by_value(cum_mask, 0, 1), dtype = tf.int32)
    return cum_mask

def process_mask(mask):
    mask = mask.numpy().astype(np.uint8)
    mask[mask > 0] = 255
    mask = cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY)
    return mask

def cal_trans(ref, adj):
    """
        calculate transfer function
        algorithm refering to wiki item: Histogram matching
    """
    table = list(range(0, 256))
    for i in list(range(1, 256)):
        for j in list(range(1, 256)):
            if ref[i] >= adj[j - 1] and ref[i] <= adj[j]:
                table[i] = j
                break
    table[255] = 255
    return table

def get_histogram(img, img_mask):
    hists = []
    img_mask = np.array(img_mask, dtype=np.uint8)
    for i in range(3):
        hist = cv2.calcHist([img],[i], img_mask, [256], [0,256])
        sum = hist.sum()
        pdf = [v / sum for v in hist]
        for i in range(1, 256):
            pdf[i] = pdf[i - 1] + pdf[i]
        hists.append(pdf)
    hists = np.array(hists)
    return hists

def preprocess(image):
    '''
    Normalize image array.
    '''
    return (tf.cast(image, tf.float32) / 255.0 - 0.5) * 2

def hist_match_func(src, ref, mask_src, mask_ref, idx_src):
    """
    Adjust the pixel values of images such that its histogram
    matches that of a target image

    Arguments:
    -----------
        source: np.ndarray
            Image to transform; the histogram is computed over the flattened
            array
        reference: np.ndarray
            Reference image; can have different dimensions to source
        index:
            the index of pixels that need to be transformed
    Returns:
    -----------
        matched: np.ndarray
            The transformed output image
    """

    img_src = np.array(src, dtype = np.uint8)
    img_ref = np.array(ref, dtype = np.uint8)

    # # CARA 1
    # result = np.zeros(oldshape, dtype = np.uint8)
    # for c in range(3):
    #     s = source[c].ravel()
    #     r = reference[c].ravel()

    #     s_values, bin_idx, s_counts = np.unique(s, return_inverse=True, return_counts=True)
    #     r_values, r_counts = np.unique(r, return_counts=True)

    #     if(len(s_counts) == 1 or len(r_counts) == 1):
    #         continue
    #     # take the cumsum of the counts and normalize by the number of pixels to
    #     # get the empirical cumulative distribution functions for the source and
    #     # template images (maps pixel value --> quantile)
    #     s_quantiles = np.cumsum(s_counts).astype(np.float64)
    #     s_quantiles /= s_quantiles[-1]
    #     r_quantiles = np.cumsum(r_counts).astype(np.float64)
    #     r_quantiles /= r_quantiles[-1]
    #     r_values = r_values

    #     # interpolate linearly to find the pixel values in the template image
    #     # that correspond most closely to the quantiles in the source image
    #     interp_value = np.zeros_like(s_values, dtype = np.float32)
    #     interp_r_values = np.interp(s_quantiles, r_quantiles, r_values)
    #     interp_value = interp_r_values
    #     result[:, :, c] = interp_value[bin_idx].reshape(oldshape[0:2])


    # # CARA 2
    # result = match_histograms(source_img, reference_img, multichannel=True)

    # CARA 3
    dst_align = [img_src[idx_src[0], idx_src[1], i] for i in range(3)]
    hist_dst = get_histogram(img_src, mask_src)
    hist_ref = get_histogram(img_ref, mask_ref)
    tables = [cal_trans(hist_dst[i], hist_ref[i]) for i in range(3)]

    mid = copy.deepcopy(dst_align)
    for i in range(0, 3):
        for k in range(0, len(idx_src[0])):
            dst_align[i][k] = tables[i][int(mid[i][k])]

    for i in range(0, 3):
        img_src[idx_src[0], idx_src[1], i] = dst_align[i]

    img_src = np.array(img_src)

    return img_src

def getMakeupGroundTruth_histogram(images, masks, reference_images, reference_masks):
    '''
    Generate the ground truth of all makeup regions.

    Args:
        images : a batch of source images.
        masks : a batch of source masks.
        reference_images : a batch of reference images.
        reference_masks : a batch of reference masks.

    Returns:
        A dictionary that contains the data information for training.
    '''

    # get source mask of each source makeup region
    hair_masks = masking_func(masks, tf.constant(id_hair, dtype = tf.int32))
    face_masks = masking_func(masks, tf.constant(id_face, dtype = tf.int32))
    brow_masks = masking_func(masks, tf.constant(id_brow, dtype = tf.int32))
    eye_masks = masking_func(masks, tf.constant(id_eye, dtype = tf.int32))
    eye_masks = tf.clip_by_value(eye_regions_func(eye_masks) - eye_masks, 0, 1)
    eye_masks = tf.clip_by_value(eye_masks - hair_masks - brow_masks, 0, 1)
    lip_masks = masking_func(masks, tf.constant(id_lip, dtype = tf.int32))

    hair_masks = process_mask(hair_masks)
    face_masks = process_mask(face_masks)
    brow_masks = process_mask(brow_masks)
    eye_masks = process_mask(eye_masks)
    lip_masks = process_mask(lip_masks)

    hair_index = np.where(hair_masks > 0)
    face_index = np.where(face_masks > 0)
    brow_index = np.where(brow_masks > 0)
    eye_index = np.where(eye_masks > 0)
    lip_index = np.where(lip_masks > 0)
    
    face = cv2.bitwise_and(images, images, mask = face_masks)
    lip = cv2.bitwise_and(images, images, mask = lip_masks)
    eye = cv2.bitwise_and(images, images, mask = eye_masks)
    brow = cv2.bitwise_and(images, images, mask = brow_masks)

    # cv2.imshow("face", face)
    # cv2.imshow("lip", lip)
    # cv2.imshow("eye", eye)
    # cv2.imshow("brow", brow)
    # cv2.waitKey(0)
    
    # get reference mask of each reference makeup region
    r_hair_masks = masking_func(reference_masks, tf.constant(id_hair, dtype = tf.int32))
    r_face_masks = masking_func(reference_masks, tf.constant(id_face, dtype = tf.int32))
    r_brow_masks = masking_func(reference_masks, tf.constant(id_brow, dtype = tf.int32))
    r_eye_masks = masking_func(reference_masks, tf.constant(id_eye, dtype = tf.int32))
    r_eye_masks = tf.clip_by_value(eye_regions_func(r_eye_masks) - r_eye_masks, 0, 1)
    r_eye_masks = tf.clip_by_value(r_eye_masks - r_hair_masks - r_brow_masks, 0, 1)
    r_lip_masks = masking_func(reference_masks, tf.constant(id_lip, dtype = tf.int32))
    
    r_hair_masks = process_mask(r_hair_masks)
    r_face_masks = process_mask(r_face_masks)
    r_brow_masks = process_mask(r_brow_masks)
    r_eye_masks = process_mask(r_eye_masks)
    r_lip_masks = process_mask(r_lip_masks)
    
    r_face = cv2.bitwise_and(reference_images, reference_images, mask = r_face_masks)
    r_lip = cv2.bitwise_and(reference_images, reference_images, mask = r_lip_masks)
    r_eye = cv2.bitwise_and(reference_images, reference_images, mask = r_eye_masks)
    r_brow = cv2.bitwise_and(reference_images, reference_images, mask = r_brow_masks)

    face_true = hist_match_func(face, r_face, face_masks, r_face_masks, face_index)
    brow_true = hist_match_func(brow, r_brow, brow_masks, r_brow_masks, brow_index)
    eye_true = hist_match_func(eye, r_eye, eye_masks, r_eye_masks, eye_index)
    lip_true = hist_match_func(lip, r_lip, lip_masks, r_lip_masks, lip_index)

    return [face_true, brow_true, eye_true, lip_true]

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
    seg_makeup = []
    seg_non_makeup = []

    # RYZEN
    makeup_directory = "D:/# Raras/GitHub/TA/dataset/RawData/images/makeup/"
    nonmakeup_directory = "D:/# Raras/GitHub/TA/dataset/RawData/images/non-makeup/"
    seg_makeup_directory = "D:/# Raras/GitHub/TA/dataset/RawData/segs/makeup/"
    seg_nonmakeup_directory = "D:/# Raras/GitHub/TA/dataset/RawData/segs/non-makeup/"

    # MAC
    # makeup_directory =  "/Users/raras/Documents/Raras/KULIAH/GITHUB/TA/dataset/RawData/images/makeup/"
    # nonmakeup_directory =  "/Users/raras/Documents/Raras/KULIAH/GITHUB/TA/dataset/RawData/images/non-makeup/"
    # seg_makeup_directory =  "/Users/raras/Documents/Raras/KULIAH/GITHUB/TA/dataset/RawData/segs/makeup/"
    # seg_nonmakeup_directory =  "/Users/raras/Documents/Raras/KULIAH/GITHUB/TA/dataset/RawData/segs/non-makeup/"

    makeup.append(makeup_directory + "12.png")
    # makeup.append(makeup_directory + "94.png")
    # makeup.append(makeup_directory + "1635.png")
    # makeup.append(makeup_directory + "103.png")
    seg_makeup.append(seg_makeup_directory + "12.png")
    seg_makeup.append(seg_makeup_directory + "94.png")
    seg_makeup.append(seg_makeup_directory + "1635.png")
    seg_makeup.append(seg_makeup_directory + "103.png")

    non_makeup.append(nonmakeup_directory + "vSYYZ25.png")
    non_makeup.append(nonmakeup_directory + "vSYYZ51.png")
    non_makeup.append(nonmakeup_directory + "vSYYZ25.png")
    non_makeup.append(nonmakeup_directory + "vSYYZ663.png")
    seg_non_makeup.append(seg_nonmakeup_directory + "vSYYZ25.png")
    seg_non_makeup.append(seg_nonmakeup_directory + "vSYYZ51.png")
    seg_non_makeup.append(seg_nonmakeup_directory + "vSYYZ25.png")
    seg_non_makeup.append(seg_nonmakeup_directory + "vSYYZ663.png")

    for i in range (len(makeup)):
        print("warping ", makeup[i], non_makeup[i])

        # Read images
        src_img = cv2.imread(makeup[i])
        dst_img = cv2.imread(non_makeup[i])
        seg_src_img = cv2.imread(seg_makeup[i], 0)
        seg_dst_img = cv2.imread(seg_non_makeup[i], 0)
        dim = (256,256)
        src_img = cv2.resize(src_img, dim, interpolation = cv2.INTER_LINEAR)
        dst_img = cv2.resize(dst_img, dim, interpolation = cv2.INTER_LINEAR)
        seg_src_img = cv2.resize(seg_src_img, dim, interpolation = cv2.INTER_NEAREST)
        seg_dst_img = cv2.resize(seg_dst_img, dim, interpolation = cv2.INTER_NEAREST)

        # cv2.imshow("src_img", src_img)
        # cv2.imshow("dst_img", dst_img)
        # cv2.imshow("seg_src_img", seg_src_img)
        # cv2.imshow("seg_dst_img", seg_dst_img)
        # cv2.waitKey(0)

        # WARPING
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
                output = getMakeupGroundTruth_warping(src_face, dst_img, seg_dst_img, src_points, dst_face["points"])
                # output = face_swap(src_face, dst_face["face"], src_points,
                #                 dst_face["points"], dst_face["shape"],
                #                 output)
            output = cv2.resize(output, dim, interpolation = cv2.INTER_AREA)
            images = [src_img, dst_img, output]
            new_im = cv2.hconcat(images)

            cv2.imshow("new image", new_im)
            cv2.waitKey(0)

            img_path = "D:/# Raras/GitHub/TA/groundtruth/coba/warping_{}.png".format(i)
            print(img_path)
            cv2.imwrite(img_path, new_im)

        # HISTOGRAM MATCHING
        img_path2 = "D:/# Raras/GitHub/TA/groundtruth/coba/matching_{}.png".format(i)
        result = getMakeupGroundTruth_histogram(dst_img, seg_dst_img, src_img, seg_src_img)

        face_true = result[0]
        brow_true = result[1]
        eye_true = result[2]
        lip_true = result[3]

        brow_true[brow_true == 0] = lip_true[brow_true == 0]
        brow_true[brow_true == 0] = eye_true[brow_true == 0]
        brow_true[brow_true == 0] = face_true[brow_true == 0]
        brow_true[brow_true == 0] = dst_img[brow_true == 0]

        cv2.imshow("src", src_img)
        cv2.imshow("dst", dst_img)
        cv2.imshow("result histogram", brow_true)
        cv2.waitKey(0)