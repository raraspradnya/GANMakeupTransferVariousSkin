#! /usr/bin/env python
import os
import cv2
import argparse
import tensorflow as tf
import numpy as np
from skimage.exposure import match_histograms

from face_detection import select_face, select_all_faces
from face_swap import face_swap
from face_parsing import get_face

def eye_regions_func(mask):
    vertical = tf.clip_by_value(tf.reduce_sum(mask, axis = 1, keepdims = True), 0, 1)
    horizontal = tf.clip_by_value(tf.reduce_sum(mask, axis = 2, keepdims = True), 0, 1)
    vertical_n = tf.clip_by_value(tf.cumsum(vertical, axis = 2), 0, 1)
    vertical_r = tf.clip_by_value(tf.cumsum(vertical, axis = 2, reverse = True), 0, 1)
    horizontal_n = tf.clip_by_value(tf.cumsum(horizontal, axis = 1), 0, 1)
    horizontal_r = tf.clip_by_value(tf.cumsum(horizontal, axis = 1, reverse = True), 0, 1)
    vertical = vertical_n * vertical_r
    horizontal = horizontal_n * horizontal_r
    return (vertical * horizontal)

def hist_match_func(source, reference):
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
    Returns:
    -----------
        matched: np.ndarray
            The transformed output image
    """

    oldshape = source.shape
    batch_size = oldshape[0]
    source = np.array(source, dtype = np.uint8)
    reference = np.array(reference, dtype = np.uint8)
    # get the set of unique pixel values and their corresponding indices and
    # counts
    result = np.zeros(oldshape, dtype = np.uint8)
        # for c in range(3):
        #     s = source[i,...,c].ravel()
        #     r = reference[i,..., c].ravel()

        #     s_values, bin_idx, s_counts = np.unique(s, return_inverse=True, return_counts=True)
        #     r_values, r_counts = np.unique(r, return_counts=True)

        #     if(len(s_counts) == 1 or len(r_counts) == 1):
        #         continue
        #     # take the cumsum of the counts and normalize by the number of pixels to
        #     # get the empirical cumulative distribution functions for the source and
        #     # template images (maps pixel value --> quantile)
        #     s_quantiles = np.cumsum(s_counts[1:]).astype(np.float64)
        #     s_quantiles /= s_quantiles[-1]
        #     r_quantiles = np.cumsum(r_counts[1:]).astype(np.float64)
        #     r_quantiles /= r_quantiles[-1]
        #     r_values = r_values[1:]

        #     # interpolate linearly to find the pixel values in the template image
        #     # that correspond most closely to the quantiles in the source image
        #     interp_value = np.zeros_like(s_values, dtype = np.float32)
        #     interp_r_values = np.interp(s_quantiles, r_quantiles, r_values)
        #     interp_value[1:] = interp_r_values
        #     result[i,...,c] = interp_value[bin_idx].reshape(oldshape[1:3])
    result = match_histograms(source[i], reference[i], multichannel=True)
    result = np.array(result, dtype=np.float32)
    return result


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

def preprocess(image):
    '''
    Normalize image array.
    '''
    return (tf.cast(image, tf.float32) / 255.0 - 0.5) * 2

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
    h, w, c = (256, 256, 3)
    face = [1, 6, 11, 12, 13]
    brow = [2, 3]
    eye = [2, 3, 4, 5]
    lip = [7, 9]
    non_makeup = [0, 4, 5, 8, 10]
    hair = [10]
    whole_face = [1, 2, 3, 4, 5, 6, 7, 8, 9]
    batch_size = 1

    # masks = tf.convert_to_tensor([masks], dtype=tf.int32)
    # reference_images = tf.convert_to_tensor([reference_images], dtype=tf.int32)
    # reference_masks = tf.convert_to_tensor([reference_masks], dtype=tf.int32)

    # get source mask of each source makeup region
    hair_masks = masking_func(masks, tf.constant(hair, dtype = tf.int32))
    face_masks = masking_func(masks, tf.constant(face, dtype = tf.int32))
    brow_masks = masking_func(masks, tf.constant(brow, dtype = tf.int32))
    eye_masks = masking_func(masks, tf.constant(eye, dtype = tf.int32))
    eye_masks = tf.clip_by_value(eye_regions_func(eye_masks), 0, 1)
    eye_masks = tf.clip_by_value(eye_masks - hair_masks - brow_masks, 0, 1)
    lip_masks = masking_func(masks, tf.constant(lip, dtype = tf.int32))

    hair_masks = process_mask(hair_masks)
    face_masks = process_mask(face_masks)
    brow_masks = process_mask(brow_masks)
    eye_masks = process_mask(eye_masks)
    lip_masks = process_mask(lip_masks)
    
    face_src = cv2.bitwise_and(images, images, mask = face_masks)
    lip_src = cv2.bitwise_and(images, images, mask = lip_masks)
    eye_src = cv2.bitwise_and(images, images, mask = eye_masks)
    brow_src = cv2.bitwise_and(images, images, mask = brow_masks)

    cv2.imshow("face_src", face_src)
    cv2.imshow("lip_src", lip_src)
    cv2.imshow("eye_src", eye_src)
    cv2.imshow("brow_src", brow_src)
    cv2.waitKey(0)  
    
    # get reference mask of each reference makeup region
    r_hair_masks = masking_func(reference_masks, tf.constant(hair, dtype = tf.int32))
    r_face_masks = masking_func(reference_masks, tf.constant(face, dtype = tf.int32))
    r_brow_masks = masking_func(reference_masks, tf.constant(brow, dtype = tf.int32))
    r_eye_masks = masking_func(reference_masks, tf.constant(eye, dtype = tf.int32))
    r_eye_masks = tf.clip_by_value(eye_regions_func(r_eye_masks) - r_eye_masks, 0, 1)
    r_eye_masks = tf.clip_by_value(r_eye_masks - r_hair_masks - r_brow_masks, 0, 1)
    r_lip_masks = masking_func(reference_masks, tf.constant(lip, dtype = tf.int32))
    
    r_hair_masks = process_mask(r_hair_masks)
    r_face_masks = process_mask(r_face_masks)
    r_brow_masks = process_mask(r_brow_masks)
    r_eye_masks = process_mask(r_eye_masks)
    r_lip_masks = process_mask(r_lip_masks)
    
    r_face = cv2.bitwise_and(images, images, mask = r_face_masks)
    r_lip = cv2.bitwise_and(images, images, mask = r_lip_masks)
    r_eye = cv2.bitwise_and(images, images, mask = r_eye_masks)
    r_brow = cv2.bitwise_and(images, images, mask = r_brow_masks)

    cv2.imshow("r_face", r_face)
    cv2.imshow("r_brow", r_brow)
    cv2.imshow("r_eye", r_eye)
    cv2.imshow("r_lip", r_lip)
    cv2.waitKey(0)    
    
    face_true = tf.py_function(hist_match_func, inp=[face_src, r_face], Tout = tf.float32)
    brow_true = tf.py_function(hist_match_func, inp=[brow_src, r_brow], Tout = tf.float32)
    eye_true = tf.py_function(hist_match_func, inp=[eye_src, r_eye], Tout = tf.float32)
    lip_true = tf.py_function(hist_match_func, inp=[lip_src, r_lip], Tout = tf.float32)

    cv2.imshow("face", face_true.numpy())
    cv2.imshow("brow", brow_true.numpy())
    cv2.imshow("eye", eye_true.numpy())
    cv2.imshow("lip", lip_true.numpy())
    cv2.waitKey(0)

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

    makeup.append("D:/# Raras/GitHub/TA/dataset/RawData/images/makeup/12.png")
    makeup.append("D:/# Raras/GitHub/TA/dataset/RawData/images/makeup/94.png")
    makeup.append("D:/# Raras/GitHub/TA/dataset/RawData/images/makeup/1635.png")
    makeup.append("D:/# Raras/GitHub/TA/dataset/RawData/images/makeup/103.png")
    seg_makeup.append("D:/# Raras/GitHub/TA/dataset/RawData/segs/makeup/12.png")
    seg_makeup.append("D:/# Raras/GitHub/TA/dataset/RawData/segs/makeup/94.png")
    seg_makeup.append("D:/# Raras/GitHub/TA/dataset/RawData/segs/makeup/1635.png")
    seg_makeup.append("D:/# Raras/GitHub/TA/dataset/RawData/segs/makeup/103.png")

    non_makeup.append("D:/# Raras/GitHub/TA/dataset/RawData/images/non-makeup/vSYYZ25.png")
    non_makeup.append("D:/# Raras/GitHub/TA/dataset/RawData/images/non-makeup/vSYYZ51.png")
    non_makeup.append("D:/# Raras/GitHub/TA/dataset/RawData/images/non-makeup/vSYYZ25.png")
    non_makeup.append("D:/# Raras/GitHub/TA/dataset/RawData/images/non-makeup/vSYYZ663.png")
    seg_non_makeup.append("D:/# Raras/GitHub/TA/dataset/RawData/segs/non-makeup/vSYYZ25.png")
    seg_non_makeup.append("D:/# Raras/GitHub/TA/dataset/RawData/segs/non-makeup/vSYYZ51.png")
    seg_non_makeup.append("D:/# Raras/GitHub/TA/dataset/RawData/segs/non-makeup/vSYYZ25.png")
    seg_non_makeup.append("D:/# Raras/GitHub/TA/dataset/RawData/segs/non-makeup/vSYYZ663.png")

    for i in range (len(makeup)):
        print("warping ", makeup[i], non_makeup[i])

        # Read images
        src_img = cv2.imread(makeup[i])
        dst_img = cv2.imread(non_makeup[i])
        seg_src_img = cv2.imread(seg_makeup[i])
        seg_dst_img = cv2.imread(seg_non_makeup[i])
        dim = (256,256)
        src_img = cv2.resize(src_img, dim, interpolation = cv2.INTER_AREA)
        dst_img = cv2.resize(dst_img, dim, interpolation = cv2.INTER_AREA)
        seg_src_img = cv2.resize(seg_src_img, dim, interpolation = cv2.INTER_AREA)
        seg_dst_img = cv2.resize(seg_dst_img, dim, interpolation = cv2.INTER_AREA)

        # cv2.imshow("src_img", src_img)
        # cv2.imshow("dst_img", dst_img)
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
                output = face_swap(src_face, dst_face["face"], src_points,
                                dst_face["points"], dst_face["shape"],
                                output)
        
            # cv2.imshow("src_img",src_img)
            # cv2.imshow("dst_img",dst_img)
            # cv2.imshow("output",output)
            # cv2.waitKey(0)
            output = cv2.resize(output, dim, interpolation = cv2.INTER_AREA)
            images = [src_img, dst_img, output]
            new_im = cv2.hconcat(images)

            img_path = "D:/# Raras/GitHub/TA/groundtruth/coba/warping_{}.png".format(i)
            print(img_path)
            cv2.imwrite(img_path, new_im)

        # # HISTOGRAM MATCHING
        # img_path2 = "D:/# Raras/GitHub/TA/groundtruth/coba/matching_{}.png".format(i)
        # result = getMakeupGroundTruth_histogram(src_img, seg_src_img, dst_img, seg_dst_img)
        # cv2.imshow("face", result[0].numpy())
        # cv2.waitKey(0)
