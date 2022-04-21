import os
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import cv2
from datetime import datetime
import math

import sys
sys.path.append('../')
from groundtruth.face_detection import select_face, select_all_faces
from groundtruth.face_swap import face_blend, face_copy

fig = plt.figure(figsize=(10,15))
def save_images(epoch, step, origin_images, pred_images, reference_images, pic_save_path):
    pos = 0
    for i in range(2):
        # origin image
        rgb_image = (origin_images[i][...,::-1] / 2 + 0.5) * 255.0 
        rgb_image = rgb_image.astype(np.uint8)
        plt.subplot(2, 3, pos+1)
        plt.imshow(rgb_image)
        plt.axis('off')

        # pred image
        rgb_image = (pred_images[i][...,::-1] / 2 + 0.5) * 255.0
        rgb_image = rgb_image.astype(np.uint8)
        plt.subplot(2, 3, pos+2)
        plt.imshow(rgb_image)
        plt.axis('off')

        # reference image
        rgb_image = (reference_images[i][...,::-1] / 2 + 0.5) * 255.0
        rgb_image = rgb_image.astype(np.uint8)
        plt.subplot(2, 3, pos+3)
        plt.imshow(rgb_image)
        plt.axis('off')

        pos += 3
    save_path = os.path.join(pic_save_path, 'epoch_{:04d}_{:04d}.png'.format(epoch, step))
    save_path = os.path.normpath(save_path)
    plt.savefig(save_path)
    plt.clf()

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
    for i in range(batch_size):
        for c in range(3):
            s = source[i,...,c].ravel()
            r = reference[i,..., c].ravel()

            s_values, bin_idx, s_counts = np.unique(s, return_inverse=True, return_counts=True)
            r_values, r_counts = np.unique(r, return_counts=True)

            if(len(s_counts) == 1 or len(r_counts) == 1):
                continue
            # take the cumsum of the counts and normalize by the number of pixels to
            # get the empirical cumulative distribution functions for the source and
            # template images (maps pixel value --> quantile)
            s_quantiles = np.cumsum(s_counts[1:]).astype(np.float64)
            s_quantiles /= s_quantiles[-1]
            r_quantiles = np.cumsum(r_counts[1:]).astype(np.float64)
            r_quantiles /= r_quantiles[-1]
            r_values = r_values[1:]

            # interpolate linearly to find the pixel values in the template image
            # that correspond most closely to the quantiles in the source image
            interp_value = np.zeros_like(s_values, dtype = np.float32)
            interp_r_values = np.interp(s_quantiles, r_quantiles, r_values)
            interp_value[1:] = interp_r_values
            result[i,...,c] = interp_value[bin_idx].reshape(oldshape[1:3])
    result = np.array(result, dtype=np.float32)
    return result

def warping_blend(source, reference):
    '''
        Warp reference (makeup face) to source image (non-makeup face)
        input:
            source : np.ndarray
                original non-makeup face
            reference : np.ndarray
                segmented makeup face (only whole_face part of the image)
        output:
            result : np.ndarray
                makeup face already warped to original non-makeup face
    '''
    oldshape = source.shape
    batch_size = oldshape[0]
    source = np.array(source, dtype = np.uint8)
    reference = np.array(reference, dtype = np.uint8)

    # counts
    result = np.zeros(oldshape, dtype = np.uint8)
    for i in range(batch_size):
        s = source[i]
        r = reference[i]
        ref_points, ref_shape, ref_face, check = select_face(r)
        if (check ==  0):
            return source
        src_faceBoxes = select_all_faces(s)
        output = s
        for k, src_face in src_faceBoxes.items():
            result[i] = (face_blend(ref_face, src_face["face"], ref_points,
                            src_face["points"], src_face["shape"],
                            output))
    result = np.array(result, dtype = np.float32)
    return result


def warping_copy(source, reference, mask_copy):
    '''
        Warp reference (makeup face) to source image (non-makeup face)
        input:
            source : np.ndarray
                original non-makeup face
            reference : np.ndarray
                segmented makeup face (only whole_face part of the image)
        output:
            result : np.ndarray
                makeup face already warped to original non-makeup face
    '''
    oldshape = source.shape
    batch_size = oldshape[0]
    source = np.array(source, dtype = np.uint8)
    reference = np.array(reference, dtype = np.uint8)
    mask_copy = np.array(mask_copy, dtype = np.uint8)

    # counts
    result = np.zeros(oldshape, dtype = np.uint8)
    for i in range(batch_size):
        s = source[i]
        r = reference[i]
        copy = mask_copy[i]
        ref_points, ref_shape, ref_face, check = select_face(r)
        if (check ==  0):
            return source
        src_faceBoxes = select_all_faces(s)
        output = s
        for k, src_face in src_faceBoxes.items():
            result[i] = (face_copy(ref_face, src_face["face"], ref_points,
                            src_face["points"], src_face["shape"],
                            output, copy))
    result = np.array(result, dtype = np.float32)
    return result

def get_eye_region(size, mask_l, mask_r):
    masks = []
    batch_size = mask_l.shape[0]
    for i in range (batch_size):
        mask = np.zeros(size.numpy(), np.uint8)
        mask_left = np.array(mask_l[i].numpy(), np.uint8)
        mask_right = np.array(mask_r[i].numpy(), np.uint8)

        mask_left[mask_left > 0] = 255
        mask_right[mask_right > 0] = 255

        ret, threshl = cv2.threshold(mask_left, 50, 255, cv2.THRESH_BINARY)
        ret, threshr = cv2.threshold(mask_right, 50, 255, cv2.THRESH_BINARY)
        contoursl, hierarchy = cv2.findContours(threshl, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        contoursr, hierarchy = cv2.findContours(threshr, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        hull_l = []
        hull_r = []
        for i in range(len(contoursl)):
            hull_l.append(cv2.convexHull(contoursl[i], False))
        for i in range(len(contoursr)):
            hull_r.append(cv2.convexHull(contoursr[i], False))

        if (len(hull_l) > 0):
            r_l = cv2.boundingRect(hull_l[0])
            center_l = ((r_l[0] + int(r_l[2] / 2), r_l[1] + int(r_l[3] / 2)))
            r_left = int(math.sqrt(r_l[2]**2 + r_l[3]**2) * 0.8)
            circle_l = cv2.circle(mask, center_l, r_left, (255, 255, 255), -1)
        if (len(hull_r) > 0):
            r_r = cv2.boundingRect(hull_r[0])
            center_r = ((r_r[0] + int(r_r[2] / 2), r_r[1] + int(r_r[3] / 2)))
            r_right = int(math.sqrt(r_r[2]**2 + r_r[3]**2) * 0.8)
            circle_r = cv2.circle(mask, center_r, r_right, (255, 255, 255), -1)
        masks.append(mask)
    result = np.array(masks, dtype = np.float32)
    return result

@tf.function
def masking_func(mask, classes_list):
    mask = tf.cast(mask, dtype = tf.int32)
    cum_mask = tf.zeros_like(mask, dtype = tf.int32)
    for i in classes_list:
        cum_mask +=  tf.cast(mask == i, dtype = tf.int32)
    cum_mask = tf.cast(tf.clip_by_value(cum_mask, 0, 1), dtype = tf.float32)
    return cum_mask

@tf.function
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

def log(epoch, gen_loss, dis_loss_X, dis_loss_Y, loss_list, loss_save_path):
    file_writer = tf.summary.create_file_writer(loss_save_path)
    file_writer.set_as_default()
    tf.summary.scalar('Generator Loss', gen_loss.numpy(), step=epoch)
    tf.summary.scalar('Discriminator X Loss', dis_loss_X.numpy(), step=epoch)
    tf.summary.scalar('Discriminator Y Loss', dis_loss_Y.numpy(), step=epoch)
    tf.summary.scalar('Adversarial Loss', loss_list[0].numpy(), step=epoch)
    tf.summary.scalar('Cycle Consistency Loss', loss_list[1].numpy(), step=epoch)
    tf.summary.scalar('Perceptual Loss', loss_list[2].numpy(), step=epoch)
    tf.summary.scalar('Makeup Loss', loss_list[3].numpy(), step=epoch)
    tf.summary.scalar('Background Loss', loss_list[4].numpy(), step=epoch)