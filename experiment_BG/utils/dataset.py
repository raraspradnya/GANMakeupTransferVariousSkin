import cv2
import glob
import numpy as np
import imgaug as ia
import tensorflow as tf
from imgaug import augmenters as dataAug
from imgaug.augmentables.segmaps import SegmentationMapsOnImage
from datetime import datetime


from utils.helper import masking_func, warping_blend, warping_copy, get_eye_region

NUM_PARALLEL_CALLS = 2
SHUFFLE_BUFFER_SIZE = 2000
PREFETCH_BUFFER_SIZE = 20

class Dataset(object):
    def __init__(self, dataset_path, image_shape, classes, batch_size, dataset_size, isTraining):
        self.image_shape = image_shape
        self.classes = classes
        self.batch_size = batch_size
        self.isTraining = isTraining
        self.dataset_size = dataset_size

        self.dataset_path = {}
        for name in dataset_path.keys():
            self.dataset_path[name] = glob.glob(dataset_path[name] + "\\*.tfrecords", recursive=True)

        if(isTraining):
            self.aug_seq = self.getAugParam()

    def __len__(self):
        return max(self.dataset_size)

    def getAugParam(self):
        '''
        Set the parameters for data augmentation. 
        '''
        seq = dataAug.Sequential([
            dataAug.Fliplr(0.5),
            #dataAug.CropAndPad(percent=(-0.25, 0.25)),
            ], random_order=True) # apply augmenters in random order
        return seq

    @tf.function
    def getData(self, example):
        '''
        Parse an example to a data.

        Args:
            example : a tfrecord example.

        Returns:
            A dictionary that contains a image and a mask.
        '''
        # parse
        parsed_features = tf.io.parse_single_example(example,features={
                                            'image': tf.io.FixedLenFeature([], tf.string, default_value=""),
                                            'mask': tf.io.FixedLenFeature([], tf.string, default_value=""),
                                            'im_height': tf.io.FixedLenFeature([], tf.int64),
                                            'im_width': tf.io.FixedLenFeature([], tf.int64),
                                            'm_height': tf.io.FixedLenFeature([], tf.int64),
                                            'm_width': tf.io.FixedLenFeature([], tf.int64),
                                            })
        h, w, c = self.image_shape[:3]
        im_h = tf.cast(parsed_features['im_height'], tf.float32)
        im_w = tf.cast(parsed_features['im_width'], tf.float32)
        m_h = tf.cast(parsed_features['m_height'], tf.float32)
        m_w = tf.cast(parsed_features['m_width'], tf.float32)

        # decode
        image = tf.io.decode_raw(parsed_features['image'], tf.uint8)
        image = tf.reshape(image, [im_h, im_w, c])
        mask = tf.io.decode_raw(parsed_features['mask'], tf.uint8)
        mask = tf.reshape(mask, [m_h, m_w, 1])

        # resize
        image = tf.image.resize(image, size = (h, w), method = tf.image.ResizeMethod.BILINEAR)
        mask = tf.image.resize(mask, size = (h, w), method = tf.image.ResizeMethod.NEAREST_NEIGHBOR)

        data = {"images" : image, "images_mask" : mask}
        return data

    def _imgaug(self, images, masks):
        '''
        Perform data augmentation by imgaug library

        Args:
            images : a batch of images.
            masks : a batch of masks.

        Returns:
            A batch of images and a batch of masks.
        '''
        images = np.array(images, dtype = np.uint8)
        masks = np.array(masks, dtype = np.int32)

        seg_maps = []
        for i in range(len(masks)):
            seg_maps.append(SegmentationMapsOnImage(masks[i], shape=images[i].shape))

        images, seg_maps = self.aug_seq(images = images, segmentation_maps=seg_maps)

        for i in range(len(masks)):
            masks[i] = seg_maps[i].get_arr()
        masks = np.array(masks, dtype = np.int32)
        return images, masks

    @tf.function
    def augmentData(self, data):
        '''
        Perform data augmentation in tensorflow.
        '''
        h, w, c = self.image_shape[:3]

        images = tf.cast(data["images"], tf.uint8)        
        images_mask = tf.cast(data["images_mask"], tf.int32)

        # data augmentation
        images, images_mask = tf.py_function(self._imgaug, inp=[images, images_mask], Tout = [tf.uint8, tf.int32])
        # set shape
        images.set_shape((self.batch_size, h , w, c))
        images_mask.set_shape((self.batch_size, h , w, 1))

        data = {"images" : images, "images_mask" : images_mask}
        return data
    
    @tf.function
    def preprocessData(self, data1, data2):
        '''
        Preprocessing a batch of data.

        Args:
            data1 : a batch of source data.
            data2 : a batch of reference data.

        Returns:
            A dictionary that contains the data information for training.
        '''
        h, w, c = self.image_shape[:3]
        images1 = data1["images"] 
        images1_mask = data1["images_mask"]
        images2 = data2["images"]
        images2_mask = data2["images_mask"]  
        # resize
        images1 = tf.image.resize(images1, size = (h, w), method = tf.image.ResizeMethod.BILINEAR)
        images1_mask = tf.image.resize(images1_mask, size = (h, w), method = tf.image.ResizeMethod.NEAREST_NEIGHBOR)
        images2 = tf.image.resize(images2, size = (h, w), method = tf.image.ResizeMethod.BILINEAR)
        images2_mask = tf.image.resize(images2_mask, size = (h, w), method = tf.image.ResizeMethod.NEAREST_NEIGHBOR)

        # Preprocessing
        # Processing mask with histogram matching
        makeup_true, makeup_masks = self.getMakeupGroundTruth(images1, images1_mask, images2, images2_mask)

        # normalize
        images1 = self.preprocess(images1)
        images1_mask = tf.cast(images1_mask, dtype = tf.float32)
        images2 = self.preprocess(images2)
        images2_mask = tf.cast(images2_mask, dtype = tf.float32)

        # processing makeup mask
        makeup_mask1 = tf.abs(1.0-masking_func(images1_mask, tf.constant(self.classes["non-makeup"], dtype = tf.int32)))
        makeup_mask1 = tf.clip_by_value(tf.cast(makeup_mask1, dtype = tf.float32), 0, 1)
        background_mask1 = tf.abs(masking_func(images1_mask, tf.constant(self.classes["non-makeup"], dtype = tf.int32)))
        background_mask1 = tf.clip_by_value(tf.cast(background_mask1, dtype = tf.float32), 0, 1)

        # return data
        data = {'images1' : images1, 'images2' : images2}
        labels = {'images1' : images1, 'images2' : images2, 'background_mask1' : background_mask1, 'makeup_mask' : makeup_mask1,
                'face_true' : makeup_true[0], 'brow_true' : makeup_true[1], 'eye_true' : makeup_true[2], 'lip_true' : makeup_true[3],
                'face_mask' : makeup_masks[0], 'brow_mask' : makeup_masks[1], 'eye_mask' : makeup_masks[2], 'lip_mask' : makeup_masks[3],
                }
        return data, labels

    def flow(self):
        '''
        TFRecordDataset flow.
        '''
        # read the files in parallel
        dataset1 = tf.data.TFRecordDataset(self.dataset_path['source'])
        dataset2 = tf.data.TFRecordDataset(self.dataset_path['reference'])

        # get a data
        dataset1 = dataset1.map(map_func = self.getData, num_parallel_calls = NUM_PARALLEL_CALLS)
        dataset2 = dataset2.map(map_func = self.getData, num_parallel_calls = NUM_PARALLEL_CALLS)

        # repeat and shuffle
        if(self.isTraining):
            dataset1 = dataset1.repeat(3)
            dataset1 = dataset1.shuffle(buffer_size = SHUFFLE_BUFFER_SIZE)
            dataset2 = dataset2.repeat()
            dataset2 = dataset2.shuffle(buffer_size = SHUFFLE_BUFFER_SIZE)
    
        # batch
        dataset1 = dataset1.batch(batch_size = self.batch_size)
        dataset1 = dataset1.prefetch(PREFETCH_BUFFER_SIZE)
        dataset2 = dataset2.batch(batch_size = self.batch_size)
        dataset2 = dataset2.prefetch(PREFETCH_BUFFER_SIZE)

        # augment a batch data
        if(self.isTraining):
            dataset1 = dataset1.map(map_func = self.augmentData, num_parallel_calls = NUM_PARALLEL_CALLS)
            dataset2 = dataset2.map(map_func = self.augmentData, num_parallel_calls = NUM_PARALLEL_CALLS)

        # prerocessing a batch of data
        dataset = tf.data.Dataset.zip((dataset1, dataset2)) 
        dataset = dataset.map(map_func = self.preprocessData, num_parallel_calls = NUM_PARALLEL_CALLS)
        return dataset

    def preprocess(self, image):
        '''
        Normalize image array.
        '''
        return (tf.cast(image, tf.float32) / 255.0 - 0.5) * 2

    @tf.function
    def getMakeupGroundTruth(self, images, masks, reference_images, reference_masks):
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
        h, w, c = self.image_shape[:3]

        # get reference mask of each reference makeup region
        r_whole_face_masks = masking_func(reference_masks, tf.constant(self.classes["whole_face"], dtype = tf.int32))
        r_whole_face = reference_images * r_whole_face_masks

        # get source mask of each source makeup region
        hair_masks = masking_func(masks, tf.constant(self.classes["hair"], dtype = tf.int32))
        face_masks = masking_func(masks, tf.constant(self.classes["face"], dtype = tf.int32))
        brow_masks = masking_func(masks, tf.constant(self.classes["brow"], dtype = tf.int32))
        eyeball_masks_l = masking_func(masks, tf.constant(self.classes["eyeball_l"], dtype = tf.int32))
        eyeball_masks_r = masking_func(masks, tf.constant(self.classes["eyeball_r"], dtype = tf.int32))
        lip_masks = masking_func(masks, tf.constant(self.classes["lip"], dtype = tf.int32))
        eye_masks = tf.py_function(get_eye_region, inp=[(h, w), eyeball_masks_l, eyeball_masks_r], Tout=tf.float32)
        eye_masks = tf.expand_dims(eye_masks, -1)

        # Get ground truth
        mask_copy = tf.clip_by_value(eye_masks + brow_masks + lip_masks, 0, 1)
        whole_face_blend = tf.py_function(warping_blend, inp=[images, r_whole_face], Tout = tf.float32)
        whole_face_copy = tf.py_function(warping_copy, inp=[images, r_whole_face, mask_copy], Tout = tf.float32)
        eye_masks2 = tf.clip_by_value(eye_masks - eyeball_masks_l - eyeball_masks_r, 0, 1)

        face_true = whole_face_blend * face_masks
        brow_true = whole_face_copy * brow_masks
        eye_true = whole_face_copy * eye_masks2
        lip_true = whole_face_copy * lip_masks

        face_true.set_shape((self.batch_size, h , w, c))
        brow_true.set_shape((self.batch_size, h , w, c))
        eye_true.set_shape((self.batch_size, h , w, c))
        lip_true.set_shape((self.batch_size, h , w, c))

        # normalize ground truth
        face_true = self.preprocess(face_true)
        brow_true = self.preprocess(brow_true)
        eye_true = self.preprocess(eye_true)
        lip_true = self.preprocess(lip_true)

        # prevent background form changing
        face_true = face_true * face_masks
        brow_true = brow_true * brow_masks
        eye_true = eye_true * eye_masks2
        lip_true = lip_true * lip_masks
        return [face_true, brow_true, eye_true, lip_true], [face_masks, brow_masks, eye_masks2, lip_masks]