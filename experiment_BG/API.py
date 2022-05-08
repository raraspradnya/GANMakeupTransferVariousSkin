import cv2
import numpy as np
import tensorflow as tf
import tensorflow_addons as tfa

class FaceBeautyModel(object):
    def __init__(self, path):
        # self.generator_model_path = "../export_models/BG150/Generator.h5"
        self.generator_model_path = path
        self.Generator = self.loadGenerator()
        self.Generator.compile()

        self.image_size = (256, 256)
        pass

    def loadGenerator(self):
        return tf.keras.models.load_model(self.generator_model_path, custom_objects = {'InstanceNormalization':tfa.layers.InstanceNormalization})

    def transfer(self, images, makeup_images, predict_batch  = 10):
        transfer_images = self.Generator.predict([images, makeup_images], batch_size = predict_batch)
        images = self.postprocessingImages(images)
        makeup_images = self.postprocessingImages(makeup_images)
        transfer_images = self.postprocessingImages(transfer_images)
        return images, makeup_images, transfer_images

    def preprocessingImages(self, images):
        images = np.array(images, dtype = np.float32)
        images = (images / 255.0  - 0.5) * 2
        return images

    def postprocessingImages(self, images):
        images = np.array(images, dtype = np.float32)
        images = (images / 2 + 0.5) * 255.0
        return images