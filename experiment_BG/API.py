import cv2
import numpy as np
import tensorflow as tf
import tensorflow_addons as tfa

class FaceBeautyModel(object):
    def __init__(self):
        self.generator_model_path = "../export_models/BG137_FIX/Generator.h5"
        self.Generator = self.loadGenerator()
        self.Generator.compile()

        self.image_size = (256, 256)
        pass

    def loadGenerator(self):
        return tf.keras.models.load_model(self.generator_model_path, custom_objects = {'InstanceNormalization':tfa.layers.InstanceNormalization})

    def transfer(self, images, makeup_images, predict_batch  = 10):
        images = self.preprocessingImages(images)
        makeup_images = self.preprocessingImages(makeup_images)

        transfer_images = self.Generator.predict([images, makeup_images], batch_size = predict_batch)
        transfer_images = self.postprocessingImages(transfer_images)
        return transfer_images

    def preprocessingImages(self, images):
        for i in range(len(images)):
            images[i] = cv2.resize(images[i], self.image_size)
        images = np.array(images, dtype = np.float32)
        images = (images / 255.0  - 0.5) * 2
        return images

    def postprocessingImages(self, images):
        images = np.array(images, dtype = np.float32)
        images = (images / 2 + 0.5) * 255.0
        return images