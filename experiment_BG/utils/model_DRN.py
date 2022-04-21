import os
import cv2
import shutil
import threading
from numpy import source
import tensorflow as tf
from datetime import datetime
import pandas as pd

from utils.module import *
from utils.losses import *
from utils.helper import *

class Model_DRN(object):
    def __init__(self, input_shape, logs_path, batch_size = 32, classes = None):
        self.input_shape = input_shape
        self.batch_size = batch_size
        self.classes = classes
        self.logs_path = logs_path
        self.model_path = os.path.join(self.logs_path, 'checkpoint')
        self.pic_save_path = os.path.join(logs_path, 'save_pic')
        self.gt_save_path = os.path.join(logs_path, 'save_gt')
        self.bg_save_path = os.path.join(logs_path, 'save_bg')
        self.loss_save_path = os.path.join(logs_path, 'metrics')

        # Define optimizer
        learning_rate_fn_1 = tf.keras.optimizers.schedules.PolynomialDecay(2e-4, 155250, 1e-6, power=1.0)
        self.model_optimizer = tf.keras.optimizers.Adam(learning_rate_fn_1, beta_1 = 0.5, beta_2=0.999)
        learning_rate_fn_2 = tf.keras.optimizers.schedules.PolynomialDecay(2e-4, 155250, 1e-6, power=1.0)
        self.discriminatorX_optimizer = tf.keras.optimizers.Adam(learning_rate_fn_2, beta_1 = 0.5, beta_2=0.999)
        learning_rate_fn_3 = tf.keras.optimizers.schedules.PolynomialDecay(2e-4, 155250, 1e-6, power=1.0)
        self.discriminatorY_optimizer = tf.keras.optimizers.Adam(learning_rate_fn_3, beta_1 = 0.5, beta_2=0.999)

        # Build Model Architecure
        vgg16_model = tf.keras.applications.vgg16.VGG16(include_top=False, weights = 'imagenet')
        vgg16_model.trainable = False
        self.feature_model = tf.keras.models.Model(inputs=vgg16_model.input, outputs=vgg16_model.get_layer('block4_conv1').output)
        self.feature_model.trainable = False        
        del vgg16_model

        self.generator = self.build_generator()
        self.model = self.build()
        self.discriminator_X = self.build_discriminator_X()
        self.discriminator_Y = self.build_discriminator_Y()

    @tf.function
    def loss_function(self, pred, labels, dis_result):
        # Get labels
        source_image, reference_image = labels["images1"], labels["images2"]
        background_mask1 = labels["background_mask1"]
        face_true, brow_true, eye_true, lip_true = labels["face_true"], labels["brow_true"], labels["eye_true"], labels["lip_true"]
        face_mask, brow_mask, eye_mask, lip_mask = labels["face_mask"], labels["brow_mask"], labels["eye_mask"], labels["lip_mask"]
    
        # Get predicted images
        transfer_image, demakeup_image  = pred["image"][0], pred["image"][1]
        cycle_reference_image, cycle_source_image  = pred["cycle_image"][0], pred["cycle_image"][1]
        
        # ADVERSARIAL LOSS
        # Get discriminator results
        fake_source = dis_result[0]
        fake_reference = dis_result[1]
        # Count loss
        alpha = 1
        adversarial_loss = Adversarial_loss_G(fake_source, fake_reference)
        adversarial_loss = adversarial_loss * alpha

        # CYCLE CONSISTENCY loss
        beta = 10
        cycle_consistency_loss = Cycle_consistency_loss(source_image, reference_image, cycle_source_image, cycle_reference_image)
        cycle_consistency_loss = cycle_consistency_loss * beta

        # PERCEPTUAL LOSS
        # Get high level feature
        feature_true = self.feature_model(source_image)
        feature = self.feature_model(transfer_image)
        # Count loss
        gamma = 0.005
        perceptual_loss = Perceptual_loss(feature_true, feature)
        perceptual_loss = perceptual_loss * gamma

        # Makeup loss
        makeup_loss = Makeup_loss(y_true = [face_true, brow_true, eye_true, lip_true], y_pred_image = transfer_image, y_mask = [face_mask, brow_mask, eye_mask, lip_mask], classes = self.classes)
        makeup_loss = makeup_loss * 10

        # Background loss
        source_background = source_image * background_mask1
        transfer_background = transfer_image * background_mask1
        background_loss = Background_loss(source_background, transfer_background)
        background_loss = background_loss * 10

        loss = adversarial_loss + cycle_consistency_loss + perceptual_loss + makeup_loss + background_loss
        return loss, [adversarial_loss, cycle_consistency_loss, perceptual_loss, makeup_loss, background_loss], [source_background, transfer_background]


    def build_generator(self):
        print("[Model DRN] Building Generator....")
        image_source = tf.keras.layers.Input(self.input_shape)
        image_reference = tf.keras.layers.Input(self.input_shape)

        # DOWNSAMPLING BRANCH
        # SOURCE BRANCH
        source = Conv2D_layer(image_source, filters = 64, kernel_size = (7, 7))
        source = ReLU_layer(source)

        # REFERENCE BRANCH
        reference = Conv2D_layer(image_reference, filters = 64, kernel_size = (7, 7))
        reference = ReLU_layer(reference)

        # CONCATENATE
        x = Concatenate_layer(source, reference, axis = 3)

        # DOWNSAMPLING
        x = Conv2D_layer(x, filters = 128, kernel_size = (3, 3))
        x = ReLU_layer(x)
        x = Dropout_layer(x, rate=0.2)
        x = Conv2D_layer(x, filters = 128, kernel_size = (3, 3))

        # DILATED RESIDUAL BLOCK - 1 
        x = Dilated_layer(x, filters = 128, kernel_size = (3, 3), dilation = 2)
        x = ReLU_layer(x)
        x = Dropout_layer(x, rate=0.2)
        x = Dilated_layer(x, filters = 128, kernel_size = (3, 3), dilation = 2)

        # DILATED RESIDUAL BLOCK - 2
        x = Dilated_layer(x, filters = 128, kernel_size = (3, 3), dilation = 4)
        x = ReLU_layer(x)
        x = Dropout_layer(x, rate=0.2)
        x = Dilated_layer(x, filters = 128, kernel_size = (3, 3), dilation = 4)

        # DILATED RESIDUAL BLOCK - 3
        x = Dilated_layer(x, filters = 128, kernel_size = (3, 3), dilation = 2)
        x = ReLU_layer(x)

        # OUTPUT
        res_source = Conv2D_layer(x, filters = 64, kernel_size = (3, 3))
        res_source = ReLU_layer(res_source)
        res_source = Conv2D_layer(res_source, filters = 3, kernel_size = (7, 7))
        res_source = tf.nn.tanh(res_source)

        res_reference = Conv2D_layer(x, filters = 64, kernel_size = (3, 3))
        res_reference = ReLU_layer(res_reference)
        res_reference = Conv2D_layer(res_reference, filters = 3, kernel_size = (7, 7))
        res_reference = tf.nn.tanh(res_reference)

        model = tf.keras.Model(inputs = [image_source, image_reference], outputs = [res_source, res_reference])
        model.summary()
        return model

    def build_discriminator_X(self):
        print("[Model DRN] Building Discriminator X....")
        image = tf.keras.layers.Input(self.input_shape)

        x = image
        x = SpectrumNormalization_Conv2D_Layer(x, filters = 64, kernel_size = (4, 4), strides = (2, 2))
        x = LeakyReLU_layer(x)
        x = SpectrumNormalization_Conv2D_Layer(x, filters = 128, kernel_size = (4, 4), strides = (2, 2))
        x = LeakyReLU_layer(x)
        x = SpectrumNormalization_Conv2D_Layer(x, filters = 256, kernel_size = (4, 4), strides = (2, 2))
        x = LeakyReLU_layer(x)
        x = SpectrumNormalization_Conv2D_Layer(x, filters = 512, kernel_size = (4, 4))
        x = LeakyReLU_layer(x)
        x = Conv2D_layer(x, filters = 1, kernel_size = (3, 3))
        x = tf.nn.sigmoid(x)

        model = tf.keras.Model(inputs = image, outputs = x)
        model.summary()
        return model

    def build_discriminator_Y(self):
        print("[Model DRN] Building Discriminator Y....")
        image = tf.keras.layers.Input(self.input_shape)

        x = image
        x = SpectrumNormalization_Conv2D_Layer(x, filters = 64, kernel_size = (4, 4), strides = (2, 2))
        x = LeakyReLU_layer(x)
        x = SpectrumNormalization_Conv2D_Layer(x, filters = 128, kernel_size = (4, 4), strides = (2, 2))
        x = LeakyReLU_layer(x)
        x = SpectrumNormalization_Conv2D_Layer(x, filters = 256, kernel_size = (4, 4), strides = (2, 2))
        x = LeakyReLU_layer(x)
        x = SpectrumNormalization_Conv2D_Layer(x, filters = 512, kernel_size = (4, 4))
        x = LeakyReLU_layer(x)
        x = Conv2D_layer(x, filters = 1, kernel_size = (3, 3))
        x = tf.nn.sigmoid(x)

        model = tf.keras.Model(inputs = image, outputs = x)
        model.summary()
        return model

    def build(self, isLoadWeight = False):
        print("[Model DRN] Building Train Model....")
        # Training flow
        image1 = tf.keras.layers.Input(self.input_shape, name = "images1")
        image2 = tf.keras.layers.Input(self.input_shape, name = "images2") 

        # transfer image
        transfer_images, demakeup_images = self.generator([image1, image2])

        # cycle consistency
        cycle_reference, cycle_source  = self.generator([demakeup_images, transfer_images])

        # Loss Function
        pred = {"image" : [transfer_images, demakeup_images],
                "cycle_image" : [cycle_reference, cycle_source]
        }

        # Model
        model = tf.keras.Model(inputs = [image1, image2], outputs = pred)
        model.summary()
        return model

    @tf.function
    def train_step(self, features, labels):
        with tf.GradientTape() as gen_tape, tf.GradientTape() as disc_tapeX, tf.GradientTape() as disc_tapeY:
            pred = self.model(features, training=True)
            real_source = self.discriminator_X(features["images1"], training=True)
            fake_source = self.discriminator_X(pred["image"][1], training=True)
            real_reference = self.discriminator_Y(features["images2"], training=True)
            fake_reference = self.discriminator_Y(pred["image"][0], training=True)

            gen_loss, loss_list, bg_images = self.loss_function(pred, labels, [fake_source, fake_reference])
            dis_loss_X = Adversarial_loss_D(real_source, fake_source)
            dis_loss_Y = Adversarial_loss_D(real_reference, fake_reference) 

        gradients_of_generator = gen_tape.gradient(gen_loss, self.model.trainable_variables)
        gradients_of_discriminator_X = disc_tapeX.gradient(dis_loss_X, self.discriminator_X.trainable_variables)
        gradients_of_discriminator_Y = disc_tapeY.gradient(dis_loss_Y, self.discriminator_Y.trainable_variables)

        self.model_optimizer.apply_gradients(zip(gradients_of_generator, self.model.trainable_variables))
        self.discriminatorX_optimizer.apply_gradients(zip(gradients_of_discriminator_X, self.discriminator_X.trainable_variables))
        self.discriminatorY_optimizer.apply_gradients(zip(gradients_of_discriminator_Y, self.discriminator_Y.trainable_variables))
        return gen_loss, dis_loss_X, dis_loss_Y, pred["image"], bg_images, loss_list

    def train(self, train_dataset, epochs = 500, pretrained_model_path = None):
        os.makedirs(self.logs_path, exist_ok = True)
        os.makedirs(self.model_path, exist_ok = True)
        os.makedirs(self.pic_save_path, exist_ok = True)
        os.makedirs(self.gt_save_path, exist_ok = True)
        os.makedirs(self.bg_save_path, exist_ok = True)
        os.makedirs(self.loss_save_path, exist_ok = True)

        # Load dataset
        train_step = len(train_dataset) // self.batch_size
        dataset = train_dataset.flow()

        current_epoch = 0
        # Load Pretrain Model
        if(pretrained_model_path != None):
            current_epoch = self.load_model(self.model, pretrained_model_path)

        print("[Model DRN] Training....")

        for epoch in range(epochs):
            step = 0
            epoch_num = epoch + 1 + current_epoch
            
            for batch_features, batch_labels in dataset:

                gen_loss, dis_loss_X, dis_loss_Y, transfer_image, bg_images, loss_list = self.train_step(batch_features, batch_labels)
                step += 1
                if(step % 10 == 0):
                    print("step : {:d}".format(step))
                    print('epoch : {0:04d}, gen loss : {1:.6f}, dis X loss : {2:.6f}, dis Y loss : {2:.6f}'.format(epoch_num, gen_loss.numpy(), dis_loss_X.numpy(), dis_loss_Y.numpy()))
                    print('adversarial : {:.3f}, cycle : {:.3f}, per : {:.3f}, makeup : {:.3f}, background : {:.3f}'.format(loss_list[0].numpy(), loss_list[1].numpy(), loss_list[2].numpy(), loss_list[3].numpy(), loss_list[4].numpy()))
                    save_images(epoch_num, step, batch_labels["face_true"].numpy(), batch_labels["lip_true"].numpy(), batch_labels["eye_true"].numpy(), self.gt_save_path)
                if(step % 200 == 0):
                    save_images(epoch_num, step, batch_features["images1"].numpy(), transfer_image[0].numpy(), batch_features["images2"].numpy(), self.pic_save_path)
                    save_images(epoch_num, step, batch_features["images1"].numpy(), bg_images[1].numpy(), transfer_image[1].numpy(), self.bg_save_path)
                    save_images(epoch_num, step, batch_labels["face_true"].numpy(), batch_labels["lip_true"].numpy(), batch_labels["eye_true"].numpy(), self.gt_save_path)
                if(step == train_step):
                    break
            model_path = os.path.join(self.model_path, "{epoch:04d}.ckpt".format(epoch = epoch_num))
            self.save_model(self.model, model_path)
            log(epoch_num, gen_loss, dis_loss_X, dis_loss_Y, loss_list, self.loss_save_path)


    def export_model(self, save_model_path, export_path):
        print("[Model DRN] Exporting Model....")
        self.load_model(self.model, save_model_path)

        # Genetor
        model_path = os.path.join(export_path, 'Generator.h5')
    
        h, w, c = self.input_shape
        image1 = tf.keras.layers.Input(self.input_shape, name = "images1")
        image2 = tf.keras.layers.Input(self.input_shape, name = "images2")

        transfer_images, demakeup_images = self.generator([image1, image2])
    
        Generator = tf.keras.Model(inputs = [image1, image2], outputs = [transfer_images, demakeup_images])
        Generator.save(model_path)

    def save_model(self, model, model_path):
        model.save_weights(model_path)
        print('[Model DRN] Save weights to {}.'.format(model_path))

    def load_model(self, model, model_path):
        model.load_weights(model_path)
        current_epoch = int(model_path[23:-5])
        print('[Model DRN] Load weights from {}.'.format(model_path))
        return current_epoch