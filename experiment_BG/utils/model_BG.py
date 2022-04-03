import os
import cv2
import shutil
import threading
import tensorflow as tf
from datetime import datetime

from utils.module import *
from utils.losses import *
from utils.helper import *

class Model_BG(object):
    def __init__(self, input_shape, logs_path, batch_size = 32, classes = None):
        self.input_shape = input_shape
        self.batch_size = batch_size
        self.classes = classes
        self.logs_path = logs_path
        self.model_path = os.path.join(self.logs_path, 'checkpoint')
        self.pic_save_path = os.path.join(logs_path, 'save_pic')
        self.gt_save_path = os.path.join(logs_path, 'save_gt')

        # Define optimizer
        learning_rate_fn_1 = tf.keras.optimizers.schedules.PolynomialDecay(2e-4, 155250, 1e-6, power=1.0)
        self.model_optimizer = tf.keras.optimizers.Adam(learning_rate_fn_1, beta_1 = 0.5, beta_2=0.999)
        learning_rate_fn_2 = tf.keras.optimizers.schedules.PolynomialDecay(2e-4, 155250, 1e-6, power=1.0)
        self.discriminator_optimizer = tf.keras.optimizers.Adam(learning_rate_fn_2, beta_1 = 0.5, beta_2=0.999)

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
        image1_true, image2_true = labels["images1"], labels["images2"]
        face_true, brow_true, eye_true, lip_true = labels["face_true"], labels["brow_true"], labels["eye_true"], labels["lip_true"]
        makeup_mask = labels["makeup_mask"]
        face_mask, brow_mask, eye_mask, lip_mask = labels["face_mask"], labels["brow_mask"], labels["eye_mask"], labels["lip_mask"]
    
        # Get predicted images
        transfer_image, demakeup_image  = pred["image"][0], pred["image"][1]
        cycle_source_image, cycle_reference_image  = pred["cycle_image"][0], pred["cycle_image"][1]
        
        # Get discriminator results
        real_source = dis_result[0]
        fake_source = dis_result[1]
        real_reference = dis_result[2]
        fake_reference = dis_result[3]

        # Adversarial loss
        alpha = 1
        adversarial_loss = Adversarial_loss(dis_loss) * alpha


        # Cycle consistency loss
        beta = 20
        cycle_consistency_loss = Cycle_consistency_loss(transfer_image, demakeup_image, cycle_source_image, cycle_reference_image)
        cycle_consistency_loss = cycle_consistency_loss * 20

        # high level feature
        feature_true = self.feature_model(image1_true)
        feature = self.feature_model(transfer_image)


        # Perceptual loss
        gamma = 0.005
        perceptual_loss = Perceptual_loss(feature_true, feature)
        perceptual_loss = perceptual_loss * gamma

        
        # Makeup loss
        makeup_loss = Makeup_loss(y_true = [face_true, brow_true, eye_true, lip_true], y_pred_image = transfer_images, y_mask = [face_mask, brow_mask, eye_mask, lip_mask], classes = self.classes)
        makeup_loss = makeup_loss * 50

        # Attention loss
        attention_loss = Attention_loss(makeup_mask, attention_mask) + Attention_loss(makeup_mask, transfer_mask) 
        attention_loss = attention_loss * 10

        # Adversarial Loss
        adversarial_loss = Adversarial_loss(dis_fake_output)
        adversarial_loss = adversarial_loss

        # KL Loss
        kl_loss = KL_loss(mean = makeup_code1[0], z_log_var = makeup_code1[1]) + KL_loss(mean = makeup_code2[0], z_log_var = makeup_code2[1])
        kl_loss = kl_loss * 0.01

        # Total Variation Loss
        tv_loss = Total_Variation_loss(attention_mask) + Total_Variation_loss(transfer_mask)
        tv_loss = tv_loss * 0.0001

        loss = reconsructed_loss + perceptual_loss + makeup_loss + IMRL_loss + attention_loss + adversarial_loss + kl_loss + tv_loss
        return loss,  [reconsructed_loss, perceptual_loss, makeup_loss, IMRL_loss, attention_loss, adversarial_loss, kl_loss, tv_loss]


    def build_generator(self):
        print("[Model BeautyGAN] Building Generator....")
        image_source = tf.keras.layers.Input(self.input_shape)
        image_reference = tf.keras.layers.Input(self.input_shape)

        # DOWNSAMPLING BRANCH
        # SOURCE BRANCH
        x = Conv2D_layer(image_source, filters = 64, kernel_size = (7, 7), strides=(2, 2))
        x = InstanceNormalization_layer(x)
        x = ReLU_layer(x)
        x = Conv2D_layer(x, filters = 128, kernel_size = (4, 4), strides=(2,2))
        x = InstanceNormalization_layer(x)
        x = ReLU_layer(x)

        # REFERENCE BRANCH
        y = Conv2D_layer(image_reference, filters = 64, kernel_size = (3, 3), strides=(2, 2))
        y = InstanceNormalization_layer(y)
        y = ReLU_layer(y)
        y = Conv2D_layer(y, filters = 128, kernel_size = (4, 4), strides=(2,2))
        y = InstanceNormalization_layer(y)
        y = ReLU_layer(y)

        # CONCATENATE
        x = Concatenate_layer(x, y, axis=1)

        # DOWNSAMPLING
        x = Conv2D_layer(x, filters = 128, kernel_size = (4, 4), strides=(2, 2))
        x = InstanceNormalization_layer(x)
        x = ReLU_layer(x)

        # RESIDUAL BLOCK
        for i in range(6):
            x = Conv2D_layer(x, filters = 256, kernel_size = (3, 3))
            x = InstanceNormalization_layer(x)
            x = ReLU_layer(x)

        # UPSAMPLING
        x = DeConv2D_layer(x, filters = 128, kernel_size = (4, 4), strides=(2, 2))
        x = InstanceNormalization_layer(x)
        x = ReLU_layer(x)
        x = DeConv2D_layer(x, filters = 64, kernel_size = (4, 4), strides=(2, 2))
        x = InstanceNormalization_layer(x)
        x = ReLU_layer(x)

        # UPSAMPLING BRANCH
        res_source = Conv2D_layer(x, filters = 64, kernel_size = (3, 3))
        res_source = InstanceNormalization_layer(res_source)
        res_source = ReLU_layer(res_source)
        res_source = Conv2D_layer(x, filters = 64, kernel_size = (3, 3))
        res_source = InstanceNormalization_layer(res_source)
        res_source = ReLU_layer(res_source)
        res_source = Conv2D_layer(x, filters = 3, kernel_size = (7, 7))
        res_source = tf.nn.tanh(res_source)

        res_reference = Conv2D_layer(x, filters = 64, kernel_size = (3, 3))
        res_reference = InstanceNormalization_layer(res_reference)
        res_reference = ReLU_layer(res_reference)
        res_reference = Conv2D_layer(x, filters = 64, kernel_size = (3, 3))
        res_reference = InstanceNormalization_layer(res_reference)
        res_reference = ReLU_layer(res_reference)
        res_reference = Conv2D_layer(x, filters = 3, kernel_size = (7, 7))
        res_reference = tf.nn.tanh(res_reference)

        model = tf.keras.Model(inputs = [image_source, image_reference], outputs = [res_source, res_reference])
        return model

    def build_discriminator_X(self):
        print("[Model BeautyGAN] Building Discriminator X....")
        image = tf.keras.layers.Input(self.input_shape)

        x = image
        x = Conv2D_layer(x, filters = 64, kernel_size = (4, 4), strides = (2, 2))
        x = LeakyReLU_layer(x)
        x = Conv2D_layer(x, filters = 128, kernel_size = (4, 4), strides = (2, 2))
        x = LeakyReLU_layer(x)
        x = Conv2D_layer(x, filters = 256, kernel_size = (4, 4), strides = (2, 2))
        x = LeakyReLU_layer(x)
        x = Conv2D_layer(x, filters = 512, kernel_size = (4, 4), strides = (2, 2))
        x = LeakyReLU_layer(x)
        x = Conv2D_layer(x, filters = 1, kernel_size = (3, 3))
        x = tf.nn.sigmoid(x)

        model = tf.keras.Model(inputs = image, outputs = x)
        return model

    def build_discriminator_Y(self):
        print("[Model BeautyGAN] Building Discriminator Y....")
        image = tf.keras.layers.Input(self.input_shape)

        x = image
        x = Conv2D_layer(x, filters = 64, kernel_size = (4, 4), strides = (2, 2))
        x = LeakyReLU_layer(x)
        x = Conv2D_layer(x, filters = 128, kernel_size = (4, 4), strides = (2, 2))
        x = LeakyReLU_layer(x)
        x = Conv2D_layer(x, filters = 256, kernel_size = (4, 4), strides = (2, 2))
        x = LeakyReLU_layer(x)
        x = Conv2D_layer(x, filters = 512, kernel_size = (4, 4), strides = (2, 2))
        x = LeakyReLU_layer(x)
        x = Conv2D_layer(x, filters = 1, kernel_size = (3, 3))
        x = tf.nn.sigmoid(x)

        model = tf.keras.Model(inputs = image, outputs = x)
        return model

    def build(self, isLoadWeight = False):
        print("[Model BeautyGAN] Building Train Model....")
        # Training flow
        image1 = tf.keras.layers.Input(self.input_shape, name = "images1")
        image2 = tf.keras.layers.Input(self.input_shape, name = "images2") 

        # transfer image
        result_source, result_reference = self.generator([image1, image2])

        # cycle consistency
        cycle_source, cycle_reference = self.generator([result_source, result_reference])

        # Loss Function
        pred = {"image" : [result_source, result_reference],
                "cycle_image" : [cycle_source, cycle_reference]
        }

        # Model
        model = tf.keras.Model(inputs = [image1, image2], outputs = pred)
        model.summary()
        return model

    @tf.function
    def train_step(self, features, labels):
        with tf.GradientTape() as gen_tape, tf.GradientTape() as disc_tape:
            pred = self.model(features, training=True)
            real_source = self.discriminator_X(features["images1"], training=True)
            fake_source = self.discriminator_X(pred["image"][0], training=True)
            real_reference = self.discriminator_Y(features["images2"], training=True)
            fake_reference = self.discriminator_Y(pred["image"][1], training=True)

            gen_loss, loss_list = self.loss_function(pred, labels, [real_source, fake_source, real_reference, fake_reference])
            dis_loss = Discriminator_loss(real_output, [fake_output1, fake_output2])

        gradients_of_generator = gen_tape.gradient(gen_loss, self.model.trainable_variables)
        gradients_of_discriminator = disc_tape.gradient(dis_loss, self.discriminator.trainable_variables)

        self.model_optimizer.apply_gradients(zip(gradients_of_generator, self.model.trainable_variables))
        self.discriminator_optimizer.apply_gradients(zip(gradients_of_discriminator, self.discriminator.trainable_variables))
        return gen_loss, dis_loss, pred["transfer_images"], loss_list

    def train(self, train_dataset, epochs = 500, pretrained_model_path = None):
        os.makedirs(self.logs_path, exist_ok = True)
        os.makedirs(self.model_path, exist_ok = True)
        os.makedirs(self.pic_save_path, exist_ok = True)
        os.makedirs(self.gt_save_path, exist_ok = True)

        # Load dataset
        train_step = len(train_dataset) // self.batch_size
        dataset = train_dataset.flow()

        # Load Pretrain Model
        if(pretrained_model_path != None):
            self.load_model(self.model, pretrained_model_path)

        print("[Model BeautyGAN] Training....")

        # Log Scalars
        logdir = "logs/scalars/" + datetime.now().strftime("%Y%m%d-%H%M%S")
        file_writer = tf.summary.create_file_writer(logdir + "/metrics")
        file_writer.set_as_default()

        for epoch in range(epochs):
            step = 0
            for batch_features, batch_labels in dataset:
                gen_loss, dis_loss, transfer_images, loss_list = self.train_step(batch_features, batch_labels)

                step += 1
                if(step % 10 == 0):
                    print ('epoch {0:04d} : gen loss : {1:.6f}, dis loss : {2:.6f}'.format(epoch + 1, gen_loss.numpy(), dis_loss.numpy()))
                    print('rec : {:.3f}, per : {:.3f}, makeup : {:.3f}, IMRL : {:.3f}, attenton : {:.3f}, adversarial : {:.3f}, kl : {:.3f}, tv : {:.3f}'.format(loss_list[0].numpy(), loss_list[1].numpy(), loss_list[2].numpy(), loss_list[3].numpy(), loss_list[4].numpy(), loss_list[5].numpy(), loss_list[6].numpy(), loss_list[7].numpy()))
                if(step % 200 == 0):
                    save_images(epoch + 1, step, batch_features["images1"].numpy(), transfer_images.numpy(), batch_features["images2"].numpy(), self.pic_save_path)
                    save_images(epoch + 1, step, batch_labels["face_true"].numpy(), batch_labels["lip_true"].numpy(), batch_labels["eye_true"].numpy(), self.gt_save_path)
                if(step == train_step):
                    break
            if (epoch + 1) % 5 == 0:
                model_path = os.path.join(self.model_path, "{epoch:04d}.ckpt".format(epoch = epoch + 1))
                self.save_model(self.model, model_path)
            tf.summary.scalar('Epoch', epoch +1)
            tf.summary.scalar('Generator Loss', gen_loss.numpy())
            tf.summary.scalar('Discriminator Loss', dis_loss.numpy())
            tf.summary.scalar('Reconstruction Loss', loss_list[0].numpy())
            tf.summary.scalar('Perceptual Loss', loss_list[1].numpy())
            tf.summary.scalar('Makeup Loss', loss_list[2].numpy())
            tf.summary.scalar('IMRL Loss', loss_list[3].numpy())
            tf.summary.scalar('Attention Loss', loss_list[4].numpy())
            tf.summary.scalar('Adversarial Loss', loss_list[5].numpy())
            tf.summary.scalar('KL Loss', loss_list[6].numpy())
            tf.summary.scalar('Total Variation Loss', loss_list[7].numpy())

    def export_model(self, save_model_path, export_path):
        print("[Model BeautyGAN] Exporting Model....")
        self.load_model(self.model, save_model_path)

        # Makeup Encoder
        model_path = os.path.join(export_path, 'MakeupEncoder.h5')
    
        h, w, c = self.input_shape
        image2 = tf.keras.layers.Input(self.input_shape, name = "images2")
        makeup_code2 = self.makeup_encoder(image2)
        context_code2 = self.context_encoder(makeup_code2)

        MakeupEncoder = tf.keras.Model(inputs = image2, outputs = context_code2)
        MakeupEncoder.save(model_path)

        # Generator
        model_path = os.path.join(export_path, 'Generator.h5')

        image1 = tf.keras.layers.Input(self.input_shape, name = "images1")
        context_code1 = tf.keras.layers.Input((8, ), name = "context_code1") 
        context_code2 = tf.keras.layers.Input((8, ), name = "context_code2")

        identity_code = self.identity_encoder(image1)
        transfer_images, transfer_mask = self.generator([identity_code, context_code1, context_code2])
        transfer_images = transfer_mask * transfer_images + (1 - transfer_mask) * image1
    
        Generator = tf.keras.Model(inputs = [image1, context_code1, context_code2], outputs = transfer_images)
        Generator.save(model_path)

    def save_model(self, model, model_path):
        model.save_weights(model_path)
        print('[Model BeautyGAN] Save weights to {}.'.format(model_path))
        result_source, result_reference = self.generator([image1, image2])

        # fake imag
        batch_size = tf.shape(identity_code)[0]
        random_code = tf.random.normal([batch_size, 8])
        random_hidden_code = self.context_encoder(random_code)
        fake_images, fake_masks = self.generator([identity_code, random_hidden_code[0], random_hidden_code[1]])
        fake_images = fake_masks * fake_images + (1 - fake_masks) * image1

        # Loss Function
        pred = {"image" : [reconsructed_images, transfer_images], "mask" : [attention_mask, transfer_mask],
                "code" : [identity_code, [code1_mean, code1_var], [code2_mean, code2_var], transfer_identity_code, transfer_makeup_code],
                "fake_images" : fake_images, "transfer_images" : transfer_images
        }

        # Model
        model = tf.keras.Model(inputs = [image1, image2], outputs = pred)
        model.summary()
        return model

    @tf.function
    def train_step(self, features, labels):
        with tf.GradientTape() as gen_tape, tf.GradientTape() as disc_tape:
            pred = self.model(features, training=True)
            real_output = self.discriminator(features["images2"], training=True)
            fake_output1 = self.discriminator(pred["fake_images"], training=True)
            fake_output2 = self.discriminator(pred["transfer_images"], training=True)

            gen_loss, loss_list = self.loss_function(pred, labels, [fake_output1, fake_output2])
            dis_loss = Discriminator_loss(real_output, [fake_output1, fake_output2])

        gradients_of_generator = gen_tape.gradient(gen_loss, self.model.trainable_variables)
        gradients_of_discriminator = disc_tape.gradient(dis_loss, self.discriminator.trainable_variables)

        self.model_optimizer.apply_gradients(zip(gradients_of_generator, self.model.trainable_variables))
        self.discriminator_optimizer.apply_gradients(zip(gradients_of_discriminator, self.discriminator.trainable_variables))
        return gen_loss, dis_loss, pred["transfer_images"], loss_list

    def train(self, train_dataset, epochs = 500, pretrained_model_path = None):
        os.makedirs(self.logs_path, exist_ok = True)
        os.makedirs(self.model_path, exist_ok = True)
        os.makedirs(self.pic_save_path, exist_ok = True)

        train_step = len(train_dataset) // self.batch_size
        dataset = train_dataset.flow()

        if(pretrained_model_path != None):
            self.load_model(self.model, pretrained_model_path)

        print("[Model BeautyGAN] Training....")
        logdir = "logs/scalars/" + datetime.now().strftime("%Y%m%d-%H%M%S")
        file_writer = tf.summary.create_file_writer(logdir + "/metrics")
        file_writer.set_as_default()
        

        for epoch in range(epochs):
            step = 0
            for batch_features, batch_labels in dataset:
                gen_loss, dis_loss, transfer_images, loss_list = self.train_step(batch_features, batch_labels)

                step += 1
                if(step % 10 == 0):
                    print ('epoch {0:04d} : gen loss : {1:.6f}, dis loss : {2:.6f}'.format(epoch + 1, gen_loss.numpy(), dis_loss.numpy()))
                    print('rec : {:.3f}, per : {:.3f}, makeup : {:.3f}, IMRL : {:.3f}, attenton : {:.3f}, adversarial : {:.3f}, kl : {:.3f}, tv : {:.3f}'.format(loss_list[0].numpy(), loss_list[1].numpy(), loss_list[2].numpy(), loss_list[3].numpy(), loss_list[4].numpy(), loss_list[5].numpy(), loss_list[6].numpy(), loss_list[7].numpy()))
                if(step % 200 == 0):
                    save_images(epoch + 1, step, batch_features["images1"].numpy(), transfer_images.numpy(), batch_features["images2"].numpy(), self.pic_save_path)
                    save_images(epoch + 1, step, batch_labels["face_true"].numpy(), batch_labels["lip_true"].numpy(), batch_labels["eye_true"].numpy(), os.path.join(self.logs_path, 'save_gt'))
                if(step == train_step):
                    break
            if (epoch + 1) % 5 == 0:
                model_path = os.path.join(self.model_path, "{epoch:04d}.ckpt".format(epoch = epoch + 1))
                self.save_model(self.model, model_path)
            tf.summary.scalar('Epoch', epoch +1)
            tf.summary.scalar('Generator Loss', gen_loss.numpy())
            tf.summary.scalar('Discriminator Loss', dis_loss.numpy())
            tf.summary.scalar('Reconstruction Loss', loss_list[0].numpy())
            tf.summary.scalar('Perceptual Loss', loss_list[1].numpy())
            tf.summary.scalar('Makeup Loss', loss_list[2].numpy())
            tf.summary.scalar('IMRL Loss', loss_list[3].numpy())
            tf.summary.scalar('Attention Loss', loss_list[4].numpy())
            tf.summary.scalar('Adversarial Loss', loss_list[5].numpy())
            tf.summary.scalar('KL Loss', loss_list[6].numpy())
            tf.summary.scalar('Total Variation Loss', loss_list[7].numpy())

    def export_model(self, save_model_path, export_path):
        print("[Model BeautyGAN] Exporting Model....")
        self.load_model(self.model, save_model_path)

        # Makeup Encoder
        model_path = os.path.join(export_path, 'MakeupEncoder.h5')
    
        h, w, c = self.input_shape
        image2 = tf.keras.layers.Input(self.input_shape, name = "images2")
        makeup_code2 = self.makeup_encoder(image2)
        context_code2 = self.context_encoder(makeup_code2)

        MakeupEncoder = tf.keras.Model(inputs = image2, outputs = context_code2)
        MakeupEncoder.save(model_path)

        # Generator
        model_path = os.path.join(export_path, 'Generator.h5')

        image1 = tf.keras.layers.Input(self.input_shape, name = "images1")
        context_code1 = tf.keras.layers.Input((8, ), name = "context_code1") 
        context_code2 = tf.keras.layers.Input((8, ), name = "context_code2")

        identity_code = self.identity_encoder(image1)
        transfer_images, transfer_mask = self.generator([identity_code, context_code1, context_code2])
        transfer_images = transfer_mask * transfer_images + (1 - transfer_mask) * image1
    
        Generator = tf.keras.Model(inputs = [image1, context_code1, context_code2], outputs = transfer_images)
        Generator.save(model_path)

    def save_model(self, model, model_path):
        model.save_weights(model_path)
        print('[Model BeautyGAN] Save weights to {}.'.format(model_path))

    def load_model(self, model, model_path):
        model.load_weights(model_path)
        print('[Model BeautyGAN] Load weights from {}.'.format(model_path))