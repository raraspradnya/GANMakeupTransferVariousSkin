import tensorflow as tf

from experiment_BG.utils.helper import masking_func

@tf.function
def Adversarial_loss_D(real, fake):
    real_loss = tf.reduce_mean(tf.square(real-1))
    fake_loss = tf.reduce_mean(tf.square(fake))
    return (real_loss + fake_loss)

@tf.function
def Adversarial_loss_G(fake_source, fake_reference):
    source_loss = tf.reduce_mean(tf.square(fake_source-1))
    reference_loss = tf.reduce_mean(tf.square(fake_reference-1))
    return (source_loss + reference_loss)

@tf.function
def Cycle_consistency_loss(source, reference, cycle_src, cycle_ref):
    source_loss = tf.reduce_mean((cycle_src - source)**2)
    reference_loss = tf.reduce_mean((cycle_ref - reference)**2)
    return source_loss + reference_loss

@tf.function
def Perceptual_loss(y_true, y_pred):
    return tf.reduce_mean((y_true - y_pred) ** 2)

@tf.function
def Makeup_loss(y_true, y_pred_image, y_mask, classes):
    y_true_face, y_true_brow, y_true_eye, y_true_lip = y_true
    face_mask, brow_mask, eye_mask, lip_mask = y_mask
    
    y_pred_face = y_pred_image * face_mask
    face_loss = tf.reduce_mean((y_true_face - y_pred_face) ** 2) * 10

    y_pred_brow = y_pred_image * brow_mask
    brow_loss = tf.reduce_mean((y_true_brow - y_pred_brow) ** 2) * 10

    y_pred_eye = y_pred_image * eye_mask
    eye_loss = tf.reduce_mean((y_true_eye - y_pred_eye) ** 2) * 20

    y_pred_lip = y_pred_image * lip_mask
    lip_loss = tf.reduce_mean((y_true_lip - y_pred_lip) ** 2) * 20
    return (face_loss + brow_loss + eye_loss + lip_loss)

@tf.function
def Total_Variation_loss(feature):
    left_loss = tf.reduce_mean(tf.abs(feature[:, 1:, :, :] - feature[:, :-1, :, :]))
    down_loss = tf.reduce_mean(tf.abs(feature[:, :, 1:, :] - feature[:, :, :-1, :]))
    TV_loss = left_loss + down_loss
    return TV_loss

@tf.function
def Attention_loss(y_true, y_pred):
    loss = tf.reduce_mean((y_true - y_pred) ** 2)
    return loss