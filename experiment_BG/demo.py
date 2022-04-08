import os
import cv2
import glob
import shutil
import numpy as np

from API import FaceBeautyModel

output_path = "Transfer"
images_path = "..\\dataset\\RawData\\images\\non-makeup\\*"
reference_image_path = "..\\dataset\\RawData\\images\\makeup\\*"

def images_demo(face_beauty_model, reference_image_path, images_path):
    # reference image

    # input images
    images_data_path = glob.glob(images_path, recursive=True)
    images_makeup_path = glob.glob(reference_image_path, recursive=True)
    images = []
    makeup_images = []
    name = []
    makeup_name = []
    for i in range(len(images_data_path)):
        image = cv2.imread(images_data_path[i])
        makeup_image = cv2.imread(images_makeup_path[i])
        images.append(image)
        makeup_images.append(makeup_image)
        name.append(images_path[images_path.rfind("\\")+1: -4])
        makeup_name.append(reference_image_path[reference_image_path.rfind("\\")+1: -4])
    
    transfer_images, demakeup_images = face_beauty_model.transfer(images, makeup_images)

    return name, makeup_name, images, makeup_images, transfer_images, demakeup_images

if __name__ == "__main__":
    face_beauty_model = FaceBeautyModel()

    # create output dir
    shutil.rmtree(output_path, ignore_errors=True)
    os.makedirs(output_path, exist_ok = True)

    name, makeup_name, image, makeup_image, transfer_images, demakeup_images = images_demo(face_beauty_model, reference_image_path, images_path)

    for i in range(len(image)):
        image[i] = cv2.resize(image[i], (256, 256))
        makeup_image[i] = cv2.resize(makeup_image[i], (256, 256))
    image = np.array(image, dtype = np.float32)
    makeup_image = np.array(makeup_image, dtype = np.float32)
    # print(transfer_images.shape)

    # save images
    for i in range(len(transfer_images)):
        path = os.path.join(output_path, '{}.png'.format(i))
        imres = cv2.hconcat([image[i], makeup_image[i], transfer_images[i], demakeup_images[i]])
        cv2.imwrite(path, imres)
    

