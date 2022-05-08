import os
import cv2
import glob
import shutil
import numpy as np
import pandas as pd 
from skimage.metrics import structural_similarity as ssim

from API import FaceBeautyModel
from utils.dataset import Dataset

output_path = "Transfer_final"
images_path = "..\\dataset\\RawData\\images\\non-makeup\\*"
reference_image_path = "..\\dataset\\RawData\\images\\makeup\\*"


params = {
        "batch_size" : 1,

        "image_shape" : (256, 256, 3),
        "classes" : {"face" : [1, 6, 11, 12, 13], "brow" : [2, 3], "eye" : [2, 3, 4, 5], "eyeball_l" : [4], "eyeball_r" : [5], "lip" : [7, 9], "non-makeup" : [0, 4, 5, 8, 10], 
                     "hair" : [10], "whole_face" : [1, 2, 3, 4, 5, 6, 7, 8, 9], "face_makeup" : [1, 2, 3, 6, 7, 9]},

        "test_dataset_size" : [100, 195],
        "test_dataset_path" : {'source' : r'../dataset/source/Test', 'reference' : r'../dataset/reference/Test'},
        }


def images_demo(face_beauty_model, source_image, reference_image):
    # reference image

    # input images
    # images_data_path = glob.glob(images_path, recursive=True)
    # images_makeup_path = glob.glob(reference_image_path, recursive=True)
    # images = []
    # makeup_images = []
    # name = []
    # makeup_name = []
    # for i in range(len(images_data_path)):
    #     image = cv2.imread(images_data_path[i])
    #     makeup_image = cv2.imread(images_makeup_path[i])
    #     images.append(image)
    #     makeup_images.append(makeup_image)
    #     name.append(images_path[images_path.rfind("\\")+1: -4])
    #     makeup_name.append(reference_image_path[reference_image_path.rfind("\\")+1: -4])
    transfer_images = []
    demakeup_images = []
    images = []
    makeup_images = []

    for i in range(len(source_image)):
        image, makeup_image, results = face_beauty_model.transfer(source_image[i], reference_image[i])
        transfer_image = results[0]
        demakeup_image = results[1]
        transfer_images.append(transfer_image)
        demakeup_images.append(demakeup_image)
        images.append(image)
        makeup_images.append(makeup_image)

    return images, makeup_images, transfer_images, demakeup_images

def getData(dataset):
    source_images = []
    reference_images = []
    for features, labels in dataset:
        source_image = features["images1"]
        reference_image = features["images2"]
        source_images.append(source_image)
        reference_images.append(reference_image)
    return source_images, reference_images

if __name__ == "__main__":
    face_beauty_model_BG = FaceBeautyModel("../export_models/BG150/Generator.h5")
    face_beauty_model_DRN = FaceBeautyModel("../export_models/DRN370/Generator.h5")

    # create output dir
    shutil.rmtree(output_path, ignore_errors=True)
    os.makedirs(output_path, exist_ok = True)
    test_dataset = Dataset(params['test_dataset_path'], image_shape = params['image_shape'], classes = params['classes'], batch_size = params['batch_size'], dataset_size = params['test_dataset_size'], isTraining = False)
    dataset = test_dataset.flow()
    source_image, reference_image = getData(dataset)

    image, makeup_image, transfer_images_BG, demakeup_images_BG = images_demo(face_beauty_model_BG, source_image, reference_image)
    image, makeup_image, transfer_images_DRN, demakeup_images_DRN = images_demo(face_beauty_model_DRN, source_image, reference_image)

    # save images
    ssim_results_bg = []
    ssim_results_drn = []

    for i in range(len(transfer_images_DRN)):
        path = os.path.join(output_path, '{}.png'.format(i))
        # cv2.imshow("source image", image[i][0])
        # cv2.imshow("reference_image", makeup_image[i][0])
        # cv2.imshow("transfer_images", transfer_images[i][0])
        # cv2.imshow("demakeup_images", demakeup_images[i][0])
        # cv2.waitKey(0)
        raw = cv2.hconcat([image[i][0], makeup_image[i][0]])
        bg = cv2.hconcat([transfer_images_BG[i][0], demakeup_images_BG[i][0]])
        drn = cv2.hconcat([transfer_images_DRN[i][0], demakeup_images_DRN[i][0]])
        imres = cv2.vconcat([raw, bg, drn])
        cv2.imwrite(path, imres)

        no_makeup = cv2.cvtColor(image[i][0], cv2.COLOR_BGR2GRAY)
        transfer_bg = cv2.cvtColor(transfer_images_BG[i][0], cv2.COLOR_BGR2GRAY)
        transfer_drn = cv2.cvtColor(transfer_images_DRN[i][0], cv2.COLOR_BGR2GRAY)
        ssim_res_BG = round(ssim(no_makeup, transfer_bg, data_range=transfer_bg.max() - transfer_bg.min()), 3)
        ssim_res_DRN = round(ssim(no_makeup, transfer_drn, data_range=transfer_drn.max() - transfer_drn.min()), 3)
        ssim_results_bg.append(ssim_res_BG)
        ssim_results_drn.append(ssim_res_DRN)

# index = np.arange(0, len(transfer_images))
print(ssim_results_bg)
print(ssim_results_drn)
ssim_results_np = np.asarray([[ssim_results_bg], [ssim_results_drn]]).reshape(-1)
pd.DataFrame(ssim_results_np).to_csv('ssim.csv')