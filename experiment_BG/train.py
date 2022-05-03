import tensorflow as tf
from utils.dataset import Dataset
from utils.model_BG import Model_BG
from utils.model_DRN import Model_DRN 

params = {
        "epochs" : 200,
        "batch_size" : 2,

        "image_shape" : (256, 256, 3),
        "classes" : {"face" : [1, 6, 11, 12, 13], "brow" : [2, 3], "eye" : [2, 3, 4, 5], "eyeball_l" : [4], "eyeball_r" : [5], "lip" : [7, 9], "non-makeup" : [0, 4, 5, 8, 10], 
                     "hair" : [10], "whole_face" : [1, 2, 3, 4, 5, 6, 7, 8, 9], "face_makeup" : [1, 2, 3, 6, 7, 9]},

        "logs_path" : 'logs/BG/06',
        "pretrained_model_path" : 'logs/BG/05/checkpoint/0137.ckpt',

        "train_dataset_path" : {'source' : r'../dataset/source/Train', 'reference' : r'../dataset/reference/Train'},
        "train_dataset_size" : [779, 1797],
        "test_dataset_path" : [r'../dataset/source/Test', r'../dataset/reference/Test'],
        }


if __name__ == "__main__":
    train_dataset = Dataset(params['train_dataset_path'], image_shape = params['image_shape'], classes = params['classes'], batch_size = params['batch_size'], dataset_size = params['train_dataset_size'], isTraining = True)

    model = Model_BG(input_shape = params['image_shape'], logs_path = params['logs_path'], batch_size = params['batch_size'], classes = params['classes'])

    model.train(train_dataset, params["epochs"], pretrained_model_path = params['pretrained_model_path'])