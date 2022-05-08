import os
import shutil

from train import params
from utils.model_DRN import Model_DRN
from utils.model_BG import Model_BG

save_model_path = "logs/DRN/14/checkpoint/0370.ckpt"
export_path = "../export_models/DRN370"

if __name__ == "__main__":
    os.makedirs(export_path, exist_ok = True)
    model = Model_DRN(input_shape = params['image_shape'], logs_path = params['logs_path'], batch_size = params['batch_size'], classes = params['classes'])

    model.export_model(save_model_path, export_path)





