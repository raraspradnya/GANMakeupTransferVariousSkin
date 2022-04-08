import os
import shutil

from train import params
from utils.model_DRN import Model_DRN

save_model_path = "logs/DRN/05/checkpoint/0050.ckpt"
export_path = "../export_models/DRN"

if __name__ == "__main__":
    os.makedirs(export_path, exist_ok = True)
    model = Model_DRN(input_shape = params['image_shape'], logs_path = params['logs_path'], batch_size = params['batch_size'], classes = params['classes'])

    model.export_model(save_model_path, export_path)





