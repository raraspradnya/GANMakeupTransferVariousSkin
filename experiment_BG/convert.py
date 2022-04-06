from utils.converter import Converter

split_rate = 0.1

# Source dataset
raw_images_path = ['../dataset/RawData/images/non-makeup']
train_dataset_path = '../dataset/source/Train'
test_dataset_path = '../dataset/source/Test'

data_converter = Converter(raw_images_path, split_rate, train_dataset_path, test_dataset_path)
data_converter.start()

# Reference dataset
raw_images_path = ['../dataset/RawData/images/makeup']
train_dataset_path = '../dataset/reference/Train'
test_dataset_path = '../dataset/reference/Test'

data_converter = Converter(raw_images_path, split_rate, train_dataset_path, test_dataset_path)
data_converter.start()