import tensorflow as tf
import cv2
import json
from google.protobuf.json_format import MessageToJson



print("Num GPUs Available: ", len(tf.config.list_physical_devices('GPU')))

raw_dataset = tf.data.TFRecordDataset("./data/reference/Train/0000.tfrecords")

for raw_record in raw_dataset.take(1):
    example = tf.train.Example()
    example.ParseFromString(raw_record.numpy())
    m = json.loads(MessageToJson(example))
    print(m['features']['feature'].keys())
    # print(m['features']['feature']['image'])
    cv2.imshow("coba", m['features']['feature']['image'])
    cv2.waitKey(0)