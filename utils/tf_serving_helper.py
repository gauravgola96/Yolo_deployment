import cv2
from grpc.beta import implementations
import tensorflow as tf
from keras.preprocessing.image import load_img, img_to_array
import numpy as np
from grpc.framework.interfaces.face.face import ExpirationError
from utils.post_processing import *
from config.config import Config
from utils.tensorflow_helpers import predict_pb2, prediction_service_pb2

current_config = Config()


class YoloHelper:
    def preprocess(self, image):
        image = img_to_array(image, )
        data = cv2.resize(image, (416, 416))
        # im_arr = img_to_array(data, )
        im_arr = data/255.0
        im_arr = np.expand_dims(im_arr, axis=0)
        return im_arr

    def predict(self, image ,threshold =0.6):
        channel = implementations.insecure_channel(current_config.TF_SERVING_HOSTNAME,
                                                   current_config.TENSORFLOW_SERVING_PORT)
        stub = prediction_service_pb2.beta_create_PredictionService_stub(channel)
        predict_request = predict_pb2.PredictRequest()
        predict_request.model_spec.name = current_config.name
        predict_request.model_spec.signature_name = current_config.signature_name
        predict_request.inputs[current_config.input_name].CopyFrom(
            tf.contrib.util.make_tensor_proto(image, dtype=tf.float32))
        try:
            result = stub.Predict(predict_request, 10.0)  # 10 secs timeout

        except ExpirationError as e:
            return {{'errors': [{'message': 'Deadline exceed or Connect Failed'}]}, \
                    {'status_code': 400}}
        else:
            return self.post_process(result,threshold)

    def post_process(self, image, threshold =0.6):
        predictions = image.outputs[current_config.output_name]
        predictions = tf.make_ndarray(predictions)
        img_arr = np.squeeze(predictions, 0)
        path = current_config.yolo_config
        meta = parse_cfg(path=path)
        meta.update(current_config.labels)

        boxes = box_contructor(meta=meta, out=img_arr, threshold=threshold)

        boxesInfo = list()
        print(len(boxes))
        for box in boxes:
            tmpBox, prob_ = process_box(b=box, h=13, w=13, threshold=threshold, meta=meta)

            if tmpBox is None:
                continue
            boxesInfo.append({'od_output': tmpBox, 'od_score': round(float(prob_ * 100), 2)})

        if len(boxesInfo) > 0:
            score_list = []
            index = []
            for i, score in enumerate(box['od_score'] for box in boxesInfo):
                score_list.append(score)
                index.append(i)
            index = np.argmax(score_list)
            name = boxesInfo[index]['od_output']
            response = {"Predicted ": name}
            return response
        return {"Predicted ": None}
