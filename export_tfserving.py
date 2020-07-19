import os
import tensorflow as tf
from tensorflow.python.saved_model import builder as saved_model_builder
from darkflow.net.build import TFNet
from config.config import Config

current_config = Config()

options = {"model": current_config.yolo_config, "load": current_config.yolo_wts, "threshold": 0.1}

tfnet = TFNet(options)

export_path = current_config.export_path

tfnet.build_model(export_path = export_path)

with tfnet.sess.graph.as_default():
    prediction_signature = tf.saved_model.signature_def_utils.build_signature_def(
        inputs={
            "input": tf.saved_model.utils.build_tensor_info(x)
        },
        outputs={
            "output": tf.saved_model.utils.build_tensor_info(pred)
        },
        method_name=tf.saved_model.signature_constants.PREDICT_METHOD_NAME
    )
    builder = saved_model_builder.SavedModelBuilder(export_path)
    builder.add_meta_graph_and_variables(
        tfnet.sess, [tf.saved_model.tag_constants.SERVING],
        signature_def_map={
            "predict": prediction_signature,
        })

    builder.save()