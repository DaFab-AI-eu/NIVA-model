# python 3.10
import json
import numpy as np
import tensorflow as tf
import tf2onnx
import onnx
import onnxruntime as rt
from eoflow.models.segmentation_unets import ResUnetA

# onnxruntime == 1.19.2, onnx == 1.12.0, tf2onnx == 1.14.0
# (versions protobuf == 3.19.6 / flatbuffers == 2.0.7 need for tensorflow == 2.11.0)

# paste your own paths to chkpt_folder / model_cfg_path
chkpt_folder = "checkpoints"
model_cfg_path = "model_cfg.json"

output_model_path = "model.onnx"
input_shape = (1024, 1024, 4)

def load_model(model_cfg_path, chkpt_folder, input_shape=(256, 256, 4)):
    # https://github.com/sentinel-hub/field-delineation/blob/main/fd/prediction.py
    input_shape = dict(features=[None, *input_shape])
    with open(model_cfg_path, 'r') as jfile:
        model_cfg = json.load(jfile)
    # initialise model from config, build, compile and load trained weights
    model = ResUnetA(model_cfg)
    model.build(input_shape)
    model.net.compile()
    model.net.load_weights(f'{chkpt_folder}/model.ckpt')
    return model


if __name__ == "__main__":
    model = load_model(model_cfg_path, chkpt_folder, input_shape=input_shape)
    # tf.float32 or tf.float64 don't matter
    input_signature = (tf.TensorSpec((None, *input_shape), tf.float64, name="features"),)
    onnx_model, _ = tf2onnx.convert.from_keras(model.net, input_signature=input_signature)
    onnx.save(onnx_model, output_model_path)

    # from example https://onnxruntime.ai/docs/tutorials/tf-get-started.html
    input1 = np.random.rand(*(1, *input_shape)).astype(np.float64)
    providers = ['CPUExecutionProvider']
    m = rt.InferenceSession(output_model_path, providers=providers)
    onnx_pred = m.run(None, {"features": input1})

    results_tf = model.net(input1)
    for ort_res, tf_res in zip(onnx_pred, results_tf):
        np.testing.assert_allclose(ort_res, tf_res, rtol=1e-5, atol=1e-6)
