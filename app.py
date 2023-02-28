#!/usr/bin/env python3.8
import argparse
import os
import cv2
import matplotlib.pyplot as plt
import onnx
import onnxruntime as rt
from flask import Flask, render_template, request
import numpy as np

from onnx_modifier import onnxModifier

app = Flask(__name__)


def pre_process(image):
    img = image.copy()
    img = img[..., ::-1]
    blob = img.astype('float32') / 127.5 - 1.0
    blob = blob.transpose(2, 0, 1)
    blob = np.expand_dims(blob, axis=0)
    return blob


@app.route('/')
def index():
    return render_template('index.html')


@app.route('/open_model', methods=['POST'])
def open_model():
    # https://blog.miguelgrinberg.com/post/handling-file-uploads-with-flask
    onnx_file = request.files['file']
    global onnx_modifier
    onnx_modifier = onnxModifier.from_name_stream(onnx_file.filename, onnx_file.stream)

    return 'OK', 200


@app.route('/show_node', methods=['POST'])
def node_analyst():
    # https://blog.miguelgrinberg.com/post/handling-file-uploads-with-flask
    name_output = request.get_json()
    new_model = onnx_modifier.add_output(name_output)

    input = new_model.graph.input[0]
    input_name = input.name
    # get type of input tensor
    tensor_type = input.type.tensor_type
    shape = []
    # check if it has a shape:
    if tensor_type.HasField("shape"):
        # iterate through dimensions of the shape:
        for d in tensor_type.shape.dim:
            shape.append(d.dim_value)
    if shape[2] == 0:
        shape[2] = 512
    if shape[3] == 0:
        shape[3] = 512

    model_proto_bytes = onnx._serialize(new_model)
    inference_session = rt.InferenceSession(model_proto_bytes)

    img_path = 'test_images'
    images = [os.path.join(img_path, i) for i in os.listdir(img_path) if
              (i.endswith('.png') or i.endswith('.jpg') or i.endswith('.jpeg'))]
    for ip in images:
        image = cv2.imread(ip)
        image = cv2.resize(image, (shape[2], shape[3]))

        im = pre_process(image)
        out = inference_session.run(list(name_output.values()), {input_name: im})[0]
        for idx, fm in enumerate(out[0]):
            p = f'results/{os.path.basename(ip).split(".")[0]}_{name_output["node_name"].replace("/", "_")}'
            os.makedirs(p, exist_ok=True)
            plt.imsave(p + f'/fm_{idx}.png', fm)

    return 'OK', 200


@app.route('/download', methods=['POST'])
def modify_and_download_model():
    modify_info = request.get_json()
    # print(modify_info)
    onnx_modifier.reload()  # allow downloading for multiple times
    onnx_modifier.modify(modify_info)
    onnx_modifier.check_and_save_model()

    return 'OK', 200


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--host', type=str, default='127.0.0.1',
                        help='the hostname to listen on. Set this to "0.0.0.0" to have the server available externally as well')
    parser.add_argument('--port', type=int, default=5000, help='the port of the webserver. Defaults to 5000.')
    parser.add_argument('--debug', type=bool, default=False, help='enable or disable debug mode.')

    args = parser.parse_args()
    return args


def main():
    args = parse_args()
    app.run(host=args.host, port=args.port, debug=args.debug)


if __name__ == '__main__':
    main()
