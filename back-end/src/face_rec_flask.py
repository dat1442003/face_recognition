from __future__ import absolute_import, division, print_function

import tensorflow as tf
import facenet
import align.detect_face
import numpy as np
import cv2
import pickle
import collections
from flask import Flask, request, jsonify
from flask_cors import CORS
import base64

tf.compat.v1.disable_eager_execution()

MINSIZE = 20
THRESHOLD = [0.6, 0.7, 0.7]
FACTOR = 0.709
IMAGE_SIZE = 182
INPUT_IMAGE_SIZE = 160
CLASSIFIER_PATH = 'Models/facemodel.pkl'
FACENET_MODEL_PATH = 'Models/20180402-114759.pb'
MTCNN_MODEL_DIR = 'src/align'

# Load The Custom Classifier
with open(CLASSIFIER_PATH, 'rb') as file:
    model, class_names = pickle.load(file)
print("Custom Classifier, Successfully loaded")

# Create a default graph
graph = tf.compat.v1.get_default_graph()

# GPU setup if available
gpu_options = tf.compat.v1.GPUOptions(per_process_gpu_memory_fraction=0.6)
sess = tf.compat.v1.Session(graph=graph, config=tf.compat.v1.ConfigProto(gpu_options=gpu_options, log_device_placement=False))

with graph.as_default():
    # Load the FaceNet model
    print('Loading feature extraction model')
    facenet.load_model(FACENET_MODEL_PATH)

    # Get input and output tensors
    images_placeholder = tf.compat.v1.get_default_graph().get_tensor_by_name("input:0")
    embeddings = tf.compat.v1.get_default_graph().get_tensor_by_name("embeddings:0")
    phase_train_placeholder = tf.compat.v1.get_default_graph().get_tensor_by_name("phase_train:0")
    embedding_size = embeddings.get_shape()[1]

    # Create MTCNN networks
    pnet, rnet, onet = align.detect_face.create_mtcnn(sess, MTCNN_MODEL_DIR)

app = Flask(__name__)
CORS(app)

@app.route('/')
def index():
    return "Server is working"

@app.route('/recog', methods=['POST'])
def recog():
    data = request.get_json()
    image_data = data['image']
    image_data = image_data.split(',')[1]  # Remove the data URL scheme part
    image = base64.b64decode(image_data)
    image = np.asarray(bytearray(image), dtype="uint8")
    image = cv2.imdecode(image, cv2.IMREAD_COLOR)

    bounding_boxes, _ = align.detect_face.detect_face(image, MINSIZE, pnet, rnet, onet, THRESHOLD, FACTOR)
    faces_found = bounding_boxes.shape[0]
    names = []

    if faces_found > 0:
        det = bounding_boxes[:, 0:4]
        bb = np.zeros((faces_found, 4), dtype=np.int32)
        for i in range(faces_found):
            bb[i][0] = det[i][0]
            bb[i][1] = det[i][1]
            bb[i][2] = det[i][2]
            bb[i][3] = det[i][3]

            cropped = image[bb[i][1]:bb[i][3], bb[i][0]:bb[i][2], :]
            scaled = cv2.resize(cropped, (INPUT_IMAGE_SIZE, INPUT_IMAGE_SIZE), interpolation=cv2.INTER_CUBIC)
            scaled = facenet.prewhiten(scaled)
            scaled_reshape = scaled.reshape(-1, INPUT_IMAGE_SIZE, INPUT_IMAGE_SIZE, 3)

            feed_dict = {images_placeholder: scaled_reshape, phase_train_placeholder: False}
            emb_array = sess.run(embeddings, feed_dict=feed_dict)

            predictions = model.predict_proba(emb_array)
            best_class_indices = np.argmax(predictions, axis=1)
            best_class_probabilities = predictions[np.arange(len(best_class_indices)), best_class_indices]
            best_name = class_names[best_class_indices[0]]

            if best_class_probabilities[0] > 0.5:
                names.append({"name": best_name, "probability": float(best_class_probabilities[0])})
            else:
                names.append({"name": "Unknown", "probability": float(best_class_probabilities[0])})

    return jsonify(names)

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=8000)
