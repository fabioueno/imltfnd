import argparse
import json
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
import tensorflow_hub as hub

from PIL import Image


def parse_arguments():
    parser = argparse.ArgumentParser()

    parser.add_argument('image_path',       action = 'store', type = str)
    parser.add_argument('model_path',       action = 'store', type = str)
    parser.add_argument('--top_k',          action = 'store', dest = 'top_k',      type = int)
    parser.add_argument('--category_names', action = 'store', dest = 'json_file',  type = str)

    args = parser.parse_args()

    return args.image_path, args.model_path, args.top_k, args.json_file

def load_model(model_path):
    return tf.keras.models.load_model('./model.h5',
                                      compile = False,
                                      custom_objects = {'KerasLayer': hub.KerasLayer})

def process_image(image):
    image = tf.image.resize(image, (224, 224))
    image /= 255
    return image.numpy()

def predict(image_path, model, top_k):
    image = Image.open(image_path)
    image = np.asarray(image)
    image = process_image(image)
    image = np.expand_dims(image, axis = 0)

    probabilities = model.predict(image)
    values, indices = tf.math.top_k(probabilities[0], top_k)
    values, indices = values.numpy(), indices.numpy() + 1

    return (values, indices)

def open_json(json_path):
    with open(json_path, 'r') as f:
        return json.load(f)

def open_image(image_path):
    image = Image.open(image_path)
    return np.asarray(image)

def convert_classes(mapper, probabilities, classes):
    classes = [mapper[str(class_index)] for class_index in classes]
    return dict(zip(classes, probabilities))

def main():
    image_path, model_path, top_k, json_file = parse_arguments()
    model = load_model(model_path)
    probabilities, classes = predict(image_path, model, top_k)
    mapper = open_json(json_file)
    print(convert_classes(mapper, probabilities, classes))

if __name__ == '__main__':
    main()