import PIL.Image as Image
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.preprocessing import image
import os
import cv2
import json

class CatDogClassifier():
    def __init__(self, model_folder_name):
        self.model = None
        self.model_json = None
        self.mobilenet_model_json = None
        self.mobilenet_model = None
        self.model_folder_name = model_folder_name

    def run(self):
        json_file = os.path.join(self.model_folder_name, 'catanddog_model.json')
        with open(json_file, 'r') as f:
            self.model_json = f.read()

        self.model = tf.keras.models.model_from_json(self.model_json)
        model_file = os.path.join(self.model_folder_name, 'catanddog_model.h5')
        self.model.load_weights(model_file)

    def mobilenet_run(self):
        mobilenet_json_file = os.path.join(self.model_folder_name, 'mobilenet.json')
        with open(mobilenet_json_file, 'r') as f:
            self.mobilenet_model_json = f.read()

        self.mobilenet_model = tf.keras.models.model_from_json(self.mobilenet_model_json)
        mobilenet_model_file = os.path.join(self.model_folder_name, 'mobilenet_model.h5')
        self.mobilenet_model.load_weights(mobilenet_model_file)

    def api(self, img):
        IMAGE_SHAPE_2 = (128, 128)
        grace_hopper = img.resize(IMAGE_SHAPE_2)
        grace_hopper = np.array(grace_hopper) / 255.0
        result = self.model.predict(grace_hopper[np.newaxis, ...])
        return result

    def prediction(self, img):
        indices = {0: 'Cat', 1: 'Dog', 2: 'Invasive carcinomar', 3: 'Normal'}
        accuracy = self.api(img)
        result = int(accuracy[0][0].round())
        label = indices[result]
        return accuracy[0][0], label

    def mobilenet_prediction(self, ori_img):

        def prepare_image(img):
            img = img.resize((224, 224))
            img_array = image.img_to_array(img)
            img_array_expanded_dims = np.expand_dims(img_array, axis=0)
            return keras.applications.mobilenet.preprocess_input(img_array_expanded_dims)

        def decode_predictions(preds, top=5, **kwargs):
            global CLASS_INDEX

            if len(preds.shape) != 2 or preds.shape[1] != 1000:
                raise ValueError('`decode_predictions` expects '
                                 'a batch of predictions '
                                 '(i.e. a 2D array of shape (samples, 1000)). '
                                 'Found array with shape: ' + str(preds.shape))

            fpath = os.path.join(self.model_folder_name, "imagenet_class_index.json")
            with open(fpath) as f:
                CLASS_INDEX = json.load(f)
            results = []
            for pred in preds:
                top_indices = pred.argsort()[-top:][::-1]
                result = [tuple(CLASS_INDEX[str(i)]) + (pred[i],) for i in top_indices]
                result.sort(key=lambda x: x[2], reverse=True)
                results.append(result)
            return results

        preprocessed_image = prepare_image(ori_img)
        predictions = self.mobilenet_model.predict(preprocessed_image)
        results_var = decode_predictions(predictions)
        return results_var[0][0][1], results_var[0][0][2]

if __name__ == "__main__":
    model_folder_name = os.getenv('MODEL_DIRECTORY')
    video_path = 'imxv4l2src device=/dev/video0 ! video/x-raw,format=I420,width=640,height=480,framerate=30/1 ! appsink'

    catdog_var = CatDogClassifier(model_folder_name)
    catdog_var.run()
    catdog_var.mobilenet_run()

    camera = cv2.VideoCapture(video_path)

    while True:
        s, i = camera.read()
        if s:
            img = Image.fromarray(i).convert('RGB')
            acc, label = catdog_var.prediction(img)

            if label == "Dog":
                accuracy = str(acc * 100)
            else:
                accuracy = str((1 - acc) * 100)

            if float(accuracy) > 98:     
                print("Cat")
            else:
                _, acc_mob = catdog_var.mobilenet_prediction(img)
                acc_mob = str(acc_mob * 100)
                if float(acc_mob) > 80:
                    print("Dog")
                else:
                    print("Unknown")