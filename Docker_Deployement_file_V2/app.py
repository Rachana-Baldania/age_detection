import sys
import os
import numpy as np
import cv2
from matplotlib import pyplot as plt


from mtcnn import MTCNN
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
global new_model
#model_path = os.environ.get('MODEL_PATH')
# Flask utils
from flask import Flask, redirect, url_for, request, render_template
from werkzeug.utils import secure_filename
from gevent.pywsgi import WSGIServer
# Define a flask app
app = Flask(__name__)

# Model saved with Keras model.save()
MODEL_PATH = f'models/age_estimation_v3_mlflow_modeladding in amu200.h5'

# Load your trained model
new_model = load_model(MODEL_PATH)


import cv2
import numpy as np
from tensorflow.keras.preprocessing import image
import matplotlib.pyplot as plt
from mtcnn import MTCNN

def load_and_preprocess_image(filepath):
    test_img = image.load_img(filepath, target_size=(198, 198))
    test_img = image.img_to_array(test_img)
    test_img = np.expand_dims(test_img, axis=0)
    test_img /= 255
    return test_img

def detect_and_crop_face(image_path):
    img = cv2.imread(image_path)
    detector = MTCNN()
    result = detector.detect_faces(img)
    
    if len(result) == 1:
        x, y, w, h = result[0]['box']
        cropped_img = img[y:y+h, x:x+w]
        return cropped_img
    else:
        return None



def predict_age_from_image(image_path, model):
    cropped_img = detect_and_crop_face(image_path)
    prb_age = []  # Initialize prb_age here
    if cropped_img is not None:
        age_pred = model.predict(load_and_preprocess_image(image_path))
        age_pred_rounded = [round(prob, 2) for prob in age_pred[0]]
        age_classes = ['20-24', '25-29', '30-34', '35-39', '40-44', '45-49', '50-54', '55-59', '60-65']
        max_prob_index = np.argmax(age_pred[0])
        max_age_class = age_classes[max_prob_index]
        
        #plt.imshow(cv2.cvtColor(cropped_img, cv2.COLOR_BGR2RGB))
        #plt.show()
        
        # print("Probabilities :", age_classes)
        # print("Probabilities :", age_pred_rounded)
        # print(f'Most probable age group: {max_age_class}')
              
        # Print predicted age probabilities
        for i, prob in enumerate(age_pred_rounded):
            prb_age.append(f"{age_classes[i]} years old: {prob:.4f}")



        # Get the index of the maximum probability
        max_age_index = np.argmax(age_pred)

        # Print predicted age
        predicted_age = age_classes[max_age_index]

        return prb_age, predicted_age
    else:
        predicted_age= ("No face detected or Multiple faces detected in the image, Please upload an image with one human face!")
        
    return prb_age, predicted_age



# Necessary
# print('Model loaded. Start serving...')

# You can also use pretrained model from Keras
# Check https://keras.io/applications/
#from keras.applications.resnet50 import ResNet50
#model = ResNet50(weights='imagenet')
#model.save('')
print('Model loaded. Check http://127.0.0.1:5000/')

# def cropFaceAndPredictAge(filepath):
#     # Load image
    
#     img = cv2.imread(filepath)
#     img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

#     # Detect faces in the image
#     detector = MTCNN()
#     faces = detector.detect_faces(img_rgb)

#     prb_age = []  # Initialize prb_age here

#     # Check if only one face is detected
#     if len(faces) == 1:
#         # Get the bounding box of the face
#         x, y, w, h = faces[0]['box']
#         # Crop the face
#         face_img = img_rgb[y:y+h, x:x+w]
#         # Resize the face image to the required size (198x198)
#         face_img_resized = cv2.resize(face_img, (198, 198))

#         # Predict age
#         age_pred = new_model.predict(np.expand_dims(face_img_resized, axis=0))[0]
#         age_classes = ['20-24', '25-29', '30-34', '35-39', '40-44', '45-49', '50-54', '55-59', '60-65']

      
#         # Print predicted age probabilities
#         for i, prob in enumerate(age_pred):
#             prb_age.append(f"{age_classes[i]} years old: {prob:.4f}")

#         # Get the index of the maximum probability
#         max_age_index = np.argmax(age_pred)

#         # Print predicted age
#         predicted_age = age_classes[max_age_index]

#         return prb_age, predicted_age
#     elif len(faces) > 1:
#         predicted_age = ("Multiple faces detected. Please upload an image with a single human face.")
#     else:
#         predicted_age = ("No face detected. Please upload an image with a human face.")
#     return prb_age, predicted_age



@app.route('/', methods=['GET'])
def index():
    # Main page
    return render_template('index.html')

@app.route('/predict', methods=['GET', 'POST'])
def upload():
    if request.method == 'POST':
        # Get the file from post request
        f = request.files['file']

        # Save the file to ./uploads
        basepath = os.path.dirname(__file__)
        file_path = os.path.join(
            basepath, 'uploads', secure_filename(f.filename))
        f.save(file_path)

        # Make prediction
        prb_age, predicted_age = predict_age_from_image(file_path,new_model)

        # Build the response string
        response = ' ' + predicted_age + '\n'
        response += '................................................................................................................................................................... ' +'\n'+' , '.join(prb_age)

        return response
        
    return None

if __name__ == '__main__':
    #app.run(debug=True)
    # Serve the app with gevent
    http_server = WSGIServer(('0.0.0.0', 5000), app)
    http_server.serve_forever()
