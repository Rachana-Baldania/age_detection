Objective : Training a ML Model for age detection using given dataset
 
Dataset:
- Images dataset
- Images metadata (directory contains .mat file which contains all metadata, this is Matlab file you can open this file with scipy.io)
    - wiki.tar.gz

Version 3,  we generally need to provide the image like this only but team would like to provide image at test side person full body image or any other image should be cropped automatically and detect the age.I have again created another model where i have cropped face images  and then trained in 9 age ranges such as 20-24, 25-29,30-34, 35-39, 40-44,45-49,50-54,55-59, 60-65, which achieved around 60% accuracy on training data. I have used different parameters and model architecture, I have tracked almost all experiments which were run on google collab in dagshub.  

Docker File: docker pull rachanasimform/age_detection:latest3 

Dagshub: https://dagshub.com/rachana/age_detection/experiments/#/experiment/m_ece39d26fb8246b2b16ec4c6a6bc6a68 

Solution Developed By Rachana Baldania:

mat.py: is used to create csv file from .mat file.

Folder "Docker_Deployement_file/requirements.txt.txt":It contains all packages related to age detection classification and prediction program 

Folder "Docker_Deployement_file/uploads": Test purpose outside images has been tested

Folder "models": Version3 of age detection through classification.

File "Code/age_estimation_v2.ipynb": It is a age estimation classification program which contains all data ML Steps such as below:
## Data Collection and Transforamtion from mat file to dataframe
## Data Cleaning and Preprocessing
## EDA (Exploratory Data Analysis)
## Image Data Generator
## Model Architecture
### Compile the Model
### Train the Model
### Evaluate the Model
## Age Estimation based on Provided Image

Through Google Collab,Model has been trained on provided data though techniques such as CNN algorithm, TensorFlow, Keras,Categorical Crossentropy, Adam, activation function: relu and softmax.


File "Code/age_estimation_v3_MLFlow_Colab.ipynb": It is a age estimation prediction program which contains all data ML Steps such as below:
## Data Collection and Transforamtion from mat file to dataframe
## Data Cleaning and Preprocessing
## EDA (Exploratory Data Analysis)
## Image Data Generator
## Model Architecture
### Compile the Model
### Train the Model
### Evaluate the Model
## Age Estimation based on Provided Image


Folder "Docker_Deployment_file" : 
It contains files related to Docker + Flask developement for deployment of application 

Thank you :) 