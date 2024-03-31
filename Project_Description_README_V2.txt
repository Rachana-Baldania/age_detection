Dataset:
- Images dataset
- Images metadata (directory contains .mat file which contains all metadata, this is Matlab file you can open this file with scipy.io)
    - wiki.tar.gz

Version 1,two best models: one for classification for 5 age ranges such as 0-24, 25-49, 50-74, 75-99, 100-124,
which achieved 74% accuracy on training data and one for regression model for age detection between 0 to 100 which was achieving mae 9.33. 
i want to experiment another model for age ranges which are in 5 groups from 20 to 65 ages so ranges will be 20-24, 25-29,30-34, 35-39, 40-44,45-49,50-54,55-59, 60-65 , also I can use MLFLow Dagshub
for google collab to check experiments  So i have created version 2 to deliver the project on 6th March 2024. 

upload any images such as full body or flowers like that so i have crop the image at test side but there will fluction of age as model is not been prepared on only face crop 
 images also + or - 5 age gap variation will be there due to constrain such as demographic person face features which might not cover in data, person biological age,features of person, model accuracy. 

Solution Developed By Rachana Baldania:

Folder "Data": wiki contains the datset 

mat.py: is used to create xlsx/csv file from .mat file.

Folder "Docker_Deployement_file/requirements.txt.txt":It contains all packages related to age detection classification and prediction program 

Folder "Docker_Deployement_file/uploads": Test purpose outside images has been tested

Folder "models": Version2 of age detection through classification.
As pe the instruction i havent used pretrained model. It contains best models best trained models using CNN algorithm and Tensorflow, keras, which are not pre trained model 
it is trained as per the given dataset which achieves around 54% accuracy for classification 

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
One can directly get the docker file through 
Pull from Docker : https://hub.docker.com/r/rachanasimform/age_detection/tags 
by using command line docker pull rachanasimform/age_detection:latest2

Also aded in bitbucket age_detection: https://bitbucket.org/simformteam/ml-poc/src/master/
Thank you :) 