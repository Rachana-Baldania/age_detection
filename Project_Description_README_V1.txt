Identifying age  by classification and prediction for wiki data 

Dataset:
- Images dataset
- Images metadata (directory contains .mat file which contains all metadata, this is Matlab file you can open this file with scipy.io)
    - wiki.tar.gz

Solution Developed By Rachana Baldania:

Folder "wiki": It contains the datset which is provided and other files such as wiki.mat and wiki.tar.gz are dataset, dataset fetch from :: https://data.vision.ee.ethz.ch/cvl/rrothe/imdb-wiki/

Folder "requirement.txt":It contains all packages related to age detection classification and prediction program 

Folder "models": I have created 2 different models to showcase my skills regarding Machine learning which is age detection through classification and prediction.
it is trained as per the given dataset which achieves around 74% accuracy for classification and
through age detection for prediction evaluated mae: 9.3365 - mse: 145.2105.  
age estimation through classification : acuuracy_bestmodel.h5
age estimation through prediction: agemodel_latest.h5

File "age_estimation_v2.ipynb": It is a age estimation classification program which contains all data ML Steps such as below:
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
Evalution Result : Acuuracy achieved around 74% and loss around 0.6
As discussed, Output shows results in age ranges such as,
1. 0-24
2. 25-49
3. 50-74
4. 75-99
5. 100-124

File "age_detection_prediction.ipynb": It is a age estimation prediction program which contains all data ML Steps such as below:
## Data Collection and Transforamtion from mat file to dataframe
## Data Cleaning and Preprocessing
## EDA (Exploratory Data Analysis)
## Image Data Generator
## Model Architecture
### Compile the Model
### Train the Model
### Evaluate the Model
## Age Estimation based on Provided Image

Through Visual Studio ,Model has been trained on provided data though techniques such as CNN algorithm, TensorFlow, Keras,loss='mse', optimizer='Adam', metrics=['mae','mse']
Output shows results in age which can be + or - 5 age number as mentioned in problem statement
Evalution Result : mae: 9.3365 - mse: 145.2105.  

Folder "Docker_Deployment_file" :
It contains files related to Docker + Flask developement for deployment of application 

Thank you :)
