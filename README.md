Identifying age  by classification and prediction for wiki data 

Dataset:
- Images dataset:https://data.vision.ee.ethz.ch/cvl/rrothe/imdb-wiki/
- Images metadata (directory contains .mat file which contains all metadata, this is Matlab file you can open this file with scipy.io)
    - wiki.tar.gz

Folder "wiki": It contains the datset which is provided and other files such as wiki.mat and wiki.tar.gz are dataset, dataset fetch from :: https://data.vision.ee.ethz.ch/cvl/rrothe/imdb-wiki/

Folder "requirement.txt":It contains all packages related to age detection classification and prediction program 

Folder "models":
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

Folder "Docker_Deployment_file" :
It contains files related to Docker + Flask developement for deployment of application 

Thank you :)
