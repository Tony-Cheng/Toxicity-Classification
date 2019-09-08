# Toxicity-Classification
This is a orginally a project for training a machine learning model to compete in the [Jigsaw Unintended Bias in Toxicity Classification](https://www.kaggle.com/c/jigsaw-unintended-bias-in-toxicity-classification). Later, an openfaas serverless api for the machine learning model is added to classify toxic texts. <br />
## How to train the machine learning model
The dataset is from [a kaggle competition](https://www.kaggle.com/c/jigsaw-unintended-bias-in-toxicity-classification). The train_model.ipynb in the folder build_model contains the code for training the machine learning model. The model currently deployed is [TransformerXL](https://arxiv.org/abs/1901.02860). <br />
## How to build the classification API
The openfaas api for classifying the texts is in the folder classify-serverless-function. To build the api image for openfaas, first follow the above steps and train a machine learning model for toxicity classification. Then, export the machine learning model and save it as Toxicity-Classification/classify-serverless-function/classify/models/text_toxicity.pkl. Next, run Toxicity-Classification/classify-serverless-function/build.sh.
