# Toxicity-Classification
This is a orginally a project for training a machine learning model to compete in the [Jigsaw Unintended Bias in Toxicity Classification](https://www.kaggle.com/c/jigsaw-unintended-bias-in-toxicity-classification). Later, an OpenFaaS serverless api for the machine learning model is added to classify toxic texts.
## How to train the machine learning model
The dataset is from [a kaggle competition](https://www.kaggle.com/c/jigsaw-unintended-bias-in-toxicity-classification). The train_model.ipynb in the folder build_model contains the code for training the machine learning model. The model currently deployed is [TransformerXL](https://arxiv.org/abs/1901.02860).
## How to build the classification API
The OpenFaaS API code for classifying texts is in the folder classify-serverless-function. To build the API image for OpenFaas, first follow the above steps and train a machine learning model for toxicity classification. Then, export the machine learning model and save it as Toxicity-Classification/classify-serverless-function/classify/models/text_toxicity.pkl. Next, run Toxicity-Classification/classify-serverless-function/build.sh to build the image.
