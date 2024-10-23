CIFAR-10 Image Classification using CNN
This repository contains my first internship project at CodeAlpha, where I developed a Convolutional Neural Network (CNN) to classify images from the CIFAR-10 dataset.

Project Overview
The CIFAR-10 dataset consists of 60,000 32x32 color images in 10 different classes (airplanes, cars, animals, etc.). This project builds a deep learning model using TensorFlow and Keras to classify these images.

Key Features:
Dataset: CIFAR-10 image dataset with 10 classes.
Model Architecture:
Three convolutional layers with ReLU activation and max-pooling.
A fully connected dense layer followed by a softmax output layer for classification.
Training: The model is trained for 3 epochs with a batch size of 64.
Evaluation: The model is evaluated on the test dataset, and accuracy is calculated.
Visualization: Sample images from the dataset and model predictions are visualized.
Technologies Used:
Python
TensorFlow
Keras
Matplotlib
How to Run:
Clone the repository.
Install the required dependencies: pip install -r requirements.txt.
Run the model training script: python cifar10_cnn.py.
Check model accuracy and predictions on test data.
Results:
Achieved a test accuracy of approximately 94%.
Feel free to fork the repository and contribute!
