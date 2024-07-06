# Hand Gesture Recognition using Convolutional Neural Networks (CNNs)

This project aims to develop a Convolutional Neural Network (CNN) model to accurately recognize American Sign Language (ASL) hand gestures. The project utilizes grayscale images of hand gestures representing ASL letters and applies various machine learning techniques to achieve high classification accuracy.

## Project Objective

To develop and train a CNN model capable of accurately classifying ASL letters based on hand gesture images. The model aims to enhance communication accessibility for individuals with hearing impairments by automating the recognition of sign language gestures.

## Dataset

The dataset consists of grayscale images of hand gestures representing ASL letters. Each image is 28x28 pixels in size, and the dataset includes labeled examples for both training and testing.

The dataset can be downloaded from Kaggle's Sign Language MNIST dataset (https://www.kaggle.com/datasets/datamunge/sign-language-mnist).

## CNN Architecture

The CNN model has the following structure:
1. **First Convolutional Layer**: Applies a set of learnable filters (kernels) to extract features such as edges and textures.
2. **Second Convolutional Layer**: Applies another set of learnable filters to capture more abstract and higher-level features.
3. **Activation Function**: Uses ReLU to introduce non-linearity.
4. **Max Pooling Layer**: Down-samples the feature maps, retaining important features while discarding less relevant information.
5. **Fully Connected Layer**: Flattens the feature maps and connects them to dense layers.
6. **Dropout Layer**: Introduces randomness to prevent overfitting.
7. **Output Layer**: Consists of 26 nodes for each ASL letter class.

## Data Augmentation

To enhance the model's robustness, data augmentation techniques are applied:
- **Rescaling**: Normalizes pixel values to the range [0, 1].
- **Zoom Range**: Randomly zooms into images by 20%.
- **Width Shift Range**: Randomly shifts images horizontally by 20% of the width.
- **Height Shift Range**: Randomly shifts images vertically by 20% of the height.

## Model Training and Evaluation

The model is compiled using the Adam optimizer and trained for 10 epochs. The training process involves monitoring changes in loss and accuracy, with plots generated to visualize these metrics over epochs.

## Results

The success of the project is evaluated based on the accuracy and loss metrics. The goal is to maximize accuracy while minimizing loss. Overfitting issues are addressed using dropout layers, and further evaluation is conducted to ensure the model's robustness.

## Requirements

- Python 3.x
- TensorFlow
- NumPy
- Pandas
- Matplotlib

## Conclusion
This project demonstrates the effective use of Convolutional Neural Networks (CNNs) for hand gesture recognition in American Sign Language (ASL). By applying data augmentation techniques and addressing overfitting, the model achieves high accuracy in classifying ASL letters, enhancing communication accessibility for individuals with hearing impairments.
