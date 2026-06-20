# Facial Emotion Recognition System using Deep Learning

## Project Overview

This project implements a Facial Emotion Recognition (FER) system capable of identifying human emotions from facial images and real-time video streams using Deep Learning and Computer Vision techniques.

The system classifies facial expressions into seven emotion categories:

* Angry
* Disgust
* Fear
* Happy
* Neutral
* Sad
* Surprise

The project includes:

* Custom CNN-based Emotion Recognition Model
* Transfer Learning Experiment using ResNet50V2
* Streamlit Web Application for Image-Based Emotion Detection
* Real-Time Webcam Emotion Detection using OpenCV
* End-to-End Deep Learning Pipeline from Training to Deployment

<img width="691" height="468" alt="image" src="https://github.com/user-attachments/assets/df290f52-cd09-4729-baee-3eeb2d82c53a" />

<img width="658" height="545" alt="image" src="https://github.com/user-attachments/assets/b6b991d8-9378-4c35-b90f-71e800ebd995" />

### **Streamlit App**

<img width="2466" height="1048" alt="Happy" src="https://github.com/user-attachments/assets/86d12931-4aa7-4557-a1e9-d98c53b311c9" />


---

## Problem Statement

Human emotions play a critical role in communication and decision-making. Automatically recognizing emotions from facial expressions can improve human-computer interaction and support applications such as:

* Virtual Assistants
* Customer Sentiment Analysis
* Mental Health Monitoring
* Smart Education Systems
* User Experience Enhancement

Facial emotion recognition is a challenging computer vision problem because:

* Emotions often exhibit subtle facial differences
* Multiple emotions share similar facial characteristics
* Lighting conditions and facial orientations vary significantly
* Expression intensity differs across individuals

Deep learning provides an effective solution by automatically learning hierarchical facial features directly from images without requiring manual feature engineering.

---

## Tech Stack

### Programming Language

* Python

### Deep Learning Frameworks

* TensorFlow
* Keras

### Computer Vision

* OpenCV

### Data Processing

* NumPy
* Pandas

### Deployment

* Streamlit

### Visualization

* Matplotlib
* Seaborn

---

## Dataset

The dataset consists of grayscale facial images categorized into seven emotion classes:

| Emotion  |
| -------- |
| Angry    |
| Disgust  |
| Fear     |
| Happy    |
| Neutral  |
| Sad      |
| Surprise |

Images were preprocessed and normalized before being used for model training.

### Data Augmentation

To improve generalization and reduce overfitting, the following augmentation techniques were applied:

* Rotation
* Width Shift
* Height Shift
* Zoom
* Horizontal Flip

The dataset was divided into training, validation, and testing sets for unbiased model evaluation.

---

# Approach 1: Custom CNN (Final Model)

A custom Convolutional Neural Network was designed specifically for low-resolution facial emotion recognition.

### Architecture Overview

Input Image (48×48 Grayscale)

↓

Convolution Layers

↓

Batch Normalization

↓

Max Pooling

↓

Dropout

↓

Fully Connected Dense Layer

↓

Softmax Output Layer (7 Classes)


### Key Components

* 6 Convolutional Layers
* Batch Normalization Layers
* Max Pooling Layers
* Dropout Regularization
* Dense Classification Layer
* Softmax Output Layer

The CNN progressively learns:

* Low-level facial edges
* Facial structures
* Emotion-specific patterns

---

## Training Strategy

The model was trained using:

* Adam Optimizer
* Categorical Cross Entropy Loss
* Batch Normalization
* Dropout Regularization

### Callbacks Used

#### ModelCheckpoint

Automatically saves the best-performing model during training.

#### ReduceLROnPlateau

Reduces learning rate when validation performance stops improving, allowing finer optimization during later training stages.

---

## Model Performance

### Validation Results

| Metric             | Score  |
| ------------------ | ------ |
| Accuracy           | 64.87% |
| Validation Loss    | 0.9820 |
| Weighted Precision | 67%    |
| Weighted Recall    | 65%    |
| Weighted F1 Score  | 65%    |

### Best Performing Classes

| Emotion  | F1 Score |
| -------- | -------- |
| Happy    | 0.87     |
| Surprise | 0.77     |

These emotions contain distinctive facial characteristics and were easier for the model to identify.

### More Challenging Classes

| Emotion | F1 Score |
| ------- | -------- |
| Fear    | 0.48     |
| Sad     | 0.50     |
| Angry   | 0.55     |

These classes frequently overlap with Neutral and other visually similar emotions.

### Common Misclassification Patterns

* Fear → Neutral
* Fear → Sad
* Sad → Neutral
* Angry → Neutral

These patterns are expected due to similarities in facial muscle movements and expression intensity.

---

# Approach 2: ResNet50V2 Transfer Learning

A second approach was implemented using ResNet50V2, a deep residual network pre-trained on the ImageNet dataset.

### Process

* Loaded pre-trained ResNet50V2 weights
* Used the network as a feature extractor
* Added custom classification layers
* Fine-tuned the model for emotion recognition

### Purpose

The objective was to compare a transfer learning approach against a custom CNN architecture specifically designed for facial emotion recognition.

---

## Why the CNN Outperformed ResNet50V2

Although transfer learning is often highly effective, the custom CNN performed better for this specific task due to several reasons:

### 1. Dataset Characteristics

FER images are:

* Low Resolution (48×48)
* Grayscale

ResNet50V2 is optimized for larger RGB images.

### 2. Domain Difference

ResNet learns general object features from ImageNet, while emotion recognition requires fine-grained facial expression analysis.

### 3. Computational Efficiency

The custom CNN:

* Trains Faster
* Requires Less Memory
* Produces Faster Inference

### 4. Real-Time Deployment Suitability

The lightweight CNN architecture was better suited for real-time webcam-based emotion recognition.

For these reasons, the custom CNN was selected as the final production model.

---

# Deployment

## Streamlit Emotion Detection App

The trained CNN model was integrated into a Streamlit application that allows users to:

* Upload facial images
* Predict emotions
* View confidence scores
* Visualize probability distributions

### Workflow

Image Upload

↓

Preprocessing

↓

CNN Prediction

↓

Emotion Classification

↓

Probability Visualization


---

## Real-Time Webcam Emotion Detection

OpenCV was used to build a real-time emotion recognition system.

### Workflow

Webcam Feed

↓

Face Detection using Haar Cascade

↓

Face Extraction

↓

Image Preprocessing

↓

CNN Prediction

↓

Live Emotion Display


The system continuously detects faces and overlays predicted emotions in real time.

---

## Project Structure

```text
├── Facial_Emotion_Recognition_Model_CNN.ipynb
├── Facial_Emotion_Recognition_Model_ResNet50V2.ipynb
├── cnn_model_new_.h5
├── app.py
├── webcam.py
├── requirements.txt
├── README.md
```

---

## Conclusion

This project successfully developed an end-to-end facial emotion recognition system capable of classifying seven human emotions from both static images and live video streams. Multiple deep learning approaches were evaluated, including a custom CNN and a transfer learning-based ResNet50V2 model. The custom CNN ultimately provided the best balance of accuracy, computational efficiency, and real-time performance, making it the preferred choice for deployment. The project demonstrates the complete machine learning lifecycle, from data preprocessing and model development to evaluation, deployment, and real-time inference.
