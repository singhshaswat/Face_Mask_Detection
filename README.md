# 😷 Face Mask Detection System — Real-Time Deep Learning Application

This project implements a real-time face mask detection system using deep learning and computer vision techniques.
It classifies detected faces into three categories:

✅ With Mask

❌ Without Mask

⚠️ Mask Worn Incorrectly

The system combines OpenCV-based face detection with an InceptionV3 CNN classifier to achieve high accuracy while maintaining real-time performance.

# 📌 Project Motivation

During the COVID-19 pandemic, enforcing mask compliance in crowded environments such as airports, hospitals, and malls became essential. Manual monitoring is inefficient and error-prone.

This project aims to:

Automate mask detection

Support real-time deployment

Reduce hardware requirements

Improve detection under real-world conditions

# 🧠 Model Architecture
🔍 Pipeline

Detect faces using OpenCV DNN (res10_300x300_ssd_iter_140000)

Crop detected face regions

Resize to 224 × 224

Preprocess images

Classify using fine-tuned InceptionV3

# 🏗️ Classification Head

Flatten

Dense (1024, ReLU)

Dropout (0.2)

Dense (512, ReLU)

Dense (3, Softmax)

# ⚙️ Training Details

Base Model: InceptionV3 (ImageNet pretrained)

Loss: Sparse Categorical Crossentropy

Optimizer: Adam

Epochs: 20

Batch Size: 16

Augmentation: Rotation, zoom, flip, shift

# 📊 Results

Final validation accuracy: 91.2%

Confusion Matrix & Metrics

From the evaluation graphs:

High precision for with_mask and without_mask

Slightly lower recall for mask_worn_incorrect due to limited dataset samples

ROC-AUC:

With Mask → 0.98

Without Mask → 0.99

Incorrect Mask → 0.91

Sample real-time detections are shown in the report images 

Face_mask_detection_report

# ⏱️ Performance

Suitable for real-time inference

Tested on GPU-enabled Kaggle environment

Optimized for deployment with lightweight inference pipelines

# 🛠️ Technologies Used

Python

TensorFlow / Keras

OpenCV

NumPy

Matplotlib

Jupyter Notebook

# Author
Shaswat Singh
