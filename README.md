# Real-Time Facial Emotion Recognition System

An AI-powered computer vision system that detects human faces and classifies emotions in real-time using Convolutional Neural Networks (CNN).

## Technologies Used
- Python
- TensorFlow / Keras
- OpenCV
- Streamlit

## Features
- Real-time webcam emotion detection
- Face detection with bounding box
- CNN-based emotion classification
- Live deployment interface

- ## Model Performance & Evaluation

The CNN-based facial emotion recognition model was evaluated on the FER dataset using standard classification metrics.

### Evaluation Metrics
- Accuracy
- Precision
- Recall
- F1-Score
- Confusion Matrix Analysis

### Overall Accuracy
The model achieved an approximate accuracy of **XX%** on the test dataset.

While the performance is promising for a baseline CNN architecture, certain emotion classes such as *Fear* and *Sad* showed higher misclassification rates due to:
- Similar facial feature patterns
- Dataset imbalance
- Low lighting variations
- Subtle expression differences

---

##  Error Analysis

From the confusion matrix analysis, we observed:

- Misclassification between *Sad* and *Neutral*
- Confusion between *Fear* and *Surprise*
- Performance degradation in low-light webcam conditions

These errors highlight the challenges of real-world emotion recognition compared to controlled datasets.

---

##  Theoretical Improvements

To enhance model performance, the following improvements are proposed:

### 1️⃣ Advanced Architectures
- Replace basic CNN with **ResNet, EfficientNet, or MobileNet**
- Use Transfer Learning instead of training from scratch

### 2️⃣ Data Augmentation
- Random brightness & contrast adjustment
- Horizontal flipping
- Rotation and zoom
- Noise injection

### 3️⃣ Class Imbalance Handling
- Use class weights
- Apply oversampling techniques
- Use Focal Loss

### 4️⃣ Hyperparameter Optimization
- Learning rate scheduling
- Batch size tuning
- Early stopping regularization

### 5️⃣ Real-World Optimization
- Face alignment preprocessing
- Histogram equalization
- Image normalization improvement

---

## Future Work

- Implement Transfer Learning with pre-trained CNN models
- Deploy the model using optimized inference pipeline
- Improve real-time performance using TensorRT
- Expand dataset diversity to enhance generalization

