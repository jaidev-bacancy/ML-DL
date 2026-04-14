# ML-DL Project: Machine Learning & Deep Learning Models

A comprehensive machine learning project featuring supervised learning, unsupervised learning, and deep learning implementations with real-world datasets and practical applications.

---

## 📋 Project Overview

This project demonstrates various machine learning algorithms applied to different datasets with proper evaluation metrics. The project is organized into three main categories:
- **Supervised Learning**: Classification and Regression
- **Unsupervised Learning**: Clustering and Dimensionality Reduction
- **Deep Learning**: Neural Network-based approaches

---

## 🗂️ Directory Structure

```
ML-DL/
├── deep learning                                                                                                                              
│   ├── ANN_MNIST.ipynb                                                                                                                        
│   ├── cnn_animal_classifier.ipynb
│   └── sentiment_analysis_lstm.ipynb
├── machine learning
│   ├── supervised
│   │   ├── logistic_regression_social_network_ads.ipynb
│   │   ├── polynomial_regression_fish_weight.ipynb
│   │   ├── random_forest_customer_satisfaction.ipynb
│   │   └── svm_diabetes_classifier.ipynb
│   └── unsupervised
│       ├── dbscan_moon.ipynb
│       └── pca_5d_dataset.ipynb
├── generate_tree.py
└── README.md
```

---

## 📊 Supervised Learning Models

### 1. Logistic Regression - Social Network Ads Classification

| Aspect | Details |
|--------|---------|
| **Algorithm** | Logistic Regression |
| **Dataset** | Social Network Ads |
| **Data Source** | `Social_Network_Ads.csv` (Kaggle: dragonheir/logistic-regression) |
| **Task** | Binary classification of user advertisement purchase likelihood |
| **Evaluation Metrics** | Accuracy: 0.89 |
| **File** | `ml/supervised/logistic_regression_social_network_ads.ipynb` |

---

### 2. Support Vector Machine - Diabetes Prediction

| Aspect | Details |
|--------|---------|
| **Algorithm** | Support Vector Machine (SVC) |
| **Dataset** | Diabetes Prediction Dataset |
| **Data Source** | `diabetes_prediction_dataset.csv` (Kaggle: iammustafatz/diabetes-prediction-dataset) |
| **Samples** | 100,000 records |
| **Features** | Age, BMI, HbA1c_level, blood_glucose_level, gender, hypertension, heart_disease, smoking_history |
| **Task** | Binary classification of diabetes presence |
| **Evaluation Metrics** | Accuracy: 0.96 |
| **File** | `ml/supervised/svm_diabetes_classifier.ipynb` |

---

### 3. Random Forest - Airline Customer Satisfaction

| Aspect | Details |
|--------|---------|
| **Algorithm** | Random Forest Classifier (max_depth=5) |
| **Dataset** | Airline Passenger Satisfaction |
| **Data Source** | `train.csv` (Kaggle: teejmahal20/airline-passenger-satisfaction) |
| **Target Variable** | Satisfaction (satisfied/not satisfied) |
| **Task** | Multi-class classification of customer satisfaction |
| **Evaluation Metrics** | Accuracy: 0.92 |
| **File** | `ml/supervised/random_forest_customer_satisfaction.ipynb` |

---

### 4. Polynomial Regression - Fish Weight Prediction

| Aspect | Details |
|--------|---------|
| **Algorithm** | Polynomial Regression (Degree 2) with StandardScaler |
| **Dataset** | Fish Market |
| **Data Source** | `Fish.csv` (Kaggle: vipullrathod/fish-market) |
| **Task** | Regression - predicting fish weight from physical measurements |
| **File** | `ml/supervised/polynomial_regression_fish_weight.ipynb` |

---

## 🔍 Unsupervised Learning Models

### 1. Principal Component Analysis (PCA) - Dimensionality Reduction

| Aspect | Details |
|--------|---------|
| **Algorithm** | Principal Component Analysis (PCA) - 3 components |
| **Dataset** | Multivariate 5D Dataset |
| **Data Source** | `PCA_5D_dataset.csv` (Kaggle: shraddha4ever20/multivariate-5d-dataset-for-pca-and-ml) |
| **Original Dimensions** | 5 features |
| **Reduced Dimensions** | 3 principal components |
| **Task** | Dimensionality reduction while preserving variance |
| **File** | `ml/unsupervised/pca_5d_dataset.ipynb` |

---

### 2. DBSCAN - Clustering with Outlier Detection

| Aspect | Details |
|--------|---------|
| **Algorithm** | Density-Based Spatial Clustering of Applications with Noise (DBSCAN) |
| **Parameters** | eps=0.3, min_samples=4 |
| **Dataset** | Moon Dataset (scikit-learn synthetic: `make_moons`) |
| **Dataset Details** | n_samples=1,000, noise=0.05, 2 classes |
| **Task** | Clustering with automatic outlier/noise point detection |
| **File** | `ml/unsupervised/dbscan_moon.ipynb` |

---

## 🧠 Deep Learning Models

### 1. LSTM - Sentiment Analysis (Twitter Data)

| Aspect | Details |
|--------|---------|
| **Algorithm** | Long Short-Term Memory (LSTM) Neural Network |
| **Architecture** | Embedding Layer → LSTM → Dense Layers |
| **Dataset** | Twitter Entity Sentiment Analysis |
| **Data Source** | `twitter_training.csv`, `twitter_validation.csv` (Kaggle: jp797498e/twitter-entity-sentiment-analysis) |
| **Task** | Multi-class sentiment classification |
| **Classes** | Positive, Negative, Neutral, Irrelevant |
| **Evaluation Metrics** | Training Accuracy, Validation Accuracy, Training Loss, Validation Loss |
| **File** | `deep learning/sentiment_analysis_lstm.ipynb` |

---

### 2. Convolutional Neural Network (CNN) - Animal Classification

| Aspect | Details |
|--------|---------|
| **Algorithm** | Convolutional Neural Network (SimpleCNN) |
| **Architecture** | Conv2d (3→32→64→128 channels) + ReLU + MaxPooling + Dense layers |
| **Dataset** | Animals10 |
| **Data Source** | `/raw-img` folder (Kaggle: alessiocorrado99/animals10) |
| **Classes** | 10 animal categories |
| **Data Split** | 80% training, 20% testing |
| **Task** | Image classification of animals |
| **Evaluation Metrics** | Test Set Accuracy: 59.09% |
| **File** | `deep learning/cnn_animal_classifier.ipynb` |

---

### 3. Artificial Neural Network (ANN) - MNIST Handwritten Digits

| Aspect | Details |
|--------|---------|
| **Algorithm** | Artificial Neural Network (ANN) - Sequential Dense Layers |
| **Architecture** | Input (784) → Hidden Layers → Output (10 classes) |
| **Dataset** | MNIST (Modified National Institute of Standards and Technology) |
| **Data Source** | `mnist_train_small.csv`, `mnist_test.csv` |
| **Input** | 28×28 pixel images (784 features) |
| **Task** | Handwritten digit classification (0-9) |
| **Training Progress** | Loss: 0.912 → 0.056 (over 14 epochs) |
| **Evaluation Metrics** | Classification Accuracy (correct predictions / total predictions) |
| **File** | `deep learning/ANN_MNIST.ipynb` |