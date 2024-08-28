American Express Data Analysis
Project Overview
This project aims to predict customer churn for American Express by analyzing customer data. The goal is to identify patterns and risk factors associated with account closure, enabling proactive measures to retain customers.

Methodology
Data Preprocessing
Handling Missing Values: Imputed missing values using appropriate techniques (e.g., mean, median, mode, imputation methods).
Outlier Detection and Treatment: Identified and addressed outliers using statistical methods or domain knowledge.
Feature Scaling: Standardized numerical features to a common scale for better model convergence.
Model Architecture
Artificial Neural Network (ANN): A sequential ANN was employed, consisting of:
Input Layer: Number of neurons determined by the number of preprocessed features.
Hidden Layers: Two hidden layers with 6 neurons each, using ReLU activation function for non-linearity.
Output Layer: Single neuron with sigmoid activation function to predict the probability of churn.
Training
Splitting Data: Dataset was divided into training (80%) and testing (20%) sets.
Batch Size: 32 samples were processed in each batch during training.
Epochs: The model was trained for 120 epochs, allowing for iterative learning.
Optimizer: Adam optimizer was used for efficient gradient descent.
Evaluation
Accuracy: Overall correct predictions.
Confusion Matrix: Detailed breakdown of model performance for each class.
Precision: Proportion of correct positive predictions.
Recall: Proportion of actual positive cases correctly identified.
F1-Score: Harmonic mean of precision and recall.
ROC-AUC: Area under the Receiver Operating Characteristic curve, measuring the model's ability to distinguish between positive and negative classes.   
Results
Model Performance: The ANN achieved an accuracy of 85.35% on the testing set.
Evaluation Metrics: Precision, recall, F1-score, and ROC-AUC were also calculated and analyzed.
Insights: The confusion matrix provided insights into potential biases or areas where the model might be struggling.
