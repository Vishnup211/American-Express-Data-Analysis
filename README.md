# American-Express-Data-Analysis

Project Overview
This project aims to predict whether an American Express customer is likely to close their account. By analyzing customer data, we can identify patterns and risk factors associated with account closure, enabling proactive measures to retain customers.

Methodology
Preprocessing:
  Handling missing values
  Outlier detection and treatment
  Feature scaling
Architecture
The implemented ANN model uses a sequential architecture with the following layers:

Input Layer: The size of this layer depends on the number of preprocessed features in your data (likely the number of columns after one-hot encoding).
Hidden Layer 1: This layer has 6 neurons with a ReLU (Rectified Linear Unit) activation function. ReLU introduces non-linearity to the model, allowing it to learn complex relationships between features.
Hidden Layer 2: Another hidden layer with 6 neurons and ReLU activation provides further abstraction and feature learning.
Output Layer: The final layer has 1 neuron with a sigmoid activation function. The sigmoid function outputs a value between 0 and 1, which can be interpreted as the probability of a customer closing their account.
Training
The model training process involved the following steps:

Splitting Data: The dataset was split into training and testing sets using a 80/20 ratio. The training set is used to train the model, and the testing set is used to evaluate its performance on unseen data.
Batch Size: A batch size of 32 was used. This means the model updates its weights based on a mini-batch of 32 data points during each training iteration.
Epochs: The model was trained for 120 epochs. An epoch represents one complete pass through the entire training dataset.
Optimizer: The Adam optimizer was used for gradient descent. Adam is an efficient optimization algorithm that adapts the learning rate for each parameter based on historical gradients.
Evaluation
The following metrics were used to assess model performance:

Accuracy: This metric measures the overall percentage of correct predictions made by the model. Your code snippet shows an accuracy of 85.35%.
Confusion Matrix: The confusion matrix provides a detailed breakdown of how the model performed on each class. It helps identify potential biases or class imbalances.
While accuracy is a good starting point, it's important to consider other metrics depending on the problem.  For imbalanced datasets (where one class is much more frequent), metrics like precision, recall, and F1-score might be more informative. Additionally, you could explore ROC-AUC (Area Under the ROC Curve) to evaluate the model's ability to discriminate between account closure and non-closure.

Results

Model Performance: Report the accuracy score (85.35%) and any other relevant metrics (e.g., precision, recall, F1-score, AUC-ROC) calculated on the test set.
Comparison: If you have a baseline model (e.g., logistic regression), compare its performance to the ANN model. Did the ANN perform significantly better?
Insights: Analyze the confusion matrix to identify potential issues. Are there any classes where the model performs poorly? This might indicate the need for further data exploration or model adjustments.
By analyzing these results, you can gain valuable insights into the effectiveness of the ANN model for predicting customer account closure.
