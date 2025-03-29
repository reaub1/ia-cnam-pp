"""
Fraud Detection using Decision Tree Classifier

This script performs fraud detection using a Decision Tree Classifier. It reads
training and testing data from CSV files, prepares the data, trains a Decision
Tree model, and evaluates its performance using a confusion matrix. The decision
tree and confusion matrix are visualized using matplotlib and seaborn.

Steps:
1. Import necessary libraries.
2. Load the training and testing datasets from CSV files.
3. Prepare the data by separating features and target variables.
4. Train a Decision Tree Classifier on the training data.
5. Predict the target variable for the testing data.
6. Visualize the decision tree and confusion matrix.

Dependencies:
- pandas
- matplotlib
- sklearn
- seaborn

Author: [Your Name]
Date: [Current Date]
"""

import pandas as pd
from matplotlib import pyplot as plt
from sklearn.metrics import confusion_matrix
from sklearn.tree import DecisionTreeClassifier, plot_tree
import seaborn as sns

# Load the data
train_data = pd.read_csv("dataset/fraud_train.csv")
test_data = pd.read_csv("dataset/fraud_test.csv")

# Prepare the data
X_train = train_data.drop("is_fraud", axis=1)
y_train = train_data["is_fraud"]

X_test = test_data.drop("is_fraud", axis=1)
y_test = test_data["is_fraud"]

# Train the Decision Tree Classifier
model = DecisionTreeClassifier()
model.fit(X_train, y_train)

# Predict the test data
y_pred = model.predict(X_test)

# Visualize the decision tree
plt.figure(figsize=(50, 50))
plot_tree(
    model,
    feature_names=X_train.columns,
    class_names=["Not Fraud", "Fraud"],
    filled=True,
    rounded=True,
)
plt.show()

# Visualize the confusion matrix
confusion_matrix = confusion_matrix(y_test, y_pred)
sns.heatmap(confusion_matrix, annot=True, fmt="d", cmap="Blues")
plt.title("Confusion Matrix Heatmap")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.show()
