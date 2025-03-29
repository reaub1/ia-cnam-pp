"""
Customer Segmentation using K-Means Clustering

This script performs customer segmentation using the K-Means clustering algorithm.
It reads customer data from a CSV file, scales the features, and applies K-Means
clustering to segment the customers into different clusters. The results are visualized
using scatter plots to show the relationship between different features.

Steps:
1. Import necessary libraries.
2. Load the dataset from a CSV file.
3. Scale the features using StandardScaler.
4. Apply K-Means clustering to the scaled data.
5. Visualize the clustering results using scatter plots.

Dependencies:
- pandas
- sklearn
- matplotlib

Author: [Your Name]
Date: [Current Date]
"""

import pandas as pd
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt

# Load the dataset
data = pd.read_csv("dataset/customer_train.csv")

# Scale the features
scaler = StandardScaler()
featured_scaled = scaler.fit_transform(data)
features_scaled_df = pd.DataFrame(featured_scaled, columns=data.columns)

# Apply K-Means clustering
k = 3
model = KMeans(n_clusters=k, random_state=42)
data["cluster"] = model.fit_predict(featured_scaled)

# Create scatter plots for different feature combinations
fig = plt.figure(figsize=(15, 5))

# Income vs Spending Score
plt.subplot(1, 3, 1)
plt.scatter(data["annual_income"], data["spending_score"], c=data["cluster"])
plt.xlabel("Annual Income")
plt.ylabel("Spending Score")
plt.title("Income vs Spending Score")

# Age vs Spending Score
plt.subplot(1, 3, 2)
plt.scatter(data["age"], data["spending_score"], c=data["cluster"])
plt.xlabel("Age")
plt.ylabel("Spending Score")
plt.title("Age vs Spending Score")

# Age vs Income
plt.subplot(1, 3, 3)
plt.scatter(data["age"], data["annual_income"], c=data["cluster"])
plt.xlabel("Age")
plt.ylabel("Annual Income")
plt.title("Age vs Income")

plt.tight_layout()
plt.show()
