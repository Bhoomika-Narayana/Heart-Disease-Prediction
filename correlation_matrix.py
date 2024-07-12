import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# Load the dataset (assuming 'heart.csv' is the dataset file)
df = pd.read_csv('heart.csv')

# Select relevant features for correlation analysis
selected_features = ['age', 'sex', 'cp', 'trestbps', 'chol', 'fbs', 'restecg', 'thalach', 'exang', 'oldpeak', 'slope', 'ca', 'thal', 'target']

# Subset the dataframe with selected features
selected_df = df[selected_features]

# Compute the correlation matrix
correlation_matrix = selected_df.corr()

# Plot the correlation matrix using seaborn
plt.figure(figsize=(12, 10))
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt=".2f", linewidths=.5)
plt.title('Correlation Matrix')
plt.show()
