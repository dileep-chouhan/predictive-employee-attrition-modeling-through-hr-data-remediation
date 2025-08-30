import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
# --- 1. Synthetic Data Generation ---
np.random.seed(42) # for reproducibility
# Generate synthetic HR data
data = {
    'Age': np.random.randint(20, 60, 1000),
    'YearsAtCompany': np.random.randint(0, 20, 1000),
    'Department': np.random.choice(['Sales', 'Marketing', 'Engineering', 'HR', 'Finance'], 1000),
    'Salary': np.random.randint(40000, 150000, 1000),
    'Attrition': np.random.choice([0, 1], 1000, p=[0.8, 0.2]) # 20% attrition rate
}
# Introduce missing values and inconsistencies
data['Salary'][np.random.choice(range(1000), 100)] = np.nan
data['YearsAtCompany'][np.random.choice(range(1000), 50)] = -1 #invalid value
df = pd.DataFrame(data)
# --- 2. Data Cleaning and Preparation ---
# Handle missing values (Salary)
imputer = SimpleImputer(strategy='median')
df['Salary'] = imputer.fit_transform(df[['Salary']])
# Handle inconsistencies (YearsAtCompany)
df['YearsAtCompany'] = df['YearsAtCompany'].apply(lambda x: 0 if x < 0 else x)
# One-hot encode categorical feature (Department)
df = pd.get_dummies(df, columns=['Department'], prefix=['Dept'])
# --- 3. Data Analysis and Visualization ---
# Feature correlation analysis
correlation_matrix = df.corr()
plt.figure(figsize=(10,8))
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm')
plt.title('Correlation Matrix of HR Data')
plt.savefig('correlation_matrix.png')
print("Plot saved to correlation_matrix.png")
# Attrition rate by department (before cleaning)
plt.figure(figsize=(10,6))
sns.countplot(x='Department', hue='Attrition', data=pd.DataFrame(data))
plt.title('Attrition Rate by Department (Before Cleaning)')
plt.savefig('attrition_before.png')
print("Plot saved to attrition_before.png")
# Attrition rate by age (after cleaning)
plt.figure(figsize=(10,6))
sns.histplot(data=df, x='Age', hue='Attrition', kde=True, multiple="stack")
plt.title('Attrition Rate by Age (After Cleaning)')
plt.savefig('attrition_after.png')
print("Plot saved to attrition_after.png")
# --- 4.  Data Splitting and Scaling (for potential model building)---
X = df.drop('Attrition', axis=1)
y = df['Attrition']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)
print("Data cleaning and preparation complete. Data is ready for predictive modeling.")