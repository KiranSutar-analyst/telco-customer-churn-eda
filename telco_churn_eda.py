#Telco Customer Churn - EDA Project
#Goal: Identify key drivers of customer churn using Python (Pandas, Matplotlib, Seaborn)

import pandas as pd 
import numpy as np 
import matplotlib.pyplot as plt 
import seaborn as sns

sns.set_style('whitegrid')
plt.rcParams['figure.figsize']=(8,5)

# 1. Load Data 

df = pd.read_csv(r"C:\Users\Kiran Suthar\Downloads\WA_Fn-UseC_-Telco-Customer-Churn.csv")
df.head()

print(df.shape)

# 2. Initial Health Check

df.info()
df.isnull().sum()

df.describe()

# 3. Clean the Data

# TotalCharges is stored as text with some blank strings - convert to numeric
df['TotalCharges']= pd.to_numeric(df['TotalCharges'], errors='coerce')

# Check how many rows became NaN after conversion
print("Rows with missing TotalCharges after conversion:", df['TotalCharges'].isnull().sum())

# Drop those rows (usually a small number, negligible impact)
df.dropna(subset=['TotalCharges'], inplace= True)

# Drop customerID - it's just an identifier, not useful for analysis
df.drop('customerID', axis=1, inplace=True)

df.reset_index(drop=True, inplace=True)
print("Final shape after cleaning:",df.shape)

# 4. Overall Churn Rate

churn_rate = df['Churn'].value_counts(normalize=True) * 100
print(churn_rate)

plt.figure(figsize=(6,5))
sns.countplot(data=df, x='Churn',hue='Churn',palette='Set2',legend=False)
plt.title('Overall Churn Distribution')
plt.xlabel('Churn')
plt.ylabel('Number of Customers')
plt.show()

# 5. Q1: Does Contract Type Affect Churn?

plt.figure(figsize=(8,5))
sns.countplot(data=df, x='Contract', hue='Churn', palette='Set2')
plt.title('Churn by Contract Type')
plt.xlabel('Contract Type')
plt.ylabel('Number of Customers')
plt.show()

contract_churn = df.groupby('Contract')['Churn'].value_counts(normalize=True).unstack() * 100
print(contract_churn.round(2))

# 6. Q2 : Does Tenure Matter?

plt.figure(figsize=(8,5))
sns.histplot(data=df,x='tenure',hue='Churn',multiple='stack',bins=30,palette='Set2')
plt.title('Churn Distribution by Tenure (Months)')
plt.xlabel('Tenure (Months)')
plt.show()

# Avg Tenure Comparison
print(df.groupby('Churn')['tenure'].mean().round(1))

# 7. Q3 : Which Services Correlate with Retention?

service = ['OnlineSecurity','TechSupport','OnlineBackup','DeviceProtection']

fig, axes = plt.subplots(2, 2, figsize=(14, 10))
axes= axes.flatten()

for i, service in enumerate(service):
    sns.countplot(data=df,x=service,hue='Churn',ax=axes[i],palette='Set2')
    axes[i].set_title(f'Churn by {service}')

plt.tight_layout()
plt.show()

# 8. Q4 : Monthly Charges vs Churn

plt.figure(figsize=(8,5))
sns.boxplot(data=df,x='Churn',y='MonthlyCharges',palette='Set2')
plt.title('Monthly Charges by Churn Status')
plt.show()

print(df.groupby('Churn')['MonthlyCharges'].mean().round(2))

# 9. Q5 : Payment Method vs Churn

plt.figure(figsize=(10,5))
sns.countplot(data=df,x='PaymentMethod',hue='Churn',palette='Set2')
plt.title('Churn by Payment Method')
plt.xticks(rotation=20)
plt.show()

# 10. Correlation Heatmap (Numeric Features)

plt.figure(figsize=(8,6))
numeric_df=df.select_dtypes(include=[np.number])
sns.heatmap(numeric_df.corr(), annot=True, cmap='coolwarm',fmt='.2f')
plt.title('Correlation Heatmap - Numeric Features')
plt.show()


# Summary of Key Findings
#
# 1. Overall churn rate: 26.58% of customers churned.
# 2. Contract type is the strongest churn driver - month-to-month customers 
#    churn at 42.71% vs 11.28% (one-year) and 2.85% (two-year contracts).
# 3. Churned customers had shorter average tenure (18.0 months) vs retained 
#    customers (37.7 months).
# 4. Churned customers paid higher average monthly charges ($74.44) vs 
#    retained customers ($61.31).
# 5. High-risk segment: month-to-month customers with short tenure and 
#    higher monthly charges are most likely to churn - suggesting targeted 
#    retention offers for this segment could reduce churn.
# 6. Customers without OnlineSecurity or TechSupport churned at ~42%, 
#    nearly 3x higher than customers who had these services (~14-15%). 
#    A similar but weaker pattern was seen for OnlineBackup and DeviceProtection.
#    This suggests bundling these services could improve retention.

