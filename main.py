import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score, 
    confusion_matrix, classification_report
)

# 1. LOAD DATASET & PREPROCESS
df = pd.read_csv("customer_churn_dataset-testing-master.csv")

# Drop non-predictive CustomerID column immediately
if 'CustomerID' in df.columns:
    df = df.drop(columns=['CustomerID'])

# Inspection
print(df.info())
print(df.head())
print(df.isnull().sum())

target_col = 'Churn'

# Categorical column encoding
categorical_cols = ['Gender', 'Subscription Type', 'Contract Length']
label_encoders = {}

for col in categorical_cols:
    le = LabelEncoder()
    df[col] = le.fit_transform(df[col].astype(str))
    label_encoders[col] = le

# Train/Test Feature Split
x = df.drop(columns=[target_col])
y = df[target_col].astype(int)

# Feature Scaling
scaler = StandardScaler()
x_scaled = scaler.fit_transform(x)

# Train-Test Split (80/20)
x_train, x_test, y_train, y_test = train_test_split(
    x_scaled, y, test_size=0.2, random_state=42, stratify=y
)

# 2. DATA ANALYSIS & VISUALIZATION (Matplotlib)
plt.figure(figsize=(11, 4.5))

# Plot 1: Churn Rate by Contract Length
plt.subplot(1, 2, 1)
contract_labels = label_encoders['Contract Length'].classes_
contract_churn = df.groupby('Contract Length')['Churn'].mean()
plt.bar(contract_labels, contract_churn.values, color=['#2b5c8f', '#d95f02', '#7570b3'])
plt.title('Churn Rate by Contract Length')
plt.xlabel('Contract Length')
plt.ylabel('Churn Rate')
plt.ylim(0, 1)

# Plot 2: Average Payment Delay by Churn Status
plt.subplot(1, 2, 2)
avg_delay = df.groupby('Churn')['Payment Delay'].mean()
plt.bar(['Active (0)', 'Churned (1)'], avg_delay.values, color=['#2ca02c', '#d62728'])
plt.title('Average Payment Delay (Days) by Churn Status')
plt.ylabel('Payment Delay (Days)')

plt.tight_layout()
plt.show()

# 3. MACHINE LEARNING
models = {
    "Logistic Regression": LogisticRegression(max_iter=1000, random_state=42),
    "Random Forest": RandomForestClassifier(n_estimators=100, max_depth=12, random_state=42)
}

results = {}
for name, model in models.items():
    model.fit(x_train, y_train)
    y_pred = model.predict(x_test)
    
    results[name] = {
        "model": model,
        "Accuracy": accuracy_score(y_test, y_pred),
        "Precision": precision_score(y_test, y_pred, zero_division=0),
        "Recall": recall_score(y_test, y_pred, zero_division=0),
        "F1 Score": f1_score(y_test, y_pred, zero_division=0),
        "CM": confusion_matrix(y_test, y_pred)
    }

# Display Model Comparison
comparison_df = pd.DataFrame(results).T.drop(columns=['model', 'CM'])
print("\n--- Model Evaluation Metrics ---")
print(comparison_df.round(4))

# Select Best Model
best_model_name = comparison_df['F1 Score'].astype(float).idxmax()
best_model = results[best_model_name]['model']
print(f"\nSelected Model: {best_model_name}")

# 4. PREDICTION FUNCTION
def predict_churn(customer_data: dict) -> tuple:
    """
    Accepts raw customer attributes dictionary and returns churn status and probability.
    """
    input_df = pd.DataFrame([customer_data])
    
    # Remove CustomerID if passed in input
    if 'CustomerID' in input_df.columns:
        input_df = input_df.drop(columns=['CustomerID'])
        
    # Apply fitted label encoders to categorical fields
    for col, le in label_encoders.items():
        if col in input_df.columns:
            input_df[col] = le.transform(input_df[col].astype(str))
            
    # Align features with model input
    input_df = input_df[x.columns]
    
    # Scale input features
    input_scaled = scaler.transform(input_df)
    
    # Generate prediction
    prediction = best_model.predict(input_scaled)[0]
    probability = best_model.predict_proba(input_scaled)[0][1]
    
    status = "Likely to Churn" if prediction == 1 else "Not Likely to Churn"
    return status, round(float(probability), 4)

# Demonstration with a sample record
sample_customer = {
    'Age': 45,
    'Gender': 'Female',
    'Tenure': 12,
    'Usage Frequency': 8,
    'Support Calls': 7,
    'Payment Delay': 25,
    'Subscription Type': 'Basic',
    'Contract Length': 'Monthly',
    'Total Spend': 300,
    'Last Interaction': 15
}

status, prob = predict_churn(sample_customer)
print(f"\nSample Prediction: {status} | Churn Probability: {prob:.2%}")