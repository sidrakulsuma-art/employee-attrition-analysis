# 1. IMPORT LIBRARIES
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, recall_score, roc_auc_score, confusion_matrix

# 2. LOAD DATASET
df = pd.read_csv("employee.csv")

# 3. BASIC CHECKS
print("Dataset Shape:", df.shape)

# 4. EDA - ATTRITION DISTRIBUTION
sns.countplot(x="Attrition", data=df)
plt.title("Employee Attrition Distribution")
plt.show()

print("Graph 1 Done")

# 5. TARGET VARIABLE
df["Attrition"] = df["Attrition"].map({"Yes": 1, "No": 0})

# 6. CONVERT ALL CATEGORICAL COLUMNS TO NUMBERS
df = pd.get_dummies(df, drop_first=True)

# 7. FEATURES AND TARGET
X = df.drop("Attrition", axis=1)
y = df["Attrition"]

# 8. TRAIN TEST SPLIT
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

# 9. LOGISTIC REGRESSION
lr = LogisticRegression(max_iter=5000)

print("Training Logistic Regression...")
lr.fit(X_train, y_train)

y_pred_lr = lr.predict(X_test)
y_prob_lr = lr.predict_proba(X_test)[:, 1]

# 10. RANDOM FOREST
rf = RandomForestClassifier(
    n_estimators=20,
    random_state=32
)

print("Training Random Forest...")
rf.fit(X_train, y_train)

y_pred_rf = rf.predict(X_test)
y_prob_rf = rf.predict_proba(X_test)[:, 1]

# 11. LOGISTIC REGRESSION RESULTS
print("\n=== LOGISTIC REGRESSION ===")
print("Accuracy:", accuracy_score(y_test, y_pred_lr))
print("Recall:", recall_score(y_test, y_pred_lr))
print("ROC-AUC:", roc_auc_score(y_test, y_prob_lr))

# 12. RANDOM FOREST RESULTS
print("\n=== RANDOM FOREST ===")
print("Accuracy:", accuracy_score(y_test, y_pred_rf))
print("Recall:", recall_score(y_test, y_pred_rf))
print("ROC-AUC:", roc_auc_score(y_test, y_prob_rf))

# 13. CONFUSION MATRIX
cm = confusion_matrix(y_test, y_pred_rf)

plt.figure(figsize=(6, 4))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")
plt.title("Confusion Matrix - Random Forest")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.show()

print("\nPROJECT COMPLETED SUCCESSFULLY")