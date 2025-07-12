import pandas as pd
import joblib

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, classification_report

# 1. Load data
df = pd.read_csv('data/obesity_dataset.csv')

# 2. Separate features and target
X = df.drop('NObeyesdad', axis=1)
y = df['NObeyesdad']

# 3. Encode categorical features (if any)
X = pd.get_dummies(X)

# 4. Feature scaling (important for KNN and Logistic Regression)
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# 5. Split into train and test sets
X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, test_size=0.2, random_state=42
)

# 6. Logistic Regression
lr = LogisticRegression(max_iter=1000)
lr.fit(X_train, y_train)
lr_pred = lr.predict(X_test)
print("Logistic Regression Accuracy:", accuracy_score(y_test, lr_pred))
print(classification_report(y_test, lr_pred))

# 7. Decision Tree
dt = DecisionTreeClassifier()
dt.fit(X_train, y_train)
dt_pred = dt.predict(X_test)
print("Decision Tree Accuracy:", accuracy_score(y_test, dt_pred))
print(classification_report(y_test, dt_pred))

# 8. KNN
knn = KNeighborsClassifier()
knn.fit(X_train, y_train)
knn_pred = knn.predict(X_test)
print("KNN Accuracy:", accuracy_score(y_test, knn_pred))
print(classification_report(y_test, knn_pred))

# 9. Save test labels and predictions for external evaluation
joblib.dump(y_test, "y_test.pkl")
joblib.dump(lr_pred, "lr_pred.pkl")
joblib.dump(dt_pred, "dt_pred.pkl")
joblib.dump(knn_pred, "knn_pred.pkl")
