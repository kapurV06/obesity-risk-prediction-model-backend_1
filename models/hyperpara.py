from sklearn.preprocessing import LabelEncoder
import pandas as pd
import joblib
import os

from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# 1. Load data
df = pd.read_csv("data/obesity_dataset.csv")

# 2. Separate features and target
X = df.drop('NObeyesdad', axis=1)
y = df['NObeyesdad']

# 3. Encode categorical features
feature_encoders = {}
for col in X.select_dtypes(include=['object']).columns:
    le = LabelEncoder()
    X[col] = le.fit_transform(X[col])
    feature_encoders[col] = le

# 4. Encode target
target_encoder = LabelEncoder()
y = target_encoder.fit_transform(y)

# 5. Feature scaling
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# 6. Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, test_size=0.4, random_state=42
)

# 7. Define hyperparameter grid for Gradient Boosting
param_grid = {
    'n_estimators': [50, 100],
    'learning_rate': [0.05, 0.1],
    'max_depth': [3, 5],
    'min_samples_split': [2, 5]
}

# 8. Perform Grid Search with 5-fold CV
grid_search = GridSearchCV(
    GradientBoostingClassifier(),
    param_grid,
    cv=5,
    n_jobs=-1,
    verbose=2
)

grid_search.fit(X_train, y_train)

# 9. Evaluate best model
print("\nBest Parameters:", grid_search.best_params_)
best_gb = grid_search.best_estimator_
gb_pred = best_gb.predict(X_test)

print("\nTuned Gradient Boosting Accuracy:", accuracy_score(y_test, gb_pred))
print("\nClassification Report:\n", classification_report(y_test, gb_pred))
print("\nConfusion Matrix:\n", confusion_matrix(y_test, gb_pred))

# 10. Save predictions and labels
joblib.dump(y_test, "y_test.pkl")
joblib.dump(gb_pred, "tuned_gradient_boosting_pred.pkl")

# 11. Save model and preprocessing tools
os.makedirs("models", exist_ok=True)
joblib.dump(best_gb, "models/tuned_gradient_boosting_model.pkl")
joblib.dump(best_gb, "models/final_obesity_model.pkl")
joblib.dump(scaler, "models/standard_scaler.pkl")
joblib.dump(X.columns, "models/feature_columns.pkl")
joblib.dump(feature_encoders, "models/feature_encoders.pkl")
joblib.dump(target_encoder, "models/target_encoder.pkl")

print("✅ Model and preprocessing tools saved successfully.")
