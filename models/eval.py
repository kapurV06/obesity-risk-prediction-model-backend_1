import joblib
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

def evaluate_model(name, y_true, y_pred):
    print(f"\n{name} Evaluation:")
    print("Accuracy:", accuracy_score(y_true, y_pred))
    print("\nClassification Report:")
    print(classification_report(y_true, y_pred))
    print("\nConfusion Matrix:")
    print(confusion_matrix(y_true, y_pred))

# Load saved files
y_test = joblib.load("y_test.pkl")
lr_pred = joblib.load("lr_pred.pkl")
dt_pred = joblib.load("dt_pred.pkl")
knn_pred = joblib.load("knn_pred.pkl")

# Evaluate each
evaluate_model("Logistic Regression", y_test, lr_pred)
evaluate_model("Decision Tree", y_test, dt_pred)
evaluate_model("KNN", y_test, knn_pred)
