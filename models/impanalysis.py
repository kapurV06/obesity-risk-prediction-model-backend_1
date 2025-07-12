import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import joblib

# 1. Load model and actual feature names
best_gb = joblib.load("models/tuned_gradient_boosting_model.pkl")
feature_columns = joblib.load("models/feature_columns.pkl")

# 2. Get importances
importances = best_gb.feature_importances_
feature_importance = pd.DataFrame({
    'Feature': feature_columns,
    'Importance': importances
}).sort_values('Importance', ascending=False)

# 3. Plot
plt.figure(figsize=(10, 6))
sns.barplot(x='Importance', y='Feature', data=feature_importance)
plt.title('Feature Importance from Gradient Boosting (Corrected)')
plt.tight_layout()
plt.show()
