# 🧠 Obesity Risk Prediction Model

A machine learning project that predicts the *obesity level* of an individual based on health, lifestyle, and demographic features using various classification and ensemble learning techniques.

---

## 📁 Project Structure


obesity-risk-prediction/
├── data/                   # Raw and processed data
├── models/                 # Saved trained models
├── notebooks/              # Jupyter notebooks for experimentation
├── images/                 # Saved plots and visualizations
├── app/                    # Deployment-related files (Flask app)
├── README.md               # Project overview and instructions


---

 📊 Dataset Description

The dataset contains *individual-level health and lifestyle data*, including:

* Demographics: Age, Gender, Height, Weight
* Habits: Eating frequency, calorie intake, physical activity, water consumption, alcohol consumption
* Target: NObeyesdad (Obesity Level classification)

Classes of target variable:

* Insufficient_Weight
* Normal_Weight
* Overweight_Level_I
* Overweight_Level_II
* Obesity_Type_I
* Obesity_Type_II
* Obesity_Type_III

> 💡 Source: UCI Machine Learning Repository (or as provided in the assignment)

---

 🚀 Methodology

 ✅ Step 1: Environment Setup

* Python 3.9+
* Required packages installed via:

bash
pip install numpy pandas matplotlib seaborn scikit-learn xgboost lightgbm catboost


---

 ✅ Step 2: Data Exploration

* Loaded the dataset using pandas
* Visualized class distribution using seaborn
* Plotted correlation heatmap

---

 ✅ Step 3: Preprocessing

* Handled missing values
* Encoded categorical features using LabelEncoder / get_dummies
* Scaled numeric features using StandardScaler
* Split data into training and test sets

---

 ✅ Step 4: Baseline Models

* *Logistic Regression*
* *Decision Tree*
* *K-Nearest Neighbors (KNN)*

Each model was evaluated using:

* Accuracy Score
* Classification Report

---

✅ Step 5: Ensemble Techniques

* *Bagging*: Random Forest
* *Boosting*: XGBoost
* *Voting Classifier*: Soft voting using LR, RF, XGB
* *Stacking*: LR, RF, XGB with SVM as final estimator

---

 ✅ Step 6: Model Evaluation

* Saved predictions using joblib
* Compared models using:

  * Accuracy
  * Classification Report
  * Confusion Matrix

---

 ✅ Step 7: Hyperparameter Tuning

Used GridSearchCV to tune:

* Random Forest
* Gradient Boosting (best performing)

Saved best estimator using:

python
joblib.dump(best_model, "models/tuned_gradient_boosting_model.pkl")


---

 ✅ Step 8: Feature Importance

* Used .feature_importances_ from Gradient Boosting
* Visualized using seaborn.barplot

---

 ✅ Step 9: Deployment Prep

* Final model selected based on performance
* Saved using joblib
* Flask-based web app created for real-time prediction
* Includes dynamic form and result display

---

 💻 How to Run

 1. Clone the repo

bash
git clone https://github.com/yourusername/obesity-risk-prediction.git
cd obesity-risk-prediction

 2. Setup Virtual Environment

bash
python3 -m venv venv
source venv/bin/activate  # Mac/Linux
venv\Scripts\activate     # Windows


3. Install Dependencies

bash
pip install -r requirements.txt


 4. Run Jupyter Notebooks

bash
jupyter notebook notebooks/


 5. Launch Flask App (optional)

bash
cd app/
python app.py


Visit http://127.0.0.1:5000 to test the web interface.

---

 📈 Results & Conclusion

* Best model: Gradient Boosting with hyperparameter tuning
* *Accuracy: \~90%+
* *Key Features*: Water intake, physical activity, calorie monitoring, screen time

> This model offers reliable classification of obesity risk levels and can be integrated into health-related applications or tools.

---

 📌 Future Improvements

* Cross-validation with stratified folds
* Use of advanced deep learning models
* UI enhancement for mobile compatibility
* Feature selection and dimensionality reduction

---

 📬 Contact

For questions or collaboration:

📧 goyalaarush232@gmail.com
📧 kapur.varish75@gmail.com