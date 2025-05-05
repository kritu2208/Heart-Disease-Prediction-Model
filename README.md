# ❤️ Heart Disease Prediction Model

A machine learning-based classification project to predict the presence of heart disease in patients based on clinical and lifestyle attributes. The goal is to support early diagnosis and improve preventive healthcare using data-driven models.

---

## 🩺 Project Objective

Heart disease remains one of the leading causes of death globally. Early detection through data analysis can help healthcare professionals make timely decisions. This project builds a predictive model using medical data to classify whether a person is likely to have heart disease.

---

## 📊 Dataset Overview

**Source:** [UCI Machine Learning Repository](https://archive.ics.uci.edu/ml/datasets/heart+Disease)

The dataset includes 14 features, such as:
- `age`, `sex`, `cp` (chest pain type)
- `trestbps` (resting blood pressure)
- `chol` (serum cholesterol)
- `fbs` (fasting blood sugar), `restecg` (resting ECG)
- `thalach` (max heart rate achieved), `exang` (exercise-induced angina)
- `oldpeak`, `slope`, `ca`, `thal`

**Target variable:**  
`target` – 0 = no heart disease, 1 = presence of heart disease

---

## 🛠️ Project Workflow

### 1. Data Exploration
- Checked data types, missing values, and unique categories
- Explored distributions and outliers using histograms and boxplots

### 2. Data Preprocessing
- Encoded categorical variables (e.g., chest pain, thal, slope)
- Scaled numerical features with `StandardScaler`
- Split data into training and testing sets

### 3. Model Training
Trained multiple classifiers including:
- Logistic Regression
- Random Forest
- K-Nearest Neighbors (KNN)
- Support Vector Machine (SVM)
- (Optional: XGBoost, Decision Tree)

### 4. Model Evaluation
- Evaluated models using accuracy, precision, recall, F1-score
- Plotted confusion matrices and ROC curves
- Selected the best-performing model for deployment

---

## 📈 Model Performance

| Model              | Accuracy | Precision | Recall | F1 Score |
|-------------------|----------|-----------|--------|----------|
| Logistic Regression | 86.0%   | 87.5%     | 85.1%  | 86.3%    |
| Random Forest       | 88.5%   | 89.2%     | 87.4%  | 88.3%    |
| KNN                 | 83.1%   | 84.1%     | 82.0%  | 83.0%    |

_The Random Forest model yielded the best balance of metrics._

---

## 📁 Project Structure

Heart-Disease-Prediction-Model/
├── heart.csv # Dataset
├── Heart_Disease_Prediction.ipynb # Full ML workflow in notebook
├── heart_model.joblib # Trained model (for saving/deployment)
├── images/ # Visuals and plots
└── README.md # Project documentation


---

## 📊 Visualizations

- **Correlation Heatmap** to understand feature importance  
- **Distribution plots** of numerical variables (e.g., age, cholesterol)  
- **Count plots** for categorical features (e.g., chest pain types)  
- **ROC Curve** and **Confusion Matrix** for model evaluation

_Examples:_

![Image](https://github.com/user-attachments/assets/546c737f-6115-4c2d-991c-7bf3b856e372)

![Image](https://github.com/user-attachments/assets/d15d9e65-a01e-4dcd-b31d-4c3aeceecf5e)

![Image](https://github.com/user-attachments/assets/05209d3c-4fdb-4001-84a0-242cbb149aff)

---

## 🔧 Technologies Used

| Category        | Tools/Libraries                        |
|----------------|----------------------------------------|
| Language        | Python                                 |
| Data Handling   | pandas, numpy                          |
| Visualization   | seaborn, matplotlib                    |
| ML Algorithms   | scikit-learn                           |
| Model Saving    | joblib                                 |
| Environment     | Jupyter Notebook                       |

---

## 🧠 Key Takeaways

- **Chest pain type**, **thal**, and **max heart rate** are strong indicators of heart disease.
- The **Random Forest** classifier outperforms other models in prediction accuracy.
- Feature scaling and proper encoding significantly improve model performance.
- A simple data pipeline can support real-world healthcare diagnostics.

---

## 🔮 Possible Extensions

- Implement cross-validation and hyperparameter tuning
- Integrate with a simple front-end or Flask API
- Expand dataset with real-world health records
- Feature importance explanation with SHAP or LIME

---


## 🙌 Acknowledgements

- Dataset from UCI Machine Learning Repository
- Medical inspiration from ongoing AI efforts in preventive healthcare

---

## ⭐ Like this Project?

If this helped or inspired you, please give it a ⭐ on GitHub!
