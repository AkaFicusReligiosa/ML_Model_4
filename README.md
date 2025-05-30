


# Heart Disease Predictor using Logistic Regression

This project implements a **Logistic Regression** model to predict the presence of heart disease in patients based on various medical attributes. Early detection can aid in preventive care and treatment planning.

---

## 🎯 Objective

To develop a reliable binary classification model that predicts whether a patient is at risk of heart disease using commonly recorded clinical parameters.

---

## 📁 Dataset

- **Source**: [UCI Heart Disease Dataset](https://archive.ics.uci.edu/ml/datasets/Heart+Disease)
- **Instances**: 303 patients
- **Features**: 13 attributes including age, sex, chest pain type, blood pressure, cholesterol, fasting blood sugar, etc.
- **Target**: Presence (`1`) or absence (`0`) of heart disease

---

## 🚀 Technologies Used

- Python
- Pandas, NumPy
- Scikit-learn
- Matplotlib / Seaborn
- Jupyter Notebook

---

## 🧠 Model Workflow

1. **Load and preprocess data**
   - Handle missing values
   - Feature scaling
   - Encoding categorical variables
2. **Split data into training and testing sets**
3. **Train Logistic Regression model**
4. **Evaluate model performance**
   - Accuracy, Precision, Recall, F1-score
   - Confusion matrix
5. **Predict on new patient data**

---

## 📊 Evaluation Metrics

- Accuracy
- Precision
- Recall
- F1-score
- Confusion Matrix

---

## 🧪 How to Run

1. Clone the repo:
   ```bash
   git clone https://github.com/yourusername/heart-disease-predictor.git
   cd heart-disease-predictor
````

2. Install required libraries:

   ```bash
   pip install -r requirements.txt
   ```

3. Run the notebook:

   ```bash
   jupyter notebook heart_disease_predictor.ipynb
   ```

---

## 🔮 Sample Prediction Code

```python
sample_data = [[63, 1, 3, 145, 233, 1, 0, 150, 0, 2.3, 0, 0, 1]]
prediction = model.predict(sample_data)
print("Heart Disease Prediction:", "Positive" if prediction[0] == 1 else "Negative")
```

---

## 📈 Results

* Model achieved approximately **85-88% accuracy** on the test dataset.
* Logistic Regression provided interpretable coefficients useful for medical insights.

---

## 📜 License

This project is licensed under the **MIT License**. See the [LICENSE](LICENSE) file for details.

---

## 🙌 Acknowledgements

* UCI Machine Learning Repository
* Scikit-learn documentation
* Open-source community contributions

```


