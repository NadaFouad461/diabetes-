# Diabetes Prediction using Machine Learning

## 📌 Project Overview

This project focuses on predicting whether a person has diabetes or not using machine learning classification algorithms.

The project uses the **Pima Indians Diabetes Dataset**, which contains medical and demographic features that can be used to predict diabetes.

The target variable is:

* `0` → No Diabetes
* `1` → Diabetes

---

## 📊 Dataset

The dataset contains **768 observations** and **9 columns**.

### Features

| Feature                    | Description                    |
| -------------------------- | ------------------------------ |
| `Pregnancies`              | Number of pregnancies          |
| `Glucose`                  | Plasma glucose concentration   |
| `BloodPressure`            | Diastolic blood pressure       |
| `SkinThickness`            | Triceps skin fold thickness    |
| `Insulin`                  | 2-Hour serum insulin           |
| `BMI`                      | Body Mass Index                |
| `DiabetesPedigreeFunction` | Diabetes hereditary risk score |
| `Age`                      | Age of the person              |
| `Outcome`                  | Target variable                |

### Target

`Outcome` is the target variable:

* `0`: No diabetes
* `1`: Diabetes

---

##  Exploratory Data Analysis (EDA)

Several visualizations were used to understand the dataset and the relationship between the features and the target variable.

### Visualizations

* Target variable distribution
* Histograms
* Boxplots
* Correlation heatmap
* Feature distributions
* Feature relationship with `Outcome`
* ROC Curve

These visualizations helped identify data distributions, possible outliers, and relationships between the features.

---

##  Data Preprocessing

The following preprocessing steps were performed:

### 1. Handling Missing Values

Some medical features contain `0` values that may represent missing measurements rather than actual medical values.

The following columns were checked:

* `Glucose`
* `BloodPressure`
* `SkinThickness`
* `Insulin`
* `BMI`

These zero values were handled as missing values and then imputed.

### 2. Train-Test Split

The dataset was divided into:

* Training set
* Testing set

`stratify=y` was used to preserve the class distribution.

### 3. Feature Scaling

`StandardScaler` was used to scale the features.

Scaling is particularly important for:

* Logistic Regression
* KNN
* SVM

---

## 🤖 Machine Learning Models

Three classification algorithms were trained and evaluated.

### 1. Logistic Regression

Logistic Regression was used as a baseline classification model for predicting diabetes.

```python
LogisticRegression()
```

---

### 2. K-Nearest Neighbors (KNN)

KNN classifies a new observation based on its nearest neighbors.

The model was implemented using:

```python
KNeighborsClassifier(
    n_neighbors=5,
    weights='distance'
)
```

---

### 3. Support Vector Machine (SVM)

SVM was used to find a decision boundary that separates the two classes.

The model was implemented using an RBF kernel:

```python
SVC(
    kernel='rbf',
    C=1.0,
    gamma='scale',
    probability=True
)
```

---

## 📈 Model Evaluation

The models were evaluated using several classification metrics:

### Accuracy

Measures the overall percentage of correct predictions.

### Precision

Measures how many of the observations predicted as diabetic were actually diabetic.

### Recall

Measures how many of the actual diabetic cases were correctly detected.

### F1-Score

Provides a balance between Precision and Recall.

### ROC-AUC

Measures the model's ability to distinguish between the two classes across different classification thresholds.

### Confusion Matrix

The confusion matrix was used to examine:

* True Positives
* True Negatives
* False Positives
* False Negatives

---

## 📊 Model Comparison

The three models were compared using:

* Accuracy
* Precision
* Recall
* F1-Score
* ROC-AUC

The comparison was performed using a Pandas DataFrame:

```python
comparison = pd.DataFrame({
    "Model": [
        "Logistic Regression",
        "KNN",
        "SVM"
    ],

    "Accuracy": [
        accuracy_score(y_test, y_pred_lr),
        accuracy_score(y_test, y_pred_knn),
        accuracy_score(y_test, y_pred_svm)
    ],

    "Precision": [
        precision_score(y_test, y_pred_lr),
        precision_score(y_test, y_pred_knn),
        precision_score(y_test, y_pred_svm)
    ],

    "Recall": [
        recall_score(y_test, y_pred_lr),
        recall_score(y_test, y_pred_knn),
        recall_score(y_test, y_pred_svm)
    ],

    "F1-Score":
```
