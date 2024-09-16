# Stroke Prediction Project

This repository contains both Python and R scripts that aim to build predictive models for stroke risk, leveraging machine learning algorithms. The project utilizes the **Stroke Prediction Dataset** from Kaggle, which contains patient-level information to predict whether a patient is likely to experience a stroke.

### Dataset:
- **Kaggle Stroke Prediction Dataset**: [Kaggle Dataset Link](https://www.kaggle.com/datasets/fedesoriano/stroke-prediction-dataset)
- **Target Variable**: `stroke`
- **Features**: Includes demographic and clinical features such as age, gender, hypertension, heart disease, glucose level, BMI, etc.

## Project Components:
- **Team Members**: Amr Ibrahim & Matthew Hureau
- **Objective**: Develop a model that accurately predicts the risk of stroke using features from the dataset, and explore various classification algorithms in both R and Python.

---

## Python Script Overview:
The Python script utilizes popular machine learning libraries, including `scikit-learn`, `pandas`, and `statsmodels`. Below is a breakdown of the script:

### Key Libraries:
- **Pandas** for data manipulation
- **Plotly** for visualizations
- **Scikit-learn** for implementing models such as decision trees, SVM, and ensemble methods
- **Statsmodels** for statistical analysis

### Workflow:
1. **Data Preprocessing**:
   - Handles missing values, outliers, and feature scaling.
   - Exploratory data analysis and visualizations using `Plotly`.
   
2. **Model Training**:
   - Trains models including Logistic Regression, SVM, Decision Trees, K-Nearest Neighbors, and Ensemble methods.
   
3. **Model Evaluation**:
   - Uses accuracy, recall, precision, and confusion matrices to evaluate models.

---

## R Script Overview:
The R script applies similar machine learning techniques using the `caret` package, with some additional neural network approaches.

### Key Libraries:
- **tidyverse** for data manipulation
- **ROSE** for addressing class imbalance
- **caret** for machine learning model training and evaluation
- **neuralnet** for implementing neural networks
- **ggplot2** for visualizations

### Workflow:
1. **Data Cleaning**:
   - Removes missing values and handles data inconsistencies.
   - Feature engineering includes dropping irrelevant columns like `id` and imputation for missing values in BMI.
   
2. **Data Visualization**:
   - Visualizes key relationships using `ggplot2` to understand trends in stroke occurrences.
   
3. **Modeling**:
   - Implements models such as logistic regression, random forest, and neural networks.

4. **Model Evaluation**:
   - Uses confusion matrices and AUC to assess model performance.

---

## How to Run the Project:

### Python:
1. Install required packages:
   ```bash
   pip install pandas plotly scikit-learn statsmodels
   ```
2. Run the script to preprocess the data, train the models, and evaluate performance.

### R:
1. Install required packages:
   ```bash
   install.packages(c("tidyverse", "ROSE", "caret", "neuralnet", "MASS", "glmnet", "randomForest", "ggplot2"))
   ```
2. Open the RMarkdown file and run all code chunks to reproduce the results.

## Project Highlights
* **Data Imbalance**: Handled using techniques such as oversampling (ROSE in R) and balancing in Python.
* **Cross-Validation**: Implemented to prevent overfitting and ensure model generalization.
* **Comparison of Multiple Models**: Tested various algorithms to find the best-performing model in terms of accuracy and recall.
