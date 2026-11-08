# Machine Learning Classification Assignment 2

## 1. Project Overview

This project implements and compares five supervised machine learning classification algorithms using the Dry Bean Dataset.

The objective is to classify dry bean varieties using numerical features extracted from bean images and evaluate the performance of different classification models.

## 2. Dataset Description

**Dataset:** Dry Bean Dataset  
**Source:** UCI Machine Learning Repository  
**Number of observations:** 13,611  
**Number of input features:** 16  
**Target variable:** Class  
**Number of classes:** 7

The dataset contains numerical measurements of dry beans, including area, perimeter, major axis length, minor axis length, aspect ratio, eccentricity and other shape-related characteristics.

## 3. Machine Learning Models

The following five classification algorithms were implemented:

1. Logistic Regression
2. Decision Tree
3. K-Nearest Neighbors (KNN)
4. Gaussian Naive Bayes
5. Support Vector Machine (SVM)

The dataset was divided into training and testing sets using an 80:20 split with stratification.

Feature scaling was applied where appropriate using Scikit-learn pipelines.

## 4. Evaluation Metrics

The models were evaluated using six metrics:

- Accuracy
- Area Under the Curve (AUC)
- Precision
- Recall
- F1 Score
- Matthews Correlation Coefficient (MCC)

Weighted averaging was used for multiclass Precision, Recall and F1 Score. AUC was calculated using the One-vs-Rest approach.

## 5. Model Comparison Results

| Model                  | Accuracy |    AUC | Precision | Recall | F1 Score |    MCC |
| ---------------------- | -------: | -----: | --------: | -----: | -------: | -----: |
| Support Vector Machine |   0.9221 | 0.9937 |    0.9224 | 0.9221 |   0.9222 | 0.9059 |
| Logistic Regression    |   0.9210 | 0.9934 |    0.9219 | 0.9210 |   0.9212 | 0.9046 |
| K-Nearest Neighbors    |   0.9166 | 0.9812 |    0.9174 | 0.9166 |   0.9168 | 0.8992 |
| Naive Bayes            |   0.8979 | 0.9902 |    0.9007 | 0.8979 |   0.8981 | 0.8773 |
| Decision Tree          |   0.8917 | 0.9330 |    0.8912 | 0.8917 |   0.8913 | 0.8691 |

## 6. Observations

**Support Vector Machine:** Achieved the highest accuracy of approximately 92.21% and the highest F1 Score and MCC. It demonstrated the best overall performance among the five models.

**Logistic Regression:** Achieved approximately 92.10% accuracy. Its performance was very close to SVM, demonstrating that it is also an effective classifier for this dataset.

**K-Nearest Neighbors:** Achieved approximately 91.66% accuracy. It provided good classification performance, although its AUC was lower than that of SVM and Logistic Regression.

**Naive Bayes:** Achieved approximately 89.79% accuracy. Although its AUC was high, its accuracy and F1 Score were lower than those of the other leading models.

**Decision Tree:** Achieved approximately 89.17% accuracy. It had the lowest accuracy and AUC among the five models in this experiment.

## 7. Overall Conclusion

Based on the evaluation results, Support Vector Machine was the best-performing model, achieving approximately 92.21% accuracy, 99.37% AUC and 92.22% F1 Score.

Logistic Regression performed very closely to SVM. Therefore, SVM is selected as the overall winner based on the observed test results.

## 8. Streamlit Application

The Streamlit application provides an interface to:

- Upload a test dataset in CSV format.
- Select one of the five trained classification models.
- Display evaluation metrics.
- View the confusion matrix.
- View the classification report.
- Compare the five models.

The application loads previously saved trained models and evaluates the uploaded test data without retraining.

## 9. Project Files

- `app.py` – Streamlit application.
- `train_models.py` – Model training and evaluation script.
- `requirements.txt` – Python dependencies.
- `test_data.csv` – Test dataset.
- `model/` – Saved trained models.
- `results/` – Model comparison results and evaluation metrics.
- `data/` – Dataset file.

## 10. GitHub Repository

https://github.com/2026ak05620-stack/bits-aiml-ml-classification-assignment-2

## 11. Execution

Install the dependencies:

```bash
pip install -r requirements.txt
```

Train the models:

```bash
python train_models.py
```

Run the Streamlit application:

```bash
streamlit run app.py
```
