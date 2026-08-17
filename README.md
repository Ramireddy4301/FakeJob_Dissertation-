# Evaluation of Machine Learning Models for Fake Job Posting Detection Using Natural Language Processing and Prototype Development

## Overview

This dissertation project develops and evaluates a machine learning system for detecting fake job postings using Natural Language Processing. Four machine learning models are compared using multiple evaluation metrics and five-fold cross-validation. The best-performing model is then integrated into a Tkinter prototype and evaluated through student feedback.

## Aim

The main aim of the dissertation is to evaluate multiple machine learning models for fake job posting detection using Natural Language Processing techniques, identify the best-performing model through multiple evaluation metrics and five-fold cross-validation, and develop and assess a prototype using human feedback.

## Dataset

The project uses the Real / Fake Job Posting Prediction Dataset from Kaggle:

https://www.kaggle.com/datasets/shivamb/real-or-fake-fake-jobposting-prediction

Dataset details:
- 17,880 job advertisements
- 18 original attributes
- Target variable: fraudulent
- 17,014 legitimate postings
- 866 fraudulent postings

## Methodology

### Data Preprocessing

The preprocessing workflow included:
- Missing-value inspection and handling
- Duplicate checking
- Combination of title, company profile, description, requirements and benefits
- Lowercase conversion
- URL removal
- HTML tag removal
- Email address removal
- Number removal
- Punctuation removal
- Whitespace normalisation

The final preprocessed dataset contained 17,880 observations and 20 columns.

### TF-IDF

TF-IDF was used to convert cleaned job-posting text into numerical features. The project used a maximum of 5,000 TF-IDF features.

### Class Imbalance

SMOTE was applied only to the training data.

After SMOTE:
- Legitimate training postings: 13,611
- Fraudulent training postings: 13,611
- Total balanced training observations: 27,222

The test data remained unchanged.

### Train-Test Split

An 80:20 stratified train-test split was used:
- Training observations: 14,304
- Testing observations: 3,576
- Random state: 42

## Machine Learning Models

The following models were implemented:
- Logistic Regression
- XGBoost
- LightGBM
- CatBoost

## Evaluation Metrics

The models were evaluated using:
- Accuracy
- Precision
- Recall
- F1-score
- ROC-AUC
- Confusion matrix
- Five-fold cross-validation

## Model Results

### Logistic Regression

- Accuracy: 97.96%
- Precision: 74.75%
- Recall: 87.28%
- F1-score: 80.53%
- ROC-AUC: 98.44%

### XGBoost

- Accuracy: 98.13%
- Precision: 86.81%
- Recall: 72.25%
- F1-score: 78.86%
- ROC-AUC: 97.72%

### LightGBM

- Accuracy: 98.52%
- Precision: 92.86%
- Recall: 75.14%
- F1-score: 83.07%
- ROC-AUC: 98.71%

### CatBoost

- Accuracy: 97.68%
- Precision: 74.46%
- Recall: 79.19%
- F1-score: 76.75%
- ROC-AUC: 97.78%

## Five-Fold Cross-Validation

Mean validation accuracy:

| Model | Mean Accuracy |
| --- | ---: |
| Logistic Regression | 98.55% |
| XGBoost | 98.49% |
| LightGBM | 99.44% |
| CatBoost | 98.14% |

LightGBM achieved the highest mean cross-validation accuracy at 99.44%.

## Best-Performing Model

LightGBM was selected as the best-performing model because it achieved the highest F1-score, accuracy, precision, ROC-AUC and mean five-fold cross-validation accuracy.

F1-score comparison:

| Model | F1-score |
| --- | ---: |
| LightGBM | 83.07% |
| Logistic Regression | 80.53% |
| XGBoost | 78.86% |
| CatBoost | 76.75% |

## Prototype

The selected LightGBM model was integrated into a Tkinter desktop prototype.

The prototype accepts:
- Job Title
- Company Profile
- Job Description
- Requirements
- Benefits

The prototype provides:
- Real job posting or fake job posting prediction
- Confidence
- Real probability
- Fake probability
- Predict button
- Reset button
- Exit button

The same text-processing and TF-IDF workflow used during model development is applied to the prototype input.

## Human Evaluation

The prototype was evaluated by 100 student participants.

Five factors were assessed using a 1 to 5 scale:
- Ease of Use
- Usefulness
- Clarity of Prediction Results
- Interface Design
- Satisfaction

Mean scores:

| Evaluation Factor | Mean Score |
| --- | ---: |
| Ease of Use | 4.46 / 5 |
| Usefulness | 4.40 / 5 |
| Clarity of Prediction Results | 4.53 / 5 |
| Interface Design | 4.45 / 5 |
| Satisfaction | 4.46 / 5 |
| Overall Rating | 4.79 / 5 |

Open comments were also collected to provide additional qualitative feedback.

## Technologies

- Python
- Jupyter Notebook
- Pandas
- NumPy
- NLTK
- Regular Expressions
- Scikit-learn
- Imbalanced-learn
- XGBoost
- LightGBM
- CatBoost
- Matplotlib
- Seaborn
- Tkinter
- Joblib


## How to Run

### Step 1

Install Python 3.x.

### Step 2

Install the main packages:

```bash
pip install pandas numpy scikit-learn imbalanced-learn xgboost lightgbm catboost matplotlib seaborn nltk joblib
```

### Step 3

Download the dataset from Kaggle and place it in the data directory.

### Step 4

Run the preprocessing and model development notebook.

### Step 5

Run model evaluation and five-fold cross-validation.

### Step 6

Save the trained LightGBM model and TF-IDF vectoriser.



## Reproducibility

A random state of 42 was used for the train-test split and SMOTE procedures. The TF-IDF vectoriser and selected LightGBM model are saved for prototype use.

## Conclusion

The project demonstrates a complete workflow for fake job posting detection, from dataset preparation and text processing to machine learning evaluation and prototype development. LightGBM was selected as the best-performing model, achieving an F1-score of 83.07% and a five-fold cross-validation accuracy of 99.44%. Human evaluation produced an overall prototype rating of 4.79 out of 5.

## Author

Student: Mallikarjuna Reddy Rami Reddy

Student ID: 34047510

Programme: MSc Big Data Analytics
