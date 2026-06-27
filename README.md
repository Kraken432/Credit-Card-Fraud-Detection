# Credit Card Fraud Detection

A machine learning project that detects fraudulent credit card transactions using multiple classification algorithms. Built as part of my AI/ML internship.

## About the Dataset

Used the [ULB Credit Card Fraud Dataset](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud) from Kaggle. It contains 284,807 transactions made by European cardholders over two days in September 2013, with only 492 fraud cases (0.17% of all transactions).

The features V1–V28 are PCA-transformed for confidentiality. Only `Amount` and `Time` are the original features.

> Download `creditcard.csv` from Kaggle and place it in the project folder before running.

## What I did

- Explored the dataset — class distribution, missing values, outliers, correlations
- Handled class imbalance using SMOTE
- Engineered new features — hour of day, amount per second, V1×V2 interaction
- Trained and compared 5 models
- Saved the best model and built a Streamlit web app for predictions

## Models Compared

| Model | Notes |
|-------|-------|
| Logistic Regression | Baseline |
| Random Forest | Strong performer |
| Decision Tree | Fast, interpretable |
| Naive Bayes | Lightweight |
| XGBoost | Best overall |

Evaluated using Accuracy, ROC-AUC, 5-Fold Cross Validation, and Precision-Recall curve.

## Project Structure

```
credit-card-fraud-detection/
├── fraud_detection.py    # Training, EDA, model comparison
├── app.py                # Streamlit web app
├── best_model.pkl        # Saved best model
└── README.md
```

## How to Run

Install dependencies:
```bash
pip install pandas numpy scikit-learn imbalanced-learn xgboost streamlit matplotlib seaborn
```

Train the model:
```bash
python fraud_detection.py
```

Run the app:
```bash
streamlit run app.py
```

## App

Enter a transaction amount and the app finds the closest matching real transaction from the dataset and predicts whether it's fraudulent or legitimate using the best trained model.

## Tech Stack

Python, Scikit-learn, XGBoost, SMOTE, Streamlit, Pandas, Matplotlib, Seaborn
