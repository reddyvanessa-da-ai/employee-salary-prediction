# Employee Salary Prediction

## Overview

This project predicts whether an individual's annual income is greater than $50K or less than or equal to $50K using machine learning. It covers the complete workflow from data preprocessing and model training to deployment using Streamlit.

## Dataset

The project uses the Adult Income Dataset from the UCI Machine Learning Repository. The dataset contains demographic and employment-related information such as age, workclass, education, occupation, marital status, capital gain, capital loss, hours worked per week, and native country.

## Features

- Missing value handling
- Outlier removal
- Label encoding of categorical features
- Feature scaling using StandardScaler
- Comparison of multiple classification models
- Streamlit-based web application for prediction

## Models Evaluated

- Logistic Regression
- K-Nearest Neighbors
- Support Vector Machine
- Random Forest
- Gradient Boosting

## Results

Gradient Boosting achieved the best performance with an accuracy of **86.72%** and was selected as the final model for deployment.

## Technologies Used

- Python
- Pandas
- NumPy
- Scikit-learn
- Matplotlib
- Seaborn
- Streamlit
- Joblib

## Project Structure

```
employee-salary-prediction
│
├── data/
├── models/
├── notebook/
├── app.py
├── requirements.txt
├── README.md
└── .gitignore
```

## Installation

Clone the repository.

```bash
git clone https://github.com/YOUR_USERNAME/employee-salary-prediction.git
```

Install the required packages.

```bash
pip install -r requirements.txt
```

Run the application.

```bash
streamlit run app.py
```

## Future Improvements

- Hyperparameter tuning
- Cross-validation
- Feature importance analysis
- Cloud deployment
