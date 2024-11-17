# Titanic Dataset Model

This project utilizes the Titanic dataset to perform data analysis and build a machine learning model to predict passenger survival. The notebook includes data preprocessing, feature engineering, and model training.

## Features

- **Dataset:** Titanic dataset containing features like `Pclass`, `Age`, `Sex`, etc.
- **Data Preprocessing:** Handling missing values, encoding categorical variables, and feature scaling.
- **Feature Engineering:** Creation of new features like `FamilySize` and `Title`.
- **Model:** Implements an XGBoost Classifier for predictions.
- **Evaluation:** Performance metrics such as accuracy, classification report, and confusion matrix.

## Prerequisites

To run this project, ensure you have the following libraries installed:

- `pandas`
- `numpy`
- `scikit-learn`
- `xgboost`
- `imbalanced-learn`

## Usage

1. Clone the repository.
2. Load the dataset (`Titanic-Dataset.csv`).
3. Execute the Jupyter Notebook to preprocess data, train the model, and evaluate performance.

## File Structure

- **Titanic_Dataset_model.ipynb**: Jupyter Notebook containing the complete workflow.
- **Titanic-Dataset.csv**: Dataset file used for training and evaluation (not included; ensure to download it).

## How to Run

1. Install the required dependencies:
   ```bash
   pip install pandas numpy scikit-learn xgboost imbalanced-learn

2. Open the notebook:
    ```bash
    jupyter notebook Titanic_Dataset_model.ipynb
3. Follow the steps in the notebook to load the dataset, preprocess the data, and train the model.


## Acknowledgments

The Titanic dataset is provided by [Kaggle](https://www.kaggle.com/datasets/yasserh/titanic-dataset).
