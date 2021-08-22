"""
In thi module are all the function to implement a machine learning models in order
to predict is a client will churn or not

Author: Hanna L.A.
Date: August 2021
"""
from sklearn.metrics import plot_roc_curve, classification_report
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import normalize
from typing import Tuple
import logging

import pandas.core.series
import shap
import joblib
from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

sns.set()


IMAGE_EDA_PATH = './images/eda'
IMAGE_RESULT_PATH = './images/results'
MODEL_PATH = './models'


def import_data(pth: str) -> pd.DataFrame:
    """
    takes a path and return a pandas dataframe
    :param pth: path to the csv
    :return: pandas dataframe
    """
    df = pd.read_csv(pth, index_col=0)
    return df


def check_null_values(df: pd.DataFrame) -> pandas.core.series.Series:
    """

    :param df: pandas dataframe
    :return: a pandas.Series with all the columns and the total null values
    """
    return df.isnull().sum()


def add_churn(df: pd.DataFrame) -> pd.DataFrame:
    df['Churn'] = df['Attrition_Flag'].apply(
        lambda x: 0 if x == 'Existing Customer' else 1)
    return df


def perform_eda(df: pd.DataFrame):
    """
    perform eda on a dataframe and save figures to images folder
    :param df: pandas dataframe
    :return: None
    """
    # figure 1
    plt.figure()
    df['Churn'].hist()
    plt.tight_layout()
    plt.savefig(Path(IMAGE_EDA_PATH, 'churn_distribution.png'))

    # figure 2
    plt.figure()
    df['Customer_Age'].hist()
    plt.tight_layout()
    plt.savefig(Path(IMAGE_EDA_PATH, 'customer_age_distribution.png'))

    # figure 3
    plt.figure()
    df['Marital_Status'].value_counts().plot(kind='bar')
    plt.tight_layout()
    plt.savefig(Path(IMAGE_EDA_PATH, 'marital_status_distribution.png'))

    # figure 4
    sns.displot(df['Total_Trans_Ct'])
    plt.tight_layout()
    plt.savefig(Path(IMAGE_EDA_PATH, 'total_transaction_distribution.png'))

    # figure
    plt.figure(figsize=(20, 10))
    sns.heatmap(df.corr(), cmap='Dark2_r', linewidths=2)
    plt.tight_layout()
    plt.savefig(Path(IMAGE_EDA_PATH, 'heatmap.png'))


def encoder_helper(df: pd.DataFrame, category_lst: list,
                   response: str) -> pd.DataFrame:
    """
    helper function to turn each categorical column into a new column with
    proportion of churn for each categorical

    :param df: pandas dataframe
    :param category_lst: columns that contain categorical features
    :param response: response name [optional argument that could be used for naming variables or index y column]
    :return: pandas dataframe with new columns
    """
    for categorical in category_lst:
        cat_groups = df.groupby(categorical).mean()[response]
        cat_values = [cat_groups.loc[val] for val in df[categorical]]
        new_categorical = '_'.join([categorical, response])
        df[new_categorical] = cat_values
    return df


def perform_feature_engineering(df: pd.DataFrame, response: str) -> Tuple[
        pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    """

    :param df: pandas dataframe
    :param response: response name [optional argument that could be used for naming variables or index y column]
    :return:
              X_train: X training data
              X_test: X testing data
              y_train: y training data
              y_test: y testing data

    """
    # features
    keep_cols = df.select_dtypes(exclude=['O']).columns.drop(
        ['CLIENTNUM', 'Churn']).to_list()
    y = df[response]
    X = df[keep_cols]

    # train test split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3,
                                                        random_state=42)
    return X_train, X_test, y_train, y_test


def train_models(X_train, X_test, y_train, y_test):
    """
    train and store model results: images + scores, and save models
    :param X_train: X training data
    :param X_test: X testing data
    :param y_train: y training data
    :param y_test: y testing data
    :return: None
    """
    # logistic regression
    lrc = LogisticRegression(solver='liblinear')
    lrc.fit(X_train, y_train)

    # random forest
    rfc = RandomForestClassifier(random_state=42)
    param_grid = {
        'n_estimators': [200, 500],
        'max_features': ['auto', 'sqrt'],
        'max_depth': [4, 5, 100],
        'criterion': ['gini', 'entropy']
    }
    cv_rfc = GridSearchCV(estimator=rfc, param_grid=param_grid, cv=5)
    cv_rfc.fit(X_train, y_train)

    # figure roc_curve
    plt.figure(figsize=(15, 8))
    ax = plt.gca()
    plot_roc_curve(cv_rfc, X_test, y_test, ax=ax, alpha=0.8)
    plot_roc_curve(lrc, X_test, y_test, ax=ax)
    plt.savefig(Path(IMAGE_RESULT_PATH, 'roc_curve_result.png'))

    # save models
    joblib.dump(cv_rfc.best_estimator_, './models/rfc_model.pkl')
    joblib.dump(lrc, './models/logistic_model.pkl')


def load_models():
    rfc_model = joblib.load('./models/rfc_model.pkl')
    lrc_model = joblib.load('./models/logistic_model.pkl')
    return lrc_model, rfc_model


def test_prediction(X_train: pd.DataFrame, X_test: pd.DataFrame,
                    lrc_model, rfc_model) -> Tuple[
        np.ndarray, np.ndarray, np.ndarray, np.ndarray]:

    y_train_pred_lr = lrc_model.predict(X_train)
    y_test_pred_lr = lrc_model.predict(X_test)
    y_train_pred_rf = rfc_model.predict(X_train)
    y_test_pred_rf = rfc_model.predict(X_test)
    return y_train_pred_lr, y_train_pred_rf, y_test_pred_lr, y_test_pred_rf


def plot_classification_report(y_train: pd.Series,
                               y_test: pd.Series,
                               y_train_pred: np.ndarray,
                               y_test_pred: np.ndarray,
                               title):
    fig = plt.figure(figsize=(6, 5))
    fig.add_subplot(1, 1, 1)
    plt.text(0.01, 1.25, title + ' Train', {'fontsize': 10},
             fontproperties='monospace')
    plt.text(0.01, 0.70, str(classification_report(y_train, y_train_pred)),
             {'fontsize': 10},
             fontproperties='monospace')
    plt.text(0.01, 0.60, title + ' Test', {'fontsize': 10},
             fontproperties='monospace')
    plt.text(0.01, 0.05, str(classification_report(y_test, y_test_pred)),
             {'fontsize': 10},
             fontproperties='monospace')
    plt.tight_layout()
    plt.axis('off')
    return fig


def classification_report_image(y_train: pd.Series,
                                y_test: pd.Series,
                                y_train_pred_lr: np.ndarray,
                                y_train_pred_rf: np.ndarray,
                                y_test_pred_lr: np.ndarray,
                                y_test_pred_rf: np.ndarray):
    """
    produces classification report for training and testing results and stores report as image
    in images folder
    :param y_train: training response values
    :param y_test: test response values
    :param y_train_pred_lr: training predictions from logistic regression
    :param y_train_pred_rf: training predictions from random forest
    :param y_test_pred_lr:  test predictions from random forest
    :param y_test_pred_rf: test predictions from random forest
    :return: None
    """
    # logistic regression
    fig_lrc = plot_classification_report(y_train, y_test, y_train_pred_lr,
                                         y_test_pred_lr, 'Logistic Regression')
    fig_lrc.savefig(Path(IMAGE_RESULT_PATH, 'logistic_results.png'))

    # random forest
    fig_rf = plot_classification_report(y_train, y_test, y_train_pred_rf,
                                        y_test_pred_rf, 'Random Forest')
    fig_rf.savefig(Path(IMAGE_RESULT_PATH, 'rf_results.png'))


def feature_importance_plot(model, X_data, output_pth):
    """
    creates and stores the feature importances in output path

    :param model: model object containing feature_importances
    :param X_data: pandas dataframe of X values
    :param output_pth: path to store the figure
    :return: None
    """
    # calculate feature importances
    importances = model.feature_importances_
    # sort feature importances in descending order
    indices = np.argsort(importances)[::-1]
    # rearrange feature names so they match the sorted feature importances
    names = [X_data.columns[i] for i in indices]

    # plot
    plt.figure(figsize=(20, 5))
    plt.bar(range(X_data.shape[1]), importances[indices])
    plt.title("Feature Importance")
    plt.ylabel('Importance')
    plt.xticks(range(X_data.shape[1]), names, rotation=90)
    plt.tight_layout()
    plt.savefig(Path(output_pth, 'feature_importance.png'))

    # shap plot
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X_data)
    plt.figure()
    shap.summary_plot(shap_values, X_data, plot_type="bar", show=False)
    plt.tight_layout()
    plt.savefig(Path(output_pth, 'feature_importance_shap.png'))


def main():
    # data
    df = import_data('data/bank_data.csv')
    df = add_churn(df)
    cat_list = ['Gender',
                'Education_Level',
                'Marital_Status',
                'Income_Category',
                'Card_Category'
                ]
    response = 'Churn'
    df = encoder_helper(df, cat_list, response)

    # feature engineering
    X_train, X_test, y_train, y_test = perform_feature_engineering(df, response)

    # train model
    # train_models(X_train, X_test, y_train, y_test)

    # models
    lrc_model, rfc_model = load_models()

    # plots
    y_train_pred_lr, y_train_pred_rf, y_test_pred_lr, y_test_pred_rf = test_prediction(X_train, X_test, lrc_model, rfc_model)
    classification_report_image(y_train, y_test, y_train_pred_lr, y_train_pred_rf, y_test_pred_lr, y_test_pred_rf)
    feature_importance_plot(rfc_model, X_test, IMAGE_RESULT_PATH)


if __name__ == '__main__':
    main()
