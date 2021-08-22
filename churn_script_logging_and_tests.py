from pathlib import Path
import logging
import pandas as pd
import churn_library_solution as cls

logging.basicConfig(
    filename='logs/churn_library.log',
    level=logging.INFO,
    filemode='w',
    format='%(name)s - %(levelname)s - %(message)s '
)


def test_import():
    """
    test import_data
    """
    try:
        df = cls.import_data('./data/bank_data.csv')
        logging.info('Testing import_data: SUCCESS')
    except FileNotFoundError as err:
        logging.error('Testing import_data: The file was not found')
        raise err
    try:
        assert df.shape[0] > 0
        assert df.shape[1] > 0
    except AssertionError as err:
        logging.error(
            'Testing import_data: The file does not appear to have rows and columns')
        raise err


def test_null_values():
    """
    test check_null_values
    """
    try:
        df = cls.import_data('./data/bank_data.csv')
        assert sum(cls.check_null_values(df).to_list()) == 0
        logging.info('Testing check_null_values: SUCCESS')
    except AssertionError as err:
        logging.error(
            'Testing check_null_values: There are null values in the dataframe')
        raise err


def test_add_churn():
    """
    test add_churn
    """
    try:
        df = cls.import_data('./data/bank_data.csv')
        df = cls.add_churn(df)
        assert 'Churn' in df.columns
        logging.info('Testing add_churn: SUCCESS')
    except AssertionError as err:
        logging.error('Testing add_churn: There is not a column called Churn')
        raise err


def test_eda():
    """
    test perform_eda
    :return:
    """
    try:
        assert Path(cls.IMAGE_EDA_PATH, 'churn_distribution.png').is_file()
        assert Path(cls.IMAGE_EDA_PATH, 'customer_age_distribution.png').is_file()
        assert Path(
            cls.IMAGE_EDA_PATH,
            'marital_status_distribution.png').is_file()
        assert Path(cls.IMAGE_EDA_PATH,
                    'total_transaction_distribution.png').is_file()
        assert Path(cls.IMAGE_EDA_PATH, 'heatmap.png').is_file()
        logging.info('Testing perform_eda: SUCCESS')
    except AssertionError as err:
        logging.error('Testing perform_eda: Some image is missing')
        raise err


def test_encoder_helper():
    """
    test encoder_helper
    :return:
    """
    try:
        df = cls.import_data('data/bank_data.csv')
        df = cls.add_churn(df)
        cat_list = ['Gender',
                    'Education_Level',
                    'Marital_Status',
                    'Income_Category',
                    'Card_Category'
                    ]
        response = 'Churn'
        df = cls.encoder_helper(df, cat_list, response)
        assert set([name + '_' + response for name in cat_list]
                   ).issubset(df.columns)
        logging.info('Testing encoder_helper: SUCCESS')
    except KeyError as err:
        logging.error(
            'Testing encoder_helper: There is not some categorical column')
        raise err
    try:
        assert df[[name + '_' + response for name in cat_list]].shape[0] > 0
        assert df[[name + '_' + response for name in cat_list]].shape[1] > 0
    except AssertionError as err:
        logging.error(
            'Testing encoder_helper: The dataframe does not appear to have rows and columns')
        raise err


def test_perform_feature_engineering():
    """
    test perform_feature_engineering
    """
    try:
        df = cls.import_data('./data/bank_data.csv')
        df = cls.add_churn(df)
        cat_list = ['Gender',
                    'Education_Level',
                    'Marital_Status',
                    'Income_Category',
                    'Card_Category'
                    ]
        response = 'Churn'
        df = cls.encoder_helper(df, cat_list, response)
        X_train, X_test, y_train, y_test = cls.perform_feature_engineering(
            df, response)
        assert isinstance(X_train, pd.DataFrame)
        assert isinstance(X_test, pd.DataFrame)
        assert isinstance(y_train, pd.Series)
        assert isinstance(y_train, pd.Series)
        logging.info(
            'Testing perform_feature_engineering: SUCCESS type feature')
    except AssertionError as err:
        logging.error(
            'Testing perform_feature_engineering: There is problem with get some feature')
        raise err
    try:
        assert X_train.shape[0] > 0
        assert X_train.shape[1] == 19
        assert X_test.shape[0] > 0
        assert X_test.shape[1] == 19
        assert y_train.shape[0] > 0
        assert y_test.shape[0] > 0
        logging.info(
            'Testing perform_feature_engineering: SUCCESS shape feature')
    except AssertionError as err:
        logging.error(
            'Testing perform_feature_engineering: The dataframe does not appear to have the right rows and columns')
        raise err


def test_train_models():
    try:
        assert Path('./models/logistic_model.pkl').is_file()
        assert Path('./models/rfc_model.pkl').is_file()
        logging.info('Testing train_model: SUCCESS')
    except AssertionError as err:
        logging.error('Testing train_model: Some model was not found')
        raise err
    try:
        assert Path('./models/logistic_model.pkl').stat().st_size != 0
        assert Path('./models/rfc_model.pkl').stat().st_size != 0
    except AssertionError as err:
        logging.error('Testing train_model: Some model is empty')
        raise err


if __name__ == "__main__":
    test_import()
    test_null_values()
    test_add_churn()
    test_eda()
    test_encoder_helper()
    test_perform_feature_engineering()
    test_train_models()
