import pandas as pd
import numpy as np

import logging

logger = logging.getLogger(__name__)

ADULT_QI_INT_COLUMNS = ['age']
CAHOUSING_QI_INT_COLUMNS = ['latitude', 'longitude', 'median_income', 'housing_median_age']
CMC_QI_INT_COLUMNS = ['Weducation', 'age', 'children']


def get_median_from_binned_data(df: pd.DataFrame, int_columns: list, df_name: str):
    """
    Get the 2 values from the range of the binned data and assign it, if the value is suppresed, assign the median of the
    dataset
    """
    for int_column in int_columns:
        if int_column in df.columns:

            ## Assign median to the suppresed values
            # Load the non-anonimized dataset
            df_non_anon = pd.read_csv(f'results/{df_name}/no_anonimization/{df_name}_anonymized_0.csv', sep=';')
            # Get the median,max and min of the non-anonimized dataset
            median = df_non_anon[int_column].median()
            df[int_column] = df[int_column].apply(lambda x: median if x == '*' else x)

            ## Try to convert to a float
            try:
                # Convert to float
                df[int_column] = df[int_column].astype(float)
            except:
                pass

            # If it is still an object, it has a range. We impute the median of the range
            if df[int_column].dtype == 'object':
                # Convert single number to invertals using ~
                df[int_column] = df[int_column].apply(lambda x: f'{x}~{x}' if type(x) == float or type(x) == np.float64 else x)
                # if the value has a + the inverval is up to the max
                logger.info(f'Getting median from binned data for column {int_column}')
                df[int_column] = df[int_column].apply(lambda x: np.mean([float(i) for i in x.split('~')]))
    return df


def drop_columns_with_only_one_unique_value(data: pd.DataFrame):
    """
    Drop columns with only one unique value
    """
    for column in data.columns:
        if len(data[column].unique()) == 1:
            logger.info(f'Dropping column {column} as it has only one unique value')
            data.drop(column, axis=1, inplace=True)
    return data


def drop_columns_if_exists(data: pd.DataFrame, columns_to_drop: list):
    """
    Drop columns if they don't exist in the dataset
    """
    for column in columns_to_drop:
        if column in data.columns:
            logger.info(f'Dropping column {column} as it does not exist in the dataset')
            data.drop(column, axis=1, inplace=True)
    return data


def convert_to_object_if_exists(data: pd.DataFrame, columns: list):
    """
    Convert the columns to object ( to be one-hot encoded)  only if they exist
    """
    for column in columns:
        if column in data.columns:
            data[column] = data[column].astype('object')
    return data


def determine_dataset(data):
    """
    Determine which dataset we are using based on the targets
    """
    if 'median_house_value' in data.columns:
        return 'cahousing'
    elif 'salary-class' in data.columns:
        return 'adult'
    elif 'method' in data.columns:
        return 'cmc'


def apply_one_hot_encoding(data: pd.DataFrame):
    # Aply one_hot_encoding
    logger.info(f'Applying one hot encoding')
    preprocessed_data = pd.get_dummies(data, drop_first=True)
    logger.info(f'Columns after one hot encoding: {preprocessed_data.dtypes}')
    return preprocessed_data


def preprocess_adult(data: pd.DataFrame):
    """
    Preprocess the adult dataset
    """
    # Drop not important columns
    columns_to_drop = ['ID']
    drop_columns_if_exists(data, columns_to_drop)

    # Drop columns with only one unique value
    data = drop_columns_with_only_one_unique_value(data)

    # Get the median from the binned data
    data = get_median_from_binned_data(df=data,
                                       int_columns=ADULT_QI_INT_COLUMNS,
                                       df_name='adult'
                                       )

    # Convert the target to 0-1
    logger.info(f'Transforming income target to binary')
    data['salary-class'] = data['salary-class'].apply(lambda x: 0 if x == '<=50K' else 1)

    # Apply one hot encoding
    preprocessed_data = apply_one_hot_encoding(data)

    return preprocessed_data


def preprocess_cahousing(data: pd.DataFrame):
    """
    Preprocess the california housing dataset, suitable for regression
    """

    # Drop columns with only one unique value
    data = drop_columns_with_only_one_unique_value(data)

    # Get the median from the binned data
    data = get_median_from_binned_data(df=data,
                                       int_columns=CAHOUSING_QI_INT_COLUMNS,
                                       df_name='cahousing'
                                       )

    # Drop not important columns
    columns_to_drop = ['ID']
    drop_columns_if_exists(data, columns_to_drop)

    # apply one-hot-encoding
    data_preprocessed = apply_one_hot_encoding(data)
    return data_preprocessed


def preprocess_cmc(data: pd.DataFrame):
    """
    Preprocess the CMC dataset, suitable for multiclass classification
    """
    # Drop columns with a unique value
    data = drop_columns_with_only_one_unique_value(data)

    # Get the median from the binned data
    data = get_median_from_binned_data(df=data,
                                       int_columns=CMC_QI_INT_COLUMNS,
                                       df_name='cmc'
                                       )

    # Drop not important columns
    columns_to_drop = ['ID']
    data = drop_columns_if_exists(data, columns_to_drop)

    # substract 1 to the target to allow xgb softmax objective
    data['method'] = data['method'].apply(lambda x: x - 1)

    # Apply one hot encoding
    preprocessed_data = apply_one_hot_encoding(data)

    return preprocessed_data


def preprocess_data(data_raw: pd.DataFrame):
    """
    Preprocess the data
    """
    # Determine the dataset
    dataset = determine_dataset(data_raw)
    logger.info(f'Dataset to be used: {dataset}')

    # Preprocess the data
    if dataset == 'adult':
        preprocessed_datasets = preprocess_adult(data_raw)
    elif dataset == 'cahousing':
        preprocessed_datasets = preprocess_cahousing(data_raw)
    elif dataset == 'cmc':
        preprocessed_datasets = preprocess_cmc(data_raw)
    else:
        raise Exception('Dataset not supported')
    return preprocessed_datasets
