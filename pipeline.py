'''
Pipeline to pre-process and analyze models for machine learning
'''
import pandas as pd
import numpy as np
from sklearn import model_selection
from sklearn import metrics
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, f1_score, \
                            precision_score, recall_score
from sklearn.exceptions import ConvergenceWarning
from sklearn.model_selection import KFold, cross_val_score
import datetime
import warnings
warnings.filterwarnings("ignore", category=ConvergenceWarning)


def read_data(datafile, datatype="csv", colheader=None):
    '''
    Reads in data file by filetype as a DataFrame

    Inputs:
        datafile (str): path to file
        datatype (str): type of file to be read in, defaults csv
        colheader (int): for excel files, row index of where column located
    
    Returns Pandas Dataframe
    '''
    if datatype == "csv":
        if colheader is None:
            colheader ="infer"
        return pd.read_csv(datafile, header=colheader)
    if datatype == "excel":
        if colheader is None:
            colheader = 0
        return pd.read_excel(datafile, header=colheader)
    if datatype == "json":
        return pd.read_json(datafile)


def analyze_first_glance(df, include_obj=False, print_stats=False):
    '''
    Provides the different values categorical columns may take on, along
    with basic stats of numerical columns (stats of numerical variables
    stored under "numerical_stats")

    Inputs:
        df (Pandas dataframe)
        include_obj (bool): whether to include objects as a type of categorical
            variable, defaults False
    Returns None, prints out statistics
    '''
    stats = {}
    categorical = df.select_dtypes(include=[object, bool, "category"])
    for col in categorical:
        stats[col] = df[col].value_counts()
        if print_stats:
            print(col + " Possible Values:")
            print(stats[col])
            print("\n")
    stats["numerical stats"] = df.describe()
    if print_stats:
        print("Stats of other variables:")
        print(stats["numerical stats"])
    return stats


def correlation(df):
    '''
    Creates correlation matrix of dataframe provided
    Inputs df (dataframe)
    Returns correlation matrix
    '''
    return df.corr()


def create_train_test(dataset, test_size=.2, random_state=1):
    '''
    Divides dataset into training and testing sets

    Inputs:
        dataset (DataFrame)
        test_size (float between 0 and 1): the proportion of dataset to include
            in test split, default to .2
        randome_state (int): shuffling applied to data before applying split,
            default 1
    Returns tuple of training set (DataFrame) and test set (DataFrame)
    '''
    train, test = model_selection.train_test_split(dataset, test_size=test_size, 
                                                   random_state=random_state)
    return train, test


def replace_missing_values(df, ignore_cols=None):
    '''
    Replaces missing values in numerical columns with median of column
    Inputs:
        df (pandas DataFrame)
        ignore_cols (list of str): any numerical columns that should not be
            filled with their median, defaults None
    Returns None, updates DataFrame
    '''
    df_numerics = df.select_dtypes(exclude=[object, bool, "category"])
    for column in df_numerics:
        if ignore_cols is None or column not in ignore_cols:
            df[column].fillna(df[column].median(), inplace=True)


def normalize(df, scaler=None, outputinc=False, outputcol=None):
    '''
    Normalizes dataframe (adapted from Nick Feamster's normalize function)
    Inputs:
        df (Pandas Dataframe)
        scaler (Scaler) :If scaler is not none, use given scaler's means and sds
                         to normalize (input for test set case); else, set 
                         scaler in function
        outputinc (bool): If output is included, set aside to ensure it does not
                          get normalized, default False
        outputcol (str): If output is included, name of output column, default
                         None
    Returns tuple of:
        Normalized DataFrame and scaler used to normalize DataFrame
    '''
    if outputinc:
        outcomes = df.loc[:,outputcol]
        df = pd.DataFrame(df.drop(outputcol, axis=1))
    if scaler is None:
      scaler = StandardScaler()
      normalized_features = scaler.fit_transform(df) 
    else:
      normalized_features = scaler.transform(df)

    normalized_df = pd.DataFrame(normalized_features)
    if outputinc:
        normalized_df[outputcol] = outcomes.tolist()

    normalized_df.index=df.index
    normalized_df.columns=df.columns

    return normalized_df, scaler


def one_hot_encode(train, test, columns=None):
    '''
    One hot encodes training and testing sets
    
    Inputs:
        train (DataFrame) training dataset
        test (DataFrame) testing dataset
        columns (list of str) specific columns to one hot encode; if None,
                one hot encodes all possible categorical/object columns
    Returns tuple of:
      one hot encoded training set and one hot encoded testing set
    '''
    train_onehot = pd.get_dummies(train, columns=columns)
    test_onehot = pd.get_dummies(test, columns=columns)
    
    train_onehot.columns = train_onehot.columns.astype(str)
    test_onehot.columns = test_onehot.columns.astype(str)

    for column in train_onehot.columns:
        if column not in test_onehot.columns:
            test_onehot[column] = 0
    to_drop = []
    for column in test_onehot.columns:
        if column not in train_onehot.columns:
            to_drop.append(column)
    test_onehot.drop(to_drop, axis=1, inplace=True)

    return train_onehot, test_onehot


def discretize(dataframe, columns, bins):
    '''
    Discretizes column(s) in a Dataframe into given bins
    Inputs:
        dataframe (DataFrame)
        columns (lst of str): list of column names to be discreted
        bins (list of int, sequence of scalars, or IntervalIndex): bins 
            corresponding to each column to be discreted
    Returns None, updates DataFrame
    '''
    for i, column in enumerate(columns):
        dataframe[column] = pd.cut(column, bins[i])


def build_classifier(model, train_features, params, outcome, inctime=False):
    '''
    Builds classifier on the model given

    Inputs:
        model (sklearn model): model to be used
        train_features (DataFrame): DataFrame of features used to train data
        params (dict): dictionary of parameters wanted for model
        outcome (Series): Series of outcome model is trying to predict
        inctime (bool): Whether to time the speed of creating model, defaults
            False
    Returns None, fits model
    '''
    if inctime:
        start = datetime.datetime.now()
    model.set_params(**params)
    model.fit(train_features, outcome)
    if inctime:
        stop = datetime.datetime.now()
        print("Time Elapsed Training:", stop - start)


def evaluate(target_test, target_predict):
    '''
    Evaluates how well the model did in predicting the wanted outcome through
    the Precision, Recall, F1 Score and Accuracy

    Inputs:
        target_test (Series): Expected outcomes for test data
        target_predict (Series): Predicted outcomes for test data
    
    Returns tuple of Precision, Recall, F1 Score and Accuracy
    '''
    MSE = metrics.mean_squared_error(target_test, target_predict)
    return MSE


def gridsearch(cv, models, grid, outcome, model_time=False):
    '''
    For each model and possible parameters provided, builds the model and then 
    evaluates the model's prediction on the test set

    Inputs:
        train_set(DataFrame): training dataset
        test_set(DataFrame): testing dataset
        models (dict): dictionary of models to try
        grid (dict): dictionary of parameters to try given a model
        outcome (str): name of outcome column

    Returns:
        Dataframe of evaluation results of each model
    '''
    # Begin timer 
    start = datetime.datetime.now()
    results = dict()

    for model_key in models.keys(): 
        results[model_key] = dict()
        for num, case in enumerate(cv):
            train_set, test_set = case[0], case[1]
            results[model_key][num] = dict()
            # Loop over parameters 
            for n, params in enumerate(grid[model_key]): 
                print("Training model:", model_key, "|", params, num)
                # Create model 
                model = models[model_key]
                train_features = train_set.drop(["Year", "School Name_x", outcome], axis=1)
                train_target = train_set.loc[:, outcome]
                build_classifier(model, train_features, params, train_target,
                                model_time)
                # Predict on testing set 
                target_predict = model.predict(test_set.drop(["Year", "School Name_x", outcome], axis=1))
                target_true = test_set.loc[:, outcome]
                # Evaluate predictions 
                r2score = evaluate(target_true, target_predict)
                results[model_key][num][n] = r2score
                print(results)
    # results = pd.DataFrame.from_dict(results, orient="index")
    # End timer
    stop = datetime.datetime.now()
    print("Time Elapsed:", stop - start)
    return results