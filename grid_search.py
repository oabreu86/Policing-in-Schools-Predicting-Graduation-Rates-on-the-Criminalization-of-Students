import pandas as pd
import pipeline
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

default_split = {0 : [[2012], 2013, 2014],
             1 : [[2012, 2013], 2014, 2015],
             2 : [[2012, 2013, 2014], 2015, 2016]}
default_ycol = "5YR Grad Rate"
default_selection_param = "RMSE"

def train_val_test_split(df, split = default_split, ycol = default_ycol):
    k = len(split)

    df_train = [pd.DataFrame(columns = list(df.columns))]*k
    df_val = [pd.DataFrame(columns = list(df.columns))]*k
    df_test = [pd.DataFrame(columns = list(df
    .columns))]*k

    for i in range(k):
        for train_yr in split[i][0]:
            df_train[i] = df_train[i].append(df[df["Year"] == train_yr])
        df_val[i] = df_val[i].append(df[df["Year"] == split[i][1]])
        df_test[i] = df_test[i].append(df[df["Year"] == split[i][2]])

    df_train_y = [None]*k
    df_train_x = [None]*k
    df_val_y = [None]*k
    df_val_x = [None]*k
    df_test_y = [None]*k
    df_test_x = [None]*k

    for i in range(k):
        df_train_y[i] = df_train[i][ycol]
        df_train_x[i] = df_train[i].drop(columns = [ycol, "Year"])
        df_val_y[i] = df_val[i][ycol]
        df_val_x[i] = df_val[i].drop(columns = [ycol, "Year"])
        df_test_y[i] = df_test[i][ycol]
        df_test_x[i] = df_test[i].drop(columns = [ycol, "Year"])

    return df_train_y, df_train_x, df_val_y, df_val_x, df_test_y, df_test_x

def normalize(df_train_x, df_val_x, df_test_x):
    k = len(df_train_x)
    train_norm = []
    valid_norm = []
    test_norm = []
    for n in range(k):
        df = pd.concat((df_train_x[n], df_val_x[n]))
        df_norm, scaler = pipeline.normalize(df)
        tr_norm = df_norm.loc[df_train_x[n].index,:]
        val_norm = df_norm.loc[df_val_x[n].index,:]
        te_norm, _ = pipeline.normalize(df_test_x[n], scaler=scaler)
        train_norm.append(tr_norm)
        valid_norm.append(val_norm)
        test_norm.append(te_norm)
    return train_norm, valid_norm, test_norm


def grid_search_time_series_cv(df_train_y, df_train_x, df_val_y, df_val_x,
                               models, p_grid, ret_int_results = False, print = False):
    k = len(df_train_y)
    val_results = [pd.DataFrame(columns = ["Model", "Params", "RMSE", "MAE", "R^2"])]*k

    for i in range(k):
        for model_key in models.keys():
            for params in p_grid[model_key]:
                if print == True:
                    print("Training model:", model_key, "|", params)
                model = models[model_key]
                model.set_params(**params)
                fitted_model = model.fit(df_train_x[i], df_train_y[i])
                test_predictions = fitted_model.predict(df_val_x[i])
                rmse = mean_squared_error(df_val_y[i], test_predictions, squared = False)
                mae = mean_absolute_error(df_val_y[i], test_predictions)
                r2 = r2_score(df_val_y[i], test_predictions)
                val_results[i] = val_results[i].append(pd.DataFrame([[model_key, params, rmse, mae, r2]],
                                                       columns = ["Model", "Params", "RMSE", "MAE", "R^2"]))

    avg_val_results = pd.DataFrame(columns = ["Model", "Params", "RMSE", "MAE", "R^2"])
    avg_val_results["Model"] = val_results[0]["Model"]
    avg_val_results["Params"] = val_results[0]["Params"]
    avg_val_results["RMSE"] = [0]*len(val_results[0])
    avg_val_results["MAE"] = [0]*len(val_results[0])
    avg_val_results["R^2"] = [0]*len(val_results[0])
    for i in range(k):
        avg_val_results["RMSE"] += val_results[i]["RMSE"]/k
        avg_val_results["MAE"] += val_results[i]["MAE"]/k
        avg_val_results["R^2"] += val_results[i]["R^2"]/k
    avg_val_results = avg_val_results.reset_index()

    if ret_int_results == True:
        return avg_val_results, val_results
    else:
        return avg_val_results

def select_best_model(avg_val_results, selection_param = default_selection_param):
    best_model = avg_val_results[avg_val_results[selection_param] == avg_val_results[selection_param].min()].iloc[0]
    return best_model

def select_model(avg_val_results, row):
    chosen_model = avg_val_results.iloc[row]

def test_model(df_train_y, df_train_x, df_val_y, df_val_x, df_test_y, df_test_x,
               chosen_model, models):
    k = len(df_train_y)
    model = models[chosen_model["Model"]]
    model.set_params(**chosen_model["Params"])

    test_results = pd.DataFrame(columns = ["CV", "RMSE", "MAE", "R^2"])

    df_tv_x = [pd.DataFrame(columns = list(df_train_x[0].columns))]*k
    df_tv_y = [pd.Series()]*k
    for i in range(k):
        df_tv_x[i] = df_tv_x[i].append(df_train_x[i]).append(df_val_x[i])
        df_tv_y[i] = df_tv_y[i].append(df_train_y[i]).append(df_val_y[i])

    for i in range(k):
        fitted_model = model.fit(df_tv_x[i], df_tv_y[i])
        test_predictions = fitted_model.predict(df_test_x[i])
        rmse = mean_squared_error(df_test_y[i], test_predictions, squared = False)
        mae = mean_absolute_error(df_test_y[i], test_predictions)
        r2 = r2_score(df_test_y[i], test_predictions)
        test_results = test_results.append(pd.DataFrame([[str(i), rmse, mae, r2]],
                                                        columns = ["CV", "RMSE", "MAE", "R^2"]))

    test_results = test_results.append(pd.DataFrame([["Average", test_results["RMSE"].mean(),
                                                      test_results["MAE"].mean(), test_results["R^2"].mean()]],
                                                      columns = ["CV", "RMSE", "MAE", "R^2"]))
    return test_results

def choose_and_test_model(df, models, p_grid, split = default_split, ycol = default_ycol, selection_param = default_selection_param):
    df_train_y, df_train_x, df_val_y, df_val_x, df_test_y, df_test_x = train_val_test_split(df, split, ycol)
    df_train_x, df_val_x, df_test_x = normalize(df_train_x,df_val_x, df_test_x)
    avg_val_results = grid_search_time_series_cv(df_train_y, df_train_x, df_val_y, df_val_x, models, p_grid)
    best = select_best_model(avg_val_results, selection_param)
    test_results = test_model(df_train_y, df_train_x, df_val_y, df_val_x, df_test_y, df_test_x, best, models)
    return test_results
