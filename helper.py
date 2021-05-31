from sklearn.linear_model import LinearRegression, Ridge, Lasso, ElasticNet
from sklearn.preprocessing import PolynomialFeatures
import pandas as pd
import numpy as np
import grid_search
import pipeline

models = {"LinearRegression" : LinearRegression(),
          "Ridge" : Ridge(),
          "Lasso" : Lasso(),
          "ElasticNet" : ElasticNet()}

def polynomial_transform(df, degree=2, interactions_only=False):
    poly = PolynomialFeatures(degree=degree, include_bias=False, 
                              interaction_only=interactions_only)
    sub_df = df.drop(["5YR Grad Rate", "Year"], axis=1)
    year_gr = df[["5YR Grad Rate", "Year"]]
    pf  = poly.fit_transform(sub_df)
    poly_df = pd.DataFrame(pf, columns=poly.get_feature_names(sub_df.columns))
    df = pd.concat([poly_df, year_gr], axis=1)
    return df


def find_features(df, model_pd):
    dfs = grid_search.train_val_test_split(df)
    df_train_y, df_train_x, df_val_y, df_val_x, df_test_y, df_test_x = dfs
    df_train_x, df_val_x, df_test_x = grid_search.normalize(df_train_x,
                                                            df_val_x, df_test_x)
    k = len(df_train_x)
    df_tv_x = [pd.DataFrame(columns = list(df_train_x[0].columns))] * k
    df_tv_y = [pd.Series()]*k

    for i in range(k):
        df_tv_x[i] = df_tv_x[i].append(df_train_x[i]).append(df_val_x[i])
        df_tv_y[i] = df_tv_y[i].append(df_train_y[i]).append(df_val_y[i])

    model = models[model_pd["Model"]]
    params = model_pd["Params"]
    model.set_params(**params)
    model.fit(df_tv_x[k-1], df_tv_y[k-1])
    n = len(model.coef_)
    coefs = pd.DataFrame(np.round(model.coef_.reshape(n, 1), decimals=2),
                         index=df_tv_x[k-1].columns, columns=["coef"])
    predictions = model.predict(df_test_x)
    results = pipeline.evaluate(df_test_y, predictions)
    print(model.intercept_)

    return coefs.sort_values(by="coef",axis=0, ascending=False), results