import pandas as pd
# to do: restrict to high schools
DATA_FILES = {"iep_2011.xls": 2010, "iep_2012.xls": 2011,
              "iep_2013.xls": 2012, "iep_2014.xls": 2013, 
              "iep_2015.xls": 2014, "iep_2016.xls": 2015}

def read_data(file, filetype):
    if filetype == "csv":
        df = pd.read_csv(file)
    elif filetype == "xls":
        df = pd.read_excel(file)
    else:
        return None
    df["Year"] = DATA_FILES[file]
    return df


def combine_dfs():
    eleven = read_data("iep_2011.xls", "xls")
    eleven = eleven.drop(['Area'], axis=1)
    twelve = read_data("iep_2012.xls", "xls")
    twelve = twelve.drop(["Unit", "Networks"], axis=1) \
                   .rename({"School Name": "School",
                            "School ID": "Unit"}, axis=1)
    thirteen = read_data("iep_2013.xls", "xls")
    thirteen = thirteen.drop(["Networks"], axis=1
                            ).rename({"School Name": "School",
                            "School ID": "Unit"}, axis=1)
    fourteen = read_data("iep_2014.xls", "xls")
    fourteen = fourteen.drop(["Network"], axis=1) \
                       .rename({"Educational Unit Name": "School",
                       "School ID": "Unit"}, axis=1)
    fifteen = read_data("iep_2015.xls", "xls")
    fifteen = fifteen.drop(["Networks"], axis=1) \
                       .rename({"Educational Unit Name": "School",
                       "School ID": "Unit"}, axis=1)
    data = (eleven, twelve, thirteen, fourteen, fifteen)
    return pd.concat(data)


def create_csv():
    df = combine_dfs()
    df.to_csv("iep.csv")