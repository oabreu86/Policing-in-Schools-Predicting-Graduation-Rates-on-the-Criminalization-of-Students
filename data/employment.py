import pandas as pd
# To do: Add way to clean and restrict to high schools
DATA_FILES = {"2009_roster.csv": 2009, "2010_roster.csv" : 2010,
              "2011_roster.csv": 2011, "2012_roster.csv": 2012,
              "2013_roster.csv": 2013, "2014_roster.xls": 2014}

def read_data(file, filetype):
    if filetype == "csv":
        df = pd.read_csv(file)
    elif filetype == "xls":
        df = pd.read_excel(file)
    else:
        return None
    df["Year"] = DATA_FILES[file]
    return df

def combine_datafile():
    nine_data = read_data("2009_roster.csv", "csv")
    nine_data = nine_data[["Position Number", "Dept/Unit Name", "Job Title", "Year"]]
    ten_data = read_data("2010_roster.csv", "csv")
    ten_data = ten_data[["Position Number", "Dept/Unit Name", "Job Title", "Year"]]
    eleven_data = read_data("2011_roster.csv", "csv")
    eleven_data = eleven_data[['Position\xa0Number','Dept/Unit\xa0Name',
                               'Job\xa0Title', "Year"]] \
                             .rename({'Dept/Unit\xa0Name': 'Dept/Unit Name',
                                      'Job\xa0Title': "Job Title",
                                      'Position\xa0Number': "Position Number"}, 
                                     axis=1)
    twelve_data = read_data("2012_roster.csv", "csv")
    twelve_data = twelve_data[['Position\xa0Number','Dept/Unit\xa0Name',
                               'Job\xa0Title', "Year"]] \
                             .rename({'Dept/Unit\xa0Name': 'Dept/Unit Name',
                                      'Job\xa0Title': "Job Title",
                                      'Position\xa0Number': "Position Number"}, 
                                     axis=1)
    thirteen_data = read_data("2013_roster.csv", "csv")
    thirteen_data =thirteen_data[['POSITION \nNUMBER', "UNIT NAME", 
                                  "JOB DESCRIPTION", "Year"]] \
                             .rename({'UNIT NAME': 'Dept/Unit Name',
                                      'JOB DESCRIPTION': "Job Title",
                                      'POSITION \nNUMBER': "Position Number"}, 
                                      axis=1)
    fourteen_data = read_data("2014_roster.xls", "xls")
    fourteen_data = fourteen_data[['POSITION\nNUMBER', "UNIT NAME",
                                   "JOB DESCRIPTION", "Year"]]\
                             .rename({'UNIT NAME': 'Dept/Unit Name',
                                      'JOB DESCRIPTION': "Job Title",
                                      'POSITION\nNUMBER': "Position Number"}, 
                                      axis=1)
    data = (nine_data, ten_data, eleven_data, twelve_data, thirteen_data,
           fourteen_data)
    combdf = pd.concat(data)
    combdf["Dept/Unit Name"] = combdf["Dept/Unit Name"].str.strip("0123456789")
                                               
    return combdf

def count_titles(df,rename_col):
    df_one = df.groupby(["Year", "Dept/Unit Name"])["Position Number"] \
                  .count().rename(rename_col)
    df = pd.get_dummies(df, columns=["Job Title"])
    subdf = df.groupby(["Year", "Dept/Unit Name"]).sum()
    return subdf.join(df_one).reset_index()


def get_police_info(df):
    FIX_STRINGS = ['\xa0', '\nSenior School Security Officer','\nh\nl\nff',
               '\nSchool Security Officer', '\nl', '\nh', '\nd']
    for x in FIX_STRINGS:
        df['Job Title'] = df['Job Title'].str.replace(x,' ').str.strip()
    police_df = df[df['Job Title'].str.contains("Security O", na=False)].dropna()
    return count_titles(police_df, "Total Security")


def get_counselor_info(df):
    FIX_STRINGS = ['\nSchool Counselor','\nGuidance Counselor Assistant']
    for x in FIX_STRINGS:
        df['Job Title'] = df['Job Title'].str.replace(x,' ').str.strip()
    couns_df = df[df["Job Title"].str.contains("Counsel.", na=False)]
    return count_titles(couns_df, "Total Counseling")


def subsect_df(df):
    police_df = get_police_info(df)
    couns_df = get_counselor_info(df)
    ndf = police_df.merge(couns_df, how='left', on=['Year', 'Dept/Unit Name'])
    ndf.iloc[:,2:] = ndf.iloc[:,2:].fillna(0)
    return ndf


def create_file():
    df = combine_datafile()
    ndf = subsect_df(df)
    ndf.to_csv("employment.csv")