# emp_iep.csv has "Dept/Unit Number" (6 digit school id), 
# "Unit_2" (4 digit School id), "School ID" (building id)

import pandas as pd

EMP_IEP = pd.read_csv("data/emp_iep.csv")
RACE_DEMO = pd.read_csv("data/RacialDemographics.csv")
RACE_COL = ["School ID", "Year"]
SUS_EXP = pd.read_csv("data/SuspensionsExpulsionsPoliceContacts.csv")
SUS_COLS = ["School ID", "School Year"]
ATT_GR = pd.read_csv("data/attendance_gradrates2.csv")
ATT_COLS = ["School ID", "Year"]
IDTYPES = {"six_sch_id": "School ID", "four_sch_id": "Unit_2",
           "building_id": "Dept/Unit Number"}

def merging(df1, df2, df2_colname,df3, df3_colname, df4, df4_colname, 
            idtype="six_sch_id"):
    df3["School Year"] = df3["School Year"] + 1
    df4["Year"] = df4["Year"] + 1
    df1["Year"] = df1["Year"] + 1
    sub_pd = df1.merge(df2, how='left', left_on=[IDTYPES[idtype], "Year"],
                       right_on=df2_colname)
    sub_pd = sub_pd.merge(df4, how='left', left_on=[IDTYPES[idtype], "Year"],
                        right_on=df4_colname)
    full_pd = sub_pd.merge(df3, how='left', left_on=[IDTYPES[idtype], "Year"],
                        right_on=df3_colname)
    full_pd = full_pd[full_pd["Year"] != 2010]
    full_pd.to_csv("school_data.csv")
    return full_pd


def run():
    return merging(EMP_IEP, ATT_GR, ATT_COLS,  SUS_EXP, SUS_COLS,
                   RACE_DEMO, RACE_COL)