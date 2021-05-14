# emp_iep.csv has "Dept/Unit Number" (6 digit school id), 
# "Unit_2" (4 digit School id), "School ID" (building id)

import pandas as pd

EMP_IEP = pd.read_csv("data/emp_iep.csv")
IDTYPES = {"six_sch_id": "Dept/Unit Number", "four_sch_id": "Unit_2",
           "building_id": "School ID"}

def merging(df1, df2, df2_colname,df3, df3_colname, 
            df2idtype="six_sch_id", df3idtype="six_sch_id"):
    sub_pd = df1.merge(df2, how='left', left_on=[IDTYPES[df2idtype], "Year"],
                       right_on=df2_colname)
    full_pd = sub_pd.merge(df3, how='left', left_on=[IDTYPES[df3idtype], "Year"],
                        right_on=df3_colname)
    full_pd.to_csv("school_data.csv")