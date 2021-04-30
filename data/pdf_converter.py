import camelot
import re
import pandas as pd
from math import ceil

files = {2009:{"2009_roster.pdf":863}, 2010:{"2010_roster.pdf":690},
         2011:{"2011_roster.pdf":719}, 2012:{"2012_roster.pdf":669},
         2013:{"2013_roster.pdf":763}}

def read_and_convert(file_name, pages_string):
        tables = camelot.read_pdf(file_name, pages=pages_string)
        df = tables[0].df
        for table in tables[1:]:
            df = pd.concat([df, table.df], ignore_index=True)
        return df


def convert_pdfs(file_dict):
    for file_name, pages in file_dict.items():
        parsed_filename = re.search(r'((?:[\w-]+))\.((?:[\w-]+))', file_name)
        next_start_page = 1
        next_df = pd.DataFrame()
        for _ in range(ceil(pages/50)):
            if (next_start_page + 49) < pages:
                end_page = next_start_page + 49
            else:
                end_page = pages
            pages_string = str(next_start_page) + '-' + str(end_page)
            print(f"reading pdf pages {pages_string}")
            df = read_and_convert(file_name, pages_string)
            df.columns = df.iloc[0]
            removal_term = df.iloc[0,2]
            df = df[df[removal_term] != removal_term]
            next_df = pd.concat([next_df, df])
            if end_page == pages:
                break
            next_start_page = end_page + 1
        print("converting to csv")
        csv_name = parsed_filename.group(1) + ".csv"
        next_df.to_csv(csv_name)
