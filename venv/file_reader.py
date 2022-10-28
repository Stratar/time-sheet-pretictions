import pandas as pd
from postfresql_import import fetch_postgresql_database
import numpy as np
import matplotlib.pyplot as plt


'''
For the initial testing, EXCEL files downloaded from the demo site are used, which are not reliable.
There is no need to stitch databases together, since no concluding results will be drawn anyway. 
The excel files can be used for debugging and testing for compatibility of the data types.
'''


def fill_blanks(df):
    for name in df.columns:
        if df[name].isnull().values.any():
            df[name] = df[name].fillna(0)

    return df


def database_from_excel():
    df = pd.read_excel('../../data/edited data/db_export.xlsx')
    return df


def rename_columns(df):

    # There are a lot of columns named description because of the inherit structure of the database. This line
    # disambiguates the column names and meanings. The new names are selected based on the original database they were
    # taken from.

    d = {'description': ['payrollcomponent', 'payrollcomponenttype', 'timecardrepresentation', 'period']}
    df = df.rename(columns=lambda c: d[c].pop(0) if c in d.keys() else c)

    d = {'startdate': ['ass_startdate', 'asscomp_startdate']}
    df = df.rename(columns=lambda c: d[c].pop(0) if c in d.keys() else c)

    d = {'enddate': ['ass_enddate', 'asscomp_enddate']}
    df = df.rename(columns=lambda c: d[c].pop(0) if c in d.keys() else c)

    d = {'active': ['ass_active', 'flex_active', 'staff_active']}
    df = df.rename(columns=lambda c: d[c].pop(0) if c in d.keys() else c)
    return df

# def datetime_fix(df):
#     for name in df.columns:
#         if "date" is in name:
#             df[name] = pd.to_datetime(df[name], format='%Y-%m-%d')
#             for i in df:
#                 i['year'] = i.name.dt.year
#                 i['month']=i.name.dt.month
#                 i['day']=i.name.dt.day
#
#     return df


def num_period(df):
    df = df[df.period.str.contains('Week')]
    print(np.unique(df['period']))
    return df


def set_date_to_idx(df):
    df['linedate'] = pd.to_datetime(df['linedate'], format='%Y-%m-%d')
    df.set_index('linedate', drop=True, append=False, inplace=True, verify_integrity=False)
    df = df.sort_index(ascending=True)
    '''df.index = pd.to_datetime(df['linedate'], format='%Y-%m-%d')
    df = df.sort_values(by=df.index)'''
    return df


def create_date_features(df):
    df = df.copy()
    df['hour'] = df.index.hour
    df['dayofweek'] = df.index.dayofweek
    df['quarter'] = df.index.quarter
    df['month'] = df.index.month
    df['year'] = df.index.year
    df['dayofyear'] = df.index.dayofyear
    df['dayofmonth'] = df.index.day
    df['weekofyear'] = df.index.isocalendar().week
    return df


def separate_by_worker(df):
    '''
    Use this segment to make different dataframes for each flexworker available in the dataset. There should be a list
    of dataframes returned, each containing all the relevant information for the selected worker.
    '''
    df_list = []
    n = len(df["flexworkerid"].unique())
    for worker in df["flexworkerid"].unique():
        df1 = df[df["flexworkerid"] == worker].copy()
        df1['amount'].plot()
        plt.show()
        if df1.empty:
            continue
        df_list.append(df1)
    exit(0)
    return df_list


def read_file(split=-1):
    # Set some options for displaying the data through pandas

    pd.set_option('display.max_rows', 500)
    pd.set_option('display.max_columns', 500)
    pd.set_option('display.width', 1000)

    # Import data and trim unnecessary rows IF the data is imported directly from an excel file
    # df1 = pd.read_excel('../../data/akyla excel/Uren export definitief.xlsx') # casting to float here won't work
    # df1 = df1.set_axis(list(df1.iloc[5]), axis='columns', copy=False)
    # df1 = df1.iloc[6:]
    # df1 = df1.iloc[:-4]

    # display(df1)
    '''
    ['Datum', 'Periodenummer', 'Aantal uren', 'Urensoort', 'Plaatsing', 'Functie', 'Inlener', 'Flexkracht']
    This is the original data that was used for the predictions of the original trials. Some will be added or removed,
    depending on their impact on the prediction.
    Removed: Plaatsing (ID/CODE), Datum, Flexkracht, Urensoort
    I have not yet checked anything with Periodenummer
    
    THIS WAS VALID FOR THE EXCEL IMPORT, NOT THE DATABASE IMPORT!!!!!!!
    '''
    df1 = fetch_postgresql_database()
    # df1 = database_from_excel()
    # df1 = rename_columns(df1)

    # x_df1 = ['Datum', 'Periodenummer', 'Aantal uren', 'Urensoort', 'Plaatsing', 'Functie', 'Inlener', 'Flexkracht']
    #     # df1 = df1.filter(items=x_df1, axis=1)

    # display(list(df1['Aantal uren']))
    # print(df1.shape)

    # Specifying these separately rather than already since the import of the excel makes them actually cast to floats
    # df1["Aantal uren"] = pd.to_numeric(df1["Aantal uren"], downcast="float")
    # df1["Periodenummer"] = pd.to_numeric(df1["Periodenummer"], downcast="float")

    # Remove the top 1% to trim the outliers
    # q = df1["totalhours"].quantile(0.99)
    # df1 = df1[df1["totalhours"] < q]
    # q = df1["amount"].quantile(0.99)
    # df1 = df1[df1["amount"] < q]
    # df1 = fill_blanks(df1)
    #num_period(df1)

    # df1 = set_date_to_idx(df1)

    # df1 = create_date_features(df1)

    # df1['totalhours'][:1000].plot()
    # plt.show()
    # exit(0)
    # Export the data to a local excel file for ease of access and clearer reading of the information
    # df1.to_excel("../../data/edited data/db_export.xlsx")
    df1.to_excel("../../data/edited data/big_db_export.xlsx")

    if split == 1:
        df_list = []
        df_list = separate_by_worker(df1)
        return df_list

    print(df1.head())
    exit(1)
    return df1
