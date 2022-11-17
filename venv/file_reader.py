import pandas as pd
from postfresql_import import fetch_postgresql_database

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


def test_database_from_csv():
    df = pd.read_csv('../../data/edited data/test_temp.csv')
    df['Date'] = pd.to_datetime(df['Date'], format='%Y-%m-%d')
    df.set_index('Date', drop=True, append=False, inplace=True, verify_integrity=False)
    df = df.sort_index(ascending=True)
    return df


def database_from_excel():
    df = pd.read_excel('../../data/edited data/big_db_export.xlsx')
    return df


def database_from_csv():
    df = pd.read_csv("../../data/edited data/worker_big_db_export.csv")
    return df


def set_date_to_idx(df):
    df['timecardline_linedate'] = pd.to_datetime(df['timecardline_linedate'], format='%Y-%m-%d')
    df.set_index('timecardline_linedate', drop=True, append=False, inplace=True, verify_integrity=False)
    df = df.sort_index(ascending=True)
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


def read_file(test=False, connection=True, store_locally=False):
    # Set some options for displaying the data through pandas

    pd.set_option('display.max_rows', 500)
    pd.set_option('display.max_columns', 500)
    pd.set_option('display.width', 1000)

    '''
    ['Datum', 'Periodenummer', 'Aantal uren', 'Urensoort', 'Plaatsing', 'Functie', 'Inlener', 'Flexkracht']
    This is the original data that was used for the predictions of the original trials. Some will be added or removed,
    depending on their impact on the prediction.
    Removed: Plaatsing (ID/CODE), Datum, Flexkracht, Urensoort
    I have not yet checked anything with Periodenummer
    
    THIS WAS VALID FOR THE EXCEL IMPORT, NOT THE DATABASE IMPORT!!!!!!!
    '''
    if connection:
        df = fetch_postgresql_database()
    else:
        # df = database_from_excel()
        df = database_from_csv()
    if test:
        df = test_database_from_csv()

    df = set_date_to_idx(df)

    df = create_date_features(df)
    if store_locally:
        # The csv is not very clear for people to read, but it is the only way to store such large amounts of data
        # could be redundant, but keep in case of no internet connection.
        df.to_csv("../../data/edited data/worker_big_db_export.csv")

    return df
