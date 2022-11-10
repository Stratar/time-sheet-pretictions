import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
import math
import statistics as stat
import pandas as pd


'''
The data needs to be handled before being forwarded to the selected model. This file contains most of the functions 
necessary to format the data in such a way that is acceptable by most algorithms.

The prefix multi refers to the function's ability to handle multiple variables as input 
'''


def multi_scaler(df_np):
    '''
    The inputs to the network is best processed with small variances, therefore scaling the inputs makes sense.
    However, the way to scale them right now relies on the last item being the variable to be predicted.
    '''
    if len(df_np.shape) > 1:
        scalers = []
        df_np = np.transpose(df_np)
        for i in range(df_np.shape[0]):
            scaler = MinMaxScaler()
            df_np[i] = scaler.fit_transform(df_np[i].reshape(-1,1)).reshape(1,-1)[0]
            scalers.append(scaler)
        df_np = np.transpose(df_np)
        scaler = scalers[-1]
        del scalers
    else:
        scaler = MinMaxScaler()
        df_np = scaler.fit_transform(df_np.reshape(-1, 1)).reshape(1, -1)[0]
    return df_np, scaler


def df_to_np(df):
    df_np = df.to_numpy()
    return df_np


# The RNN needs data with the format of [samples, time steps, features]
# Here, we create N samples, input_sequence_length time steps per sample, and f features
def partition_dataset(data, in_win_size, out_win_size, step=1):
    x, y = [], []
    data_len = data.shape[0]
    for i in range(in_win_size, data_len - out_win_size, step):
        x.append([data[i - in_win_size:i]])  # contains input_sequence_length values 0-input_sequence_length * columns
        y.append([data[i:i + out_win_size]])  # contains the prediction values for validation

    # Convert the x and y to numpy arrays
    x = np.array(x).reshape(len(x),in_win_size, 1)
    y = np.array(y).reshape(len(y),out_win_size, 1)

    return x, y


def multi_partition_dataset(data, in_win_size, out_win_size, step=1):
    x, y = [], []
    data_len = data.shape[0]
    for i in range(in_win_size, data_len - out_win_size, step):
        row = [r for r in data[i - in_win_size:i]]
        x.append(row)
        label = [r[-1] for r in data[i:i + out_win_size]]
        y.append(label)

    n_dims = data.shape[1]
    # Convert the x and y to numpy arrays
    x = np.array(x).reshape(len(x),in_win_size, n_dims)
    y = np.array(y).reshape(len(y),out_win_size, 1)

    return x, y


def train_val_split(df_np, in_win_size, out_win_size=1):
    '''
    Use this when the data is already in numpy format
    '''
    train_data_length = math.ceil(df_np.shape[0] * 0.8)
    val_data_length = math.ceil(df_np.shape[0])
    train_data = df_np[:train_data_length]
    val_data = df_np[train_data_length - in_win_size :val_data_length]

    return train_data, val_data


def get_test_set_np(df_list, size=0.8, in_win_size=7):
    if len(df_list) == 1:
        data_length = math.ceil(df_list[0].shape[0] * size)
        test_data = df_list[0][data_length - in_win_size:]
        df_list[0] = df_list[0][:data_length]
    elif len(df_list) > 1:
        data_length = math.ceil(len(df_list) * size)
        test_data = df_list[-1]
        df_list = df_list[:-1]

    return df_list, test_data


def convert_and_scale(df_list):

    for i in range(len(df_list)):
        df_list[i] = df_to_np(df_list[i])
        df_list[i], scaler = multi_scaler(df_list[i])

    return df_list, scaler


def create_subsets(df, features, split=-1):
    df_list = []
    if split == 1:
        df_list = get_flex_groups(df, features)
    else:
        df_list.append(df)

    return df_list


def fill_gaps(df_list, dt_inputs=False):
    for df in df_list:
        cols = df.columns
        df1 = df.drop(columns=cols[-1]).resample('1D').ffill()
        if dt_inputs:
            df1['dayofweek'] = df1.index.dayofweek
            df1['weekofyear'] = df1.index.isocalendar().week
            df1['dayofyear'] = df1.index.dayofyear
        df2 = df.drop(columns=cols[:-1]).resample('1D').mean().fillna(0)
        df = pd.concat([df1, df2], axis=1)
    return df_list


def uniform_data_types(df_list):
    cols = df_list[0].columns
    for df in df_list:
        for name in cols:
            if df[name].dtype == 'UInt32':
                df[name] = df[name].astype("int64")

    return df_list


def remove_double_indices(df):
    '''
    This function is currently hardcoded, assuming that the database received has these fixed names
    There must be a better way of doing this, to apply to all scenarios
    '''
    return df.groupby(df.index).agg({'assignment_functionname': 'first',
                                   'assignment_startdate': 'first',
                                   'assignment_enddate': 'first',
                                   'assignment_active': 'first',
                                    'assignment_deleted':'first',
                                    'assignment_flexworkerid':'first',
                                    'assignmentcomponent_startdate':'first',
                                    'assignmentcomponent_enddate':'first',
                                    'assignmentcomponent_wage':'first',
                                    'timecardline_starttime':'first',
                                    'timecardline_endtime':'first',
                                    'timecardline_resttime':'first',
                                    'timecardline_amount':sum,
                                    'timecard_totalhours':'first',
                                    'timecard_totalexpense':'first',
                                    'payrollcomponent_description':'first',
                                    'payrollcomponenttype_description':'first',
                                    'flexworkerbase_flexworkertype':'first',
                                    'flexworkerbase_active':'first',
                                    'flexworkerbase_status':'first',
                                    'staffingcustomer_companyname': 'first',
                                    'staffingcustomer_active': 'first',
                                    'staffingcustomer_region':'first',
                                    'timecardrepresentation_description':'first',
                                    'period_description':'first',
                                    'hour':'first',
                                    'dayofweek':'first',
                                    'quarter':'first',
                                    'month':'first',
                                    'year':'first',
                                    'dayofyear':'first',
                                    'dayofmonth':'first',
                                    'weekofyear':'first'})


def get_flex_groups(df, features):
    '''
    Use this segment to make different dataframes for each flexworker available in the dataset. There should be a list
    of dataframes returned, each containing all the relevant information for the selected worker.
    Keep in mind that the column name may change, depending on the dataset that is being used.
    '''
    name_grouping = "assignment_flexworkerid"
    company_grouping = "staffingcustomer_companyname"
    df_list = []
    group_df = df.groupby(name_grouping)
    i = 0
    flexworker_collection = df[name_grouping].unique()
    sizes = []
    for flexworkerid in flexworker_collection:
        df1 = group_df.get_group(flexworkerid)
        sizes.append(df1.shape[0])
        if len(df1[company_grouping].unique()) > 1:
            group_comp = df1.groupby(company_grouping)
            j=0
            for company in df1[company_grouping].unique():
                df2 = group_comp.get_group(company)
                df2 = remove_double_indices(df2)
                length = len(df2)
                if not (df2[features[-1]] == 0).all():
                    df_list.append(df2[features])
                    # df_list.append(df2[features[0]])
            #         df2.to_excel(f"../../data/edited data/workers exp/worker{i}_company{j}.xlsx")
            #     j+=1
            # i+=1
            continue
        df1 = remove_double_indices(df1)
        length = len(df1)
        if not (df1[features[-1]] == 0).all():
            df_list.append(df1[features]) # For now drop the ids, but they are needed later!
            # df_list.append(df1[features[0]]) # For now drop the ids, but they are needed later!
            # df_list.append(df1.drop(columns=grouping)) # For now drop the ids, but they are needed later!

        #     df1.to_excel(f"../../data/edited data/workers exp/worker{i}.xlsx")
        # i+=1

    # df_list[-1].to_excel("../../data/edited data/workers exp/worker.xlsx")
    print(min(sizes))
    print(max(sizes))
    print(stat.median(sizes))
    print(stat.mean(sizes))
    return df_list


def get_flex_groups_original(df, features):
    '''
    Use this segment to make different dataframes for each flexworker available in the dataset. There should be a list
    of dataframes returned, each containing all the relevant information for the selected worker.
    Keep in mind that the column name may change, depending on the dataset that is being used.
    '''
    name_grouping = "assignment_flexworkerid"
    company_grouping = "staffingcustomer_companyname"
    df_list = []
    max = 0
    group_df = df.groupby(name_grouping)
    i = 0
    for flexworkerid in df[name_grouping].unique():
        print(flexworkerid)
        df1 = group_df.get_group(flexworkerid)
        if len(df1[company_grouping].unique()) > 1:
            group_comp = df1.groupby(company_grouping)
            j=0
            for company in df1[company_grouping].unique():
                print(company)
                df2 = group_comp.get_group(company)
                df2 = remove_double_indices(df2)
                length = len(df2)
                if length > max and not (df2[features[-1]] == 0).all():
                    max = length
                    if df_list:df_list.pop()
                    df_list.append(df2[features[0]])
            #     df2.to_excel(f"../../data/edited data/workers exp/worker{i}_company{j}.xlsx")
            #     j+=1
            # i+=1
            continue

        df1 = remove_double_indices(df1)
        length = len(df1)
        if length > max and not (df1[features[-1]] == 0).all():
            max = length
            if df_list: df_list.pop()
            df_list.append(df1[features[0]]) # For now drop the ids, but they are needed later!
            # df_list.append(df1.drop(columns=grouping)) # For now drop the ids, but they are needed later!

        #     df1.to_excel(f"../../data/edited data/workers exp/worker{i}.xlsx")
        # i+=1
        # if df1.empty:
        #     continue
        # df_list.append(df1)
    # df_list[0].plot()
    # plt.show()
    # print(df_list[0])
    # df_list[0] = df_list[0][features[0]]

    df_list[0].to_excel("../../data/edited data/workers exp/worker.xlsx")
    return df_list