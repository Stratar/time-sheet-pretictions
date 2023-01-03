import numpy as np
from sklearn.preprocessing import MinMaxScaler
import math
import pandas as pd
import holidays
from datetime import datetime


'''
The data needs to be handled before being forwarded to the selected model. This file contains most of the functions 
necessary to format the data in such a way that is acceptable by most algorithms.

The prefix multi refers to the function's ability to handle multiple variables as input 
'''


def data_scaler(data_list):
    scalers = []
    for df_np in data_list:
        individual_scalers = []
        df_np = np.transpose(df_np)
        for i, column in enumerate(df_np):
            scaler = MinMaxScaler()
            original_shape = df_np[i].shape
            df_np[i] = scaler.fit_transform(df_np[i].reshape((-1,1))).reshape(original_shape)
            individual_scalers.append(scaler)
        scalers.append(individual_scalers)
        df_np = np.transpose(df_np)
    return data_list[0], data_list[1], data_list[2], data_list[3], data_list[4], data_list[5], scalers


def multi_scaler(df_list, scalers):
    '''
    The inputs to the network is best processed with small variances, therefore scaling the inputs makes sense.
    However, the way to scale them right now relies on the last item being the variable to be predicted.
    '''
    for df_np in df_list:
        if len(df_np.shape) > 1:
            df_np = np.transpose(df_np)
            for i, scaler in enumerate(scalers):
                df_np[i] = scaler.transform(df_np[i].reshape(-1,1)).reshape(1,-1)[0]
            df_np = np.transpose(df_np)
        else:
            scaler = MinMaxScaler()
            df_np = scaler.fit_transform(df_np.reshape(-1, 1)).reshape(1, -1)[0]
    return df_list


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


def get_test_set_np(df_list, size=0.8, in_win_size=7, out_win_size=14):
    if len(df_list) == 1:
        data_length = math.ceil(df_list[0].shape[0] * size)
        if df_list[0].shape[0] - data_length < in_win_size + out_win_size: data_length= df_list[0].shape[0] - \
                                                                                        (in_win_size + out_win_size)
        test_data = df_list[0][data_length - in_win_size:]
        df_list[0] = df_list[0][:data_length]
    elif len(df_list) > 1:
        test_data = df_list[-1]
        df_list = df_list[:-1]

    return df_list, test_data


def convert_and_scale(df_list):
    categories = df_list[0].columns # Get the number of columns that would need to be scaled
    category_dict = {c:[] for c in categories}
    for i in range(len(df_list)):
        for category in categories:
            category_dict[category].append((df_to_np(df_list[i][category])))
        df_list[i] = df_to_np(df_list[i])

    '''
    The ideal structure of scalers that we're going for is: 
    List of scalers per timesheet in list ->[ 
    List of the corresponding variable scalers from current timesheet ->[
    
    ]]
    '''

    scalers = []
    disable = True
    if not disable:
        for category in categories:
            scaler = MinMaxScaler()
            # This is equivalent to doing the reshape(-1, 1) in np.ndarray, but that can handle up to 32 dims, so it crashes
            flat = [[item] for sublist in category_dict[category] for item in sublist]
            scalers.append(scaler.fit(flat))

        df_list = multi_scaler(df_list, scalers)

    return df_list, scalers


def create_subsets(df, features, split=-1, company_split=True):
    df_list = []
    if split == 1:
        df_list = get_flex_groups(df, features)
        # df_list = get_flex_groups(df, features, company_split=company_split)
    else:
        df_list.append(df)

    return df_list


def fill_gaps(df, dt_inputs=False):
    cols = df.columns
    df1 = df.drop(columns=cols[-1]).resample('1D').ffill()
    if dt_inputs:
        if 'dayofweek' in cols: df1['dayofweek'] = df1.index.dayofweek
        if 'weekofyear' in cols: df1['weekofyear'] = df1.index.isocalendar().week
        if 'dayofyear' in cols: df1['dayofyear'] = df1.index.dayofyear
    df2 = df.drop(columns=cols[:-1]).resample('1D').mean().fillna(0)
    df = pd.concat([df1, df2], axis=1)
    return df


def uniform_data_types(df_list):
    cols = df_list[0].columns
    for df in df_list:
        for name in cols:
            if df[name].dtype == 'UInt32':
                df[name] = df[name].astype("int64")

    return df_list


def remove_double_indices(df, cnt):
    '''
    This function is currently hardcoded, assuming that the database received has these fixed names
    There must be a better way of doing this, to apply to all scenarios
    '''
    # More conditions for data squeezing can be added
    if cnt < 0: # Just to see the differences between double indices
        print(f"CNT: {cnt}")
        print(f"the size before grouping: {df.shape}")
        group = df.groupby(df.index).agg({k: sum if k == 'timecardline_amount' else 'first' for k in df.columns})
        print(f"the size after grouping: {group.shape}")
        print("------------------------------------------")
    return df.groupby(df.index).agg({k:sum if k == 'timecardline_amount' else 'first' for k in df.columns})


def get_assignment_dates(df):
    dates_dict = {'start': [], 'end': []}
    for company, company_group in df.groupby('staffingcustomer_companyname'):
        s, e = [], []
        for start, end in zip(company_group['assignment_startdate'].unique(),
                              company_group['assignment_enddate'].unique()):
            s.append(start)
            e.append(end)
        dates_dict['start'].append(s)
        dates_dict['end'].append(e)
    return dates_dict


def add_holidays(df):
    # df.loc[:,'is_holiday'] = 0
    df.insert(0, 'is_holiday', np.zeros(df.shape[0]))
    for date in np.unique(df.index.values):
        edit_date = str(date)[:10]
        edit_date = datetime.strptime(edit_date, "%Y-%m-%d").date()
        if edit_date in holidays.NL(years=[2020, 2021, 2022]).keys(): df.loc[date, 'is_holiday'] = 1
    return df


def add_total_active_assignments(df, dates_dict):
    df.loc[:,'active_assignments'] = 0
    for date in np.unique(df.index.values):
        # df = add_holidays(df, date)
        for start, end in zip(dates_dict['start'], dates_dict['end']):
            if (s <= date <= e for s, e, in zip(start, end)):
                # This may be copying and overwriting previously encountered values, because it creates a copy of
                # the original dataframe column, consider making an independent list that gets appended to the dataframe instead.
                df.loc[date, 'active_assignments'] += 1
    return df


def get_flex_groups(df, features, store_locally=False, company_split=True):
    '''
    Use this segment to make different dataframes for each flexworker available in the dataset. There should be a list
    of dataframes returned, each containing all the relevant information for the selected worker.
    Keep in mind that the column name may change, depending on the dataset that is being used.
    '''
    name_grouping = "assignment_flexworkerid"
    company_grouping = "staffingcustomer_companyname"
    df_list = []
    group_df = df.groupby(name_grouping)
    holiday_bool = False
    i = 0
    flexworker_collection = df[name_grouping].unique()
    cnt = 0
    for flexworkerid in flexworker_collection:
        df1 = group_df.get_group(flexworkerid)
        if len(df1[company_grouping].unique()) > 1:
            group_comp = df1.groupby(company_grouping)
            j=0
            dates_dict = get_assignment_dates(df1)
            df1 = add_total_active_assignments(df1, dates_dict)
            if company_split:
                for company in df1[company_grouping].unique():
                    df2 = group_comp.get_group(company)
                    df2 = remove_double_indices(df2, cnt)
                    ins = ''
                    if 'is_holiday' in features:
                        holiday_bool = True
                        for idx, val in enumerate(features):
                            if val == 'is_holiday':
                                ins = features.pop(idx)
                    df2 = fill_gaps(df2[features], dt_inputs=True)
                    df2 = add_holidays(df2)
                    if holiday_bool: features.insert(0, ins)
                    if not (df2[features[-1]] == 0).all():
                        df_list.append(df2[features])
                        if store_locally: df2.to_excel(f"../../data/edited data/workers exp/worker{i}_company{j}.xlsx")
                    j+=1
                    cnt+=1
                i+=1
                continue
        dates_dict = get_assignment_dates(df1)
        df1 = add_total_active_assignments(df1, dates_dict)
        df1 = remove_double_indices(df1, cnt)
        if 'is_holiday' in features:
            holiday_bool = True
            for idx, val in enumerate(features):
                if val == 'is_holiday':
                    ins = features.pop(idx)
        df1 = fill_gaps(df1[features], dt_inputs=True)
        df1 = add_holidays(df1)
        if holiday_bool: features.insert(0, ins)
        zero_amount_bool = (df1[features[-1]] == 0).all()
        if not zero_amount_bool:
            df_list.append(df1[features]) # For now drop the ids, but they are needed later!
            if store_locally: df1.to_excel(f"../../data/edited data/workers exp/worker{i}.xlsx")
        i+=1
        cnt+=1

    if store_locally: df_list[-1].to_excel("../../data/edited data/workers exp/worker example.xlsx")

    return df_list


def convert_data(df, features, split=True, legacy=False):
    df_list = create_subsets(df, features, split=split)
    # df_list = add_support_variables(df_list)
    '''
    Scale all input variables that would be used for the forecast. The last two flexworkers are the ones that are being
    used as the validation and test data. 
    The only data to be fitted for scaling is the test data, which will then be used to scale the rest of the data.
    Grouping done over flexworkers as G. The categories within G that we're interested in, are the ones that will be 
    scaled.
    '''
    # df_list = fill_gaps_original(df_list, dt_inputs=True)
    df_list = uniform_data_types(df_list)
    if legacy: df_list, scaler = convert_and_scale_legacy(df_list)
    else: df_list, scaler = convert_and_scale(df_list)
    return df_list, scaler


def data_split(df_np, in_win_size, out_win_size, mode):
    array_type=False
    if type(df_np) == np.ndarray: array_type=True

    if array_type: df_np, test_data = get_test_set_np([df_np], in_win_size=in_win_size, out_win_size=out_win_size)
    elif not array_type: df_np, test_data = get_test_set_np(df_np, in_win_size=in_win_size, out_win_size=out_win_size)

    df_np, val_data = get_test_set_np(df_np, in_win_size=in_win_size, out_win_size=out_win_size)

    if array_type: df_np = df_np[0]  # Maybe this looks clumsy but it works

    if mode == 0:
        x_test, y_test = partition_dataset(test_data, in_win_size, out_win_size)
        x_val, y_val = partition_dataset(val_data, in_win_size, out_win_size)
    elif mode == 1 or mode == 3:
        x_test, y_test = multi_partition_dataset(test_data, in_win_size, out_win_size)
        x_val, y_val = multi_partition_dataset(val_data, in_win_size, out_win_size)

    return df_np, x_test, y_test, x_val, y_val