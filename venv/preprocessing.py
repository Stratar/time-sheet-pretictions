import numpy as np
from sklearn.preprocessing import MinMaxScaler
import math
import pandas as pd


'''
The data needs to be handled before being forwarded to the selected model. This file contains most of the functions 
necessary to format the data in such a way that is acceptable by most algorithms.

The prefix multi refers to the function's ability to handle multiple variables as input 
'''


def multi_scaler_legacy(df_np):
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


def multi_scaler(df_list, scalers):
    '''
    The inputs to the network is best processed with small variances, therefore scaling the inputs makes sense.
    However, the way to scale them right now relies on the last item being the variable to be predicted.
    '''
    for df_np in df_list:
        if len(df_np.shape) > 1:
            df_np = np.transpose(df_np)
            for i, scaler in enumerate(scalers):
                df_np[i] = scaler.fit_transform(df_np[i].reshape(-1,1)).reshape(1,-1)[0]
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


def get_test_set_np(df_list, size=0.8, in_win_size=7):
    print(len(df_list))
    if len(df_list) == 1:
        data_length = math.ceil(df_list[0].shape[0] * size)
        test_data = df_list[0][data_length - in_win_size:]
        df_list[0] = df_list[0][:data_length]
    elif len(df_list) > 1:
        test_data = df_list[-1]
        df_list = df_list[:-1]

    return df_list, test_data


def convert_and_scale_legacy(df_list):

    for i in range(len(df_list)):
        df_list[i] = df_to_np(df_list[i])
        df_list[i], scaler = multi_scaler_legacy(df_list[i])

    return df_list, scaler


def convert_and_scale(df_list):
    categories = df_list[0].columns # Get the number of columns that would need to be scaled
    category_dict = {c:[] for c in categories}
    for i in range(len(df_list)):
        for category in categories:
            category_dict[category].append((df_to_np(df_list[i][category])))
        df_list[i] = df_to_np(df_list[i])
    scalers = []
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
        df_list = get_flex_groups(df, features, company_split=company_split)
    else:
        df_list.append(df)

    return df_list


def fill_gaps(df_list, dt_inputs=False):
    for cnt, df in enumerate(df_list):
        if cnt < 0:
            print(f"FILL GAPS CNT: {cnt}")
            print(f"the size before grouping: {df.shape}")
        cols = df.columns
        df1 = df.drop(columns=cols[-1]).resample('1D').ffill()
        if dt_inputs:
            df1['dayofweek'] = df1.index.dayofweek
            df1['weekofyear'] = df1.index.isocalendar().week
            df1['dayofyear'] = df1.index.dayofyear
        df2 = df.drop(columns=cols[:-1]).resample('1D').mean().fillna(0)
        df_list[cnt] = pd.concat([df1, df2], axis=1)
        if cnt < 0: # Just to see the differences between double indices
            print(df1.head())
            print(df1.shape)
            print(df2.head())
            print(df2.shape)
            print(f"the size after grouping: {df.shape}")
            print("------------------------------------------")
    return df_list


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

    new_group = df.groupby([name_grouping, company_grouping], group_keys=True)['assignment_startdate', 'assignment_enddate'].apply(lambda x: x)
    print(new_group.index)
    print(new_group.index.unique(level=0))
    for flexworker in new_group.index.unique(level=0):
        print("-----------------------------------------------------------------")
        flexworker_sheet = new_group.loc[flexworker]
        print(len(flexworker_sheet.index.unique(level=0)))
        print("-----------------------------------------------------------------")
        for company in flexworker_sheet.index.unique(level=0):
            company_sheet = flexworker_sheet.loc[company]
            assignment_start_end_dates = [company_sheet.loc[:,'assignment_startdate'].unique(), company_sheet.loc[:,'assignment_enddate'].unique()]
            print(assignment_start_end_dates)
    exit()

    i = 0
    flexworker_collection = df[name_grouping].unique()
    sizes = []
    cnt = 0
    for flexworkerid in flexworker_collection:
        df1 = group_df.get_group(flexworkerid)
        sizes.append(df1.shape[0])
        if len(df1[company_grouping].unique()) > 1:
            group_comp = df1.groupby(company_grouping)
            j=0
            if company_split:
                for company in df1[company_grouping].unique():
                    df2 = group_comp.get_group(company)
                    df2 = remove_double_indices(df2, cnt)
                    if not (df2[features[-1]] == 0).all():
                        df_list.append(df2[features])
                        if store_locally: df2.to_excel(f"../../data/edited data/workers exp/worker{i}_company{j}.xlsx")
                    j+=1
                    cnt+=1
                i+=1
                continue
        df1 = remove_double_indices(df1, cnt)
        if not (df1[features[-1]] == 0).all():
            df_list.append(df1[features]) # For now drop the ids, but they are needed later!
            if store_locally: df1.to_excel(f"../../data/edited data/workers exp/worker{i}.xlsx")
        i+=1
        cnt+=1

    if store_locally: df_list[-1].to_excel("../../data/edited data/workers exp/worker example.xlsx")

    return df_list


def add_support_variables(df_list):
    for df in df_list:
        print(df.head())
        print("--------------------------")
        df["amount_sum"] = df["timecardline_amount"].copy()



        all_years = df["year"].unique()
        all_weeks = df["weekofyear"].unique()

        df1 = df.groupby(["year", "weekofyear"]).agg({k:sum if k == 'amount_sum' else 'first' for k in ["amount_sum", "weekofyear"]}).drop(["weekofyear"], axis=1)
        print(df1)
        idx = df[df["dayofweek"]==0].index.values[0]
        print(idx)
        df = df.loc[idx:,:]
        print(df.head())
        df4 = df.loc[:, ["amount_sum"]].rolling(7, step=7).sum()
        # df4["amount_sum"] += hour_val
        print("start")
        print(df4)
        print("done")
        df = pd.concat([df, df4], axis=1)
        df["amount_sum"] = df["amount_sum"].shift(-1).fillna(method='bfill')
        print(df)
        exit()

        # for year in all_years:
        #     df2 = df1.loc[year]
        #     print(df2)
        #     for week in all_weeks:
        #         df3 = df2.loc[week]
        #         hour_val = df3.iloc[0]
        #         print(week)
        #         print(hour_val)
        #         '''
        #         At this point we know the total amount for each week of each year, so the dataframe needs to be filled
        #         in for rows up to the end of the selected week.
        #         '''
        # print(df1)
        # df["amount sum"] = np.where(df["weekofyear"]==df1.index, df1["amount_sum"""])
        # print(df)


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
    df_list = fill_gaps(df_list, dt_inputs=True)
    df_list = uniform_data_types(df_list)
    if legacy: df_list, scaler = convert_and_scale_legacy(df_list)
    else: df_list, scaler = convert_and_scale(df_list)
    return df_list, scaler


def data_split(df_np, in_win_size, out_win_size, mode):
    if type(df_np) == np.ndarray: array_type=True

    if array_type: df_np, test_data = get_test_set_np([df_np], in_win_size=in_win_size)
    elif not array_type: df_np, test_data = get_test_set_np(df_np, in_win_size=in_win_size)

    df_np, val_data = get_test_set_np(df_np, in_win_size=in_win_size)

    if array_type: df_np = df_np[0]  # Maybe this looks clumsy but it works

    if mode == 0:
        x_test, y_test = partition_dataset(test_data, in_win_size, out_win_size)
        x_val, y_val = partition_dataset(val_data, in_win_size, out_win_size)
    elif mode == 1:
        x_test, y_test = multi_partition_dataset(test_data, in_win_size, out_win_size)
        x_val, y_val = multi_partition_dataset(val_data, in_win_size, out_win_size)

    return df_np, x_test, y_test, x_val, y_val