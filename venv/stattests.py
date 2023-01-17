import matplotlib.pyplot as plt
import seaborn as sns
import statsmodels.api as sm
import pandas as pd
from scipy.stats import f_oneway
from preprocessing import create_subsets, fill_gaps
import numpy as np


def check_correlation(df):
    # Check correlation between database variables######################################################################
    X = pd.get_dummies(data=df[['Datum', 'Urensoort', 'Plaatsing', 'Functie', 'Inlener', 'Flexkracht']]
                        , drop_first=True)
    df = pd.concat([df["Aantal uren"], X], axis=1)
    print(df.head())
    # x = df.values
    # corr_mat = np.corrcoef(x.T)
    corr_mat = df.corr()
    # sns.heatmap(corr_mat, annot=True) # This crashes
    # plt.show()
    # Find correlation between total hours worked and independent variables, anything above 0 has some sort of
    # correlation, otherwise it doesn't. The threshold can be adjusted.
    corr_list = corr_mat.unstack().sort_values(kind="quicksort")
    print(corr_list["Aantal uren"][corr_list["Aantal uren"]>0.05])


def group_means(df):
    # Group statistics by (a) certain categorical value(s) and get the corresponding means(or other stats if interested)
    print(df.groupby(["Functie", "Inlener", "Flexkracht"]).mean())
    print(df.groupby(["Functie", "Inlener"])["Aantal uren"].sum().unstack().reset_index().fillna(0).set_index("Functie"))
    # Get the data distributions
    fig = sns.displot(df, x="timecardline_amount", hue="staffingcustomer_staffingcustomerid")
    plt.show(fig)


def anova(df, histogram_anova):
    # print("---------------------------------------")
    # print("---------------------------------------")
    # print("                 ANOVA")
    # Calculate the statistical significance between the selected variables
    filtered_columns = []
    for item in df.columns:
        # Due to some type errors between the enddate date format, as well as some binary values, the final list is not
        # as representative as it should/could be
        try:
            groupped = df.groupby(item)['timecardline_amount'].apply(list)
            results = f_oneway(*groupped)
            # print(f"For {item} the results are: {results}")
            if results[1] < 0.05:
                histogram_anova.append(item)
                filtered_columns.append(item)
                # print("******************Relation between " + item + ":")
                # print(results)
                # print("\n")
        except Exception as e:
            print(e)
            continue
    # print(filtered_columns)
    return filtered_columns


def make_boxplot(df):
    for feature in df.columns:
        try:
            sns.boxplot(x=df[feature])
            plt.show()
        except Exception as e:
            print(e)


def make_lineplot(df):
    sns.lineplot(data=df, x=df.index, y="timecardline_amount", hue="quarter",
                 style="assignment_flexworkerid", markers=True)
    plt.show()
    # try:
    #     sns.lineplot(data=df, x=df.index, y="timecardline_amount", hue="assignment_startdate",
    #                  style="assignment_flexworkerid", markers=True)
    #     plt.show()
    # except:
    #     print("One worker is displayed.")
    # try:
    #     sns.lineplot(data=df, x=df.index, y="timecardline_amount", hue="assignment_enddate",
    #                  style="assignment_flexworkerid", markers=True)
    #     plt.show()
    # except:
    #     print("One worker is displayed.")


def general_statistics(df, cnt):

    # Get an overview of the imported dataframe

    # df = df.resample('1D').mean().fillna(0)
    cols = df.columns
    print("Dropping the final one")
    print(df.drop(columns=cols[-1]).head())
    df1 = df.drop(columns=cols[-1]).resample('1D').ffill()
    print("The last input")
    print(df.drop(columns=cols[:-1]).head())
    df2 = df.drop(columns=cols[:-1]).resample('1D').mean().fillna(0)
    print("The sizes of the two dataframes to be concatenated:")
    print(df1.shape)
    print(df2.shape)
    df = pd.concat([df1, df2], axis=1)
    print(df.iloc[:,-1].head())

    print("---------------------------------------")
    print("---------------------------------------")
    print(df.head())
    print("---------------------------------------")
    print("---------------------------------------")
    print(df.info())
    print("---------------------------------------")
    print("---------------------------------------")

    # Check what the dtypes of the variables stored in the database are
    print("The dtypes are:")
    print(df.dtypes)
    print("---------------------------------------")
    print("---------------------------------------")

    # A general description of the numeric variables in the dataset
    print(df.describe())
    print("---------------------------------------")
    print("---------------------------------------")

    print("Quantile analysis")
    amount_data=df['timecardline_amount']
    print(f"25%:\t{amount_data.quantile(0.25)}\n"
          f"50%:\t{amount_data.quantile(0.5)}\n"
          f"75%:\t{amount_data.quantile(0.75)}\n"
          f"99.9%:\t{amount_data.quantile(0.999)}\n"
          f"std:\t{amount_data.std()}\n"
          f"mean:\t{amount_data.mean()}")

    print("---------------------------------------")
    print("---------------------------------------")

    print(f"Number of inputs: {df.shape}")

    # make_boxplot(df)
    print(f"\n----------------------------------------------\n"
          f"            Input number {cnt}:"
          f"\n----------------------------------------------")

    make_lineplot(df)

    # correlated_list = anova(df)
    # if ["timecardline_linedate"] not in correlated_list:
    #     correlated_list += ["timecardline_linedate"]
    # return df[correlated_list]
    # Get the number of categories available
    # print(pd.value_counts(df["Inlener"], normalize=True))


def stat_mode_initialiser(df, i=0):
    FEATURES = ['payrollcomponent', 'flexworkerids', 'quarter', 'weekofyear', 'assignmentcomponent_startdate',
                'assignmentcomponent_enddate', 'assignment_flexworkerid',
                'staffingcustomer_staffingcustomerid', 'timecard_totalhours', 'timecardline_amount']
    print("Using statistic analysis mode")
    df_list = create_subsets(df, FEATURES)
    histogram_anova=[]
    # df_list = fill_gaps(df_list, dt_inputs=True)
    for cnt, df in enumerate(df_list):
        if cnt < i: continue
        df = fill_gaps(df, dt_inputs=True)
        print(f"\n**********************************************\n"
              f"            Input number {cnt}:"
              f"\n**********************************************")
        print(f"Worker: {df[FEATURES[0]][0]}")
        print(f"Company: {df[FEATURES[1]][0]}")
        # make_boxplot(df)
        # anova(df, histogram_anova)
        general_statistics(df, cnt)
        
    histogram_anova = np.array(histogram_anova)
    unique_items = np.unique(histogram_anova)
    print(unique_items)
    print(len(unique_items))
    exit()