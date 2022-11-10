import matplotlib.pyplot as plt
import seaborn as sns
import statsmodels.api as sm
from mlxtend.frequent_patterns import apriori, association_rules
import pandas as pd
from scipy.stats import f_oneway
from preprocessing import uniform_data_types


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


def normality_check(df):
    # Check if normal distribution. If not, try sqrt, cube root, log2, or log10
    fig = sm.qqplot(df["Aantal uren"], line='45')
    plt.show(fig)


def distribution_check(df):
    # Get the data distributions
    fig = sns.displot(df, x="Aantal uren", hue="Inlener")
    plt.show(fig)


def anova(df):
    # Calculate the statistical significance between the selected variables
    filtered_columns = []
    for item in df.columns:
        # Due to some type errors between the enddate date format, as well as some binary values, the final list is not
        # as representative as it should/could be
        try:
            groupped = df.groupby(item)['timecardline_amount'].apply(list)
            results = f_oneway(*groupped)
            if results[1] < 0.05:
                filtered_columns.append(item)
                print("******************Relation between " + item + ":")
                print(results)
                print("\n")
        except:
            continue
    print(filtered_columns)
    '''
    df1 = df.rename(columns={"Aantal uren":"Uren"})
    model = ols(formula="Uren ~ C(Functie)", data=df1)
    aov_table = sm.stats.anova_lm(model.fit(), typ=2)
    # Added some extra stats that may not be necessary, focus mostly on the F score and PR
    aov_table['mean_sq'] = aov_table[:]['sum_sq'] / aov_table[:]['df']

    aov_table['eta_sq'] = aov_table[:-1]['sum_sq'] / sum(aov_table['sum_sq'])

    aov_table['omega_sq'] = (aov_table[:-1]['sum_sq'] - (aov_table[:-1]['df'] * aov_table['mean_sq'][-1])) / (
                sum(aov_table['sum_sq']) + aov_table['mean_sq'][-1])

    cols = ['sum_sq', 'df', 'mean_sq', 'F', 'PR(>F)', 'eta_sq', 'omega_sq']
    aov = aov_table[cols]
    print(aov)
    '''
    return filtered_columns


def make_boxplot(df):
    # sns.boxplot(x=df['totalhours'])
    # plt.show()
    sns.boxplot(x=df['timecardline_amount'])
    plt.show()
    try:
        sns.boxplot(x=df['staffingcustomer_companyname'])
        plt.show()
    except:
        print("No company name to show.")
    # sns.boxplot(data=df, x="period", y="totalhours")
    # plt.show()


def make_lineplot(df):
    sns.lineplot(data=df, x=df.index, y="timecardline_amount", hue="weekofyear",
                 style="assignment_flexworkerid", markers=True)
    plt.show()
    try:
        sns.lineplot(data=df, y="timecardline_amount", hue="assignment_flexworkerid", markers=True)
        plt.show()
    except:
        print("One worker is displayed.")
    try:
        sns.lineplot(data=df, y="timecardline_amount" , hue="staffingcustomer_companyname", markers=True)
        plt.show()
    except:
        print("One company is displayed.")


def general_statistics(df):

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

    # Get the median of the numeric values
    print("The median is:")
    print(df.median())
    print("---------------------------------------")
    print("---------------------------------------")

    # A general description of the numeric variables in the dataset
    print(df.describe())
    print("---------------------------------------")
    print("---------------------------------------")
    print(f"Assignment start dates: {df.iloc[:,0].unique()}")
    print(f"Assignment end date: {df.iloc[:,1].unique()}")
    print(f"Assignment component start date: {df.iloc[:,2].unique()}")
    print(f"Assignment component end date: {df.iloc[:,3].unique()}")
    print(f"Worker: {df.iloc[0,4]}")
    print(f"Company: {df.iloc[0,5]}")
    print(f"Number of inputs: {df.shape}")

    # make_boxplot(df)

    make_lineplot(df)

    # correlated_list = anova(df)
    # if ["timecardline_linedate"] not in correlated_list:
    #     correlated_list += ["timecardline_linedate"]
    # return df[correlated_list]
    # Get the number of categories available
    # print(pd.value_counts(df["Inlener"], normalize=True))