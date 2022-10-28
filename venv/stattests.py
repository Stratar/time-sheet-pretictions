import matplotlib.pyplot as plt
import seaborn as sns
import statsmodels.api as sm
from mlxtend.frequent_patterns import apriori, association_rules
import pandas as pd
from scipy.stats import f_oneway


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


def apriori(df):
    # Apriori attempt##################################################################################################
    df_grouped = df.groupby(["Flexkracht", "Functie"])["Aantal uren"].sum().unstack().reset_index().fillna(0).set_index(
        "Flexkracht")
    print(df_grouped)

    def hot_encode(x):
        if (x <= 0):
            return 0
        if (x >= 1):
            return 1

    df_enc = df_grouped.applymap(hot_encode)
    print(df_enc)

    # Building the model
    frq_items = apriori(df_enc, min_support=0.05, use_colnames=True, verbose=1)
    print("apriori: ")
    print(frq_items)
    rules = association_rules(frq_items, metric="lift", min_threshold=0.1)
    rules = rules.sort_values(["confidence", "lift"], ascending=[False, False])
    print("rules: ")
    print(rules)


def anova(df):
    # Calculate the statistical significance between the selected variables
    filtered_columns = []
    for item in df.columns:
        # Due to some type errors between the enddate date format, as well as some binary values, the final list is not
        # as representative as it should/could be
        try:
            groupped = df.groupby(item)['amount'].apply(list)
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


def log_reg(df):
    fig = sns.regplot(x=pd.get_dummies(["functionname"], drop_first=True), y=df["totalhours"], data=df, logistic=True)
    plt.show(fig)


def make_boxplot(df):
    sns.boxplot(x=df['totalhours'])
    plt.show()
    sns.boxplot(x=df['amount'])
    plt.show()
    sns.boxplot(data=df, x="period", y="totalhours")
    plt.show()
    sns.boxplot(data=df, x="companyname", y="totalhours")
    plt.show()
    sns.boxplot(data=df, x="functionname", y="totalhours")
    plt.show()


def make_lineplot(df):
    sns.lineplot(x=df["period"].sort_values(), y=df["totalhours"])
    plt.show()


def general_statistics(df):

    # Get an overview of the imported dataframe
    print(df.head())
    print(df.info())

    # Check what the dtypes of the variables stored in the database are
    print(df.dtypes)

    # Get the median of the numeric values
    print(df.median())

    # A general description of the numeric variables in the dataset
    print(df.describe())

    make_boxplot(df)

    make_lineplot(df)

    exit(1)

    correlated_list = anova(df)
    if ["linedate"] not in correlated_list:
        correlated_list += ["linedate"]
    return df[correlated_list]
    # Get the number of categories available
    # print(pd.value_counts(df["Inlener"], normalize=True))