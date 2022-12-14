import os

import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
import sys


def bar_plotter(df):
    print(df)
    plt.style.use('seaborn')
    plt.rcParams['axes.grid'] = True
    train_predictions = df.iloc[:, -6]
    train_actuals = df.iloc[:, -5][~np.isnan(df.iloc[:, -5])]
    val_predictions = df.iloc[:, -4][~np.isnan(df.iloc[:, -4])]
    val_actuals = df.iloc[:, -3][~np.isnan(df.iloc[:, -3])]
    test_predictions = df.iloc[:, -2][~np.isnan(df.iloc[:, -2])]
    test_actuals = df.iloc[:, -1][~np.isnan(df.iloc[:, -1])]
    days = ['Sunday', 'Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday']
    test_predictions = test_predictions.values.reshape(-1,7)
    test_actuals = test_actuals.values.reshape(-1,7)

    barWidth = 0.25
    fig = plt.subplots(figsize=(12, 8))

    # Set position of bar on X axis
    br1 = np.arange(len(test_actuals[0]))
    br2 = [x + barWidth for x in br1]

    # Make the plot
    plt.bar(br1, test_actuals[0], color='r', width=barWidth,
            edgecolor='grey', label='actuals')
    plt.bar(br2, test_predictions[0].round(1), color='g', width=barWidth,
            edgecolor='grey', label='predictions')

    # Adding Xticks
    plt.xlabel('Branch', fontweight='bold', fontsize=15)
    plt.ylabel('Students passed', fontweight='bold', fontsize=15)
    plt.xticks([r + barWidth for r in range(len(test_actuals[0]))], days)

    plt.legend()
    plt.show()


def plotter(df):
    train_predictions = df.iloc[:, -6]
    train_actuals = df.iloc[:, -5][~np.isnan(df.iloc[:, -5])]
    val_predictions = df.iloc[:, -4][~np.isnan(df.iloc[:, -4])]
    val_actuals = df.iloc[:, -3][~np.isnan(df.iloc[:, -3])]
    test_predictions = df.iloc[:, -2][~np.isnan(df.iloc[:, -2])]
    test_actuals = df.iloc[:, -1][~np.isnan(df.iloc[:, -1])]
    plt.style.use('seaborn')
    plt.rcParams['axes.grid'] = True
    figure, axis = plt.subplots(5, sharex=True, sharey=True)
    axis[0].plot(np.arange(len(train_predictions)), train_predictions.round(), label="Predictions")
    axis[0].plot(np.arange(len(train_actuals)), train_actuals, label="Actuals")
    axis[0].title.set_text('Train Run')
    axis[1].plot(np.arange(len(val_predictions)), val_predictions.round(), label="Predictions")
    axis[1].plot(np.arange(len(val_actuals)), val_actuals, label="Actuals")
    axis[1].title.set_text('Value Run')
    axis[2].plot(np.arange(len(test_predictions)), test_predictions.round(), label="Predictions")
    axis[2].plot(np.arange(len(test_actuals)), test_actuals, label="Actuals")
    axis[2].title.set_text('Test Run')
    axis[3].fill_between(np.arange(len(train_predictions)), train_predictions.round(), train_actuals)
    axis[3].title.set_text('Painted Train Difference')
    axis[4].fill_between(np.arange(len(test_predictions)), test_predictions.round(), test_actuals)
    axis[4].title.set_text('Painted Test Difference')
    lines = []
    labels = []

    for ax in figure.axes:
        Line, Label = ax.get_legend_handles_labels()
        # print(Label)
        lines.extend(Line)
        labels.extend(Label)

    figure.legend(lines, labels, loc='upper right')
    plt.show()


if __name__ == '__main__':
    iteration = 11
    version = 'd'
    if len(sys.argv) > 1:
        iteration = int(sys.argv[1])
    # df = pd.read_excel(f"../../data/results/RNN/multivariate_AdvGRU_timecardline_amount/{iteration}{version}.xlsx")
    df = pd.read_excel(f"../../data/results/RNN/multivariate_AdvLSTM_timecardline_amount/{iteration}{version}.xlsx")

    print(df.head())

    test_loss = df["mse test"][0]
    val_loss = df["mse val"][0]
    train_loss = df["mse train"][0]
    time = df["train time"][0]
    layer_size = df["layer size"][0]
    layer_number = df["layer number"][0]
    lr = df["learning rate"][0]
    epochs = df["epochs"][0]
    in_win_size = df["input shape"][0]
    print(f"---------------------------------------------------------------------------------------------------\n"
          f"|*train loss: {round(train_loss, 3)}\t|*layer size: {layer_size}\t|*input window size: {in_win_size}\t|\n"
          f"|*value loss: {round(val_loss, 3)}\t|*layer number: {layer_number}\t|\n"
          f"|*test loss: {round(test_loss, 3)}\t|*epochs: {epochs}\t\t|\n"
          f"|*time: {round(time/60, 2)} mins\t|*learning rate: {lr}\t|\n"
          f"---------------------------------------------------------------------------------------------------\n")

    plotter(df)
    # bar_plotter(df)
    # sns.lineplot(data=df, x=df.index, y="timecardline_amount", hue="staffingcustomer_companyname",
    #              style="assignment_flexworkerid", markers=True)