import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import math
import xgboost as xgb
from sklearn.metrics import mean_squared_error


def start_boosted_tree(df, in_win_size, target):
    # df_np = df.to_numpy()
    train_data_length = math.ceil(df.shape[0] * 0.85)
    train_data = df[:train_data_length]
    test_data = df[train_data_length - in_win_size:]

    x_train = train_data.loc[:, df.columns!=target]
    y_train = train_data.loc[:, target]

    x_test = test_data.loc[:, df.columns!=target]
    y_test = test_data.loc[:, target]

    reg = xgb.XGBRegressor(base_score=0.5, booster='gbtree',
                           n_estimators=1000,
                           early_stopping_rounds=50,
                           objective='reg:linear',
                           max_depth=4,
                           learning_rate=0.001)

    reg.fit(x_train, y_train, eval_set=[(x_train, y_train),
                            (x_test, y_test)], verbose=100)

    fi = pd.DataFrame(data=reg.feature_importances_,
                      index=reg.feature_names_in_,
                      columns=['importance'])
    fi.sort_values('importance').plot(kind='barh', title='Feature Importance')
    plt.show()

    x = len(df)
    num_index = range(0, x, 1)
    print(num_index)
    df = df.reset_index()

    prediction = reg.predict(x_test)
    df_pred = pd.DataFrame(prediction, columns=['prediction'])
    # test_data['prediction'] = prediction
    # df = df.merge(df_pred[['prediction']], how='left', left_index=True, right_index=True)
    ax = df[[target]].plot(figsize=(10, 5))
    df_pred['prediction'].plot(ax=ax, style='-')
    plt.legend(['Truth Data', 'Predictions'])
    ax.set_title('Raw Dat and Prediction')
    plt.show()

    score = np.sqrt(mean_squared_error(test_data[target], df_pred['prediction']))
    print(f'RMSE Score on Test set: {score:0.2f}')

    test_data['error'] = np.abs(test_data[target] - df_pred['prediction'])
    test_data['date'] = test_data.index.date
    test_data.groupby(['date'])['error'].mean().sort_values(ascending=False).head(10)