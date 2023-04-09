from statsmodels.tsa.ar_model import AutoReg

from Drivers import get_data, forecast_accuracy, optimal_lag, getDataTablesFigures
from statsmodels.tsa.arima.model import ARIMA
import numpy as np
import pandas as pd

########################################################################################################################
# LOAD DATA
########################################################################################################################
# Import data into 60% training, 20% validation and 20% test sets
lag=5
data, df, df_diff, date_train, date_test, date_train_diff, date_test_diff, depo, swap_dates, spread, spread_diff = get_data(lag=lag)

X_train = df[0]
X_val = df[1]
X_test = df[2]
Y_train = df[3]
Y_val = df[4]
Y_test = df[5]

X_train_diff = df_diff[0]
X_val_diff = df_diff[1]
X_test_diff = df_diff[2]
Y_train_diff = df_diff[3]
Y_val_diff = df_diff[4]
Y_test_diff = df_diff[5]

# Combine the training and validation sets
Y_tv = np.vstack((Y_train, Y_val))
Y_tv_diff = np.vstack((Y_train_diff, Y_val_diff))
Y_diff = np.vstack((Y_tv_diff, Y_test_diff))

X_tv = np.vstack((X_train, X_val))
X_tv_diff = np.vstack((X_train_diff, X_val_diff))
X_diff = np.vstack((X_tv_diff, X_test_diff))

# Return tables and figures for Data section
getDataTablesFigures(data[0], data[1], pd.DataFrame(Y_diff), pd.DataFrame(X_diff), depo, swap_dates)

#getDataTablesFigures(data[2], data[3], pd.DataFrame(X_diff), depo, swap_dates) # input: df_swap, df_drivers, swap_diff# input: df_swap, df_drivers, swap_diff

def forecast_acc(forecast, actual):
    mae = np.mean(np.abs(forecast - actual))  # MAE
    mse = np.mean((forecast - actual) ** 2)  # MSE
    rmse = np.mean((forecast - actual) ** 2) ** .5  # RMSE
    pd.DataFrame([[mae, mse, rmse]], columns=['mae', 'mse', 'rmse'])

###### Results with differences ######


def getForecast(x, y, n_train, n_tv, n_test, h, diff_lag):

    # Intitialize variables
    w = n_tv
    f_i = []
    f_iX = []
    p_i = []
    p_iX = []
    q_iX = []
    y_test = pd.DataFrame(y[n_tv:])
    error = []
    errorX = []

    for i in range(0, y.shape[1]):

        p_AR1 = optimal_lag(x_train=0, x_tv=0, y_train=y[:n_train,i], y_tv=y[:n_tv,i], maxlags_p=10, maxlags_q=10, indicator=0)
        p_ARX1, q_ARX1 = optimal_lag(x_train=x[:n_train], x_tv=x[:n_tv], y_train=y[:n_train,i], y_tv=y[:n_tv,i], maxlags_p=10, maxlags_q=10, indicator=1)
        p_i.append(p_AR1)
        p_iX.append(p_ARX1)
        q_iX.append(q_ARX1)

        f_k = []
        f_kX = []

        # Perform rolling window forecasts
        for k in range(n_test - h + 1):
            print(i)
            print(k)

            # Fit both models
            AR = AutoReg(endog=y[:k+h+w-1, i], lags=p_AR1).fit()
            ARX = AutoReg(endog=y[:k+h+w-1, i], exog=x[:k+h+w-1], lags=p_ARX1).fit()

            # # Forecast out-of-sample
            preds_AR = AR.predict(start=w+k, end=w+k+h-1, dynamic=True)
            y_hatAR = preds_AR[h-1]
            f_k.append(y_hatAR)

            # Forecast out-of-sample
            preds_ARX = ARX.predict(start=w+k, end=w+k+h-1, exog=x[:w+k+h-1], exog_oos=x[w:k+h+w+k],
                                    dynamic=True)
            y_hatARX = preds_ARX[h-1]
            f_kX.append(y_hatARX)
        error.append(f_k -y_test.iloc[h-1:,i])
        errorX.append(f_kX -y_test.iloc[h-1:,i])
        acc_AR = forecast_accuracy(f_k, y_test.iloc[h-1:,i], df_indicator=0)
        acc_ARX = forecast_accuracy(f_kX, y_test.iloc[h-1:,i], df_indicator=0)

        f_i.append(acc_AR)
        f_iX.append(acc_ARX)

    t=1
    resultsAR = pd.DataFrame(np.concatenate(f_i), columns=['MEA', 'RMSE'])
    resultsAR['lag p'] = pd.DataFrame(p_i)

    resultsARX = pd.DataFrame(np.concatenate(f_iX), columns=['MEA', 'RMSE'])
    resultsARX['lag p'] = pd.DataFrame(p_iX)
    resultsARX['lag q'] = pd.DataFrame(q_iX)

    return resultsAR, resultsARX, error, errorX


########################################################################################################################
# RUN CODE
########################################################################################################################

x = np.vstack((X_tv, X_test))
y = np.vstack((Y_tv, Y_test))
pdAR1, pdARX1, error1, errorX1 = getForecast(x, y, n_train=len(Y_train), n_tv=len(Y_tv), n_test=len(Y_test), h=1, diff_lag=lag)
pdAR5, pdARX5, error5, errorX5 = getForecast(x, y, n_train=len(Y_train), n_tv=len(Y_tv), n_test=len(Y_test), h=5, diff_lag=lag)
pdAR10, pdARX10, error10, errorX10 = getForecast(x, y, n_train=len(Y_train), n_tv=len(Y_tv), n_test=len(Y_test), h=10, diff_lag=lag)
pdAR20, pdARX20, error20, errorX20 = getForecast(x, y, n_train=len(Y_train), n_tv=len(Y_tv), n_test=len(Y_test), h=20, diff_lag=lag)


y_diff = np.vstack((Y_tv_diff, Y_test_diff))
pdAR1_diff, pdARX1_diff, error1_diff, errorX1_diff = getForecast(x, y_diff, n_train=len(Y_train_diff), n_tv=len(Y_tv_diff), n_test=len(Y_test_diff), h=1, diff_lag=lag)
pdAR5_diff, pdARX5_diff, error5_diff, errorX5_diff = getForecast(x, y_diff, n_train=len(Y_train_diff), n_tv=len(Y_tv_diff), n_test=len(Y_test_diff), h=5, diff_lag=lag)
pdAR10_diff, pdARX10_diff, error10_diff, errorX10_diff = getForecast(x, y_diff, n_train=len(Y_train_diff), n_tv=len(Y_tv_diff), n_test=len(Y_test_diff), h=10, diff_lag=lag)
pdAR20_diff, pdARX20_diff, error20_diff, errorX20_diff = getForecast(x, y_diff, n_train=len(Y_train_diff), n_tv=len(Y_tv_diff), n_test=len(Y_test_diff), h=20, diff_lag=lag)

t = 1



t=1