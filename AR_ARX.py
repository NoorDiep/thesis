from Drivers import get_data, forecast_accuracy, optimal_lag, getDataTablesFigures
from statsmodels.tsa.arima.model import ARIMA
import numpy as np
import pandas as pd

########################################################################################################################
# LOAD DATA
########################################################################################################################
# Import data into 60% training, 20% validation and 20% test sets
data, df, df_diff, date_train, date_test, date_train_diff, date_test_diff, depo, swap_dates, spread, spread_diff = get_data(lag=5)

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


def getForecast(x, y, n_train, n_tv, n_test, h):


    w = n_train

    f_i = []
    f_iX = []
    p_ARi = []
    p_ARXi = []
    q_ARXi = []
    resultsAR = []
    resultsARX = []



    for i in range(0, y.shape[1]):

        p_AR1 = optimal_lag(x_train=0, x_tv=0, y_train=y[:n_train,i], y_tv=y[:n_tv,i], maxlags_p=30, maxlags_q=15, indicator=0)
        p_ARX1, q_ARX1 = optimal_lag(x_train=x[:n_train], x_tv=x[:n_tv], y_train=y[:n_train,i], y_tv=y[:n_tv,i], maxlags_p=30, maxlags_q=15, indicator=1)
        p_ARi.append(p_AR1)
        p_ARXi.append(p_ARX1)
        q_ARXi.append(q_ARX1)

        f_k = []
        f_kX = []
        #xtrain = pd.DataFrame(X_tv_diff).shift(q_ARX).dropna()
        AR = ARIMA(y[:n_tv,i], order=(p_AR1, 0, 0)).fit()
        ARX = ARIMA(y[:n_tv,i], exog=x[:n_tv], order=(p_ARX1, 0, 0)).fit()

        # Perform rolling window forecasts
        for k in range(n_test - h + 1):
            print(i)
            print(k)
            # # Forecast out-of-sample
            preds_AR = AR.predict(start=len(Y_tv[i])+k, end=len(Y_tv[i])+k + h-1, dynamic=True) # Dynamic equal to False means direct forecasts
            preds_AR = pd.DataFrame(preds_AR)
            preds_AR.index = date_test.iloc[k:k+h]

            test = y[len(Y_tv[i])+k:len(Y_tv[i])+k + h]
            test = pd.DataFrame(test)
            test.index = date_test.iloc[k:k + h]

            preds_AR = preds_AR.T.reset_index(drop=True).T
            acc_AR = forecast_accuracy(preds_AR[0], test[0], df_indicator=0)
            f_k.append(acc_AR)

            # Forecast out-of-sample
            preds_ARX = ARX.predict(start=len(Y_tv[i]) + k, end=len(Y_tv[i]) + k + h-1 , exog=x[k-w:k+h-w+q_ARX1+k-1],
                                    dynamic=True)  # Dynamic equal to False means direct forecasts
            preds_ARX = pd.DataFrame(preds_ARX)
            preds_ARX.index = date_test.iloc[k:k + h]
            preds_ARX = preds_ARX.T.reset_index(drop=True).T
            test = y[len(Y_tv[i]) + k:len(Y_tv[i]) + k + h]
            test = pd.DataFrame(test)
            test.index = date_test.iloc[k:k + h]

            # Get forecast accuracy
            acc_ARX = forecast_accuracy(preds_ARX[0], test[0], df_indicator=0)
            f_kX.append(acc_ARX)

        f_i = pd.DataFrame(np.concatenate(f_k), columns=['MEA', 'MSE', 'RMSE'])
        means = np.mean(f_i)
        means['lag p'] = p_AR1
        f_iX = pd.DataFrame(np.concatenate(f_kX), columns=['MEA', 'MSE', 'RMSE'])
        meansX = np.mean(f_iX)
        meansX['lag p'] = p_ARX1
        meansX['lag q'] = q_ARX1
        print(means)
        print(meansX)
        resultsAR.append(means)
        resultsARX.append(meansX)
        print('optimal p AR:', p_AR1)
        print('optimal p, q:', p_ARX1, q_ARX1)

    resultsAR1 = pd.DataFrame(resultsAR)
    resultsARX1 = pd.DataFrame(resultsARX)
    return resultsAR1, resultsARX1


########################################################################################################################
# RUN CODE
########################################################################################################################

x = np.vstack((X_tv, X_test))
y = np.vstack((Y_tv, Y_test))
#pdAR1, pdARX1 = getForecast(x, y, n_train=len(Y_train), n_tv=len(Y_tv), n_test=len(Y_test), h=1)
#pdAR5, pdARX5 = getForecast(x, y, n_train=len(Y_train), n_tv=len(Y_tv), n_test=len(Y_test), h=5)
pdAR10, pdARX10 = getForecast(x, y, n_train=len(Y_train), n_tv=len(Y_tv), n_test=len(Y_test), h=10)
#pdAR30, pdARX30 = getForecast(x, y, n_train=len(Y_train), n_tv=len(Y_tv), n_test=len(Y_test), h=30)


y_diff = np.vstack((Y_tv_diff, Y_test_diff))
#pdAR1_diff, pdARX1_diff = getForecast(x, y_diff, n_train=len(Y_train), n_tv=len(Y_tv), n_test=len(Y_test), h=1)
#pdAR5_diff, pdARX5_diff = getForecast(x, y_diff, n_train=len(Y_train), n_tv=len(Y_tv), n_test=len(Y_test), h=5)
pdAR10_diff, pdARX10_diff = getForecast(x, y_diff, n_train=len(Y_train), n_tv=len(Y_tv), n_test=len(Y_test), h=10)
#pdAR30_diff, pdARX30_diff = getForecast(x, y_diff, n_train=len(Y_train), n_tv=len(Y_tv), n_test=len(Y_test), h=30)

t = 1