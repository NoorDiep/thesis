import itertools
from Drivers import get_data, forecast_accuracy, optimal_lag, getDataTablesFigures
import numpy as np
import pandas as pd
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM


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
#getDataTablesFigures(data[0], data[1], pd.DataFrame(Y_diff), pd.DataFrame(X_diff), depo, swap_dates)

#getDataTablesFigures(data[2], data[3], pd.DataFrame(X_diff), depo, swap_dates) # input: df_swap, df_drivers, swap_diff# input: df_swap, df_drivers, swap_diff

def forecast_acc(forecast, actual):
    mae = np.mean(np.abs(forecast - actual))  # MAE
    mse = np.mean((forecast - actual) ** 2)  # MSE
    rmse = np.mean((forecast - actual) ** 2) ** .5  # RMSE
    pd.DataFrame([[mae, mse, rmse]], columns=['mae', 'mse', 'rmse'])

########################################################################################################################
# RUN CODE
########################################################################################################################

def forecastNNARX(x_train, x_tv, x_test, y_train, y_tv, y_test, y_swap, n_train, n_tv, n_test, units, h):
    results = []


    w = n_tv
    x = np.vstack((x_tv, x_test))
    y = np.vstack((y_tv, y_test))
    f_i = []
    f_iX = []
    error = []
    errorX = []

    # reshape input to be 3D [samples, timesteps, features]; each line turns into an 'sub-array'
    df_x = x.reshape((x.shape[0], 1, x.shape[1]))
    model = Sequential()
    model.add(LSTM(units, input_shape=(df_x.shape[1], df_x.shape[2])))
    model.add(Dense(1))
    model.compile(loss='mae', optimizer='adam')

    epochs_list = [10, 15, 20, 25, 30, 35]
    batch_size_list = [2, 4, 6, 8]
    params = {}
    for epochs, batch_size in itertools.product(epochs_list, batch_size_list):
        fitted = model.fit(df_x[:n_train], y_train, epochs=epochs, batch_size=batch_size, shuffle=False,
                                 validation_data=(df_x[n_train:n_tv], y_tv[n_train:]))
        fitted_loss = fitted.history['val_loss'][-1]
        params[(epochs, batch_size)] = fitted_loss

    opt_params = min(params, key=params.get)
    #opt_params = [5,32]


    print('Opt params:', opt_params)

    for i in range(0, y_train.shape[1]):
        f_k = []
        for k in range(n_test - h + 1):
            model.fit(df_x[:k+w], y[:k+w], epochs=opt_params[0], batch_size=opt_params[1], shuffle=False)
            print(i,k)
            # # Forecast out-of-sample
            test_Xk = df_x[k+w:k + h +w + k]
            y_pred = model.predict(test_Xk)
            # y_pred = y_pred.reshape(1,-1)
            y_hat = y_pred[h - 1]
            f_k.append(y_hat[0])

        f_i.append(f_k)
    f_i = pd.DataFrame(f_i).T
    f_i.columns = y_test.columns
    y_test = y_test.iloc[h-1:,].reset_index(drop=True, inplace=True)
    error = f_i - y_test
    results = forecast_accuracy(f_i, y_test, df_indicator=1)
    return results, opt_params, error

y_swap = np.vstack((Y_tv_diff, Y_test_diff))
results10_n3, opt_params10_3, error10_3 = forecastNNARX(X_train_diff, X_tv_diff, X_test_diff, Y_train_diff, Y_tv_diff, Y_test_diff, y_swap,
                        n_train=len(Y_train_diff), n_tv=len(Y_tv_diff), n_test=len(Y_test_diff), units=3,  h=10)
# results10_n20, opt_params10_20, error10_20 = forecastNNARX(X_train_diff, X_tv_diff, X_test_diff, Y_train_diff, Y_tv_diff, Y_test_diff, y_swap,
#                         n_train=len(Y_train_diff), n_tv=len(Y_tv_diff), n_test=len(Y_test_diff), units=20,  h=10)
# results10_n50, opt_params10_50, error10_50 = forecastNNARX(X_train_diff, X_tv_diff, X_test_diff, Y_train_diff, Y_tv_diff, Y_test_diff, y_swap,
#                         n_train=len(Y_train_diff), n_tv=len(Y_tv_diff), n_test=len(Y_test_diff), units=50,  h=10)

results1_n3, opt_params1_3, error1_3 = forecastNNARX(X_train_diff, X_tv_diff, X_test_diff, Y_train_diff, Y_tv_diff, Y_test_diff, y_swap,
                        n_train=len(Y_train_diff), n_tv=len(Y_tv_diff), n_test=len(Y_test_diff), units=3,  h=1)
# results1_n20, opt_params1_20, error1_20 = forecastNNARX(X_train_diff, X_tv_diff, X_test_diff, Y_train_diff, Y_tv_diff, Y_test_diff, y_swap,
#                         n_train=len(Y_train_diff), n_tv=len(Y_tv_diff), n_test=len(Y_test_diff), units=20,  h=1)
# results1_n50, opt_params1_50, error1_50 = forecastNNARX(X_train_diff, X_tv_diff, X_test_diff, Y_train_diff, Y_tv_diff, Y_test_diff, y_swap,
#                         n_train=len(Y_train_diff), n_tv=len(Y_tv_diff), n_test=len(Y_test_diff), units=50,  h=1)

results5_n3, opt_params5_3, error5_3 = forecastNNARX(X_train_diff, X_tv_diff, X_test_diff, Y_train_diff, Y_tv_diff, Y_test_diff, y_swap,
                        n_train=len(Y_train_diff), n_tv=len(Y_tv_diff), n_test=len(Y_test_diff), units=3,  h=5)
# results5_n20, opt_params5_20, error5_20 = forecastNNARX(X_train_diff, X_tv_diff, X_test_diff, Y_train_diff, Y_tv_diff, Y_test_diff, y_swap,
#                         n_train=len(Y_train_diff), n_tv=len(Y_tv_diff), n_test=len(Y_test_diff), units=20,  h=5)
# results5_n50, opt_params5_50, error5_50 = forecastNNARX(X_train_diff, X_tv_diff, X_test_diff, Y_train_diff, Y_tv_diff, Y_test_diff, y_swap,
#                         n_train=len(Y_train_diff), n_tv=len(Y_tv_diff), n_test=len(Y_test_diff), units=50,  h=5)

results20_n3, opt_params20_3, error20_3 = forecastNNARX(X_train_diff, X_tv_diff, X_test_diff, Y_train_diff, Y_tv_diff, Y_test_diff, y_swap,
                        n_train=len(Y_train_diff), n_tv=len(Y_tv_diff), n_test=len(Y_test_diff), units=3,  h=20)
# results20_n20, opt_params20_20, error20_20 = forecastNNARX(X_train_diff, X_tv_diff, X_test_diff, Y_train_diff, Y_tv_diff, Y_test_diff, y_swap,
#                         n_train=len(Y_train_diff), n_tv=len(Y_tv_diff), n_test=len(Y_test_diff), units=20,  h=20)
# results20_n50, opt_params20_50, error20_50 = forecastNNARX(X_train_diff, X_tv_diff, X_test_diff, Y_train_diff, Y_tv_diff, Y_test_diff, y_swap,
#                         n_train=len(Y_train_diff), n_tv=len(Y_tv_diff), n_test=len(Y_test_diff), units=50,  h=20)


y_swap = np.vstack((Y_tv, Y_test))
results10L_n3, opt_params10L_3, error10_3l = forecastNNARX(X_train, X_tv, X_test, Y_train, Y_tv, Y_test, y_swap,
                        n_train=len(Y_train), n_tv=len(Y_tv), n_test=len(Y_test), units=3,  h=10)
# results10L_n20, opt_params10L_20, error10_20l = forecastNNARX(X_train, X_tv, X_test, Y_train, Y_tv_diff, Y_test, y_swap,
#                         n_train=len(Y_train), n_tv=len(Y_tv), n_test=len(Y_test), units=20,  h=10)
# results10L_n50, opt_params10L_50, error10_50l = forecastNNARX(X_train, X_tv, X_test, Y_train, Y_tv, Y_test, y_swap,
#                         n_train=len(Y_train), n_tv=len(Y_tv),n_test=len(Y_test), units=50,  h=10)

results1L_n3, opt_params1L_3, error1_3l = forecastNNARX(X_train, X_tv, X_test, Y_train, Y_tv, Y_test, y_swap,
                        n_train=len(Y_train), n_tv=len(Y_tv), n_test=len(Y_test), units=3,  h=1)
# results1L_n20, opt_params1L_20, error1_20l = forecastNNARX(X_train, X_tv, X_test, Y_train, Y_tv, Y_test, y_swap,
#                         n_train=len(Y_train), n_tv=len(Y_tv), n_test=len(Y_test), units=20,  h=1)
# results1L_n50, opt_params1L_50, error1_50l = forecastNNARX(X_train, X_tv, X_test, Y_train, Y_tv, Y_test, y_swap,
#                         n_train=len(Y_train), n_tv=len(Y_tv), n_test=len(Y_test), units=50,  h=1)

results5L_n3, opt_params5L_3, error5_3l = forecastNNARX(X_train, X_tv, X_test, Y_train, Y_tv, Y_test, y_swap,
                        n_train=len(Y_train), n_tv=len(Y_tv), n_test=len(Y_test), units=3,  h=5)
# results5L_n20, opt_params5L_20, error5_20l = forecastNNARX(X_train, X_tv, X_test, Y_train, Y_tv, Y_test, y_swap,
#                         n_train=len(Y_train), n_tv=len(Y_tv), n_test=len(Y_test), units=20,  h=5)
# results5L_n50, opt_params5L_50, error5_50l = forecastNNARX(X_train, X_tv, X_test, Y_train, Y_tv, Y_test, y_swap,
#                         n_train=len(Y_train), n_tv=len(Y_tv), n_test=len(Y_test), units=50,  h=5)

results20L_n3, opt_params20L_3, error20_3l = forecastNNARX(X_train, X_tv, X_test, Y_train, Y_tv, Y_test, y_swap,
                        n_train=len(Y_train), n_tv=len(Y_tv), n_test=len(Y_test), units=3,  h=20)
# results20L_n20, opt_params20L_20, error20_20l = forecastNNARX(X_train, X_tv, X_test, Y_train, Y_tv, Y_test, y_swap,
#                         n_train=len(Y_train), n_tv=len(Y_tv), n_test=len(Y_test), units=20,  h=20)
# results20L_n50, opt_params20L_50, error20_50l = forecastNNARX(X_train, X_tv, X_test, Y_train, Y_tv, Y_test, y_swap,
#                         n_train=len(Y_train), n_tv=len(Y_tv), n_test=len(Y_test), units=50,  h=20)
t=1
