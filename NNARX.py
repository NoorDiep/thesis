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

def forecastNNARX(x_train, x_tv, x_test, y_train, y_tv, y_test, y_swap, n_train, n_tv, n_test, hidden_nodes, h):
    results = []


    w = n_train
    x = np.vstack((x_tv, x_test))
    y = np.vstack((y_tv, y_test))

    # reshape input to be 3D [samples, timesteps, features]; each line turns into an 'sub-array'
    df_x = x.reshape((x.shape[0], 1, x.shape[1]))

    # define the model architecture
    # model = Sequential()
    # model.add(Dense(int(hidden_nodes), input_dim=int(input_nodes)))
    # model.add(Dense(int(output_nodes)))
    # model.compile(loss='mse', optimizer='adam', metrics=['accuracy'])
    model = Sequential()
    model.add(LSTM(hidden_nodes, input_shape=(df_x.shape[1], df_x.shape[2])))
    model.add(Dense(1))
    model.compile(loss='mae', optimizer='adam')

    #
    # epochs_list = [50, 100, 150, 200, 250]
    # batch_size_list = [8, 16, 24, 32, 40, 48, 56]
    # params = {}
    # for epochs, batch_size in itertools.product(epochs_list, batch_size_list):
    #     fitted = model.fit(df_x[:n_train], y_train, epochs=epochs, batch_size=batch_size, shuffle=False,
    #                              validation_data=(df_x[n_train:n_tv], y_tv[n_train:]))
    #     fitted_loss = fitted.history['val_loss'][-1]
    #     params[(epochs, batch_size)] = fitted_loss
    #
    # opt_params = min(params, key=params.get)
    opt_params = [100,32]
    model.fit(df_x[:n_tv], y_tv, epochs=opt_params[0], batch_size=opt_params[1], shuffle=False)

    print('Opt params:', opt_params)

    for i in range(0, y_train.shape[1]):
        f_k = []
        for k in range(n_test - h + 1):
            print(i,k)
            # # Forecast out-of-sample
            test_Xk = df_x[k - w:k + h - w + k]
            y_pred = model.predict(test_Xk)
            preds = pd.DataFrame(y_pred)

            test = y[n_tv+k:n_tv+k + h,i]
            test = pd.DataFrame(test)

            acc_AR = forecast_accuracy(preds[0], test[0], df_indicator=0)
            f_k.append(acc_AR)
        f_i = pd.DataFrame(np.concatenate(f_k), columns=['MEA', 'MSE', 'RMSE'])
        means = np.mean(f_i)
        results.append(means)
    results1 = pd.DataFrame(results)
    return results1, opt_params

y_swap = np.vstack((Y_tv_diff, Y_test_diff))
results10_n3 = forecastNNARX(X_train_diff, X_tv_diff, X_test_diff, Y_train_diff, Y_tv_diff, Y_test_diff, y_swap,
                        n_train=len(Y_train_diff), n_tv=len(Y_tv_diff), n_test=len(Y_test_diff), hidden_nodes=3,  h=10)
results10_n2 = forecastNNARX(X_train_diff, X_tv_diff, X_test_diff, Y_train_diff, Y_tv_diff, Y_test_diff, y_swap,
                        n_train=len(Y_train_diff), n_tv=len(Y_tv_diff), n_test=len(Y_test_diff), hidden_nodes=2,  h=10)
results10_n4 = forecastNNARX(X_train_diff, X_tv_diff, X_test_diff, Y_train_diff, Y_tv_diff, Y_test_diff, y_swap,
                        n_train=len(Y_train_diff), n_tv=len(Y_tv_diff), n_test=len(Y_test_diff), hidden_nodes=4,  h=10)
