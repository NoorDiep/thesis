
from keras.layers import Dense
from keras import Input
from keras import Model
from statsmodels.tsa.ar_model import AutoReg
from Drivers import get_data, forecast_accuracy, optimal_lag, getDataTablesFigures
import itertools
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
def buildAE(y_train, y_tv, y_test, n):
    # https://www.kaggle.com/code/saivarunk/dimensionality-reduction-using-keras-auto-encoder

    ncol = y_train.shape[1]

    encoding_dim = n
    input_dim = Input(shape=(ncol,))

    # Encoder Layers
    encoded = Dense(encoding_dim, activation='tanh')(input_dim)

    # Decoder Layers
    decoded = Dense(ncol, activation='linear')(encoded)

    # Combine Encoder and Deocoder layers
    autoencoder = Model(inputs=input_dim, outputs=decoded)
    encoder = Model(inputs=input_dim, outputs=encoded)
    encoded_input = Input(shape=(encoding_dim,))
    encoded_train = pd.DataFrame(encoder.predict(y_train))
    encoded_train = encoded_train.add_prefix('feature_')


    decoder_layer = autoencoder.layers[-1]
    decoder = Model(inputs=encoded_input, outputs=decoder_layer(encoded_input))



    encoded_tv = pd.DataFrame(encoder.predict(y_tv))
    encoded_test = pd.DataFrame(encoder.predict(y_test))
    encoded_test = encoded_test.add_prefix('feature_')


    autoencoder.compile(optimizer='adam', loss='mse')

    epochs_list = [250, 300, 350, 400, 450, 500]
    batch_size_list = [40, 48, 56, 64]
    params = {}
    for epochs, batch_size in itertools.product(epochs_list, batch_size_list):
        fitted = autoencoder.fit(y_train, y_train, epochs=epochs, batch_size=batch_size, shuffle=False, validation_data=(y_tv, y_tv))
        fitted_loss = fitted.history['val_loss'][-1]
        params[(epochs, batch_size)] = fitted_loss

    opt_params = min(params, key=params.get)
    #opt_params = [10,48]
    autoencoder.fit(y_tv, y_tv, epochs=opt_params[0], batch_size=opt_params[1], shuffle=False)

    return encoded_train, encoded_tv, encoded_test, decoder, opt_params



def getForecastAE(decoder, x, y, y_swap, n_train, n_tv, n_test, h):
    w = n_tv
    f_i = []
    f_iX = []
    p_i = []
    p_iX = []
    q_iX = []

    y_test = pd.DataFrame(y[n_tv:])



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
            preds_AR = AR.predict(start=w+k, end=w+k+h-1, dynamic=False)
            y_hatAR = preds_AR[h-1]
            f_k.append(y_hatAR)

            # Forecast out-of-sample
            preds_ARX = ARX.predict(start=w+k, end=w+k+h-1, exog=x[:w+k+h-1], exog_oos=x[w:k+h+w+k],
                                    dynamic=False)
            y_hatARX = preds_ARX[h-1]
            f_kX.append(y_hatARX)
        f_i.append(f_k)
        f_iX.append(f_kX)

    preds_AR = decoder.predict(pd.DataFrame(f_i).T)
    preds_ARX = decoder.predict(pd.DataFrame(f_iX).T)

    acc_AR = forecast_accuracy(pd.DataFrame(preds_AR), pd.DataFrame(y_swap[-n_test + h - 1:]), df_indicator=1)
    acc_ARX = forecast_accuracy(pd.DataFrame(preds_ARX), pd.DataFrame(y_swap[-n_test + h - 1:]), df_indicator=1)

    resultsAR = pd.DataFrame(acc_AR, columns=['MEA', 'MSE'])
    resultsAR['lag p'] = pd.DataFrame(p_i)

    resultsARX = pd.DataFrame(acc_ARX, columns=['MEA', 'MSE'])
    resultsARX['lag p'] = pd.DataFrame(p_iX)
    resultsARX['lag q'] = pd.DataFrame(q_iX)

    return resultsAR, resultsARX


ae_train, ae_tv, ae_test, decoder, opt_parameters = buildAE(Y_train_diff, Y_tv_diff, Y_test_diff, n=3)
y = np.vstack((pd.DataFrame(ae_tv), pd.DataFrame(ae_test)))
x = np.vstack((X_tv, X_test))
y_swap = np.vstack((Y_tv_diff, Y_test_diff))

AE3_AR10, AE3_ARX10 = getForecastAE(decoder, x, y, y_swap, n_train=len(ae_train), n_tv=len(ae_tv), n_test=len(ae_test), h=10)
AE3_AR1, AE3_ARX1 = getForecastAE(decoder, x, y, y_swap, n_train=len(ae_train), n_tv=len(ae_tv), n_test=len(ae_test), h=1)
AE3_AR5, AE3_ARX5 = getForecastAE(decoder, x, y, y_swap, n_train=len(ae_train), n_tv=len(ae_tv), n_test=len(ae_test), h=5)
AE3_AR20, AE3_ARX20 = getForecastAE(decoder, x, y, y_swap, n_train=len(ae_train), n_tv=len(ae_tv), n_test=len(ae_test), h=20)
t=1

ae_train, ae_tv, ae_test, decoder, opt_parameters = buildAE(Y_train_diff, Y_tv_diff, Y_test_diff, n=2)
y = np.vstack((pd.DataFrame(ae_tv), pd.DataFrame(ae_test)))
x = np.vstack((X_tv, X_test))
y_swap = np.vstack((Y_tv_diff, Y_test_diff))

AE2_AR10, AE2_ARX10 = getForecastAE(decoder, x, y, y_swap, n_train=len(ae_train), n_tv=len(ae_tv), n_test=len(ae_test), h=10)
AE2_AR1, AE2_ARX1 = getForecastAE(decoder, x, y, y_swap, n_train=len(ae_train), n_tv=len(ae_tv), n_test=len(ae_test), h=1)
AE2_AR5, AE2_ARX5 = getForecastAE(decoder, x, y, y_swap, n_train=len(ae_train), n_tv=len(ae_tv), n_test=len(ae_test), h=5)
AE2_AR20, AE2_ARX20 = getForecastAE(decoder, x, y, y_swap, n_train=len(ae_train), n_tv=len(ae_tv), n_test=len(ae_test), h=20)
t=1

ae_train, ae_tv, ae_test, decoder, opt_parameters = buildAE(Y_train_diff, Y_tv_diff, Y_test_diff, n=4)
y = np.vstack((pd.DataFrame(ae_tv), pd.DataFrame(ae_test)))
x = np.vstack((X_tv, X_test))
y_swap = np.vstack((Y_tv_diff, Y_test_diff))

AE4_AR10, AE4_ARX10 = getForecastAE(decoder, x, y, y_swap, n_train=len(ae_train), n_tv=len(ae_tv), n_test=len(ae_test), h=10)
AE4_AR1, AE4_ARX1 = getForecastAE(decoder, x, y, y_swap, n_train=len(ae_train), n_tv=len(ae_tv), n_test=len(ae_test), h=1)
AE4_AR5, AE4_ARX5 = getForecastAE(decoder, x, y, y_swap, n_train=len(ae_train), n_tv=len(ae_tv), n_test=len(ae_test), h=5)
AE4_AR20, AE4_ARX20 = getForecastAE(decoder, x, y, y_swap, n_train=len(ae_train), n_tv=len(ae_tv), n_test=len(ae_test), h=20)
t=1




ae_train, ae_tv, ae_test, decoder, opt_parameters = buildAE(Y_train, Y_tv, Y_test, n=3)
y = np.vstack((pd.DataFrame(ae_tv), pd.DataFrame(ae_test)))
x = np.vstack((X_tv, X_test))
y_swap = np.vstack((Y_tv, Y_test))

AE3_AR10l, AE3_ARX10l = getForecastAE(decoder, x, y, y_swap, n_train=len(ae_train), n_tv=len(ae_tv), n_test=len(ae_test), h=10)
AE3_AR1l, AE3_ARX1l = getForecastAE(decoder, x, y, y_swap, n_train=len(ae_train), n_tv=len(ae_tv), n_test=len(ae_test), h=1)
AE3_AR5l, AE3_ARX5l = getForecastAE(decoder, x, y, y_swap, n_train=len(ae_train), n_tv=len(ae_tv), n_test=len(ae_test), h=5)
AE3_AR20l, AE3_ARX20l = getForecastAE(decoder, x, y, y_swap, n_train=len(ae_train), n_tv=len(ae_tv), n_test=len(ae_test), h=20)
t=1

ae_train, ae_tv, ae_test, decoder, opt_parameters = buildAE(Y_train, Y_tv, Y_test, n=2)
y = np.vstack((pd.DataFrame(ae_tv), pd.DataFrame(ae_test)))
x = np.vstack((X_tv, X_test))
y_swap = np.vstack((Y_tv, Y_test))

AE2_AR10l, AE2_ARX10l = getForecastAE(decoder, x, y, y_swap, n_train=len(ae_train), n_tv=len(ae_tv), n_test=len(ae_test), h=10)
AE2_AR1l, AE2_ARX1l = getForecastAE(decoder, x, y, y_swap, n_train=len(ae_train), n_tv=len(ae_tv), n_test=len(ae_test), h=1)
AE2_AR5l, AE2_ARX5l = getForecastAE(decoder, x, y, y_swap, n_train=len(ae_train), n_tv=len(ae_tv), n_test=len(ae_test), h=5)
AE2_AR20l, AE2_ARX20l = getForecastAE(decoder, x, y, y_swap, n_train=len(ae_train), n_tv=len(ae_tv), n_test=len(ae_test), h=20)
t=1

ae_train, ae_tv, ae_test, decoder, opt_parameters = buildAE(Y_train, Y_tv, Y_test, n=4)
y = np.vstack((pd.DataFrame(ae_tv), pd.DataFrame(ae_test)))
x = np.vstack((X_tv, X_test))
y_swap = np.vstack((Y_tv, Y_test))

AE4_AR10l, AE4_ARX10l = getForecastAE(decoder, x, y, y_swap, n_train=len(ae_train), n_tv=len(ae_tv), n_test=len(ae_test), h=10)
AE4_AR1l, AE4_ARX1l = getForecastAE(decoder, x, y, y_swap, n_train=len(ae_train), n_tv=len(ae_tv), n_test=len(ae_test), h=1)
AE4_AR5l, AE4_ARX5l = getForecastAE(decoder, x, y, y_swap, n_train=len(ae_train), n_tv=len(ae_tv), n_test=len(ae_test), h=5)
AE4_AR20l, AE4_ARX20l = getForecastAE(decoder, x, y, y_swap, n_train=len(ae_train), n_tv=len(ae_tv), n_test=len(ae_test), h=20)
t=1
