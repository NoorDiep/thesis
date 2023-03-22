from Drivers import get_data, forecast_accuracy, optimal_lag, getDataTablesFigures
from statsmodels.tsa.arima.model import ARIMA
import numpy as np
import pandas as pd
from keras.layers import Dense,Conv2D,MaxPooling2D,UpSampling2D
from keras import Input, Model
from keras import Model
import itertools
from sklearn.model_selection import GridSearchCV

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

    epochs_list = [250, 300, 350, 400]
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


    w = n_train




    p_ARi = []
    p_ARXi = []
    q_ARXi = []
    resultsAR1 = []
    resultsAR2 = []
    resultsAR3 = []
    resultsAR4 = []
    resultsAR5 = []
    resultsAR7 = []
    resultsAR10 = []
    resultsAR15 = []
    resultsAR20 = []
    resultsAR30 = []

    resultsARX1 = []
    resultsARX2 = []
    resultsARX3 = []
    resultsARX4 = []
    resultsARX5 = []
    resultsARX7 = []
    resultsARX10 = []
    resultsARX15 = []
    resultsARX20 = []
    resultsARX30 = []


    resultsARX = []



    for i in range(0, y.shape[1]):


        p_AR1 = optimal_lag(x_train=0, x_tv=0, y_train=y[:n_train,i], y_tv=y[:n_tv,i], maxlags_p=10, maxlags_q=15, indicator=0)
        p_ARX1, q_ARX1 = optimal_lag(x_train=x[:n_train], x_tv=x[:n_tv], y_train=y[:n_train,i], y_tv=y[:n_tv,i], maxlags_p=10, maxlags_q=15, indicator=1)
        p_ARi.append(p_AR1)
        p_ARXi.append(p_ARX1)
        q_ARXi.append(q_ARX1)

        f_k1 = []
        f_k2 = []
        f_k3 = []
        f_k4 = []
        f_k5 = []
        f_k7 = []
        f_k10 = []
        f_k15 = []
        f_k20 = []
        f_k30 = []

        f_kX1 = []
        f_kX2 = []
        f_kX3 = []
        f_kX4 = []
        f_kX5 = []
        f_kX7 = []
        f_kX10 = []
        f_kX15 = []
        f_kX20 = []
        f_kX30 = []

        # Perform rolling window forecasts
        for k in range(n_test - h + 1):
            print(i)
            print(k)
            # # Forecast out-of-sample
            f_j = []
            for j in range(y.shape[1]):
                train = y[k:w + k, j]

                AR = ARIMA(train, order=(p_AR1, 0, 0)).fit()

                preds_AE = AR.predict(start=n_tv + k, end=n_tv + k + h - 1,
                                      dynamic=False)  # Dynamic equal to False means direct forecasts
                f_j.append(preds_AE)
            preds_AR = pd.DataFrame(f_j)
            preds_AR_s = decoder.predict(preds_AR.T)

            test = y_swap[n_tv+k:n_tv+k + h]
            test = pd.DataFrame(test)

            preds_AR = preds_AR.T.reset_index(drop=True).T
            acc_AR = forecast_accuracy(preds_AR_s, test, df_indicator=1)
            f_k1.append(acc_AR.iloc[0])
            f_k2.append(acc_AR.iloc[1])
            f_k3.append(acc_AR.iloc[2])
            f_k4.append(acc_AR.iloc[3])
            f_k5.append(acc_AR.iloc[4])
            f_k7.append(acc_AR.iloc[5])
            f_k10.append(acc_AR.iloc[6])
            f_k15.append(acc_AR.iloc[7])
            f_k20.append(acc_AR.iloc[8])
            f_k30.append(acc_AR.iloc[9])

            # Forecast out-of-sample
            f_jX = []
            for j in range(y.shape[1]):
                ytrain = y[k + q_ARX1:w + k, j]
                xtrain = pd.DataFrame(x[k:w + k]).shift(q_ARX1).dropna()

                # Construct the model with optimized parameters
                ARX = ARIMA(ytrain, exog=xtrain, order=(p_AR1, 0, 0)).fit()
                # Forecast out-of-sample
                preds_AE = ARX.predict(start=n_tv + k, end=n_tv + k + h - 1, dynamic=False, exog=x[k - w - n_test:k + h - w + q_ARX1 + k])  # Dynamic equal to False means direct forecasts
                f_jX.append(preds_AE)
            preds_ARX = pd.DataFrame(f_jX)
            preds_ARX_s = decoder.predict(preds_ARX.T)

            test = y_swap[n_tv + k:n_tv + k + h]
            test = pd.DataFrame(test)

            # Get forecast accuracy
            acc_ARX = forecast_accuracy(preds_ARX_s, test, df_indicator=1)
            f_kX1.append(acc_ARX.iloc[0])
            f_kX2.append(acc_ARX.iloc[1])
            f_kX3.append(acc_ARX.iloc[2])
            f_kX4.append(acc_ARX.iloc[3])
            f_kX5.append(acc_ARX.iloc[4])
            f_kX7.append(acc_ARX.iloc[5])
            f_kX10.append(acc_ARX.iloc[6])
            f_kX15.append(acc_ARX.iloc[7])
            f_kX20.append(acc_ARX.iloc[8])
            f_kX30.append(acc_ARX.iloc[9])

        f_i1 = pd.DataFrame(f_k1)
        means1 = np.mean(f_i1)
        means1['lag p'] = p_AR1
        f_iX1 = pd.DataFrame(f_kX1)
        meansX1 = np.mean(f_iX1)
        meansX1['lag p'] = p_ARX1
        meansX1['lag q'] = q_ARX1

        f_i2 = pd.DataFrame(f_k2)
        means2 = np.mean(f_i2)
        means2['lag p'] = p_AR1
        f_iX2 = pd.DataFrame(f_kX2)
        meansX2 = np.mean(f_iX2)
        meansX2['lag p'] = p_ARX1
        meansX2['lag q'] = q_ARX1

        f_i3 = pd.DataFrame(f_k3)
        means3 = np.mean(f_i3)
        means3['lag p'] = p_AR1
        f_iX3 = pd.DataFrame(f_kX3)
        meansX3 = np.mean(f_iX3)
        meansX3['lag p'] = p_ARX1
        meansX3['lag q'] = q_ARX1

        f_i4 = pd.DataFrame(f_k4)
        means4 = np.mean(f_i4)
        means4['lag p'] = p_AR1
        f_iX4 = pd.DataFrame(f_kX4)
        meansX4 = np.mean(f_iX4)
        meansX4['lag p'] = p_ARX1
        meansX4['lag q'] = q_ARX1

        f_i5 = pd.DataFrame(f_k5)
        means5 = np.mean(f_i5)
        means5['lag p'] = p_AR1
        f_iX5 = pd.DataFrame(f_kX5)
        meansX5 = np.mean(f_iX5)
        meansX5['lag p'] = p_ARX1
        meansX5['lag q'] = q_ARX1

        f_i7 = pd.DataFrame(f_k7)
        means7 = np.mean(f_i7)
        means7['lag p'] = p_AR1
        f_iX7 = pd.DataFrame(f_kX7)
        meansX7 = np.mean(f_iX7)
        meansX7['lag p'] = p_ARX1
        meansX7['lag q'] = q_ARX1

        f_i10 = pd.DataFrame(f_k10)
        means10 = np.mean(f_i10)
        means10['lag p'] = p_AR1
        f_iX10 = pd.DataFrame(f_kX10)
        meansX10 = np.mean(f_iX10)
        meansX10['lag p'] = p_ARX1
        meansX10['lag q'] = q_ARX1

        f_i15 = pd.DataFrame(f_k15)
        means15 = np.mean(f_i15)
        means15['lag p'] = p_AR1
        f_iX15 = pd.DataFrame(f_kX15)
        meansX15 = np.mean(f_iX15)
        meansX15['lag p'] = p_ARX1
        meansX15['lag q'] = q_ARX1

        f_i20 = pd.DataFrame(f_k20)
        means20 = np.mean(f_i20)
        means20['lag p'] = p_AR1
        f_iX20 = pd.DataFrame(f_kX20)
        meansX20 = np.mean(f_iX20)
        meansX20['lag p'] = p_ARX1
        meansX20['lag q'] = q_ARX1


        f_i30 = pd.DataFrame(f_k30)
        means30 = np.mean(f_i30)
        means30['lag p'] = p_AR1
        f_iX30 = pd.DataFrame(f_kX30)
        meansX30 = np.mean(f_iX30)
        meansX30['lag p'] = p_ARX1
        meansX30['lag q'] = q_ARX1

        resultsAR1.append(means1)
        resultsAR2.append(means2)
        resultsAR3.append(means3)
        resultsAR4.append(means4)
        resultsAR5.append(means5)
        resultsAR7.append(means7)
        resultsAR10.append(means10)
        resultsAR15.append(means15)
        resultsAR20.append(means20)
        resultsAR30.append(means30)

        resultsARX1.append(meansX1)
        resultsARX2.append(meansX2)
        resultsARX3.append(meansX3)
        resultsARX4.append(meansX4)
        resultsARX5.append(meansX5)
        resultsARX7.append(meansX7)
        resultsARX10.append(meansX10)
        resultsARX15.append(meansX15)
        resultsARX20.append(meansX20)
        resultsARX30.append(meansX30)

    AR = pd.DataFrame([np.mean(pd.DataFrame(resultsAR1)), np.mean(pd.DataFrame(resultsAR2)), np.mean(pd.DataFrame(resultsAR3)),
          np.mean(pd.DataFrame(resultsAR4)),np.mean(pd.DataFrame(resultsAR5)), np.mean(pd.DataFrame(resultsAR7)),
        np.mean(pd.DataFrame(resultsAR10)),np.mean(pd.DataFrame(resultsAR15)),np.mean(pd.DataFrame(resultsAR20)), np.mean(pd.DataFrame(resultsAR30))])
    ARX = pd.DataFrame(
        [np.mean(pd.DataFrame(resultsARX1)), np.mean(pd.DataFrame(resultsARX2)), np.mean(pd.DataFrame(resultsARX3)), np.mean(pd.DataFrame(resultsARX4)), np.mean(pd.DataFrame(resultsARX5)),
        np.mean(pd.DataFrame(resultsARX7)), np.mean(pd.DataFrame(resultsARX10)), np.mean(pd.DataFrame(resultsARX15)), np.mean(pd.DataFrame(resultsARX20)), np.mean(pd.DataFrame(resultsARX30))])
    return AR,ARX


ae_train, ae_tv, ae_test, decoder, opt_parameters = buildAE(Y_train_diff, Y_tv_diff, Y_test_diff, n=3)
y = np.vstack((pd.DataFrame(ae_tv), pd.DataFrame(ae_test)))
x = np.vstack((X_tv, X_test))
y_swap = np.vstack((Y_tv_diff, Y_test_diff))

AE_AR10, AE_ARX10 = getForecastAE(decoder, x, y, y_swap, n_train=len(ae_train), n_tv=len(ae_tv), n_test=len(ae_test), h=10)
AE_AR1, AE_ARX1 = getForecastAE(decoder, x, y, y_swap, n_train=len(ae_train), n_tv=len(ae_tv), n_test=len(ae_test), h=1)
AE_AR5, AE_ARX5 = getForecastAE(decoder, x, y, y_swap, n_train=len(ae_train), n_tv=len(ae_tv), n_test=len(ae_test), h=5)
AE_AR30, AE_ARX30 = getForecastAE(decoder, x, y, y_swap, n_train=len(ae_train), n_tv=len(ae_tv), n_test=len(ae_test), h=30)
t=1
