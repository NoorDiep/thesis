from sklearn.model_selection import train_test_split
import pandas as pd
import numpy as np
from matplotlib import pyplot
from sklearn.model_selection import train_test_split, GridSearchCV
from statsmodels.tools import add_constant
from statsmodels.stats.outliers_influence import variance_inflation_factor
import matplotlib.pyplot as plt
import matplotlib.dates as dates
import matplotlib.ticker as ticker
from Drivers import read_file, difference_series, getCSSED, optimal_lag, forecast_accuracy
import statsmodels.api as sm
from statsmodels.tsa.ar_model import AutoReg
from keras.layers import Dense
from keras import Input
from keras import Model
import itertools
from sklearn.decomposition import PCA
from keras.models import Sequential
from keras.layers import LSTM


def get_data(lag):

    # Load all data
    df_swap, dates = read_file(file='SwapDriverData1.xlsx', sheet='Swap')
    df_drivers = pd.read_excel('Driver.xlsx')
    depo = read_file(file='SwapDriverData1.xlsx', sheet='DEPO')
    spread = pd.read_excel('Spread.xlsx')
    spread = spread.drop('Date', axis=1)

    spread = spread.dropna()
    # df_swap = df_swap[1015:]
    # df_drivers = df_drivers[1015:]
    # dates = dates[1015:]

    #df_drivers = df_drivers.drop('Stress', axis=1)
    #f_drivers = df_drivers.drop('EcSu', axis=1)
    #df_drivers = df_drivers.drop('Sent', axis=1)
    #df_drivers = df_drivers.drop('PoUn', axis=1)
    #df_drivers = df_drivers.drop('News', axis=1)
    df_drivers = df_drivers.drop('Vol', axis=1)
    #df_drivers = df_drivers.drop('Infl', axis=1)
    df_drivers = df_drivers.drop('Depo', axis=1)
    df_drivers = df_drivers.drop('GB_P', axis=1)
    df_drivers = df_drivers.drop('GB_BA', axis=1)
    df_drivers = df_drivers.drop('Swap spread', axis=1)
    df_drivers  = df_drivers.drop('Date', axis=1)


    diff_swap = difference_series(df_swap, lag)
    df_drivers_diff = difference_series(df_drivers, lag)
    df_spread_diff = difference_series(spread, lag)
    #df_drivers_diff = df_drivers.iloc[lag:]  # Cut off first n observations

    # Get descriptive statistics, corelation tables, adf test results and figures for Data Section
    #getDataTablesFigures()

    # Create train, validation and test set
    df_swap = df_swap.iloc[lag:]
    df_drivers = df_drivers.iloc[lag:]
    df_drivers = df_drivers.reset_index(drop=True)
    df_swap_dates = df_swap
    df_swap = df_swap.reset_index(drop=True)
    spread = spread.reset_index(drop=True)

    # Construct final df

    df_drivers = pd.DataFrame([df_drivers['EcSu'],df_drivers['Sent'],df_drivers['Stress'],df_drivers['PoUn'],df_drivers['News'],df_drivers['Infl']]).T

    # print(sm.OLS(df_swap['10Y'], add_constant(df_drivers)).fit().summary())
    # print(sm.OLS(df_swap['1Y'], add_constant(df_drivers)).fit().summary())
    # print(sm.OLS(df_swap['30Y'], add_constant(df_drivers)).fit().summary())
    #
    # print(sm.OLS(diff_swap['10Y'], add_constant(df_drivers)).fit().summary())
    # print(sm.OLS(diff_swap['1Y'], add_constant(df_drivers)).fit().summary())
    # print(sm.OLS(diff_swap['30Y'], add_constant(df_drivers)).fit().summary())
    #print(sm.OLS(diff_swap['10Y'], add_constant(df_drivers_diff)).fit().summary())

    # Selected set for descriptive statistics
    # df_crisis = diff_swap[:1433];
    # df_post_crisis = diff_swap[1433:]
    # get_ds(df_crisis)
    # get_ds(df_post_crisis)

    x_train, x_test, y_train, y_test = train_test_split(df_drivers, df_swap, test_size=0.2, shuffle=False)
    x_train, x_val, y_train, y_val = train_test_split(x_train, y_train, test_size=0.25, shuffle=False)

    x_train_diff, x_test_diff, y_train_diff, y_test_diff = train_test_split(df_drivers_diff, diff_swap, test_size=0.2, shuffle=False)
    x_train_diff, x_val_diff, y_train_diff, y_val_diff = train_test_split(x_train_diff, y_train_diff, test_size=0.25, shuffle=False)

    data_full = [df_swap, df_drivers, diff_swap, df_drivers_diff]
    data = [x_train, x_val, x_test, y_train, y_val, y_test]
    data_diff = [x_train_diff, x_val_diff, x_test_diff, y_train_diff, y_val_diff, y_test_diff]
    date_train = dates[:x_train.shape[0]]
    date_val = dates[x_train.shape[0]:x_val.index[-1]+1]
    date_test = dates[x_val.index[-1]+1:]
    date_train_diff = dates[lag:y_train_diff.shape[0]]
    date_test_diff = dates[y_train_diff.shape[0]:len(dates)-lag]

    return data_full, data, data_diff, date_train, date_test, date_train_diff, date_test_diff, depo, df_swap_dates, spread, df_spread_diff

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
            preds_AR = AR.predict(start=w+k, end=w+k+h-1, dynamic=False)
            y_hatAR = preds_AR[h-1]
            f_k.append(y_hatAR)

            # Forecast out-of-sample
            preds_ARX = ARX.predict(start=w+k, end=w+k+h-1, exog=x[:w+k+h-1], exog_oos=x[w:k+h+w+k],
                                    dynamic=False)
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

    resultsAR = pd.DataFrame(acc_AR, columns=['MEA', 'RMSE'])
    resultsAR['lag p'] = pd.DataFrame(p_i)

    resultsARX = pd.DataFrame(acc_ARX, columns=['MEA', 'RMSE'])
    resultsARX['lag p'] = pd.DataFrame(p_iX)
    resultsARX['lag q'] = pd.DataFrame(q_iX)

    error = pd.DataFrame(preds_AR) - pd.DataFrame(y_swap[-n_test + h - 1:])
    errorX = pd.DataFrame(preds_ARX) - pd.DataFrame(y_swap[-n_test + h - 1:])

    return resultsAR, resultsARX, pd.DataFrame(pd.DataFrame(error)).T, pd.DataFrame(pd.DataFrame(errorX)).T

def getPC(y_train, y_tv, y_test, n):
    mu_total=[]

    pca = PCA(n_components=n)
    PC_train = pca.fit_transform(y_tv)
    PC_train_df = pd.DataFrame(PC_train)

    names_pcas = [f"PCA Component {i}" for i in range(1, n + 1, 1)]
    scree = pd.DataFrame(list(zip(names_pcas, pca.explained_variance_ratio_)),
                         columns=["Component", "Explained Variance Ratio"])


    y_trainPC = np.dot(PC_train, pca.components_) + np.array(np.mean(y_train, axis=0))
    PC_tv = pca.transform(y_tv)
    y_tvPC = np.dot(PC_tv, pca.components_) + np.array(np.mean(y_tv, axis=0))
    PC_tv_df = pd.DataFrame(PC_tv)
    PC_test = pca.transform(y_test)
    y_testPC = np.dot(PC_test, pca.components_) + np.array(np.mean(y_test, axis=0))
    PC_test_df = pd.DataFrame(PC_test)

    mu_total.append(np.array(np.mean(y_train, axis=0)))
    mu_total.append(np.array(np.mean(y_tv, axis=0)))
    mu_total.append(np.array(np.mean(y_test, axis=0)))
    return PC_train_df, PC_tv_df, PC_test_df, mu_total, pca.components_, scree


def getForecastPCA(means, PCs, x, y, y_swap, n_train, n_tv, n_test, h):


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

    preds_AR = np.dot(pd.DataFrame(f_i).T, PCs) + means[1]
    preds_ARX = np.dot(pd.DataFrame(f_iX).T, PCs) + means[1]

    resultsAR = forecast_accuracy(pd.DataFrame(preds_AR), pd.DataFrame(y_swap[-n_test + h - 1:]), df_indicator=1)
    resultsARX = forecast_accuracy(pd.DataFrame(preds_ARX), pd.DataFrame(y_swap[-n_test + h - 1:]), df_indicator=1)
    resultsAR['lag p'] = pd.DataFrame(p_i)
    resultsARX['lag p'] = pd.DataFrame(p_iX)
    resultsARX['lag q'] = pd.DataFrame(q_iX)

    error = pd.DataFrame(preds_AR) - pd.DataFrame(y_swap[-n_test + h - 1:])
    errorX = pd.DataFrame(preds_ARX) - pd.DataFrame(y_swap[-n_test + h - 1:])

    return resultsAR, resultsARX, pd.DataFrame(error), pd.DataFrame(errorX)

def forecastLSTM(x_train, x_tv, x_test, y_train, y_tv, y_test, y_swap, n_train, n_tv, n_test, units, h):
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

    epochs_list = [5, 10, 15, 20, 25, 30, 35]
    batch_size_list = [2, 4, 6, 8]
    params = {}
    for epochs, batch_size in itertools.product(epochs_list, batch_size_list):
        fitted = model.fit(df_x[:n_train], y_train, epochs=epochs, batch_size=batch_size, shuffle=False,
                                 validation_data=(df_x[n_train:n_tv], y_tv[n_train:]))
        fitted_loss = fitted.history['val_loss'][-1]
        params[(epochs, batch_size)] = fitted_loss

    opt_params = min(params, key=params.get)
    # opt_params = [1,1000]

    model.fit(df_x[:w], y[:w], epochs=opt_params[0], batch_size=opt_params[1], shuffle=False)
    print('Opt params:', opt_params)

    for i in range(0, y_train.shape[1]):
        f_k = []
        for k in range(n_test - h + 1):
            # model.fit(df_x[:k+w], y[:k+w], epochs=opt_params[0], batch_size=opt_params[1], shuffle=False)
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
    test = y_test.iloc[h-1:,]
    test = test.reset_index(drop=True)
    error = f_i - test
    results = forecast_accuracy(f_i, test, df_indicator=1)
    return results, opt_params, error

def getCSSED(errorB, errorM):

    cssed = np.cumsum(errorB**2-errorM**2)

    return cssed

def getCSSEDplot(df, h, dates):
    color = ["blue", "lightblue", "deepskyblue", "dodgerblue", "steelblue", "mediumblue", "darkblue", "slategrey",
             "gray", "black"]
    df.columns = ['1Y', '2Y','3Y', '4Y','5Y', '7Y','10Y', '15Y','20Y', '30Y']

    outperform = df['10Y'] > 0
    outperform = np.array(outperform).astype(int)

    i = 0
    plt.ioff()
    for column in df:
        ax = df[column].plot(figsize=(15, 5), lw=1, color=color[i], label=column)
        i = i + 1
    ax.pcolorfast(ax.get_xlim(), ax.get_ylim(),
                  pd.DataFrame(outperform).values[np.newaxis],
                  cmap='Greys', alpha=0.3)

    #plt.title("Forecast horizon: h = ", h)
    plt.legend(ncol=2)
    plt.xlabel("Date")
    plt.ylabel("")
    plt.show()


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
#getDataTablesFigures(data[0], data[1], pd.DataFrame(Y_diff), pd.DataFrame(X_diff), depo, swap_dates)

x = np.vstack((X_tv, X_test))
y = np.vstack((Y_tv, Y_test))
y_diff = np.vstack((Y_tv_diff, Y_test_diff))

print(sm.OLS(y[:,6], add_constant(x)).fit().summary())
print(sm.OLS(y[:,0], add_constant(x)).fit().summary())
print(sm.OLS(y[:,9], add_constant(x)).fit().summary())

print(sm.OLS(y_diff[:,6], add_constant(x)).fit().summary())
print(sm.OLS(y_diff[:,0], add_constant(x)).fit().summary())
print(sm.OLS(y_diff[:,9], add_constant(x)).fit().summary())


########################################################################################################################
# RUN CODE
########################################################################################################################

####### AR - ARX #######


pdAR1, pdARX1, error1, errorX1 = getForecast(x, y, n_train=len(Y_train), n_tv=len(Y_tv), n_test=len(Y_test), h=1, diff_lag=lag)
pdAR5, pdARX5, error5, errorX5 = getForecast(x, y, n_train=len(Y_train), n_tv=len(Y_tv), n_test=len(Y_test), h=5, diff_lag=lag)
pdAR10, pdARX10, error10, errorX10 = getForecast(x, y, n_train=len(Y_train), n_tv=len(Y_tv), n_test=len(Y_test), h=10, diff_lag=lag)
pdAR20, pdARX20, error20, errorX20 = getForecast(x, y, n_train=len(Y_train), n_tv=len(Y_tv), n_test=len(Y_test), h=20, diff_lag=lag)



pdAR1_diff, pdARX1_diff, error1_diff, errorX1_diff = getForecast(x, y_diff, n_train=len(Y_train), n_tv=len(Y_tv), n_test=len(Y_test), h=1, diff_lag=lag)
pdAR5_diff, pdARX5_diff, error5_diff, errorX5_diff = getForecast(x, y_diff, n_train=len(Y_train), n_tv=len(Y_tv), n_test=len(Y_test), h=5, diff_lag=lag)
pdAR10_diff, pdARX10_diff, error10_diff, errorX10_diff = getForecast(x, y_diff, n_train=len(Y_train), n_tv=len(Y_tv), n_test=len(Y_test), h=10, diff_lag=lag)
pdAR20_diff, pdARX20_diff, error20_diff, errorX20_diff = getForecast(x, y_diff, n_train=len(Y_train), n_tv=len(Y_tv), n_test=len(Y_test), h=20, diff_lag=lag)

t = 1

######### PCA ##########

PC1_train, PC1_tv, PC1_test, mu1, var_expl1, scree1 = getPC(Y_train_diff, Y_tv_diff, Y_test_diff,1)
PC2_train, PC2_tv, PC2_test, mu2, var_expl2, scree2 = getPC(Y_train_diff, Y_tv_diff, Y_test_diff,2)
PC3_train, PC3_tv, PC3_test, mu3, var_expl3, scree3 = getPC(Y_train_diff, Y_tv_diff, Y_test_diff,3)
PC4_train, PC4_tv, PC4_test, mu4, var_expl4, scree4 = getPC(Y_train_diff, Y_tv_diff, Y_test_diff,4)
PC5_train, PC5_tv, PC5_test, mu5, var_expl5, scree5 = getPC(Y_train_diff, Y_tv_diff, Y_test_diff,5)


x = np.vstack((X_tv, X_test))
y1 = np.vstack((PC1_tv, PC1_test))
y2 = np.vstack((PC2_tv, PC2_test))
y3 = np.vstack((PC3_tv, PC3_test))
y4 = np.vstack((PC4_tv, PC4_test))
y5 = np.vstack((PC5_tv, PC5_test))
y_swap = np.vstack((Y_tv_diff, Y_test_diff))


PCA3_AR10, PCA3_ARX10, error3_10, errorX3_10 = getForecastPCA(mu3, var_expl3, x, y3, y_swap, n_train=len(PC3_train), n_tv=len(PC3_tv), n_test=len(PC3_test), h=10)
# print(PCA3_AR10, PCA3_ARX10)
# PCA4_AR10, PCA4_ARX10, error4_10, errorX4_10 = getForecastPCA(mu4, var_expl4, x, y4, y_swap, n_train=len(PC4_train), n_tv=len(PC4_tv), n_test=len(PC4_test), h=10)
# print(PCA4_AR10, PCA4_ARX10)
# PCA2_AR10, PCA2_ARX10, error2_10, errorX2_10 = getForecastPCA(mu2, var_expl2, x, y2, y_swap, n_train=len(PC2_train), n_tv=len(PC2_tv), n_test=len(PC2_test), h=10)
# print(PCA2_AR10, PCA2_ARX10)
# PCA5_AR10, PCA5_ARX10, error5_10, errorX5_10 = getForecastPCA(mu5, var_expl5, x, y5, y_swap, n_train=len(PC5_train), n_tv=len(PC5_tv), n_test=len(PC5_test), h=10)
# PCA1_AR10, PCA1_ARX10, error1_10, errorX1_10= getForecastPCA(mu1, var_expl1, x, y1, y_swap, n_train=len(PC1_train), n_tv=len(PC1_tv), n_test=len(PC1_test), h=10)
#
#
# PCA1_AR1, PCA1_ARX1, error1_1, errorX1_1 = getForecastPCA(mu1, var_expl1, x, y1, y_swap, n_train=len(PC1_train), n_tv=len(PC1_tv), n_test=len(PC1_test), h=1)
# PCA2_AR1, PCA2_ARX1, error2_1, errorX2_1 = getForecastPCA(mu2, var_expl2, x, y2, y_swap, n_train=len(PC2_train), n_tv=len(PC2_tv), n_test=len(PC2_test), h=1)
PCA3_AR1, PCA3_ARX1, error3_1, errorX3_1 = getForecastPCA(mu3, var_expl3, x, y3, y_swap, n_train=len(PC3_train), n_tv=len(PC3_tv), n_test=len(PC3_test), h=1)
# PCA4_AR1, PCA4_ARX1, error4_1, errorX4_1 = getForecastPCA(mu4, var_expl4, x, y4, y_swap, n_train=len(PC4_train), n_tv=len(PC4_tv), n_test=len(PC4_test), h=1)
# PCA5_AR1, PCA5_ARX1, error5_1, errorX5_1 = getForecastPCA(mu5, var_expl5, x, y5, y_swap, n_train=len(PC5_train), n_tv=len(PC5_tv), n_test=len(PC5_test), h=1)
#
# PCA1_AR5, PCA1_ARX5, error1_5, errorX1_5 = getForecastPCA(mu1, var_expl1, x, y1, y_swap, n_train=len(PC1_train), n_tv=len(PC1_tv), n_test=len(PC1_test), h=5)
# PCA2_AR5, PCA2_ARX5, error2_5, errorX2_5 = getForecastPCA(mu2, var_expl2, x, y2, y_swap, n_train=len(PC2_train), n_tv=len(PC2_tv), n_test=len(PC2_test), h=5)
PCA3_AR5, PCA3_ARX5, error3_5, errorX3_5 = getForecastPCA(mu3, var_expl3, x, y3, y_swap, n_train=len(PC3_train), n_tv=len(PC3_tv), n_test=len(PC3_test), h=5)
# PCA4_AR5, PCA4_ARX5, error4_5, errorX4_5 = getForecastPCA(mu4, var_expl4, x, y4, y_swap, n_train=len(PC4_train), n_tv=len(PC4_tv), n_test=len(PC4_test), h=5)
# PCA5_AR5, PCA5_ARX5, error5_5, errorX5_5 = getForecastPCA(mu5, var_expl5, x, y5, y_swap, n_train=len(PC5_train), n_tv=len(PC5_tv), n_test=len(PC5_test), h=5)
#
# PCA1_AR20, PCA1_ARX20, error1_20, errorX1_20 = getForecastPCA(mu1, var_expl1, x, y1, y_swap, n_train=len(PC1_train), n_tv=len(PC1_tv), n_test=len(PC1_test), h=20)
# PCA2_AR20, PCA2_ARX20, error2_20, errorX2_20 = getForecastPCA(mu2, var_expl2, x, y2, y_swap, n_train=len(PC2_train), n_tv=len(PC2_tv), n_test=len(PC2_test), h=20)
PCA3_AR20, PCA3_ARX20, error3_20, errorX3_20 = getForecastPCA(mu3, var_expl3, x, y3, y_swap, n_train=len(PC3_train), n_tv=len(PC3_tv), n_test=len(PC3_test), h=20)
# PCA4_AR20, PCA4_ARX20, error4_20, errorX4_20 = getForecastPCA(mu4, var_expl4, x, y4, y_swap, n_train=len(PC4_train), n_tv=len(PC4_tv), n_test=len(PC4_test), h=20)
# PCA5_AR20, PCA5_ARX20, error5_20, errorX5_20 = getForecastPCA(mu5, var_expl5, x, y5, y_swap, n_train=len(PC5_train), n_tv=len(PC5_tv), n_test=len(PC5_test), h=20)



PC1l_train, PC1l_tv, PC1l_test, mu1l, var_expl1l, scree1l = getPC(Y_train, Y_tv, Y_test,1)
PC2l_train, PC2l_tv, PC2l_test, mu2l, var_expl2l, scree2l = getPC(Y_train, Y_tv, Y_test,2)
PC3l_train, PC3l_tv, PC3l_test, mu3l, var_expl3l, scree3l = getPC(Y_train, Y_tv, Y_test,3)
PC4l_train, PC4l_tv, PC4l_test, mu4l, var_expl4l, scree4l = getPC(Y_train, Y_tv, Y_test,4)
PC5l_train, PC5l_tv, PC5l_test, mu5l, var_expl5l, scree5l = getPC(Y_train, Y_tv, Y_test,5)


x = np.vstack((X_tv, X_test))
y1l = np.vstack((PC1l_tv, PC1l_test))
y2l = np.vstack((PC2l_tv, PC2l_test))
y3l = np.vstack((PC3l_tv, PC3l_test))
y4l = np.vstack((PC4l_tv, PC4l_test))
y5l = np.vstack((PC5l_tv, PC5l_test))
y_swapl = np.vstack((Y_tv, Y_test))


PCA3l_AR10, PCA3l_ARX10, error3_10l, errorX3_10l = getForecastPCA(mu3l, var_expl3l, x, y3l, y_swapl, n_train=len(PC3l_train), n_tv=len(PC3l_tv), n_test=len(PC3l_test), h=10)
# print(PCA3l_AR10, PCA3l_ARX10)
# PCA4l_AR10, PCA4l_ARX10, error4_10l, errorX4_10l = getForecastPCA(mu4l, var_expl4l, x, y4l, y_swapl, n_train=len(PC4l_train), n_tv=len(PC4l_tv), n_test=len(PC4l_test), h=10)
# print(PCA4l_AR10, PCA4l_ARX10)
# PCA2l_AR10, PCA2l_ARX10, error2_10l, errorX2_10l = getForecastPCA(mu2l, var_expl2l, x, y2l, y_swapl, n_train=len(PC2l_train), n_tv=len(PC2l_tv), n_test=len(PC2l_test), h=10)
# print(PCA2l_AR10, PCA2l_ARX10)
# PCA5l_AR10, PCA5l_ARX10, error5_10l, errorX5_10l = getForecastPCA(mu5l, var_expl5l, x, y5l, y_swapl, n_train=len(PC5l_train), n_tv=len(PC5l_tv), n_test=len(PC5l_test), h=10)
# PCA1l_AR10, PCA1l_ARX10, error1_10l, errorX1_10l = getForecastPCA(mu1l, var_expl1l, x, y1l, y_swapl, n_train=len(PC1l_train), n_tv=len(PC1l_tv), n_test=len(PC1l_test), h=10)



PCA3l_AR1, PCA3l_ARX1, error3_1l, errorX3_1l = getForecastPCA(mu3l, var_expl3l, x, y3l, y_swapl, n_train=len(PC3l_train), n_tv=len(PC3l_tv), n_test=len(PC3l_test), h=1)
# PCA4l_AR1, PCA4l_ARX1, error4_1l, errorX4_1l = getForecastPCA(mu4l, var_expl4l, x, y4l, y_swapl, n_train=len(PC4l_train), n_tv=len(PC4l_tv), n_test=len(PC4l_test), h=1)
# PCA2l_AR1, PCA2l_ARX1, error2_1l, errorX2_1l = getForecastPCA(mu2l, var_expl2l, x, y2l, y_swapl, n_train=len(PC2l_train), n_tv=len(PC2l_tv), n_test=len(PC2l_test), h=1)
# PCA5l_AR1, PCA5l_ARX1, error5_1l, errorX5_1l = getForecastPCA(mu5l, var_expl5l, x, y5l, y_swapl, n_train=len(PC5l_train), n_tv=len(PC5l_tv), n_test=len(PC5l_test), h=1)
# PCA1l_AR1, PCA1l_ARX1, error1_1l, errorX1_1l = getForecastPCA(mu1l, var_expl1l, x, y1l, y_swapl, n_train=len(PC1l_train), n_tv=len(PC1l_tv), n_test=len(PC1l_test), h=1)

PCA3l_AR5, PCA3l_ARX5, error3_5l, errorX3_5l = getForecastPCA(mu3l, var_expl3l, x, y3l, y_swapl, n_train=len(PC3l_train), n_tv=len(PC3l_tv), n_test=len(PC3l_test), h=5)
# PCA4l_AR5, PCA4l_ARX5, error4_5l, errorX4_5l = getForecastPCA(mu4l, var_expl4l, x, y4l, y_swapl, n_train=len(PC4l_train), n_tv=len(PC4l_tv), n_test=len(PC4l_test), h=5)
# PCA2l_AR5, PCA2l_ARX5, error2_5l, errorX2_5l = getForecastPCA(mu2l, var_expl2l, x, y2l, y_swapl, n_train=len(PC2l_train), n_tv=len(PC2l_tv), n_test=len(PC2l_test), h=5)
# PCA5l_AR5, PCA5l_ARX5, error5_5l, errorX5_5l = getForecastPCA(mu5l, var_expl5l, x, y5l, y_swapl, n_train=len(PC5l_train), n_tv=len(PC5l_tv), n_test=len(PC5l_test), h=5)
# PCA1l_AR5, PCA1l_ARX5, error1_5l, errorX1_5l = getForecastPCA(mu1l, var_expl1l, x, y1l, y_swapl, n_train=len(PC1l_train), n_tv=len(PC1l_tv), n_test=len(PC1l_test), h=5)

PCA3l_AR20, PCA3l_ARX20, error3_20l, errorX3_20l = getForecastPCA(mu3l, var_expl3l, x, y3l, y_swapl, n_train=len(PC3l_train), n_tv=len(PC3l_tv), n_test=len(PC3l_test), h=20)
# PCA4l_AR20, PCA4l_ARX20, error4_20l, errorX4_20l = getForecastPCA(mu4l, var_expl4l, x, y4l, y_swapl, n_train=len(PC4l_train), n_tv=len(PC4l_tv), n_test=len(PC4l_test), h=20)
# PCA2l_AR20, PCA2l_ARX20, error2_20l, errorX2_20l = getForecastPCA(mu2l, var_expl2l, x, y2l, y_swapl, n_train=len(PC2l_train), n_tv=len(PC2l_tv), n_test=len(PC2l_test), h=20)
# PCA5l_AR20, PCA5l_ARX20, error5_20l, errorX5_20l = getForecastPCA(mu5l, var_expl5l, x, y5l, y_swapl, n_train=len(PC5l_train), n_tv=len(PC5l_tv), n_test=len(PC5l_test), h=20)
# PCA1l_AR20, PCA1l_ARX20, error1_20l, errorX1_20l = getForecastPCA(mu1l, var_expl1l, x, y1l, y_swapl, n_train=len(PC1l_train), n_tv=len(PC1l_tv), n_test=len(PC1l_test), h=20)
t=1


######### Autoencoder ############
ae_train, ae_tv, ae_test, decoder, opt_parameters3 = buildAE(Y_train_diff, Y_tv_diff, Y_test_diff, n=3)
y = np.vstack((pd.DataFrame(ae_tv), pd.DataFrame(ae_test)))
x = np.vstack((X_tv, X_test))
y_swap = np.vstack((Y_tv_diff, Y_test_diff))

AE3_AR10, AE3_ARX10, error3AE_10, errorX3AE_10 = getForecastAE(decoder, x, y, y_swap, n_train=len(ae_train), n_tv=len(ae_tv), n_test=len(ae_test), h=10)
AE3_AR1, AE3_ARX1, error3AE_1, errorX3AE_1 = getForecastAE(decoder, x, y, y_swap, n_train=len(ae_train), n_tv=len(ae_tv), n_test=len(ae_test), h=1)
AE3_AR5, AE3_ARX5, error3AE_5, errorX3AE_5 = getForecastAE(decoder, x, y, y_swap, n_train=len(ae_train), n_tv=len(ae_tv), n_test=len(ae_test), h=5)
AE3_AR20, AE3_ARX20, error3AE_20, errorX3AE_20 = getForecastAE(decoder, x, y, y_swap, n_train=len(ae_train), n_tv=len(ae_tv), n_test=len(ae_test), h=20)
t=1

# ae_train, ae_tv, ae_test, decoder, opt_parameters2 = buildAE(Y_train_diff, Y_tv_diff, Y_test_diff, n=2)
# y = np.vstack((pd.DataFrame(ae_tv), pd.DataFrame(ae_test)))
# x = np.vstack((X_tv, X_test))
# y_swap = np.vstack((Y_tv_diff, Y_test_diff))
#
# AE2_AR10, AE2_ARX10, error2_10, errorX2_10 = getForecastAE(decoder, x, y, y_swap, n_train=len(ae_train), n_tv=len(ae_tv), n_test=len(ae_test), h=10)
# AE2_AR1, AE2_ARX1, error2_1, errorX2_1 = getForecastAE(decoder, x, y, y_swap, n_train=len(ae_train), n_tv=len(ae_tv), n_test=len(ae_test), h=1)
# AE2_AR5, AE2_ARX5, error2_5, errorX2_5 = getForecastAE(decoder, x, y, y_swap, n_train=len(ae_train), n_tv=len(ae_tv), n_test=len(ae_test), h=5)
# AE2_AR20, AE2_ARX20, error2_20, errorX2_20 = getForecastAE(decoder, x, y, y_swap, n_train=len(ae_train), n_tv=len(ae_tv), n_test=len(ae_test), h=20)
# t=1
#
# ae_train, ae_tv, ae_test, decoder, opt_parameters4 = buildAE(Y_train_diff, Y_tv_diff, Y_test_diff, n=4)
# y = np.vstack((pd.DataFrame(ae_tv), pd.DataFrame(ae_test)))
# x = np.vstack((X_tv, X_test))
# y_swap = np.vstack((Y_tv_diff, Y_test_diff))
#
# AE4_AR10, AE4_ARX10, error4_10, errorX4_10 = getForecastAE(decoder, x, y, y_swap, n_train=len(ae_train), n_tv=len(ae_tv), n_test=len(ae_test), h=10)
# AE4_AR1, AE4_ARX1, error4_1, errorX4_1 = getForecastAE(decoder, x, y, y_swap, n_train=len(ae_train), n_tv=len(ae_tv), n_test=len(ae_test), h=1)
# AE4_AR5, AE4_ARX5, error4_5, errorX4_5 = getForecastAE(decoder, x, y, y_swap, n_train=len(ae_train), n_tv=len(ae_tv), n_test=len(ae_test), h=5)
# AE4_AR20, AE4_ARX20, error4_20, errorX4_20 = getForecastAE(decoder, x, y, y_swap, n_train=len(ae_train), n_tv=len(ae_tv), n_test=len(ae_test), h=20)
# t=1




ae_train, ae_tv, ae_test, decoder, opt_parameters3l = buildAE(Y_train, Y_tv, Y_test, n=3)
y = np.vstack((pd.DataFrame(ae_tv), pd.DataFrame(ae_test)))
x = np.vstack((X_tv, X_test))
y_swap = np.vstack((Y_tv, Y_test))

AE3_AR10l, AE3_ARX10l, error3AE_10l, errorX3AE_10l = getForecastAE(decoder, x, y, y_swap, n_train=len(ae_train), n_tv=len(ae_tv), n_test=len(ae_test), h=10)
AE3_AR1l, AE3_ARX1l, error3AE_1l, errorX3AE_1l = getForecastAE(decoder, x, y, y_swap, n_train=len(ae_train), n_tv=len(ae_tv), n_test=len(ae_test), h=1)
AE3_AR5l, AE3_ARX5l, error3AE_5l, errorX3AE_5l = getForecastAE(decoder, x, y, y_swap, n_train=len(ae_train), n_tv=len(ae_tv), n_test=len(ae_test), h=5)
AE3_AR20l, AE3_ARX20l, error3AE_20l, errorX3AE_20l = getForecastAE(decoder, x, y, y_swap, n_train=len(ae_train), n_tv=len(ae_tv), n_test=len(ae_test), h=20)
t=1
#
# ae_train, ae_tv, ae_test, decoder, opt_parameters2l = buildAE(Y_train, Y_tv, Y_test, n=2)
# y = np.vstack((pd.DataFrame(ae_tv), pd.DataFrame(ae_test)))
# x = np.vstack((X_tv, X_test))
# y_swap = np.vstack((Y_tv, Y_test))
#
# AE2_AR10l, AE2_ARX10l, error2_10l, errorX2_10l = getForecastAE(decoder, x, y, y_swap, n_train=len(ae_train), n_tv=len(ae_tv), n_test=len(ae_test), h=10)
# AE2_AR1l, AE2_ARX1l, error2_1l, errorX2_1l = getForecastAE(decoder, x, y, y_swap, n_train=len(ae_train), n_tv=len(ae_tv), n_test=len(ae_test), h=1)
# AE2_AR5l, AE2_ARX5l, error2_5l, errorX2_5l = getForecastAE(decoder, x, y, y_swap, n_train=len(ae_train), n_tv=len(ae_tv), n_test=len(ae_test), h=5)
# AE2_AR20l, AE2_ARX20l, error2_20l, errorX2_20l = getForecastAE(decoder, x, y, y_swap, n_train=len(ae_train), n_tv=len(ae_tv), n_test=len(ae_test), h=20)
# t=1
#
# ae_train, ae_tv, ae_test, decoder, opt_parameters4l = buildAE(Y_train, Y_tv, Y_test, n=4)
# y = np.vstack((pd.DataFrame(ae_tv), pd.DataFrame(ae_test)))
# x = np.vstack((X_tv, X_test))
# y_swap = np.vstack((Y_tv, Y_test))
#
# AE4_AR10l, AE4_ARX10l, error4_10l, errorX4_10l = getForecastAE(decoder, x, y, y_swap, n_train=len(ae_train), n_tv=len(ae_tv), n_test=len(ae_test), h=10)
# AE4_AR1l, AE4_ARX1l, error4_1l, errorX4_1l = getForecastAE(decoder, x, y, y_swap, n_train=len(ae_train), n_tv=len(ae_tv), n_test=len(ae_test), h=1)
# AE4_AR5l, AE4_ARX5l, error4_5l, errorX4_5l = getForecastAE(decoder, x, y, y_swap, n_train=len(ae_train), n_tv=len(ae_tv), n_test=len(ae_test), h=5)
# AE4_AR20l, AE4_ARX20l, error4_20l, errorX4_20l = getForecastAE(decoder, x, y, y_swap, n_train=len(ae_train), n_tv=len(ae_tv), n_test=len(ae_test), h=20)
# t=1

#### NARX ###


y_swap = np.vstack((Y_tv_diff, Y_test_diff))
#results10_n3, opt_params10_3, error10_3 = forecastNNARX(X_train_diff, X_tv_diff, X_test_diff, Y_train_diff, Y_tv_diff, Y_test_diff, y_swap,
#                        n_train=len(Y_train_diff), n_tv=len(Y_tv_diff), n_test=len(Y_test_diff), units=3,  h=10)
results10_n20, opt_params10_20, error10_20 = forecastLSTM(X_train_diff, X_tv_diff, X_test_diff, Y_train_diff, Y_tv_diff, Y_test_diff, y_swap,
                        n_train=len(Y_train_diff), n_tv=len(Y_tv_diff), n_test=len(Y_test_diff), units=20,  h=10)
# results10_n50, opt_params10_50, error10_50 = forecastLSTM(X_train_diff, X_tv_diff, X_test_diff, Y_train_diff, Y_tv_diff, Y_test_diff, y_swap,
#                          n_train=len(Y_train_diff), n_tv=len(Y_tv_diff), n_test=len(Y_test_diff), units=50,  h=10)

#results1_n3, opt_params1_3, error1_3 = forecastNNARX(X_train_diff, X_tv_diff, X_test_diff, Y_train_diff, Y_tv_diff, Y_test_diff, y_swap,
#                        n_train=len(Y_train_diff), n_tv=len(Y_tv_diff), n_test=len(Y_test_diff), units=3,  h=1)
results1_n20, opt_params1_20, error1_20 = forecastLSTM(X_train_diff, X_tv_diff, X_test_diff, Y_train_diff, Y_tv_diff, Y_test_diff, y_swap,
                        n_train=len(Y_train_diff), n_tv=len(Y_tv_diff), n_test=len(Y_test_diff), units=20,  h=1)
# results1_n50, opt_params1_50, error1_50 = forecastLSTM(X_train_diff, X_tv_diff, X_test_diff, Y_train_diff, Y_tv_diff, Y_test_diff, y_swap,
#                          n_train=len(Y_train_diff), n_tv=len(Y_tv_diff), n_test=len(Y_test_diff), units=50,  h=1)

#results5_n3, opt_params5_3, error5_3 = forecastNNARX(X_train_diff, X_tv_diff, X_test_diff, Y_train_diff, Y_tv_diff, Y_test_diff, y_swap,
#                        n_train=len(Y_train_diff), n_tv=len(Y_tv_diff), n_test=len(Y_test_diff), units=3,  h=5)
results5_n20, opt_params5_20, error5_20 = forecastLSTM(X_train_diff, X_tv_diff, X_test_diff, Y_train_diff, Y_tv_diff, Y_test_diff, y_swap,
                        n_train=len(Y_train_diff), n_tv=len(Y_tv_diff), n_test=len(Y_test_diff), units=20,  h=5)
# results5_n50, opt_params5_50, error5_50 = forecastLSTM(X_train_diff, X_tv_diff, X_test_diff, Y_train_diff, Y_tv_diff, Y_test_diff, y_swap,
#                          n_train=len(Y_train_diff), n_tv=len(Y_tv_diff), n_test=len(Y_test_diff), units=50,  h=5)

#results20_n3, opt_params20_3, error20_3 = forecastNNARX(X_train_diff, X_tv_diff, X_test_diff, Y_train_diff, Y_tv_diff, Y_test_diff, y_swap,
#                        n_train=len(Y_train_diff), n_tv=len(Y_tv_diff), n_test=len(Y_test_diff), units=3,  h=20)
results20_n20, opt_params20_20, error20_20 = forecastLSTM(X_train_diff, X_tv_diff, X_test_diff, Y_train_diff, Y_tv_diff, Y_test_diff, y_swap,
                        n_train=len(Y_train_diff), n_tv=len(Y_tv_diff), n_test=len(Y_test_diff), units=20,  h=20)
# results20_n50, opt_params20_50, error20_50 = forecastLSTM(X_train_diff, X_tv_diff, X_test_diff, Y_train_diff, Y_tv_diff, Y_test_diff, y_swap,
#                          n_train=len(Y_train_diff), n_tv=len(Y_tv_diff), n_test=len(Y_test_diff), units=50,  h=20)

print("here")
y_swap = np.vstack((Y_tv, Y_test))
#results10L_n3, opt_params10L_3, error10_3l = forecastNNARX(X_train, X_tv, X_test, Y_train, Y_tv, Y_test, y_swap,
#                        n_train=len(Y_train), n_tv=len(Y_tv), n_test=len(Y_test), units=3,  h=10)
results10L_n20, opt_params10L_20, error10_20l = forecastLSTM(X_train, X_tv, X_test, Y_train, Y_tv_diff, Y_test, y_swap,
                       n_train=len(Y_train), n_tv=len(Y_tv), n_test=len(Y_test), units=20,  h=10)
# results10L_n50, opt_params10L_50, error10_50l = forecastLSTM(X_train, X_tv, X_test, Y_train, Y_tv, Y_test, y_swap,
#                          n_train=len(Y_train), n_tv=len(Y_tv),n_test=len(Y_test), units=50,  h=10)

#results1L_n3, opt_params1L_3, error1_3l = forecastNNARX(X_train, X_tv, X_test, Y_train, Y_tv, Y_test, y_swap,
#                        n_train=len(Y_train), n_tv=len(Y_tv), n_test=len(Y_test), units=3,  h=1)
results1L_n20, opt_params1L_20, error1_20l = forecastLSTM(X_train, X_tv, X_test, Y_train, Y_tv, Y_test, y_swap,
                        n_train=len(Y_train), n_tv=len(Y_tv), n_test=len(Y_test), units=20,  h=1)
# results1L_n50, opt_params1L_50, error1_50l = forecastLSTM(X_train, X_tv, X_test, Y_train, Y_tv, Y_test, y_swap,
#                          n_train=len(Y_train), n_tv=len(Y_tv), n_test=len(Y_test), units=50,  h=1)
print("here1")
#results5L_n3, opt_params5L_3, error5_3l = forecastNNARX(X_train, X_tv, X_test, Y_train, Y_tv, Y_test, y_swap,
#                        n_train=len(Y_train), n_tv=len(Y_tv), n_test=len(Y_test), units=3,  h=5)
results5L_n20, opt_params5L_20, error5_20l = forecastLSTM(X_train, X_tv, X_test, Y_train, Y_tv, Y_test, y_swap,
                        n_train=len(Y_train), n_tv=len(Y_tv), n_test=len(Y_test), units=20,  h=5)
# results5L_n50, opt_params5L_50, error5_50l = forecastLSTM(X_train, X_tv, X_test, Y_train, Y_tv, Y_test, y_swap,
#                          n_train=len(Y_train), n_tv=len(Y_tv), n_test=len(Y_test), units=50,  h=5)

#results20L_n3, opt_params20L_3, error20_3l = forecastNNARX(X_train, X_tv, X_test, Y_train, Y_tv, Y_test, y_swap,
#                        n_train=len(Y_train), n_tv=len(Y_tv), n_test=len(Y_test), units=3,  h=20)
results20L_n20, opt_params20L_20, error20_20l = forecastLSTM(X_train, X_tv, X_test, Y_train, Y_tv, Y_test, y_swap,
                        n_train=len(Y_train), n_tv=len(Y_tv), n_test=len(Y_test), units=20,  h=20)
# results20L_n50, opt_params20L_50, error20_50l = forecastLSTM(X_train, X_tv, X_test, Y_train, Y_tv, Y_test, y_swap,
#                          n_train=len(Y_train), n_tv=len(Y_tv), n_test=len(Y_test), units=50,  h=20)
t=1




########## CSSED plots ##########
CSSED_AR5 = getCSSED(error5, error5_diff)

CSSED_AR_ARX_h1 = getCSSED(error1[6], errorX1[6])
CSSED_AR_ARX_h5 = getCSSED(error5[6], errorX5[6])
CSSED_AR_ARX_h10 = getCSSED(error10[6], errorX10[6])
CSSED_AR_ARX_h20 = getCSSED(error20[6], errorX20[6])

CSSED_AR_ARX_diff_h1 = getCSSED(error1_diff[6], errorX1_diff[6])
CSSED_AR_ARX_diff_h5 = getCSSED(error5_diff[6], errorX5_diff[6])
CSSED_AR_ARX_diff_h10 = getCSSED(error10_diff[6], errorX10_diff[6])
CSSED_AR_ARX_diff_h20 = getCSSED(error20_diff[6], errorX20_diff[6])

CSSED_AR_PCA_AR_h1 = getCSSED(error1[6], error3_1l[6])
CSSED_AR_PCA_AR_h5 = getCSSED(error1[6], error3_5l[6])
CSSED_AR_PCA_AR_h10 = getCSSED(error1[6], error3_10l[6])
CSSED_AR_PCA_AR_h20 = getCSSED(error1[6], error3_20l[6])

CSSED_AR_PCA_AR_diff_h1 = getCSSED(error1_diff[6], error3_1[6])
CSSED_AR_PCA_AR_diff_h5 = getCSSED(error1_diff[6], error3_5[6])
CSSED_AR_PCA_AR_diff_h10 = getCSSED(error1_diff[6], error3_10[6])
CSSED_AR_PCA_AR_diff_h20 = getCSSED(error1_diff[6], error3_20[6])

CSSED_AR_PCA_ARX_h1 = getCSSED(error1[6], errorX3_1l[6])
CSSED_AR_PCA_ARX_h5 = getCSSED(error1[6], errorX3_5l[6])
CSSED_AR_PCA_ARX_h10 = getCSSED(error1[6], errorX3_10l[6])
CSSED_AR_PCA_ARX_h20 = getCSSED(error1[6], errorX3_20l[6])

CSSED_AR_PCA_ARX_diff_h1 = getCSSED(error1_diff[6], errorX3_1[6])
CSSED_AR_PCA_ARX_diff_h5 = getCSSED(error1_diff[6], errorX3_5[6])
CSSED_AR_PCA_ARX_diff_h10 = getCSSED(error1_diff[6], errorX3_10[6])
CSSED_AR_PCA_ARX_diff_h20 = getCSSED(error1_diff[6], errorX3_20[6])

CSSED_AE_PCA_AR_h1 = getCSSED(error1[6], error3AE_1l[6])
CSSED_AE_PCA_AR_h5 = getCSSED(error1[6], error3AE_5l[6])
CSSED_AE_PCA_AR_h10 = getCSSED(error1[6], error3AE_10l[6])
CSSED_AE_PCA_AR_h20 = getCSSED(error1[6], error3AE_20l[6])

CSSED_AE_PCA_AR_diff_h1 = getCSSED(error1_diff[6], error3AE_1[6])
CSSED_AE_PCA_AR_diff_h5 = getCSSED(error1_diff[6], error3AE_5[6])
CSSED_AE_PCA_AR_diff_h10 = getCSSED(error1_diff[6], error3AE_10[6])
CSSED_AE_PCA_AR_diff_h20 = getCSSED(error1_diff[6], error3AE_20[6])

CSSED_AE_PCA_ARX_h1 = getCSSED(error1[6], errorX3AE_1l[6])
CSSED_AE_PCA_ARX_h5 = getCSSED(error1[6], errorX3AE_5l[6])
CSSED_AE_PCA_ARX_h10 = getCSSED(error1[6], errorX3AE_10l[6])
CSSED_AE_PCA_ARX_h20 = getCSSED(error1[6], errorX3AE_20l[6])

CSSED_AE_PCA_ARX_diff_h1 = getCSSED(error1_diff[6], errorX3AE_1[6])
CSSED_AE_PCA_ARX_diff_h5 = getCSSED(error1_diff[6], errorX3AE_5[6])
CSSED_AE_PCA_ARX_diff_h10 = getCSSED(error1_diff[6], errorX3AE_10[6])
CSSED_AE_PCA_ARX_diff_h20 = getCSSED(error1_diff[6], errorX3AE_20[6])

CSSED_AR_LSTM_h1 = getCSSED(error1[6], error1_20l['10Y'])
CSSED_AR_LSTM_h5 = getCSSED(error5[6], error5_20l['10Y'])
CSSED_AR_LSTM_h10 = getCSSED(error10[6], error10_20l['10Y'])
CSSED_AR_LSTM_h20 = getCSSED(error20[6], error20_20l['10Y'])

CSSED_AR_LSTM_diff_h1 = getCSSED(error1_diff[6], error1_20['10Y'])
CSSED_AR_LSTM_diff_h5 = getCSSED(error5_diff[6], error5_20['10Y'])
CSSED_AR_LSTM_diff_h10 = getCSSED(error10_diff[6], error10_20['10Y'])
CSSED_AR_LSTM_diff_h20 = getCSSED(error20_diff[6], error20_20['10Y'])


t=1

CSSED_AR_ARX_h1 = getCSSED(error1, errorX1)
CSSED_AR_PCA_AR_h1 = getCSSED(error1, error3_1l)
CSSED_AR_PCA_ARX_h1 = getCSSED(error1, errorX3_1l)
CSSED_AE_PCA_AR_h1 = getCSSED(error1, error3AE_1l)
CSSED_AE_PCA_ARX_h1 = getCSSED(error1, errorX3AE_1l)
CSSED_AR_LSTM_h1 = getCSSED(error1, error1_20l)

df1 = pd.DataFrame([CSSED_AR_ARX_h1,CSSED_AR_PCA_AR_h1, CSSED_AR_PCA_ARX_h1, CSSED_AE_PCA_AR_h1,
                   CSSED_AE_PCA_ARX_h1, CSSED_AR_LSTM_h1]).T
df1.columns = ['ARX', 'PCA-AR', 'PCA-ARX', 'AE-AR', 'AE-ARX', 'LSTM']


df1_diff = pd.DataFrame([CSSED_AR_ARX_diff_h1,CSSED_AR_PCA_AR_diff_h1, CSSED_AR_PCA_ARX_diff_h1, CSSED_AE_PCA_AR_diff_h1,
                   CSSED_AE_PCA_ARX_diff_h1, CSSED_AR_LSTM_diff_h1]).T
df1_diff.columns = ['ARX', 'PCA-AR', 'PCA-ARX', 'AE-AR', 'AE-ARX', 'LSTM']

df5 = pd.DataFrame([CSSED_AR_ARX_h5,CSSED_AR_PCA_AR_h5, CSSED_AR_PCA_ARX_h5, CSSED_AE_PCA_AR_h5,
                   CSSED_AE_PCA_ARX_h5, CSSED_AR_LSTM_h5]).T
df5.columns = ['ARX', 'PCA-AR', 'PCA-ARX', 'AE-AR', 'AE-ARX', 'LSTM']


df5_diff = pd.DataFrame([CSSED_AR_ARX_diff_h5,CSSED_AR_PCA_AR_diff_h5, CSSED_AR_PCA_ARX_diff_h5, CSSED_AE_PCA_AR_diff_h5,
                   CSSED_AE_PCA_ARX_diff_h5, CSSED_AR_LSTM_diff_h5]).T
df5_diff.columns = ['ARX', 'PCA-AR', 'PCA-ARX', 'AE-AR', 'AE-ARX', 'LSTM']

df10= pd.DataFrame([CSSED_AR_ARX_h10,CSSED_AR_PCA_AR_h10, CSSED_AR_PCA_ARX_h10, CSSED_AE_PCA_AR_h10,
                   CSSED_AE_PCA_ARX_h10, CSSED_AR_LSTM_h10]).T
df10.columns = ['ARX', 'PCA-AR', 'PCA-ARX', 'AE-AR', 'AE-ARX', 'LSTM']


df10_diff = pd.DataFrame([CSSED_AR_ARX_diff_h10,CSSED_AR_PCA_AR_diff_h10, CSSED_AR_PCA_ARX_diff_h10, CSSED_AE_PCA_AR_diff_h10,
                   CSSED_AE_PCA_ARX_diff_h10, CSSED_AR_LSTM_diff_h10]).T
df10_diff.columns = ['ARX', 'PCA-AR', 'PCA-ARX', 'AE-AR', 'AE-ARX', 'LSTM']

df20 = pd.DataFrame([CSSED_AR_ARX_h20,CSSED_AR_PCA_AR_h20, CSSED_AR_PCA_ARX_h20, CSSED_AE_PCA_AR_h20,
                   CSSED_AE_PCA_ARX_h20, CSSED_AR_LSTM_h20]).T
df20.columns = ['ARX', 'PCA-AR', 'PCA-ARX', 'AE-AR', 'AE-ARX', 'LSTM']


df20_diff = pd.DataFrame([CSSED_AR_ARX_diff_h20,CSSED_AR_PCA_AR_diff_h20, CSSED_AR_PCA_ARX_diff_h20, CSSED_AE_PCA_AR_diff_h20,
                   CSSED_AE_PCA_ARX_diff_h20, CSSED_AR_LSTM_diff_h20]).T
df20_diff.columns = ['ARX', 'PCA-AR', 'PCA-ARX', 'AE-AR', 'AE-ARX', 'LSTM']