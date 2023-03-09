from statsmodels.tsa.vector_ar.var_model import VAR
from sklearn.decomposition import PCA
import numpy as np
from statsmodels.tsa.arima.model import ARIMA

from Drivers import get_data, getDataTablesFigures, optimal_lag, plot_forecast, forecast_accuracy, buildAE

import pandas as pd


# Import data into 60% training, 20% validation and 20% test sets
data, df, df_diff, date_train, date_test, date_train_diff, date_test_diff = get_data(lag=5)


X_train = df[0]
X_val = df[1]
X_test = df[2]
Y_train = df[3]
Y_val = df[4]
Y_test = df[5]


X_train_diff  = df_diff [0]
X_val_diff  = df_diff [1]
X_test_diff  = df_diff [2]
Y_train_diff  = df_diff [3]
Y_val_diff  = df_diff [4]
Y_test_diff  = df_diff [5]

# Combine the training and validation sets
Y_tv = np.vstack((Y_train, Y_val))
Y_tv_diff = np.vstack((Y_train_diff, Y_val_diff))
Y_diff = np.vstack((Y_tv_diff,Y_test_diff))

X_tv = np.vstack((X_train, X_val))
X_tv_diff = np.vstack((X_train_diff, X_val_diff))

# Return tables and figures for Data section
#getDataTablesFigures(data[0], data[1], pd.DataFrame(Y_diff)) # input: df_swap, df_drivers, swap_diff

########################################################################################################################
# FORECASTING METHODS
########################################################################################################################


"""
Univariate autoregressive model AR
"""
def getAR(x_train, y_train, y_tv, y_test, h, plot_pred, dates):
    # Strip indices of dataframes
    try:
        y_train1 = y_train.reset_index(drop=True)
        y_test = y_test.T.reset_index(drop=True).T
    except:
        pass
    # Stack all data sets to one series
    y = np.vstack((pd.DataFrame(y_tv), pd.DataFrame(y_test)))

    # Define variables
    n_train = y_train.shape[0]
    n_tv = y_tv.shape[0]
    n = y_test.shape[0]
    w = y_tv.shape[0]
    f_measures = []
    means = []
    results = []
    idx_p =[]

    # Get the number of columns
    try:
        cols = y_train1.shape[1]
    except:
        #
        cols = 1

    # Loop over each column
    for i in range(cols):
        print(i)
        f_i = []
        # Compute the optimal model; 1 indicating ARX model, AR model otherwise
        idx_UAR = optimal_lag(x_train=0, x_tv=0, y_train=y[:n_train,i], y_tv=y[:n_tv,i], maxlags_p=10, maxlags_q=10, indicator=0)

        # Perform rolling window forecasts
        for k in range(n - h + 1):
            print(k)
            train = y[k:w+k, i]

            AR = ARIMA(train, order=(idx_UAR, 0, 0)).fit()

            # Forecast out-of-sample
            preds_AR = AR.predict(start=len(y_tv)+k, end=len(y_tv)+k + h - 1, dynamic=False) # Dynamic equal to False means direct forecasts
            preds_AR = pd.DataFrame(preds_AR)
            preds_AR.index = dates.iloc[k:k+h]
            test = y[len(y_tv)+k:len(y_tv)+k + h, i]
            test = pd.DataFrame(test)
            test.index = dates.iloc[k:k + h]

            # Plot the prediction vs test data
            if plot_pred:
                plot_forecast(preds_AR, test)

            # Get forecast accuracy
            acc_AR = forecast_accuracy(preds_AR, test, df_indicator=0)
            f_i.append(acc_AR)
            results.append(preds_AR)
        f_i = pd.DataFrame(np.concatenate(f_i), columns=['MEA', 'MSE', 'RMSE'])
        f_measures.append(f_i)
        means.append(np.mean(f_i))
        idx_p.append(idx_UAR)

    return f_measures, means, results, idx_p


"""
Univariate autoregressive model with exogenous inputs ARX
"""
def getARX(x_train, x_tv, x_test, y_train, y_tv, y_test, h, plot_pred, dates):
    # Strip indices of dataframes
    try:
        y_train1 = y_train.reset_index(drop=True)
    except:
        pass
    # Stack all data sets to one series
    x = np.vstack((x_tv, x_test))
    y = np.vstack((pd.DataFrame(y_tv), pd.DataFrame(y_test)))

    # Define variables
    n_train = y_train.shape[0]
    n_tv = y_tv.shape[0]
    n = y_test.shape[0]
    w = y_tv.shape[0]
    f_measures = []
    means = []
    results = []
    idx_p_opt = []
    idx_q_opt = []

    # Get the number of columns
    try:
        cols = y_train1.shape[1]
    except:
        cols = 1

    for i in range(cols):
        f_i = []
        # Compute the optimal model; 1 indicating ARX model, AR model otherwise
        idx_p, idx_q = optimal_lag(x_train=x[:n_train], x_tv=x[:n_tv], y_train=y[:n_train,i], y_tv=y[:n_tv,i], maxlags_p=10, maxlags_q=10, indicator=1)
        idx_p_opt.append(idx_p)
        idx_q_opt.append(idx_q)
        print("optimal p is", idx_p, " optimal q is ", idx_q)
        for k in range(n - h):
            print(i, k)
            # Select rolling window for each out-of-sample period k
            ytrain = y[k+idx_q:w + k, i]
            xtrain = pd.DataFrame(x[k:w + k]).shift(idx_q).dropna()

            # Construct the model with optimized parameters
            ARX = ARIMA(ytrain, exog=xtrain,  order=(idx_p, 0, 0)).fit()

            # Forecast out-of-sample
            preds_ARX = ARX.predict(start=len(y_tv)+k, end=len(y_tv)+k + h - 1, dynamic=False, exog=x[k-w:k+h-w+idx_q+k]) # Dynamic equal to False means direct forecasts
            preds_ARX = pd.DataFrame(preds_ARX)
            preds_ARX.index = dates.iloc[k:k+h]
            test = y[len(y_tv)+k:len(y_tv)+k + h, i]
            test = pd.DataFrame(test)
            test.index = dates.iloc[k:k + h]

            # Plot the prediction vs test data
            if plot_pred:
                plot_forecast(preds_ARX, test)

            # Get forecast accuracy
            acc_AR = forecast_accuracy(np.array(preds_ARX), np.array(test), df_indicator=0)
            f_i.append(acc_AR)
            results.append(preds_ARX)
        f_i = pd.DataFrame(np.concatenate(f_i), columns=['MEA', 'MSE', 'RMSE'])
        f_measures.append(f_i)
        means.append(np.mean(f_i))
    return f_measures, means, results, idx_p_opt, idx_q_opt


"""
Vector autoregressive model
"""
def getVAR(x_train, x_test, y_train, y_test, plot_pred, column, dates):
    #https://www.analyticsvidhya.com/blog/2021/08/vector-autoregressive-model-in-python/
    # Create model
    model = VAR(endog=y_train)

    # Determine optimal lag
    res = model.select_order(maxlags=10)
    print(res.summary())
    lag_order = res.bic
    #lag_order = 5

    # Fit model with optimal lag
    results = model.fit(maxlags=lag_order, ic='bic', verbose=1)
    print('Optimal lag is: ', lag_order)
    #print(results.summary())

    # Forecast out of sample
    lagged_Values = y_train.values[-lag_order:]
    pred = results.forecast(y=lagged_Values, steps=len(y_test))
    df_forecast = pd.DataFrame(data=pred, index=dates, columns=['1Y', '2Y', '3Y', '4Y','5Y', '7Y', '10Y', '15Y', '20Y', '30Y'])
    y_test = y_test.set_index(dates)
    if plot_pred:
        plot_forecast(df_forecast[column], y_test[column])
    # Get forecast accuracy
    print('VAR results: ')
    forecast_accuracy(df_forecast,y_test,df_indicator=1)

"""
Vector autoregressive model with exogenous inputs
"""
def getVARX(x_train, x_test, y_train, y_test, plot_pred, column, dates):
    #https://www.analyticsvidhya.com/blog/2021/08/vector-autoregressive-model-in-python/
    # Create model
    x_train = x_train.reset_index(drop=True)
    model = VAR(endog=y_train, exog=x_train)

    # Determine optimal lag
    res = model.select_order(maxlags=10)
    print(res.summary())
    lag_order = res.bic
    #lag_order = 5

    # Fit model with optimal lag
    results = model.fit(maxlags=lag_order, ic='bic', verbose=1)
    print('Optimal lag is: ', lag_order)
    #print(results.summary())

    # Forecast out of sample
    lagged_Values = y_train.values[-lag_order:]
    pred = results.forecast(y=lagged_Values, steps=len(y_test), exog_future=x_test)
    df_forecast = pd.DataFrame(data=pred, index=dates, columns=['1Y', '2Y', '3Y', '4Y','5Y', '7Y', '10Y', '15Y', '20Y', '30Y'])
    y_test = y_test.set_index(dates)
    if plot_pred:
        plot_forecast(df_forecast[column], y_test[column])
    # Get forecast accuracy
    print('VARX results: ')
    forecast_accuracy(df_forecast,y_test,df_indicator=1)

"""
Principal component analysis AR
"""
def getARPCA(means, PCs, y_train, y_tv, y_test, y_s, h, plot_pred, dates):

    # Stack all data sets to one series
    y = np.vstack((pd.DataFrame(y_tv), pd.DataFrame(y_test)))

    # Define variables
    n_train = y_train.shape[0]
    n_tv = y_tv.shape[0]
    n = y_test.shape[0]
    w = y_tv.shape[0]
    f_measures = []
    results = []
    idx_p =[]
    mean = []

    # Get the number of columns
    try:
        cols = y_train.shape[1]
    except:
        #
        cols = 1

    # Loop over each column
    for i in range(cols):
        print(i)
        f_mae = []
        f_mse = []
        f_rmse = []
        f_i = []
        # Compute the optimal model; 1 indicating ARX model, AR model otherwise
        idx = optimal_lag(x_train=0, x_tv=0, y_train=y[:n_train,i], y_tv=y[:n_tv,i], maxlags_p=10, maxlags_q=10, indicator=0)

        # Perform rolling window forecasts
        for k in range(n - h + 1):
            print(k)
            f_k =[]
            for i in range(cols):
                train = y[k:w+k, i]

                AR = ARIMA(train, order=(idx, 0, 0)).fit()

                # Forecast out-of-sample
                preds_PC = AR.predict(start=len(y_tv)+k, end=len(y_tv)+k + h - 1, dynamic=False) # Dynamic equal to False means direct forecasts
                f_k.append(preds_PC)
            preds_AR = pd.DataFrame(f_k)
            preds_AR_s = np.dot(preds_AR.T, PCs) + means[0]
            preds_AR_s= pd.DataFrame(preds_AR_s)
            preds_AR_s.index = dates.iloc[k:k+h]
            test = y_s[len(y_tv)+k:len(y_tv)+k + h, ]
            test = pd.DataFrame(test)
            test.index = dates.iloc[k:k + h]

            # Plot the prediction vs test data
            if plot_pred:
                plot_forecast(preds_AR_s, test)

            # Get forecast accuracy
            acc_AR = forecast_accuracy(preds_AR_s, test, df_indicator=0)
            f_mae.append(np.mean(acc_AR.iloc[0]['mae']))
            f_mse.append(np.mean(acc_AR.iloc[0]['mse']))
            f_rmse.append(np.mean(acc_AR.iloc[0]['rmse']))
            results.append(preds_AR_s)
            fmsr = [np.mean(f_mae), np.mean(f_mse), np.mean(f_rmse)]
            f_i.append(pd.DataFrame(fmsr).T)
            t=1
        f = pd.DataFrame(np.concatenate(f_i), columns=['MEA', 'MSE', 'RMSE'])
        f_measures.append(f)
        mean.append(np.mean(f))
        idx_p.append(idx)

    return f_measures, mean, results, idx_p
"""
Principal component analysis ARX
"""
def getARXPCA(means, PCs, y_train, y_tv, y_test, y_s, h, plot_pred, dates):

    # Stack all data sets to one series
    y = np.vstack((pd.DataFrame(y_tv), pd.DataFrame(y_test)))

    # Define variables
    n_train = y_train.shape[0]
    n_tv = y_tv.shape[0]
    n = y_test.shape[0]
    w = y_tv.shape[0]
    f_measures = []
    results = []
    idx_p =[]
    mean = []

    # Get the number of columns
    try:
        cols = y_train.shape[1]
    except:
        #
        cols = 1

    # Loop over each column
    for i in range(cols):
        print(i)
        f_mae = []
        f_mse = []
        f_rmse = []
        f_i = []
        # Compute the optimal model; 1 indicating ARX model, AR model otherwise
        idx = optimal_lag(x_train=0, x_tv=0, y_train=y[:n_train,i], y_tv=y[:n_tv,i], maxlags_p=10, maxlags_q=10, indicator=0)

        # Perform rolling window forecasts
        for k in range(n - h + 1):
            print(k)
            f_k =[]
            for i in range(cols):
                train = y[k:w+k, i]

                AR = ARIMA(train, order=(idx, 0, 0)).fit()

                # Forecast out-of-sample
                preds_PC = AR.predict(start=len(y_tv)+k, end=len(y_tv)+k + h - 1, dynamic=False) # Dynamic equal to False means direct forecasts
                f_k.append(preds_PC)
            preds_AR = pd.DataFrame(f_k)
            preds_AR_s = np.dot(preds_AR.T, PCs) + means[0]
            preds_AR_s= pd.DataFrame(preds_AR_s)
            preds_AR_s.index = dates.iloc[k:k+h]
            test = y_s[len(y_tv)+k:len(y_tv)+k + h, ]
            test = pd.DataFrame(test)
            test.index = dates.iloc[k:k + h]

            # Plot the prediction vs test data
            if plot_pred:
                plot_forecast(preds_AR_s, test)

            # Get forecast accuracy
            acc_AR = forecast_accuracy(preds_AR_s, test, df_indicator=0)
            f_mae.append(np.mean(acc_AR.iloc[0]['mae']))
            f_mse.append(np.mean(acc_AR.iloc[0]['mse']))
            f_rmse.append(np.mean(acc_AR.iloc[0]['rmse']))
            results.append(preds_AR_s)
            fmsr = [np.mean(f_mae), np.mean(f_mse), np.mean(f_rmse)]
            f_i.append(pd.DataFrame(fmsr).T)
            t=1
        f = pd.DataFrame(np.concatenate(f_i), columns=['MEA', 'MSE', 'RMSE'])
        f_measures.append(f)
        mean.append(np.mean(f))
        idx_p.append(idx)

    return f_measures, mean, results, idx_p

"""
Principal component analysis
"""
def getPCA(x_train, y_train, y_tv, y_test, n_max, method):
    # https://www.geeksforgeeks.org/implementing-pca-in-python-with-scikit-learn/?ref=rp

    pca = PCA()
    pca.fit(y_train)

    # Calculate the explained variance for each component
    variance = pca.explained_variance_ratio_
    # Calculate the cumulative explained variance
    cumulative_variance = np.cumsum(variance)
    # Determine the number of components needed to explain a given amount of variance
    n = np.argmax(cumulative_variance >= 0.95) + 1
    scree_df = []
    f_total = []
    p_total = []
    q_total = []
    means = []
    y = np.vstack((pd.DataFrame(y_tv), pd.DataFrame(y_test)))
    for n in range(1,n_max+1): #range(1,n_max+1):
        print("Component: ", n)
        pca = PCA(n_components=n)

        PC_train = pca.fit_transform(y_tv)
        y_trainPC = np.dot(PC_train, pca.components_) + np.array(np.mean(y_train, axis=0))

        names_pcas = [f"PCA Component {i}" for i in range(1, n+1, 1)]
        scree = pd.DataFrame(list(zip(names_pcas, pca.explained_variance_ratio_)), columns=["Component", "Explained Variance Ratio"])
        scree_df.append(scree)
        print(scree)


        PC_tv = pca.transform(y_test)
        y_tvPC = np.dot(PC_tv, pca.components_) + np.array(np.mean(y_tv, axis=0))
        PC_tv_df = pd.DataFrame(PC_tv)
        PC_test = pca.transform(y_test)
        y_testPC = np.dot(PC_test, pca.components_) + np.array(np.mean(y_test, axis=0))
        PC_test_df = pd.DataFrame(PC_test)


        means.append(np.array(np.mean(y_train, axis=0)))
        means.append(np.array(np.mean(y_tv, axis=0)))
        means.append(np.array(np.mean(y_test, axis=0)))

        if method == "AR":
            f, f_mean, results, p = getARPCA(means,  pca.components_, PC_train, PC_tv, PC_test, y, h=10, plot_pred=0, dates=date_test)
            f_total.append(f_mean)
            p_total.append(p)
            q_total = None
            t=1
        else:
            f, f_mean, results, p, q = getARX(x_train, X_tv, X_test, PC_train, PC_tv, PC_test, h=10, plot_pred=0, dates=date_test)
            f_total.append(f_mean)
            p_total.append(p)
            q_total.append(q)
    t=1
    return f_total, p_total, q_total

"""
Autoencoder factor analysis
"""
def getAE(x_train, x_test, y_train, y_test, plot_pred, dates, forecast_method):
    ae_train, ae_test = buildAE(y_train, y_test)

    if forecast_method == 'AR':
        print("AE_AR forecast")
        getAR(ae_train, y_train, y_test, plot_pred=0, dates=dates)
    else:
        print("AE_ARX forecast")
        getARX(x_train, x_test, ae_train, ae_test, plot_pred=0, dates=dates)

    # # Compute the optimal model; 1 indicating ARX model, AR model otherwise
    # idx, AR = optimal_lag(ae_train, y_train, 24, 0)
    #
    # AR.summary()
    #
    # # Forecast out-of-sample
    # preds_AR = AR.predict(start=len(y_train), end=len(y_train) + len(y_test) - 1)
    # preds_AR.index = y_test.index
    #
    # # Plot the prediction vs test data
    # if plot_pred:
    #     plot_forecast(preds_AR, y_test["10Y"])
    #
    # # Get forecast accuracy
    # acc_AR = forecast_accuracy(preds_AR, y_test["10Y"], df_indicato=0)

    return


########################################################################################################################
# RUN CODE
########################################################################################################################

###### Results with levels ######
f_ARh1, f_ARh1_mean, resultsARh1, idx_ph1 = getAR(X_train, Y_train, Y_tv, Y_test, h=1, plot_pred=0, dates=date_test)
print('AR', f_ARh1, f_ARh1_mean)
f_ARh1_d, f_ARh1_d_mean, resultsARh1_d, idx_ph1_d = getAR(X_train_diff, Y_train_diff, Y_tv_diff, Y_test_diff, h=1, plot_pred=0, dates=date_test_diff)
print('AR_d', f_ARh1_d, f_ARh1_d_mean)
f_ARh5, f_ARh5_mean, resultsARh5, idx_ph5 = getAR(X_train, Y_train, Y_tv, Y_test, h=5, plot_pred=0, dates=date_test)
print('AR', f_ARh5, f_ARh5_mean)
f_ARh5_d, f_ARh5_d_mean, resultsARh5_d, idx_ph5_d = getAR(X_train_diff, Y_train_diff, Y_tv_diff, Y_test_diff, h=5, plot_pred=0, dates=date_test_diff)
print('AR_d', f_ARh5_d, f_ARh5_d_mean)
# f_ARX, f_ARX_mean, resultsARX, p, q = getARX(X_train, X_tv, X_test, Y_train, Y_tv, Y_test, h=10, plot_pred=0, dates=date_test)
# print('ARX', f_ARX, f_ARX_mean)
# f_ARX_d, f_ARX_d_mean, resultsARX_d, p_d, q_d = getARX(X_train_diff, X_tv_diff, X_test_diff, Y_train_diff, Y_tv_diff, Y_test_diff, h=10, plot_pred=0, dates=date_test)
# print('ARX_d', f_ARX_d, f_ARX_d_mean)
# t=1
# getVAR(X_train, X_test, Y_train, Y_test, plot_pred=0, column='30Y', dates=date_test)
# getVARX(X_train, X_test, Y_train, Y_test, plot_pred=0, column='30Y', dates=date_test)
f_PCA_mean, optPPCA, optQPCA = getPCA(X_train, Y_train, Y_tv, Y_test, n_max=5,  method='AR')
f_PCA_mean, optPPCA, optQPCA = getPCA(X_train, Y_train_diff, Y_tv_diff, Y_test_diff, n_max=5,  method='AR')
t=1
#resultsPCA_ARX = getPCA(X_train, X_test, Y_train, Y_test, dates=date_test, forecast_method='ARX')
#resultsAE_AR = getAE(X_train, X_test, Y_train, Y_test, plot_pred=0, dates=date_test, forecast_method='AR')
#resultsAE_ARX = getAE(X_train, X_test, Y_train, Y_test, plot_pred=0, dates=date_test, forecast_method='ARX')


###### Results with differences ######
#resultsAR_diff = getAR(X_train_diff, Y_train_diff, Y_test_diff, plot_pred=0, dates=date_test_diff)
#resultsARX_diff = getARX(X_train_diff, X_test_diff, Y_train_diff, Y_test_diff, plot_pred=0, dates=date_test_diff)
#resultsVAR_diff = getVAR(X_train_diff, X_test_diff, Y_train_diff, Y_test_diff, plot_pred=0, column='30Y', dates=date_test_diff)
#resultsVARX_diff = getVARX(X_train_diff, X_test_diff, Y_train_diff, Y_test_diff, plot_pred=0, column='30Y', dates=date_test_diff)
t=1