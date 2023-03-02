from statsmodels.tsa.vector_ar.var_model import VAR
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import numpy as np

from Drivers import get_data, getDataTablesFigures, optimal_lag, plot_forecast, forecast_accuracy, plot_components, buildAE

import pandas as pd


from keras.layers import Dense
from keras import Model, Input
from keras import Sequential

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

# combine the training and validation sets
Y_tv = np.vstack((Y_train, Y_val))
Y_tv_diff = np.vstack((Y_train_diff, Y_val_diff))


#getDataTablesFigures(data[0], data[1], data[2]) # input: df_swap, df_drivers, swap_diff

########################################################################################################################
# FORECASTING METHODS
########################################################################################################################


"""
Univariate autoregressive model
"""


def getAR(x_train, y_train, y_tv, y_test, plot_pred, dates):
    # Strip indices of dataframes
    y_train1 = y_train.reset_index(drop=True)
    dep = y_train1.T.reset_index(drop=True).T



    results = []
    for i in range(len(y_train)):
        print(i)
        # Compute the optimal model; 1 indicating ARX model, AR model otherwise
        idx_UAR, AR = optimal_lag(x_train=0, x_tv=0, y_train= dep[i],y_tv= y_tv[i], maxlags=10, indicator=0)
        AR.summary()

        # Forecast out-of-sample
        preds_AR = AR.predict(start=len(y_train), end=len(y_train) + len(y_test) - 1)
        preds_AR = pd.DataFrame(preds_AR)
        preds_AR.index = dates
        y_test = y_test.set_index(dates)

        # Plot the prediction vs test data
        if plot_pred:
            plot_forecast(preds_AR, y_test[i])

        # Get forecast accuracy
        acc_AR = forecast_accuracy(preds_AR, y_test[i], df_indicator=0)

        results.append([idx_UAR, acc_AR])
        print(acc_AR)

    return results


"""
Univariate autoregressive model with exogenous inputs
"""
def getARX(x_train, x_test, y_train, y_test, plot_pred, dates):
    results = []
    for column in y_train:
        print(column)
        idx_UARX, ARX = optimal_lag(x_train, y_train[column], 10, 1) # Compute the optimal model; 1 indicating ARX model, AR model otherwise
        ARX.summary()

        # Forecast out-of-sample
        preds_ARX = ARX.predict(exog_oos=x_test,start=len(y_train), end=len(y_train)+len(y_test)-1)
        preds_ARX.index = dates
        y_test = y_test.set_index(dates)
        if plot_pred:
            # Plot the prediction vs test data
            plot_forecast(preds_ARX, y_test[column])

        # Get forecast accuracy
        acc_ARX = forecast_accuracy(preds_ARX, y_test[column], df_indicator=0)

        results.append([idx_UARX, acc_ARX])
        print(acc_ARX)

    return results

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
Principal component analysis
"""
def getPCA(x_train, x_test, y_train, y_test, dates, forecast_method):
    # https://www.geeksforgeeks.org/implementing-pca-in-python-with-scikit-learn/?ref=rp

    pca = PCA(n_components=3)

    PC_train = pca.fit_transform(y_train)
    PC_train_df = pd.DataFrame(PC_train, columns=['pc1', 'pc2', 'pc3'])
    PC_test = pca.transform(y_test)
    PC_test_df = pd.DataFrame(PC_test, columns=['pc1', 'pc2', 'pc3'])

    # pca = PCA(n_components=3)
    # PC = pca.fit_transform(x)
    # principalDF = pd.DataFrame(data=PC, columns=['pc1', 'pc2', 'pc3'])
    # finalDf = pd.concat([principalDF, df[['Attrition_Flag']]], axis=1)
    # Xfinal = finalDf[['pc1', 'pc2', 'pc3']]
    # yfinal = finalDf['Attrition_Flag']
    # X_train, X_test, y_train, y_test = train_test_split(Xfinal, yfinal, test_size=0.3)
    # logistic = LogisticRegression()
    # logistic.fit(X=X_train, y=y_train)
    # logistic.predict(X_test)
    # score_3 = logistic.score(X_test, y_test)

    explained_variance = pca.explained_variance_ratio_


    #plot_components(PC_train_df, y_train)

    if forecast_method == 'AR':
        print("PCA_AR forecast")
        getAR(PC_train_df, y_train, y_test, plot_pred=0, dates=dates)
    else:
        print("PCA_ARX forecast")
        getARX(x_train, x_test, PC_train_df, PC_test_df, plot_pred=0, dates=dates)

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
resultsAR = getAR(X_train, Y_train, Y_tv, Y_test, plot_pred=0, dates=date_test)
resultsAR_diff = getAR(X_train_diff, Y_train_diff, Y_tv_diff, Y_test_diff, plot_pred=0, dates=date_test_diff)
#resultsARX = getARX(X_train, X_test, Y_train, Y_test, plot_pred=1, dates=date_test)
# getVAR(X_train, X_test, Y_train, Y_test, plot_pred=0, column='30Y', dates=date_test)
# getVARX(X_train, X_test, Y_train, Y_test, plot_pred=0, column='30Y', dates=date_test)
#resultsPCA_AR = getPCA(X_train, X_test, Y_train, Y_test, dates=date_test, forecast_method='AR')
#resultsPCA_ARX = getPCA(X_train, X_test, Y_train, Y_test, dates=date_test, forecast_method='ARX')
#resultsAE_AR = getAE(X_train, X_test, Y_train, Y_test, plot_pred=0, dates=date_test, forecast_method='AR')
#resultsAE_ARX = getAE(X_train, X_test, Y_train, Y_test, plot_pred=0, dates=date_test, forecast_method='ARX')


###### Results with differences ######
#resultsAR_diff = getAR(X_train_diff, Y_train_diff, Y_test_diff, plot_pred=0, dates=date_test_diff)
#resultsARX_diff = getARX(X_train_diff, X_test_diff, Y_train_diff, Y_test_diff, plot_pred=0, dates=date_test_diff)
#resultsVAR_diff = getVAR(X_train_diff, X_test_diff, Y_train_diff, Y_test_diff, plot_pred=0, column='30Y', dates=date_test_diff)
#resultsVARX_diff = getVARX(X_train_diff, X_test_diff, Y_train_diff, Y_test_diff, plot_pred=0, column='30Y', dates=date_test_diff)
t=1