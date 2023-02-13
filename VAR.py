from statsmodels.tsa.vector_ar.var_model import VAR
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

from Drivers import get_data, getDataTablesFigures, optimal_lag, plot_forecast, forecast_accuracy, plot_components, buildAE

import pandas as pd


from keras.layers import Dense
from keras import Model, Input
from keras import Sequential

data, df, df_diff = get_data(lag=5)

X_train = df[0]
X_test = df[1]
Y_train = df[2]
Y_test = df[3]

x_train_diff = df_diff[0]
x_test_diff = df_diff[1]
y_train_diff = df_diff[2]
y_test_diff = df_diff[3]

#getDataTablesFigures(data[0], data[1], data[2]) # input: df_swap, df_drivers, swap_diff

########################################################################################################################
# FORECASTING METHODS
########################################################################################################################


"""
Univariate autoregressive model
"""
def getAR(x_train, y_train, y_test, plot_pred):

    results = []
    for column in y_train:
        print(column)
        # Compute the optimal model; 1 indicating ARX model, AR model otherwise
        idx_UAR, AR = optimal_lag(x_train, y_train[column], 24, 0)
        AR.summary()

        # Forecast out-of-sample
        preds_AR = AR.predict(start=len(y_train), end=len(y_train) + len(y_test) - 1)
        preds_AR.index = y_test.index

        # Plot the prediction vs test data
        if plot_pred:
            plot_forecast(preds_AR, y_test[column])

        # Get forecast accuracy
        acc_AR = forecast_accuracy(preds_AR, y_test[column], df_indicator=0)

        results.append([idx_UAR, acc_AR])

    return results


"""
Univariate autoregressive model with exogenous inputs
"""
def getARX(x_train, x_test, y_train, y_test, plot_pred):
    results = []
    for column in y_train:
        print(column)
        idx_UARX, ARX = optimal_lag(x_train, y_train[column], 24, 1) # Compute the optimal model; 1 indicating ARX model, AR model otherwise
        ARX.summary()

        # Forecast out-of-sample
        preds_ARX = ARX.predict(exog_oos=x_test,start=len(y_train), end=len(y_train)+len(y_test)-1)
        preds_ARX.index = y_test.index

        if plot_pred:
            # Plot the prediction vs test data
            plot_forecast(preds_ARX, y_test[column])

        # Get forecast accuracy
        acc_ARX = forecast_accuracy(preds_ARX, y_test[column], df_indicator=0)

        results.append([idx_UARX, acc_ARX])

    return results

"""
Vector autoregressive model
"""
def getVAR():
    #https://www.analyticsvidhya.com/blog/2021/08/vector-autoregressive-model-in-python/
    # Create model
    model = VAR(y_train)

    # Determine optimal lag
    res = model.select_order(maxlags=15)
    print(res.summary())
    lag_order = res.aic

    # Fit model with optimal lag
    results = model.fit(maxlags=lag_order, ic='aic')
    print(results.summary())

    # Forecast out of sample
    lagged_Values = y_train.values[-lag_order:]
    pred = results.forecast(y=lagged_Values, steps=len(y_test))
    df_forecast = pd.DataFrame(data=pred, index=y_test.index, columns=['1Y', '2Y', '3Y', '4Y','5Y', '7Y', '10Y', '15Y', '20Y', '30Y'])

    # Get forecast accuracy
    forecast_accuracy(df_forecast,y_test,df_indicator=1)


"""
Principal component analysis
"""
def getPCA(x_train, x_test, y_train, y_test):
    # https://www.geeksforgeeks.org/implementing-pca-in-python-with-scikit-learn/?ref=rp

    pca = PCA(n_components=3)

    PC_train = pca.fit_transform(x_train)
    PC_train_df = pd.DataFrame(PC_train, columns=['pc1', 'pc2', 'pc3'])
    PC_test = pca.transform(x_test)

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


    plot_components(PC_train_df, y_train)

    getAR(PC_train_df, PC_test_df, Y_test, plot_pred=0)

"""
Autoencoder factor analysis
"""
def getAE(x_train, x_test, y_train, y_test, plot_pred):
    ae_factors = buildAE(y_train, y_test)

    # Compute the optimal model; 1 indicating ARX model, AR model otherwise
    idx, AR = optimal_lag(ae_factors, y_train, 24, 0)

    AR.summary()

    # Forecast out-of-sample
    preds_AR = AR.predict(start=len(y_train), end=len(y_train) + len(y_test) - 1)
    preds_AR.index = y_test.index

    # Plot the prediction vs test data
    if plot_pred:
        plot_forecast(preds_AR, y_test["10Y"])

    # Get forecast accuracy
    acc_AR = forecast_accuracy(preds_AR, y_test["10Y"], df_indicato=0)

    return idx, acc_AR


########################################################################################################################
# RUN CODE
########################################################################################################################

resultsAR = getAR(X_train, Y_train, Y_test, plot_pred=0)
resultsARX = getARX(X_train, X_test, Y_train, Y_test, plot_pred=0)
getVAR(X_train, X_test, Y_train, Y_test)
#getPCA(X_train, X_test, Y_train, Y_test)
#getAE(plot_pred = 0)
