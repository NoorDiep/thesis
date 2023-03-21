from Drivers import get_data, forecast_accuracy, optimal_lag, getDataTablesFigures
from statsmodels.tsa.arima.model import ARIMA
from sklearn.decomposition import PCA
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

########################################################################################################################
# RUN CODE
########################################################################################################################

###### Results with differences ######
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


    w = n_train

    f_i = []
    f_iX = []
    p_ARi = []
    p_ARXi = []
    q_ARXi = []
    resultsAR = []
    resultsARX = []



    for i in range(y.shape[1]):

        p_AR1 = optimal_lag(x_train=0, x_tv=0, y_train=y[:n_train,i], y_tv=y[:n_tv,i], maxlags_p=15, maxlags_q=10, indicator=0)
        p_ARX1, q_ARX1 = optimal_lag(x_train=x[:n_train,i], x_tv=x[:n_tv,i], y_train=y[:n_train,i], y_tv=y[:n_tv,i], maxlags_p=15, maxlags_q=10, indicator=1)

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
        f_j = []
        for j in range(y.shape[1]):
            train = y[k:w + k, j]

            AR = ARIMA(train, order=(p_ARi[j], 0, 0)).fit()

            preds_PC = AR.predict(start=n_tv + k, end=n_tv+ k + h - 1, dynamic=False)  # Dynamic equal to False means direct forecasts
            f_j.append(preds_PC)
        preds_PC = AR.predict(start=n_tv + k, end=n_tv + k + h - 1, dynamic=False)

        # Forecast out-of-sample
        #preds_PC = AR.predict(start=n_tv + k, end=n_tv + k + h - 1, dynamic=False)  # Dynamic equal to False means direct forecasts
        preds_AR = pd.DataFrame(f_j)
        #preds_AR = pd.DataFrame(preds_PC)
        preds_AR_s = np.dot(preds_AR.T, PCs) + means[1]
        preds_AR_s = pd.DataFrame(preds_AR_s)

        test = y_swap[n_tv+k:n_tv+k + h]
        test = pd.DataFrame(test)

        acc_AR = forecast_accuracy(preds_AR_s[0], test[0], df_indicator=0)
        f_k.append(acc_AR)

        # Forecast out-of-sample
        #preds_PC = ARX.predict(start=n_tv+k, end=n_tv+k + h - 1, dynamic=False, exog=x[k-w:k+h-w+q_ARX1+k])

        f_jX = []
        for j in range(y.shape[1]):
            ytrain = y[k + q_ARXi[j]:w + k, j]
            xtrain = pd.DataFrame(x[k:w + k]).shift(q_ARXi[j]).dropna()

            # Construct the model with optimized parameters
            ARX = ARIMA(ytrain, exog=xtrain, order=(p_ARi[j], 0, 0)).fit()
            # Forecast out-of-sample
            preds_PC = ARX.predict(start=n_tv + k, end=n_tv + k + h - 1, dynamic=False, exog=x[k - w:k + h - w + q_ARXi[j] + k])  # Dynamic equal to False means direct forecasts
            f_jX.append(preds_PC)

        preds_ARX = pd.DataFrame(f_jX)
        #preds_PC = ARX.predict(start=n_tv + k, end=n_tv + k + h - 1, dynamic=False,
        #                       exog=x[k - w:k + h - w + k])  # Dynamic equal to False means direct forecasts
        #preds_ARX = pd.DataFrame(preds_PC)
        preds_ARX_s = np.dot(preds_ARX.T, PCs) + means[1]
        preds_ARX_s = pd.DataFrame(preds_ARX_s)

        test = y_swap[n_tv + k:n_tv + k + h]
        test = pd.DataFrame(test)


        # Get forecast accuracy
        acc_ARX = forecast_accuracy(preds_ARX_s[0], test[0], df_indicator=0)
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

# PCA1_AR10, PCA1_ARX10 = getForecastPCA(mu1, var_expl1, x, y1, y_swap, n_train=len(PC1_train), n_tv=len(PC1_tv), n_test=len(PC1_test), h=10)
# PCA2_AR10, PCA2_ARX10 = getForecastPCA(mu2, var_expl2, x, y2, y_swap, n_train=len(PC2_train), n_tv=len(PC2_tv), n_test=len(PC2_test), h=10)
# PCA3_AR10, PCA3_ARX10 = getForecastPCA(mu3, var_expl3, x, y3, y_swap, n_train=len(PC3_train), n_tv=len(PC3_tv), n_test=len(PC3_test), h=10)
# PCA4_AR10, PCA4_ARX10 = getForecastPCA(mu4, var_expl4, x, y4, y_swap, n_train=len(PC4_train), n_tv=len(PC4_tv), n_test=len(PC4_test), h=10)
# PCA5_AR10, PCA5_ARX10 = getForecastPCA(mu5, var_expl5, x, y5, y_swap, n_train=len(PC5_train), n_tv=len(PC5_tv), n_test=len(PC5_test), h=10)


#PCA1_AR1, PCA1_ARX1 = getForecastPCA(mu1, var_expl1, x, y1, y_swap, n_train=len(PC1_train), n_tv=len(PC1_tv), n_test=len(PC1_test), h=1)
PCA2_AR1, PCA2_ARX1 = getForecastPCA(mu2, var_expl2, x, y2, y_swap, n_train=len(PC2_train), n_tv=len(PC2_tv), n_test=len(PC2_test), h=1)
PCA3_AR1, PCA3_ARX1 = getForecastPCA(mu3, var_expl3, x, y3, y_swap, n_train=len(PC3_train), n_tv=len(PC3_tv), n_test=len(PC3_test), h=1)
PCA4_AR1, PCA4_ARX1 = getForecastPCA(mu4, var_expl4, x, y4, y_swap, n_train=len(PC4_train), n_tv=len(PC4_tv), n_test=len(PC4_test), h=1)
#PCA5_AR1, PCA5_ARX1 = getForecastPCA(mu5, var_expl5, x, y5, y_swap, n_train=len(PC5_train), n_tv=len(PC5_tv), n_test=len(PC5_test), h=1)

#PCA1_AR5, PCA1_ARX5 = getForecastPCA(mu1, var_expl1, x, y1, y_swap, n_train=len(PC1_train), n_tv=len(PC1_tv), n_test=len(PC1_test), h=5)
PCA2_AR5, PCA2_ARX5 = getForecastPCA(mu2, var_expl2, x, y2, y_swap, n_train=len(PC2_train), n_tv=len(PC2_tv), n_test=len(PC2_test), h=5)
PCA3_AR5, PCA3_ARX5 = getForecastPCA(mu3, var_expl3, x, y3, y_swap, n_train=len(PC3_train), n_tv=len(PC3_tv), n_test=len(PC3_test), h=5)
PCA4_AR5, PCA4_ARX5 = getForecastPCA(mu4, var_expl4, x, y4, y_swap, n_train=len(PC4_train), n_tv=len(PC4_tv), n_test=len(PC4_test), h=5)
PCA5_AR5, PCA5_ARX5 = getForecastPCA(mu5, var_expl5, x, y5, y_swap, n_train=len(PC5_train), n_tv=len(PC5_tv), n_test=len(PC5_test), h=5)

t=1