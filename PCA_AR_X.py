from sklearn.decomposition import PCA
from statsmodels.tsa.ar_model import AutoReg

from Drivers import get_data, forecast_accuracy, optimal_lag, getDataTablesFigures
from statsmodels.tsa.arima.model import ARIMA
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
            preds_AR = AR.predict(start=w+k, end=w+k+h-1, dynamic=True)
            y_hatAR = preds_AR[h-1]
            f_k.append(y_hatAR)

            # Forecast out-of-sample
            preds_ARX = ARX.predict(start=w+k, end=w+k+h-1, exog=x[:w+k+h-1], exog_oos=x[w:k+h+w+k],
                                    dynamic=True)
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

PC1_train, PC1_tv, PC1_test, mu1, var_expl1, scree1 = getPC(Y_train_diff, Y_tv_diff, Y_test_diff,1)
PC2_train, PC2_tv, PC2_test, mu2, var_expl2, scree2 = getPC(Y_train_diff, Y_tv_diff, Y_test_diff,2)
PC3_train, PC3_tv, PC3_test, mu3, var_expl3, scree3 = getPC(Y_train_diff, Y_tv_diff, Y_test_diff,3)
PC4_train, PC4_tv, PC4_test, mu4, var_expl4, scree4 = getPC(Y_train_diff, Y_tv_diff, Y_test_diff,4)
PC5_train, PC5_tv, PC5_test, mu5, var_expl5, scree5 = getPC(Y_train_diff, Y_tv_diff, Y_test_diff,5)
PC6_train, PC6_tv, PC6_test, mu6, var_expl6, scree6 = getPC(Y_train_diff, Y_tv_diff, Y_test_diff,6)
PC7_train, PC7_tv, PC7_test, mu7, var_expl7, scree7 = getPC(Y_train_diff, Y_tv_diff, Y_test_diff,7)


x = np.vstack((X_tv, X_test))
y1 = np.vstack((PC1_tv, PC1_test))
y2 = np.vstack((PC2_tv, PC2_test))
y3 = np.vstack((PC3_tv, PC3_test))
y4 = np.vstack((PC4_tv, PC4_test))
y5 = np.vstack((PC5_tv, PC5_test))
y6 = np.vstack((PC6_tv, PC6_test))
y7 = np.vstack((PC7_tv, PC7_test))
y_swap = np.vstack((Y_tv_diff, Y_test_diff))


PCA3_AR10, PCA3_ARX10, error3_10, errorX3_10 = getForecastPCA(mu3, var_expl3, x, y3, y_swap, n_train=len(PC3_train), n_tv=len(PC3_tv), n_test=len(PC3_test), h=10)
print(PCA3_AR10, PCA3_ARX10)
PCA4_AR10, PCA4_ARX10, error4_10, errorX4_10 = getForecastPCA(mu4, var_expl4, x, y4, y_swap, n_train=len(PC4_train), n_tv=len(PC4_tv), n_test=len(PC4_test), h=10)
print(PCA4_AR10, PCA4_ARX10)
PCA2_AR10, PCA2_ARX10, error2_10, errorX2_10 = getForecastPCA(mu2, var_expl2, x, y2, y_swap, n_train=len(PC2_train), n_tv=len(PC2_tv), n_test=len(PC2_test), h=10)
print(PCA2_AR10, PCA2_ARX10)
PCA5_AR10, PCA5_ARX10, error5_10, errorX5_10 = getForecastPCA(mu5, var_expl5, x, y5, y_swap, n_train=len(PC5_train), n_tv=len(PC5_tv), n_test=len(PC5_test), h=10)
PCA1_AR10, PCA1_ARX10, error1_10, errorX1_10= getForecastPCA(mu1, var_expl1, x, y1, y_swap, n_train=len(PC1_train), n_tv=len(PC1_tv), n_test=len(PC1_test), h=10)
PCA6_AR10, PCA6_ARX10, error6_10, errorX6_10= getForecastPCA(mu6, var_expl6, x, y6, y_swap, n_train=len(PC6_train), n_tv=len(PC6_tv), n_test=len(PC6_test), h=10)
PCA7_AR10, PCA7_ARX10, error7_10, errorX7_10= getForecastPCA(mu7, var_expl7, x, y7, y_swap, n_train=len(PC7_train), n_tv=len(PC7_tv), n_test=len(PC7_test), h=10)


PCA1_AR1, PCA1_ARX1, error1_1, errorX1_1 = getForecastPCA(mu1, var_expl1, x, y1, y_swap, n_train=len(PC1_train), n_tv=len(PC1_tv), n_test=len(PC1_test), h=1)
PCA2_AR1, PCA2_ARX1, error2_1, errorX2_1 = getForecastPCA(mu2, var_expl2, x, y2, y_swap, n_train=len(PC2_train), n_tv=len(PC2_tv), n_test=len(PC2_test), h=1)
PCA3_AR1, PCA3_ARX1, error3_1, errorX3_1 = getForecastPCA(mu3, var_expl3, x, y3, y_swap, n_train=len(PC3_train), n_tv=len(PC3_tv), n_test=len(PC3_test), h=1)
PCA4_AR1, PCA4_ARX1, error4_1, errorX4_1 = getForecastPCA(mu4, var_expl4, x, y4, y_swap, n_train=len(PC4_train), n_tv=len(PC4_tv), n_test=len(PC4_test), h=1)
PCA5_AR1, PCA5_ARX1, error5_1, errorX5_1 = getForecastPCA(mu5, var_expl5, x, y5, y_swap, n_train=len(PC5_train), n_tv=len(PC5_tv), n_test=len(PC5_test), h=1)
PCA6_AR1, PCA6_ARX1, error6_1, errorX6_1 = getForecastPCA(mu6, var_expl6, x, y6, y_swap, n_train=len(PC6_train), n_tv=len(PC6_tv), n_test=len(PC6_test), h=1)
PCA7_AR1, PCA7_ARX1, error7_1, errorX7_1 = getForecastPCA(mu7, var_expl7, x, y7, y_swap, n_train=len(PC7_train), n_tv=len(PC7_tv), n_test=len(PC7_test), h=1)


PCA1_AR5, PCA1_ARX5, error1_5, errorX1_5 = getForecastPCA(mu1, var_expl1, x, y1, y_swap, n_train=len(PC1_train), n_tv=len(PC1_tv), n_test=len(PC1_test), h=5)
PCA2_AR5, PCA2_ARX5, error2_5, errorX2_5 = getForecastPCA(mu2, var_expl2, x, y2, y_swap, n_train=len(PC2_train), n_tv=len(PC2_tv), n_test=len(PC2_test), h=5)
PCA3_AR5, PCA3_ARX5, error3_5, errorX3_5 = getForecastPCA(mu3, var_expl3, x, y3, y_swap, n_train=len(PC3_train), n_tv=len(PC3_tv), n_test=len(PC3_test), h=5)
PCA4_AR5, PCA4_ARX5, error4_5, errorX4_5 = getForecastPCA(mu4, var_expl4, x, y4, y_swap, n_train=len(PC4_train), n_tv=len(PC4_tv), n_test=len(PC4_test), h=5)
PCA5_AR5, PCA5_ARX5, error5_5, errorX5_5 = getForecastPCA(mu5, var_expl5, x, y5, y_swap, n_train=len(PC5_train), n_tv=len(PC5_tv), n_test=len(PC5_test), h=5)
PCA6_AR5, PCA6_ARX5, error6_5, errorX6_5 = getForecastPCA(mu6, var_expl6, x, y6, y_swap, n_train=len(PC6_train), n_tv=len(PC6_tv), n_test=len(PC6_test), h=5)
PCA7_AR5, PCA7_ARX5, error7_5, errorX7_5 = getForecastPCA(mu7, var_expl7, x, y7, y_swap, n_train=len(PC7_train), n_tv=len(PC7_tv), n_test=len(PC7_test), h=5)

PCA1_AR20, PCA1_ARX20, error1_20, errorX1_20 = getForecastPCA(mu1, var_expl1, x, y1, y_swap, n_train=len(PC1_train), n_tv=len(PC1_tv), n_test=len(PC1_test), h=20)
PCA2_AR20, PCA2_ARX20, error2_20, errorX2_20 = getForecastPCA(mu2, var_expl2, x, y2, y_swap, n_train=len(PC2_train), n_tv=len(PC2_tv), n_test=len(PC2_test), h=20)
PCA3_AR20, PCA3_ARX20, error3_20, errorX3_20 = getForecastPCA(mu3, var_expl3, x, y3, y_swap, n_train=len(PC3_train), n_tv=len(PC3_tv), n_test=len(PC3_test), h=20)
PCA4_AR20, PCA4_ARX20, error4_20, errorX4_20 = getForecastPCA(mu4, var_expl4, x, y4, y_swap, n_train=len(PC4_train), n_tv=len(PC4_tv), n_test=len(PC4_test), h=20)
PCA5_AR20, PCA5_ARX20, error5_20, errorX5_20 = getForecastPCA(mu5, var_expl5, x, y5, y_swap, n_train=len(PC5_train), n_tv=len(PC5_tv), n_test=len(PC5_test), h=20)
PCA6_AR20, PCA6_ARX20, error6_20, errorX6_20 = getForecastPCA(mu6, var_expl6, x, y6, y_swap, n_train=len(PC6_train), n_tv=len(PC6_tv), n_test=len(PC6_test), h=20)
PCA7_AR20, PCA7_ARX20, error7_20, errorX7_20 = getForecastPCA(mu7, var_expl7, x, y7, y_swap, n_train=len(PC7_train), n_tv=len(PC7_tv), n_test=len(PC7_test), h=20)


PC1l_train, PC1l_tv, PC1l_test, mu1l, var_expl1l, scree1l = getPC(Y_train, Y_tv, Y_test,1)
PC2l_train, PC2l_tv, PC2l_test, mu2l, var_expl2l, scree2l = getPC(Y_train, Y_tv, Y_test,2)
PC3l_train, PC3l_tv, PC3l_test, mu3l, var_expl3l, scree3l = getPC(Y_train, Y_tv, Y_test,3)
PC4l_train, PC4l_tv, PC4l_test, mu4l, var_expl4l, scree4l = getPC(Y_train, Y_tv, Y_test,4)
PC5l_train, PC5l_tv, PC5l_test, mu5l, var_expl5l, scree5l = getPC(Y_train, Y_tv, Y_test,5)
PC6l_train, PC6l_tv, PC6l_test, mu6l, var_expl6l, scree6l = getPC(Y_train, Y_tv, Y_test,6)
PC7l_train, PC7l_tv, PC7l_test, mu7l, var_expl7l, scree7l = getPC(Y_train, Y_tv, Y_test,7)


x = np.vstack((X_tv, X_test))
y1l = np.vstack((PC1l_tv, PC1l_test))
y2l = np.vstack((PC2l_tv, PC2l_test))
y3l = np.vstack((PC3l_tv, PC3l_test))
y4l = np.vstack((PC4l_tv, PC4l_test))
y5l = np.vstack((PC5l_tv, PC5l_test))
y6l = np.vstack((PC6l_tv, PC6l_test))
y7l = np.vstack((PC7l_tv, PC7l_test))
y_swapl = np.vstack((Y_tv, Y_test))


PCA3l_AR10, PCA3l_ARX10, error3_10l, errorX3_10l = getForecastPCA(mu3l, var_expl3l, x, y3l, y_swapl, n_train=len(PC3l_train), n_tv=len(PC3l_tv), n_test=len(PC3l_test), h=10)
print(PCA3l_AR10, PCA3l_ARX10)
PCA4l_AR10, PCA4l_ARX10, error4_10l, errorX4_10l = getForecastPCA(mu4l, var_expl4l, x, y4l, y_swapl, n_train=len(PC4l_train), n_tv=len(PC4l_tv), n_test=len(PC4l_test), h=10)
print(PCA4l_AR10, PCA4l_ARX10)
PCA2l_AR10, PCA2l_ARX10, error2_10l, errorX2_10l = getForecastPCA(mu2l, var_expl2l, x, y2l, y_swapl, n_train=len(PC2l_train), n_tv=len(PC2l_tv), n_test=len(PC2l_test), h=10)
print(PCA2l_AR10, PCA2l_ARX10)
PCA5l_AR10, PCA5l_ARX10, error5_10l, errorX5_10l = getForecastPCA(mu5l, var_expl5l, x, y5l, y_swapl, n_train=len(PC5l_train), n_tv=len(PC5l_tv), n_test=len(PC5l_test), h=10)
PCA1l_AR10, PCA1l_ARX10, error1_10l, errorX1_10l = getForecastPCA(mu1l, var_expl1l, x, y1l, y_swapl, n_train=len(PC1l_train), n_tv=len(PC1l_tv), n_test=len(PC1l_test), h=10)
PCA6l_AR10, PCA6l_ARX10, error6_10l, errorX6_10l = getForecastPCA(mu6l, var_expl6l, x, y6, y_swap, n_train=len(PC6l_train), n_tv=len(PC6l_tv), n_test=len(PC6l_test), h=10)
PCA7l_AR10, PCA7l_ARX10, error7_10l, errorX7_10l = getForecastPCA(mu7l, var_expl7l, x, y7, y_swap, n_train=len(PC7l_train), n_tv=len(PC7l_tv), n_test=len(PC7l_test), h=10)


PCA3l_AR1, PCA3l_ARX1, error3_1l, errorX3_1l = getForecastPCA(mu3l, var_expl3l, x, y3l, y_swapl, n_train=len(PC3l_train), n_tv=len(PC3l_tv), n_test=len(PC3l_test), h=1)
PCA4l_AR1, PCA4l_ARX1, error4_1l, errorX4_1l = getForecastPCA(mu4l, var_expl4l, x, y4l, y_swapl, n_train=len(PC4l_train), n_tv=len(PC4l_tv), n_test=len(PC4l_test), h=1)
PCA2l_AR1, PCA2l_ARX1, error2_1l, errorX2_1l = getForecastPCA(mu2l, var_expl2l, x, y2l, y_swapl, n_train=len(PC2l_train), n_tv=len(PC2l_tv), n_test=len(PC2l_test), h=1)
PCA5l_AR1, PCA5l_ARX1, error5_1l, errorX5_1l = getForecastPCA(mu5l, var_expl5l, x, y5l, y_swapl, n_train=len(PC5l_train), n_tv=len(PC5l_tv), n_test=len(PC5l_test), h=1)
PCA1l_AR1, PCA1l_ARX1, error1_1l, errorX1_1l = getForecastPCA(mu1l, var_expl1l, x, y1l, y_swapl, n_train=len(PC1l_train), n_tv=len(PC1l_tv), n_test=len(PC1l_test), h=1)
PCA6l_AR1, PCA6l_ARX1, error6_1l, errorX6_1l = getForecastPCA(mu6l, var_expl6l, x, y6, y_swap, n_train=len(PC6l_train), n_tv=len(PC6l_tv), n_test=len(PC6l_test), h=1)
PCA7l_AR1, PCA7l_ARX1, error7_1l, errorX7_1l = getForecastPCA(mu7l, var_expl7l, x, y7, y_swap, n_train=len(PC7l_train), n_tv=len(PC7l_tv), n_test=len(PC7l_test), h=1)

PCA3l_AR5, PCA3l_ARX5, error3_5l, errorX3_5l = getForecastPCA(mu3l, var_expl3l, x, y3l, y_swapl, n_train=len(PC3l_train), n_tv=len(PC3l_tv), n_test=len(PC3l_test), h=5)
PCA4l_AR5, PCA4l_ARX5, error4_5l, errorX4_5l = getForecastPCA(mu4l, var_expl4l, x, y4l, y_swapl, n_train=len(PC4l_train), n_tv=len(PC4l_tv), n_test=len(PC4l_test), h=5)
PCA2l_AR5, PCA2l_ARX5, error2_5l, errorX2_5l = getForecastPCA(mu2l, var_expl2l, x, y2l, y_swapl, n_train=len(PC2l_train), n_tv=len(PC2l_tv), n_test=len(PC2l_test), h=5)
PCA5l_AR5, PCA5l_ARX5, error5_5l, errorX5_5l = getForecastPCA(mu5l, var_expl5l, x, y5l, y_swapl, n_train=len(PC5l_train), n_tv=len(PC5l_tv), n_test=len(PC5l_test), h=5)
PCA1l_AR5, PCA1l_ARX5, error1_5l, errorX1_5l = getForecastPCA(mu1l, var_expl1l, x, y1l, y_swapl, n_train=len(PC1l_train), n_tv=len(PC1l_tv), n_test=len(PC1l_test), h=5)
PCA6l_AR5, PCA6l_ARX5, error6_5l, errorX6_5l = getForecastPCA(mu6l, var_expl6l, x, y6, y_swap, n_train=len(PC6l_train), n_tv=len(PC6l_tv), n_test=len(PC6l_test), h=5)
PCA7l_AR5, PCA7l_ARX5, error7_5l, errorX7_5l = getForecastPCA(mu7l, var_expl7l, x, y7, y_swap, n_train=len(PC7l_train), n_tv=len(PC7l_tv), n_test=len(PC7l_test), h=5)

PCA3l_AR20, PCA3l_ARX20, error3_20l, errorX3_20l = getForecastPCA(mu3l, var_expl3l, x, y3l, y_swapl, n_train=len(PC3l_train), n_tv=len(PC3l_tv), n_test=len(PC3l_test), h=20)
PCA4l_AR20, PCA4l_ARX20, error4_20l, errorX4_20l = getForecastPCA(mu4l, var_expl4l, x, y4l, y_swapl, n_train=len(PC4l_train), n_tv=len(PC4l_tv), n_test=len(PC4l_test), h=20)
PCA2l_AR20, PCA2l_ARX20, error2_20l, errorX2_20l = getForecastPCA(mu2l, var_expl2l, x, y2l, y_swapl, n_train=len(PC2l_train), n_tv=len(PC2l_tv), n_test=len(PC2l_test), h=20)
PCA5l_AR20, PCA5l_ARX20, error5_20l, errorX5_20l = getForecastPCA(mu5l, var_expl5l, x, y5l, y_swapl, n_train=len(PC5l_train), n_tv=len(PC5l_tv), n_test=len(PC5l_test), h=20)
PCA1l_AR20, PCA1l_ARX20, error1_20l, errorX1_20l = getForecastPCA(mu1l, var_expl1l, x, y1l, y_swapl, n_train=len(PC1l_train), n_tv=len(PC1l_tv), n_test=len(PC1l_test), h=20)
PCA6l_AR20, PCA6l_ARX20, error6_20l, errorX6_20l = getForecastPCA(mu6l, var_expl6l, x, y6, y_swap, n_train=len(PC6l_train), n_tv=len(PC6l_tv), n_test=len(PC6l_test), h=20)
PCA7l_AR20, PCA7l_ARX20, error7_20l, errorX7_20l = getForecastPCA(mu7l, var_expl7l, x, y7, y_swap, n_train=len(PC7l_train), n_tv=len(PC7l_tv), n_test=len(PC7l_test), h=20)

t=1