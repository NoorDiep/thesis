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



    for i in range(y.shape[1]):
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

                AR = ARIMA(train, order=(p_AR1, 0, 0)).fit()

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
            #preds_PC = ARX.predict(start=n_tv+k, end=n_tv+k + h - 1, dynamic=False, exog=x[k-w:k+h-w+q_ARX1+k])

            f_jX = []
            for j in range(y.shape[1]):
                ytrain = y[k + q_ARX1:w + k, j]
                xtrain = pd.DataFrame(x[k:w + k]).shift(q_ARX1).dropna()

                # Construct the model with optimized parameters
                ARX = ARIMA(ytrain, exog=xtrain, order=(p_AR1, 0, 0)).fit()
                # Forecast out-of-sample
                preds_PC = ARX.predict(start=n_tv + k, end=n_tv + k + h - 1, dynamic=False, exog=x[k - w:k + h - w + q_ARX1 + k])  # Dynamic equal to False means direct forecasts
                f_jX.append(preds_PC)

            preds_ARX = pd.DataFrame(f_jX)
            preds_ARX_s = np.dot(preds_ARX.T, PCs) + means[1]
            preds_ARX_s = pd.DataFrame(preds_ARX_s)

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
    ARX = pd.DataFrame([np.mean(pd.DataFrame(resultsARX1)), np.mean(pd.DataFrame(resultsARX2)), np.mean(pd.DataFrame(resultsARX3)), np.mean(pd.DataFrame(resultsARX4)), np.mean(pd.DataFrame(resultsARX5)),
        np.mean(pd.DataFrame(resultsARX7)), np.mean(pd.DataFrame(resultsARX10)), np.mean(pd.DataFrame(resultsARX15)), np.mean(pd.DataFrame(resultsARX20)), np.mean(pd.DataFrame(resultsARX30))])
    return AR,ARX


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


PCA3_AR10, PCA3_ARX10 = getForecastPCA(mu3, var_expl3, x, y3, y_swap, n_train=len(PC3_train), n_tv=len(PC3_tv), n_test=len(PC3_test), h=10)
print(PCA3_AR10, PCA3_ARX10)
PCA4_AR10, PCA4_ARX10 = getForecastPCA(mu4, var_expl4, x, y4, y_swap, n_train=len(PC4_train), n_tv=len(PC4_tv), n_test=len(PC4_test), h=10)
print(PCA4_AR10, PCA4_ARX10)
PCA2_AR10, PCA2_ARX10 = getForecastPCA(mu2, var_expl2, x, y2, y_swap, n_train=len(PC2_train), n_tv=len(PC2_tv), n_test=len(PC2_test), h=10)
print(PCA2_AR10, PCA2_ARX10)
PCA5_AR10, PCA5_ARX10 = getForecastPCA(mu5, var_expl5, x, y5, y_swap, n_train=len(PC5_train), n_tv=len(PC5_tv), n_test=len(PC5_test), h=10)
PCA1_AR10, PCA1_ARX10 = getForecastPCA(mu1, var_expl1, x, y1, y_swap, n_train=len(PC1_train), n_tv=len(PC1_tv), n_test=len(PC1_test), h=10)



PCA1_AR1, PCA1_ARX1 = getForecastPCA(mu1, var_expl1, x, y1, y_swap, n_train=len(PC1_train), n_tv=len(PC1_tv), n_test=len(PC1_test), h=1)
PCA2_AR1, PCA2_ARX1 = getForecastPCA(mu2, var_expl2, x, y2, y_swap, n_train=len(PC2_train), n_tv=len(PC2_tv), n_test=len(PC2_test), h=1)
PCA3_AR1, PCA3_ARX1 = getForecastPCA(mu3, var_expl3, x, y3, y_swap, n_train=len(PC3_train), n_tv=len(PC3_tv), n_test=len(PC3_test), h=1)
PCA4_AR1, PCA4_ARX1 = getForecastPCA(mu4, var_expl4, x, y4, y_swap, n_train=len(PC4_train), n_tv=len(PC4_tv), n_test=len(PC4_test), h=1)
PCA5_AR1, PCA5_ARX1 = getForecastPCA(mu5, var_expl5, x, y5, y_swap, n_train=len(PC5_train), n_tv=len(PC5_tv), n_test=len(PC5_test), h=1)

#PCA1_AR5, PCA1_ARX5 = getForecastPCA(mu1, var_expl1, x, y1, y_swap, n_train=len(PC1_train), n_tv=len(PC1_tv), n_test=len(PC1_test), h=5)
PCA2_AR5, PCA2_ARX5 = getForecastPCA(mu2, var_expl2, x, y2, y_swap, n_train=len(PC2_train), n_tv=len(PC2_tv), n_test=len(PC2_test), h=5)
PCA3_AR5, PCA3_ARX5 = getForecastPCA(mu3, var_expl3, x, y3, y_swap, n_train=len(PC3_train), n_tv=len(PC3_tv), n_test=len(PC3_test), h=5)
PCA4_AR5, PCA4_ARX5 = getForecastPCA(mu4, var_expl4, x, y4, y_swap, n_train=len(PC4_train), n_tv=len(PC4_tv), n_test=len(PC4_test), h=5)
PCA5_AR5, PCA5_ARX5 = getForecastPCA(mu5, var_expl5, x, y5, y_swap, n_train=len(PC5_train), n_tv=len(PC5_tv), n_test=len(PC5_test), h=5)
t=1
