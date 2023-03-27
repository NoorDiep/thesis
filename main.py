from statsmodels.tools import add_constant

from Autoencoder import getForecastAE, buildAE
from Drivers import get_data, forecast_accuracy, optimal_lag, getDataTablesFigures
import numpy as np
import pandas as pd
import statsmodels.api as sm
from statsmodels.tools import add_constant
from AR_ARX import getForecast
from PCA import getForecastPCA, getPC

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
#getDataTablesFigures(data[0], data[1], pd.DataFrame(Y_diff), pd.DataFrame(X_diff), depo, swap_dates)

x = np.vstack((X_tv, X_test))
y = np.vstack((Y_tv, Y_test))
y_diff = np.vstack((Y_tv_diff, Y_test_diff))

print(sm.OLS(y['10Y'], add_constant(x)).fit().summary())
print(sm.OLS(y['1Y'], add_constant(x)).fit().summary())
print(sm.OLS(y['30Y'], add_constant(x)).fit().summary())

print(sm.OLS(y_diff['10Y'], add_constant(x)).fit().summary())
print(sm.OLS(y_diff['1Y'], add_constant(x)).fit().summary())
print(sm.OLS(y_diff['30Y'], add_constant(x)).fit().summary())


########################################################################################################################
# RUN CODE
########################################################################################################################

####### AR - ARX #######

x = np.vstack((X_tv, X_test))
y = np.vstack((Y_tv, Y_test))
pdAR1, pdARX1, error1, errorX1 = getForecast(x, y, n_train=len(Y_train), n_tv=len(Y_tv), n_test=len(Y_test), h=1, diff_lag=lag)
pdAR5, pdARX5, error5, errorX5 = getForecast(x, y, n_train=len(Y_train), n_tv=len(Y_tv), n_test=len(Y_test), h=5, diff_lag=lag)
pdAR10, pdARX10, error10, errorX10 = getForecast(x, y, n_train=len(Y_train), n_tv=len(Y_tv), n_test=len(Y_test), h=10, diff_lag=lag)
pdAR20, pdARX20, error20, errorX20 = getForecast(x, y, n_train=len(Y_train), n_tv=len(Y_tv), n_test=len(Y_test), h=20, diff_lag=lag)


y_diff = np.vstack((Y_tv_diff, Y_test_diff))
pdAR1_diff, pdARX1_diff, error1_diff, errorX1_diff = getForecast(x, y_diff, n_train=len(Y_train), n_tv=len(Y_tv), n_test=len(Y_test), h=1, diff_lag=lag)
pdAR5_diff, pdARX5_diff, error5_diff, errorX5_diff = getForecast(x, y_diff, n_train=len(Y_train), n_tv=len(Y_tv), n_test=len(Y_test), h=5, diff_lag=lag)
pdAR10_diff, pdARX10_diff, error10_diff, errorX10_diff = getForecast(x, y_diff, n_train=len(Y_train), n_tv=len(Y_tv), n_test=len(Y_test), h=10, diff_lag=lag)
pdAR20_diff, pdARX20_diff, error20_diff, errorX20_diff = getForecast(x, y_diff, n_train=len(Y_train), n_tv=len(Y_tv), n_test=len(Y_test), h=20, diff_lag=lag)

t = 1

####### PCA #######

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
print(PCA3_AR10, PCA3_ARX10)
PCA4_AR10, PCA4_ARX10, error4_10, errorX4_10 = getForecastPCA(mu4, var_expl4, x, y4, y_swap, n_train=len(PC4_train), n_tv=len(PC4_tv), n_test=len(PC4_test), h=10)
print(PCA4_AR10, PCA4_ARX10)
PCA2_AR10, PCA2_ARX10, error2_10, errorX2_10 = getForecastPCA(mu2, var_expl2, x, y2, y_swap, n_train=len(PC2_train), n_tv=len(PC2_tv), n_test=len(PC2_test), h=10)
print(PCA2_AR10, PCA2_ARX10)
PCA5_AR10, PCA5_ARX10, error5_10, errorX5_10 = getForecastPCA(mu5, var_expl5, x, y5, y_swap, n_train=len(PC5_train), n_tv=len(PC5_tv), n_test=len(PC5_test), h=10)
PCA1_AR10, PCA1_ARX10, error1_10, errorX1_10= getForecastPCA(mu1, var_expl1, x, y1, y_swap, n_train=len(PC1_train), n_tv=len(PC1_tv), n_test=len(PC1_test), h=10)



PCA1_AR1, PCA1_ARX1, error1_1, errorX1_1 = getForecastPCA(mu1, var_expl1, x, y1, y_swap, n_train=len(PC1_train), n_tv=len(PC1_tv), n_test=len(PC1_test), h=1)
PCA2_AR1, PCA2_ARX1, error2_1, errorX2_1 = getForecastPCA(mu2, var_expl2, x, y2, y_swap, n_train=len(PC2_train), n_tv=len(PC2_tv), n_test=len(PC2_test), h=1)
PCA3_AR1, PCA3_ARX1, error3_1, errorX3_1 = getForecastPCA(mu3, var_expl3, x, y3, y_swap, n_train=len(PC3_train), n_tv=len(PC3_tv), n_test=len(PC3_test), h=1)
PCA4_AR1, PCA4_ARX1, error4_1, errorX4_1 = getForecastPCA(mu4, var_expl4, x, y4, y_swap, n_train=len(PC4_train), n_tv=len(PC4_tv), n_test=len(PC4_test), h=1)
PCA5_AR1, PCA5_ARX1, error5_1, errorX5_1 = getForecastPCA(mu5, var_expl5, x, y5, y_swap, n_train=len(PC5_train), n_tv=len(PC5_tv), n_test=len(PC5_test), h=1)

PCA1_AR5, PCA1_ARX5, error1_5, errorX1_5 = getForecastPCA(mu1, var_expl1, x, y1, y_swap, n_train=len(PC1_train), n_tv=len(PC1_tv), n_test=len(PC1_test), h=5)
PCA2_AR5, PCA2_ARX5, error2_5, errorX2_5 = getForecastPCA(mu2, var_expl2, x, y2, y_swap, n_train=len(PC2_train), n_tv=len(PC2_tv), n_test=len(PC2_test), h=5)
PCA3_AR5, PCA3_ARX5, error3_5, errorX3_5 = getForecastPCA(mu3, var_expl3, x, y3, y_swap, n_train=len(PC3_train), n_tv=len(PC3_tv), n_test=len(PC3_test), h=5)
PCA4_AR5, PCA4_ARX5, error4_5, errorX4_5 = getForecastPCA(mu4, var_expl4, x, y4, y_swap, n_train=len(PC4_train), n_tv=len(PC4_tv), n_test=len(PC4_test), h=5)
PCA5_AR5, PCA5_ARX5, error5_5, errorX5_5 = getForecastPCA(mu5, var_expl5, x, y5, y_swap, n_train=len(PC5_train), n_tv=len(PC5_tv), n_test=len(PC5_test), h=5)

PCA1_AR20, PCA1_ARX20, error1_20, errorX1_20 = getForecastPCA(mu1, var_expl1, x, y1, y_swap, n_train=len(PC1_train), n_tv=len(PC1_tv), n_test=len(PC1_test), h=20)
PCA2_AR20, PCA2_ARX20, error2_20, errorX2_20 = getForecastPCA(mu2, var_expl2, x, y2, y_swap, n_train=len(PC2_train), n_tv=len(PC2_tv), n_test=len(PC2_test), h=20)
PCA3_AR20, PCA3_ARX20, error3_20, errorX3_20 = getForecastPCA(mu3, var_expl3, x, y3, y_swap, n_train=len(PC3_train), n_tv=len(PC3_tv), n_test=len(PC3_test), h=20)
PCA4_AR20, PCA4_ARX20, error4_20, errorX4_20 = getForecastPCA(mu4, var_expl4, x, y4, y_swap, n_train=len(PC4_train), n_tv=len(PC4_tv), n_test=len(PC4_test), h=20)
PCA5_AR20, PCA5_ARX20, error5_20, errorX5_20 = getForecastPCA(mu5, var_expl5, x, y5, y_swap, n_train=len(PC5_train), n_tv=len(PC5_tv), n_test=len(PC5_test), h=20)



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
print(PCA3l_AR10, PCA3l_ARX10)
PCA4l_AR10, PCA4l_ARX10, error4_10l, errorX4_10l = getForecastPCA(mu4l, var_expl4l, x, y4l, y_swapl, n_train=len(PC4l_train), n_tv=len(PC4l_tv), n_test=len(PC4l_test), h=10)
print(PCA4l_AR10, PCA4l_ARX10)
PCA2l_AR10, PCA2l_ARX10, error2_10l, errorX2_10l = getForecastPCA(mu2l, var_expl2l, x, y2l, y_swapl, n_train=len(PC2l_train), n_tv=len(PC2l_tv), n_test=len(PC2l_test), h=10)
print(PCA2l_AR10, PCA2l_ARX10)
PCA5l_AR10, PCA5l_ARX10, error5_10l, errorX5_10l = getForecastPCA(mu5l, var_expl5l, x, y5l, y_swapl, n_train=len(PC5l_train), n_tv=len(PC5l_tv), n_test=len(PC5l_test), h=10)
PCA1l_AR10, PCA1l_ARX10, error1_10l, errorX1_10l = getForecastPCA(mu1l, var_expl1l, x, y1l, y_swapl, n_train=len(PC1l_train), n_tv=len(PC1l_tv), n_test=len(PC1l_test), h=1)


######### AE ###########



