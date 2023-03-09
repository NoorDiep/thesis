from VAR import getAR, getARX, getPCA, getAE
from Drivers import get_data

import numpy as np

########################################################################################################################
# LOAD DATA
########################################################################################################################
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
# RUN CODE
########################################################################################################################

###### Results with levels ######
f_ARh1, f_ARh1_mean, resultsARh1, idx_ph1 = getAR(X_train, Y_train, Y_tv, Y_test, h=1, plot_pred=0, dates=date_test)
print('AR', f_ARh1, f_ARh1_mean)
f_ARh1_d, f_ARh1_d_mean, resultsARh1_d, idx_ph1_d = getAR(X_train_diff, Y_train_diff, Y_tv_diff, Y_test_diff, h=1, plot_pred=0, dates=date_test_diff)
print('AR_d', f_ARh1_d, f_ARh1_d_mean)
f_ARh5, f_ARh5_mean, resultsARh5, idx_ph5 = getAR(X_train, Y_train, Y_tv, Y_test, h=1, plot_pred=0, dates=date_test)
print('AR', f_ARh5, f_ARh5_mean)
f_ARh5_d, f_ARh5_d_mean, resultsARh5_d, idx_ph5_d = getAR(X_train_diff, Y_train_diff, Y_tv_diff, Y_test_diff, h=1, plot_pred=0, dates=date_test_diff)
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
