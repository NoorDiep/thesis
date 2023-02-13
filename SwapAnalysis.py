from statsmodels.stats.outliers_influence import variance_inflation_factor
from statsmodels.tools import add_constant
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

from Drivers import read_file, get_ds, get2dplot, get3dplot, difference_series, adf_test, pacf_plot, predict_ar, optimal_lag, forecast_accuracy, plot_forecast
import pandas as pd
import numpy as np
import statsmodels.tsa.stattools as ts
import statsmodels.api as sm
from mpl_toolkits.mplot3d import axes3d
import matplotlib.dates as dates
import matplotlib.ticker as ticker
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from statsmodels.tsa.ar_model import AutoReg
from datetime import datetime
from statsmodels.tsa.api import VAR
from statsmodels.tsa.stattools import adfuller
from statsmodels.tools.eval_measures import rmse, aic
#from pyneurgen.neuralnet import NeuralNet
#from pyneurgen.recurrent import NARXRecurrent
from keras.models import Sequential
from keras.layers import Dense
from matplotlib import pyplot


from gekko import GEKKO
import matplotlib.pyplot as plt
#from sysidentpy.polynomial_basis import PolynomialNarmax
from torch import nn
from sysidentpy.neural_network import NARXNN
from sysidentpy.basis_function import Polynomial
from sysidentpy.utils.plotting import plot_residues_correlation, plot_results
from sysidentpy.residues.residues_correlation import compute_residues_autocorrelation
from sysidentpy.residues.residues_correlation import compute_cross_correlation
from sysidentpy.utils.narmax_tools import regressor_code
from sklearn.ensemble import GradientBoostingRegressor
from sysidentpy.metrics import root_relative_squared_error
from sysidentpy.general_estimators import NARX
from sysidentpy.metrics import mean_squared_error





# Load all data
df_swap, dates = read_file(file='SwapDriverData1.xlsx', sheet='Swap')
df_drivers = read_file(file='SwapDriverData1.xlsx', sheet='Driver')
diff_swap5 = difference_series(df_swap, 5)
df_drivers_diff = df_drivers.iloc[5:] # Cut off first 5 observations

# Get descriptive statistics, corelation tables, adf test results and figures for Data Section
#getDataTablesFigures()

# Create train and test set
df_drivers = df_drivers.reset_index(drop=True)
x_train, x_test, y_train, y_test = train_test_split(df_drivers, df_swap, test_size=0.2, shuffle=False)

X_train_diff, X_test_diff, y_train_diff, y_test_diff = train_test_split(df_drivers_diff, diff_swap5, test_size=0.2, shuffle=False)

#########################################
# Univariate autoregressive model
#########################################
#pacf_plot(df_swap, 25)

# # Instantiate and fit AR model
# idx_UARX, ARX = optimal_lag(x_train, y_train, 24, 1) # Compute the optimal model; 1 indicating ARX model, AR model otherwise
# idx_UAR, AR = optimal_lag(x_train, y_train, 24, 0)
#
# ARX.summary()
# AR.summary()
#
# # Forecasting
# #ARX.predict(ARX.params, start=n_train+1, end=n)
# preds_ARX = ARX.predict(exog_oos=x_test,start=len(y_train), end=len(y_train)+len(y_test)-1)
# preds_AR = AR.predict(start=len(y_train), end=len(y_train)+len(y_test)-1)
#
#
# preds_ARX.index = y_test.index
# preds_AR.index = y_test.index
#
# # Plot the prediction vs test data
# plot_forecast(preds_ARX, y_test["10Y"])
# plot_forecast(preds_AR, y_test["10Y"])
#
#
#
# # Predict against test data
#
# acc_ARX = forecast_accuracy(preds_ARX, y_test["10Y"])
# acc_AR = forecast_accuracy(preds_AR, y_test["10Y"])

# #########################################
# PCA
# #########################################

# Standardize data
sc= StandardScaler()

X_train = sc.fit_transform(x_train)
X_test = sc.transform(x_test)

pca = PCA(n_components=3)

X_train = pca.fit_transform(X_train)
X_train = pd.DataFrame(X_train)
X_test = pca.transform(x_test)

explained_variance = pca.explained_variance_ratio_


X_train1 = X_train.reset_index(drop=True)
y_train1 = y_train.reset_index(drop=True)
dep = y_train1.iloc[:, 6]
dep.name = 0
AIC = []
maxlags = 24
for i in range(1, maxlags, 1):
    # Strip indices of dataframes
    X_train1 = X_train.reset_index(drop=True)
    y_train1 = y_train.reset_index(drop=True)
    #y_train1 = y_train1.T.reset_index(drop=True).T
    dep = y_train1.iloc[:, 6]
    dep.name = 0

    # Construct the model for different lags
    model = AutoReg(endog=dep, exog=X_train1, lags=i)

    result = model.fit()
    print('Lag Order =', i)
    print('AIC : ', result.aic)
    print('BIC : ', result.bic)
    print('FPE : ', result.fpe)
    print('HQIC: ', result.hqic, '\n')
    AIC.append(result.aic)
idx = AIC.index(min(AIC))

PCA_AR = AutoReg(endog=dep, exog=X_train1, lags=idx).fit()

preds_AR = PCA_AR.predict(exog_oos=y_test, start=len(y_train), end=len(y_train)+len(y_test)-1)

t=1
# #########################################
# # NEURAL NETWORK ARX
# #########################################

# NARX = PolynomialNarmax(non_degree=2, order_selection=True,
#                         ylag=2, xlag=[[1, 2], [1, 2]],
#                         info_criteria='aic', estimator='least_squares')
# NARX.fit(x_train, y_train)
# yhat = NARX.predict(x_test, y_test)
# rrse = root_relative_squared_error(y_test, yhat)
# print(rrse)
#
# results = pd.DataFrame(NARX.results(err_precision=8, dtype='dec'), columns=['Regressors', 'Parameters', 'ERR'])
# print(results)
# ee, ex, extras, lam = NARX.residuals(x_test, y_test, yhat)
# NARX.plot_result(y_test, yhat, ee, ex)

#
# basis_function=Polynomial(degree=1)
#
# # Get optimal params
# regressors = regressor_code(X=x_train,
#     xlag=[2,3],
#     ylag=[2,4],
#     model_type="NARMAX",
#     model_representation="neural_network",
#     basis_function=basis_function
# )
#
# n_features = regressors.shape[0] # the number of features of the NARX net
# n_features

#
# # Building NARX neural network
# basis_function = Polynomial(degree=1)
#
# gb_narx = NARX(
#     base_estimator=GradientBoostingRegressor(
#         loss='quantile',
#         alpha=0.90,
#         n_estimators=250,
#         max_depth=10,
#         learning_rate=.1,
#         min_samples_leaf=9,
#         min_samples_split=9
#         ),
#     xlag=2,
#     ylag=2,
#     basis_function=basis_function,
#     model_type="NARMAX"
# )
#
# gb_narx.fit(X=x_train, y=y_train)
# yhat = gb_narx.predict(X=x_test, y=y_test)
# print(mean_squared_error(y_test, yhat))
#
# plot_results(y=y_test, yhat=yhat, n=200)
# ee = compute_residues_autocorrelation(y_test, yhat)
# plot_residues_correlation(data=ee, title="Residues", ylabel="$e^2$")
# x1e = compute_cross_correlation(y_test, yhat, x_test)
# plot_residues_correlation(data=x1e, title="Residues", ylabel="$x_1e$")

# # NARXRecurrent
# input_nodes, hidden_nodes, output_nodes = 10, 2, 10
# output_order, incoming_weight_from_output = 10, .6
# input_order, incoming_weight_from_input = 7, .4
#
# # init neural network
# net = NeuralNet()
# net.init_layers(input_nodes, [hidden_nodes], output_nodes,
#                 NARXRecurrent(output_order, incoming_weight_from_output,
#                               input_order, incoming_weight_from_input))
# net.randomize_network()
# net.set_halt_on_extremes(True)
#
# # set constrains and rates
# net.set_random_constraint(.5)
# net.set_learnrate(.1)
#
#
# t=1

# # set inputs and outputs
# net.set_all_inputs(all_inputs)
# net.set_all_targets(all_targets)
#
# # set lengths
# length = len(all_inputs)
# learn_end_point = int(length * .8)

# # set ranges
# # net.set_learn_range(0, learn_end_point)
# # net.set_test_range(learn_end_point + 1, length - 1)
#
# # add activation to layer 1
# net.layers[1].set_activation_type('tanh')
#
# # fit data to model
# net.learn(epochs=150, show_epoch_results=True, random_testing=False)
#
# # define mean squared error
# mse = net.test()
# print( "Testing mse = ", mse)
#
# # define data
# x = [item[0][0] * 200.0 for item in net.get_test_data()]
# y = [item[0][0] for item in net.test_targets_activations]
# y_true = [item[1][0] for item in net.test_targets_activations]
#
# # plot results
# plot_results(x, y, y_true)

past=2
model = Sequential()
#model.add(Dense(2, activation='relu',use_bias=False, input_dim=2*past))
model.add(Dense(2, activation='linear',use_bias=False, input_dim=2*past))
model.add(Dense(1))
model.compile(optimizer='adam', loss='mse')
history=model.fit(x_train, y_train, epochs=50, batch_size=20, verbose=2)


network_prediction = model.predict(x_test)
from numpy import linalg as LA
Y_test=np.reshape(y_test, (y_test.shape[0],1))
error=network_prediction-Y_test

# this is the measure of the prediction performance in percents
error_percentage=LA.norm(error,2)/LA.norm(Y_test,2)*100

plt.figure()
plt.plot(Y_test, 'b', label='Real output')
plt.plot(network_prediction,'r', label='Predicted output')
plt.xlabel('Discrete time steps')
plt.ylabel('Output')
plt.legend()
plt.savefig('prediction_offline.png')
#plt.show()

###############################################################################
#       plot training and validation curves
###############################################################################

loss=history.history['loss']
val_loss=history.history['val_loss']
epochs=range(1,len(loss)+1)
plt.figure()
plt.plot(epochs, loss,'b', label='Training loss')
plt.plot(epochs, val_loss,'r', label='Validation loss')
plt.title('Training and validation losses')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.xscale('log')
#plt.yscale('log')
plt.legend()
plt.savefig('loss_curves.png')
#plt.show()