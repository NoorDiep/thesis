import pandas as pd
import numpy as np
from matplotlib import pyplot
from sklearn.model_selection import train_test_split, GridSearchCV
from statsmodels.tools import add_constant
from statsmodels.stats.outliers_influence import variance_inflation_factor
import matplotlib.pyplot as plt
from Drivers import read_file, difference_series, getCSSED, optimal_lag, forecast_accuracy
import statsmodels.api as sm
from statsmodels.tsa.ar_model import AutoReg

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

x = np.vstack((X_tv, X_test))
y = np.vstack((Y_tv, Y_test))
y_diff = np.vstack((Y_tv_diff, Y_test_diff))

print(sm.OLS(y[:,6], add_constant(x)).fit().summary())
print(sm.OLS(y[:,0], add_constant(x)).fit().summary())
print(sm.OLS(y[:,9], add_constant(x)).fit().summary())

print(sm.OLS(y_diff[:,6], add_constant(x)).fit().summary())
print(sm.OLS(y_diff[:,0], add_constant(x)).fit().summary())
print(sm.OLS(y_diff[:,9], add_constant(x)).fit().summary())


t=1