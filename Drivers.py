import pandas as pd
import numpy as np
from matplotlib import pyplot
from sklearn.model_selection import train_test_split, GridSearchCV

from statsmodels.tools import add_constant
from statsmodels.stats.outliers_influence import variance_inflation_factor

import matplotlib.pyplot as plt
import matplotlib.dates as dates
import matplotlib.ticker as ticker
import statsmodels.tsa.stattools as ts
from statsmodels.graphics.tsaplots import plot_pacf
from statsmodels.tsa.ar_model import AutoReg
from statsmodels.tsa.arima.model import ARIMA
import statsmodels.api as sm
from sklearn.metrics import mean_absolute_error, mean_squared_error
from mpl_toolkits.mplot3d import Axes3D

import keras
from keras import layers
from keras.layers import Dense,Conv2D,MaxPooling2D,UpSampling2D
from keras import Input, Model
from keras import Model
from keras.wrappers.scikit_learn import KerasClassifier
from keras import Sequential
from keras.datasets import mnist



def read_file(file: str, sheet: str):
    df = pd.read_excel(open(file, 'rb'),
                  sheet_name=sheet)

    dates = df.iloc[:, 0]
    if sheet == "Swap":
        df_cleaned = df.iloc[:, 1:11]
        df_cleaned.index = dates
        return df_cleaned, dates
    else:
        df_cleaned = df.iloc[:, 1:8]
        df_cleaned.index = dates
        return df_cleaned


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

    # df_drivers = df_drivers.drop('Stress', axis=1)
    # df_drivers = df_drivers.drop('EcSu', axis=1)
    # df_drivers = df_drivers.drop('Sent', axis=1)
    # df_drivers = df_drivers.drop('PoUn', axis=1)
    # df_drivers = df_drivers.drop('News', axis=1)
    df_drivers = df_drivers.drop('Vol', axis=1)
    # df_drivers = df_drivers.drop('Infl', axis=1)
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

    df_drivers = pd.DataFrame([df_drivers['EcSu'],df_drivers_diff['Sent'],df_drivers['Stress'],df_drivers['PoUn'],df_drivers_diff['News'],df_drivers['Infl']]).T

    # print(sm.OLS(df_swap['10Y'], add_constant(df_drivers)).fit().summary())
    # print(sm.OLS(df_swap['1Y'], add_constant(df_drivers)).fit().summary())
    # print(sm.OLS(df_swap['30Y'], add_constant(df_drivers)).fit().summary())
    #
    # print(sm.OLS(diff_swap['10Y'], add_constant(df_drivers)).fit().summary())
    # print(sm.OLS(diff_swap['1Y'], add_constant(df_drivers)).fit().summary())
    # print(sm.OLS(diff_swap['30Y'], add_constant(df_drivers)).fit().summary())
    #print(sm.OLS(diff_swap['10Y'], add_constant(df_drivers_diff)).fit().summary())

    # Selected set for descriptive statistics
    df_crisis = diff_swap[:1433];
    df_post_crisis = diff_swap[1433:]
    get_ds(df_crisis)
    get_ds(df_post_crisis)

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

def getDataTablesFigures(df_swap, df_drivers, diff_swap, diff_drivers, depo, df_swap_dates):
    # Get descriptive statistics and correlation matrix
    ds_swap = get_ds(df_swap)
    ds_drivers = get_ds(df_drivers)
    ds_diff = get_ds(diff_swap)
    ds_diffX = get_ds(diff_drivers)

    corr_drivers = df_drivers.corr().round(3)
    corr_drivers = diff_drivers.corr().round(3)

    # Construct VIF, first include constant to do so
    X = add_constant(df_drivers)
    VIF = pd.Series([variance_inflation_factor(X.values, i) for i in range(X.shape[1])], index=X.columns)

    X_diff = add_constant(diff_drivers)
    VIF_diff = pd.Series([variance_inflation_factor(X_diff.values, i) for i in range(X_diff.shape[1])], index=X_diff.columns)

    getadf_values_exog(df_drivers)

    # Get correlation matrices
    corr_swap = df_swap.corr()
    corr_swap_diff = diff_swap.corr()

    # Get 2D and 3D plots of the swap rate
    # get2dplot(df_swap, depo, df_swap_dates, 0)
    #get2dplot(diff_swap, depo, df_swap_dates, 0)
    # get3dplot(df_swap, 0)

    # Perform ADF test
    adf, adf_diff = adf_test(df_swap, diff_swap)
    adfX, adfX_diff = adf_test(df_drivers, diff_drivers)

    # Print all output for Data Section
    print('Descriptive statistics swap rate ', ds_swap)
    print('Descriptive statistics drivers ', ds_drivers)
    print('Descriptive statistics differenced swap ', ds_diff)
    print('Descriptive statistics differenced drivers ', ds_diffX)
    print('Correlation drivers ', corr_drivers.to_latex())
    print('Correlation diff drivers ', corr_drivers.to_latex())
    print('VIF:', VIF)
    print('VIF diff:', VIF_diff)
    print('ADF test results swap in levels', adf)
    print('ADF test results swap differences', adf_diff)
    print('ADF test results swap in levels', adfX)
    print('ADF test results swap differences', adfX_diff)

    return corr_swap, corr_swap_diff



def get_ds(df: pd.DataFrame):
    # Calculate descriptive statistics
    mean = df.mean()
    sd = df.std()
    min = df.min()
    q25 = df.quantile(0.25)
    median = df.median()
    q75 = df.quantile(0.75)
    max = df.max()
    skew = df.skew()
    kurt = df.kurt()

    # Get autocorrelations
    acf1_series = []
    acf5_series = []
    acf10_series = []
    pacf1_series = []
    pacf5_series = []
    pacf10_series = []
    for column in df:
        acf1 = ts.acf(df[column], nlags=1)[1]
        acf5 = ts.acf(df[column], nlags=5)[5]
        acf10 = ts.acf(df[column], nlags=10)[10]
        acf1_series.append(acf1)
        acf5_series.append(acf5)
        acf10_series.append(acf10)

        pacf1 = ts.pacf(df[column], nlags=1)[1]
        pacf5 = ts.pacf(df[column], nlags=5)[5]
        pacf10 = ts.pacf(df[column], nlags=10)[10]
        pacf1_series.append(pacf1)
        pacf5_series.append(pacf5)
        pacf10_series.append(pacf10)

    # Concatenate
    ds = [mean, sd, min, q25, median, q75, max, skew, kurt,
          acf1_series, acf5_series, acf10_series, pacf1_series, pacf5_series, pacf10_series]

    # Transform list to DataFrame
    tab = np.array(ds)
    table = tab.reshape((15, len(df.columns)))
    des_stats_table = pd.DataFrame(table, columns=df.columns.values)
    des_stats_table.index = ['mean', 'sd', 'min', '25%', 'median', '75%', 'max', 'skewness',
                             'kurtosis', 'AC(1)', 'AC(5)', 'AC(10)', 'PAC(1)', 'PAC(5)', 'PAC(10)']
    des_stats_table = des_stats_table.round(3)
    #latex = des_stats_table.to_latex()
    return des_stats_table


def pacf_plot(df: pd.DataFrame, maxlags: int):
    for column in df:
        pacf = plot_pacf(df[column], lags=maxlags, method='ywm')
        plt.show()


def get2dplot(df: pd.DataFrame, depo_df: pd.DataFrame, save: bool):
    color = ["blue", "lightblue", "deepskyblue", "dodgerblue", "steelblue", "mediumblue", "darkblue", "slategrey",
             "gray", "black"]
    i = 0
    for column in df:
        df[column].plot(figsize=(15, 5), lw=1, color=color[i], label=column)
        i = i + 1
    depo_df.plot(figsize=(15, 5), lw=1, color="red", label=column)
    plt.legend(ncol=2)
    plt.xlabel("Date")
    plt.ylabel("Eurozone swap rate")
    #plt.title("The Eurzone term structure of interest swap rate")
    if bool:
        plt.savefig('swap2D.png')
    plt.show()
    plt.close()


def get3dplot(df: pd.DataFrame, save: bool):
    # Create numpy rec array
    dfn = df.to_records()
    type(dfn)
    dfn

    # Get tenor values
    header = []
    for name in dfn.dtype.names[1:]:
        maturity = name.split("Y")[0]
        header.append(maturity)

    x = []  # Dates
    y = []  # Maturities
    z = []  # Rates

    # Convert dates from datetime to numeric
    for dt in dfn.Date:
        dt_num = dates.date2num(dt)
        x.append([dt_num for i in range(len(dfn.dtype.names) - 1)])
    # print ('x_data: ', x[1:len(df_swap.index)])

    # Fill maturity and rates lists
    for row in dfn:
        y.append(header)
        z.append(list(row.tolist()[1:]))

    # Convert lists to np.array
    x = np.array(x, dtype='f')
    y = np.array(y, dtype='f')
    z = np.array(z, dtype='f')

    fig = plt.figure(figsize=(10, 5))
    ax = fig.add_subplot(111, projection='3d')
    ax.plot_surface(x, y, z, rstride=2, cstride=1, cmap='magma', vmin=np.nanmin(z), vmax=np.nanmax(z))
    #ax.set_title('The Eurozone term structure of interest swap rates')
    ax.set_ylabel('Maturity')
    ax.set_zlabel('Swap rate')

    ax.w_xaxis.set_major_formatter(ticker.FuncFormatter(format_date))
    for tl in ax.w_xaxis.get_ticklabels():
        tl.set_ha('right')
        tl.set_rotation(40)

    if bool:
        plt.savefig('swap3D.jpeg')
    plt.show()


def format_date(x, pos=None):
    return dates.num2date(x).strftime('%Y')


def difference_series(df: list, lag: int):
    np_diff = np.diff(df, n=lag, axis=0)
    df_diff = pd.DataFrame(np_diff, columns=df.columns.values)
    return df_diff


def adf_test(df, df_diff):
    ADF_df = getadf_values(df, 0)
    ADF_diff = getadf_values(df_diff, 1)

    return ADF_df, ADF_diff

def getadf_values(df: list, indicator_diff):
    ADF_c = []
    ADF_ct = []
    for column in df:
        #print('Index is ', column)
        adf_c = ts.adfuller(df[column], regression="c", autolag="BIC", maxlag=30)  # constant only
        adf_ct = ts.adfuller(df[column], regression="ct", autolag="BIC",maxlag=30)  # constant and trend
        #print('Constant only: ', adf_c[0:2])
        #print('Constant and trend: ', adf_ct[0:2])
        ADF_c.append(adf_c[0:3])
        ADF_ct.append(adf_ct[0:3])
    concat = [ADF_c, ADF_ct]
    tab = np.array(concat)
    table = tab.reshape(len(ADF_ct), 6)

    # Define column names
    if indicator_diff == 1:
        cols = [['ADF Stat C Diff', 'p-value C Diff', 'lag C Diff', 'ADF Stat CT Diff', 'p-value CT Diff', 'lag CT Diff']]
    elif indicator_diff == 0:
        cols = [['ADF Stat C', 'p-value C', 'lag C', 'ADF Stat CT', 'p-value CT', 'lag C',]]

    # Return as pd.DataFrame
    results = pd.DataFrame(table, columns=cols)
    results.index = df.columns.values
    results = results.round(3)
    return results

def getadf_values_exog(df: list):
    ADF_n = []

    for column in df:
        if column == 'Stress' or column == 'Infl' or column == 'Sent':
            adf_c = ts.adfuller(df[column], regression="n", autolag="BIC", maxlag=30)  # constant  # constant and trend
            ADF_n.append(adf_c[0:3])

        else:
            adf_c = ts.adfuller(df[column], regression="c", autolag="BIC", maxlag=30)  # constant  # constant and trend
            ADF_n.append(adf_c[0:3])

    tab = np.array(ADF_n)
    table = tab.reshape(len(ADF_n), 3)

    # Define column names
    cols = [['ADF Stat N', 'p-value N', 'lags N']]

    # Return as pd.DataFrame
    results = pd.DataFrame(table, columns=cols)
    results.index = df.columns.values
    results = results.round(3)
    return results

def optimal_lag(x_train, x_tv, y_train, y_tv, maxlags_p, maxlags_q, indicator):

    BIC = []
    BICX = []
    idx_q =[]
    q_opt = 0

    for i in range(1,maxlags_p+1,1):
        print(i)
        # Construct the model for different lags
        if indicator == 1:
            BIC_inter = []
            for j in range(1, maxlags_q+1,1):
                # Create lagged exogenous variables
                x_train_lagged = pd.DataFrame(x_train).shift(j).dropna()
                x_tv_lagged = pd.DataFrame(x_tv).shift(j).dropna()
                # ARX model
                model = AutoReg(endog=y_train[:y_train.size-j], exog=x_train_lagged, lags=i)

                try:
                    result = model.fit()
                except:
                    result = []
                    print('Error at lag ', i)
                    pass

                BIC_inter.append(result.bic)
            q = BIC_inter.index(min(BIC_inter)) + 1
            idx_q.append(q)
            BICX.append(BIC_inter[q-1])
        else:
            # AR model
            model = AutoReg(endog=y_train, lags=i)
            #model = ARIMA(y_train, order=(i, 0, 0))

            try:
                result = model.fit()
            except:
                print('Error at lag ', i)
                pass

            BIC.append(result.bic)

    if indicator == 1:
        p_optX = BICX.index(min(BICX)) + 1
        q_opt = idx_q[p_optX - 1]
        return p_optX, q_opt
    else:
        p_opt = BIC.index(min(BIC)) + 1
        return p_opt


def plot_forecast(pred, test):
    plt.plot(pred, label="forecast")
    plt.plot(test, color='red', label="actual")
    plt.legend(loc="upper left")
    plt.show()


# Forecast accuracy measures
def forecast_accuracy(forecast, actual, df_indicator):
    if df_indicator:
        errors = []

        for column in actual:
            MAE = np.mean(np.abs(forecast[column] - actual[column]))
            RMSE = np.mean((forecast[column] - actual[column]) ** 2) ** 0.5

            errors.append([MAE,RMSE])


        return pd.DataFrame(errors, columns=['MEA', 'RMSE'])

    else:
        mae = np.mean(np.abs(forecast - actual))    # MAE
        mse = np.mean((forecast - actual) ** 2)     # MSE

        return pd.DataFrame([[mae, np.sqrt(mse)]], columns=['mae','rmse'])

def getCSSED(errorB, errorM, data):
    cssde = []

    for idx in range(errorB.shape[0]):
        cssde.append(np.cumsum(errorB[:idx]^2-errorM[:idx]^2))

    cssde = pd.DataFrame(cssde)

    return cssde



def plot_components(x, data):

    fig = plt.figure(figsize=(10, 10))

    # choose projection 3d for creating a 3d graph
    axis = fig.add_subplot(111, projection='3d')

    # x[:,0]is pc1,x[:,1] is pc2 while x[:,2] is pc3
    axis.scatter(x['pc1'], x['pc2'], x['pc3'], c=data['10Y'],  cmap='plasma')
    axis.set_xlabel("PC1", fontsize=10)
    axis.set_ylabel("PC2", fontsize=10)
    axis.set_zlabel("PC3", fontsize=10)
    legend1 = axis.legend( loc="lower left", title="Classes")
    axis.add_artist(legend1)
    plt.show()









