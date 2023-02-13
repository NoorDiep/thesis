import pandas as pd
import numpy as np
from matplotlib import pyplot
from sklearn.model_selection import train_test_split

from statsmodels.tools import add_constant
from statsmodels.stats.outliers_influence import variance_inflation_factor

import matplotlib.pyplot as plt
import matplotlib.dates as dates
import matplotlib.ticker as ticker
import statsmodels.tsa.stattools as ts
from statsmodels.graphics.tsaplots import plot_pacf
from statsmodels.tsa.ar_model import AutoReg
from statsmodels.tsa.arima.model import ARIMA
from mpl_toolkits.mplot3d import Axes3D

import keras
from keras import layers
from keras.layers import Dense,Conv2D,MaxPooling2D,UpSampling2D
from keras import Input, Model
from keras import Model
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
    df_drivers = read_file(file='SwapDriverData1.xlsx', sheet='Driver')
    diff_swap = difference_series(df_swap, lag)
    df_drivers_diff = df_drivers.iloc[lag:]  # Cut off first n observations

    # Get descriptive statistics, corelation tables, adf test results and figures for Data Section
    # getDataTablesFigures()

    # Create train and test set
    df_drivers = df_drivers.reset_index(drop=True)
    x_train, x_test, y_train, y_test = train_test_split(df_drivers, df_swap, test_size=0.2, shuffle=False)
    x_train_diff, x_test_diff, y_train_diff, y_test_diff = train_test_split(df_drivers_diff, diff_swap, test_size=0.2, shuffle=False)

    data_full = [df_swap, df_drivers, diff_swap, df_drivers_diff]
    data = [x_train, x_test, y_train, y_test]
    data_diff = [x_train_diff, x_test_diff, y_train_diff, y_test_diff]

    return data_full, data, data_diff

def getDataTablesFigures(df_swap, df_drivers, diff_swap):
    # Get descriptive statistics and corrlation matrix
    ds_swap, ds_swap_latex = get_ds(df_swap)
    ds_drivers, ds_drivers_latex = get_ds(df_drivers)

    corr_drivers = df_drivers.corr().round(3)

    # Construct VIF, first include constant to do so
    X = add_constant(df_drivers)
    VIF =  pd.Series([variance_inflation_factor(X.values, i)  for i in range(X.shape[1])], index=X.columns)

    # Get 2D and 3D plots of the swap rate
    get2dplot(df_swap, 1)
    get3dplot(df_swap, 1)

    # Perform ADF test
    adf, adf_diff = adf_test(df_swap, diff_swap)

    # Print all output for Data Section
    print('Descriptive statistics swap rate ', ds_swap)
    print('Descriptive statistics drivers ', ds_drivers)
    print('Correlation drivers ', corr_drivers.to_latex())
    print('VIF:', VIF)
    print('ADF test results swap in levels', adf)
    print('ADF test results swap differences', adf_diff)


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
    latex = des_stats_table.to_latex()
    return des_stats_table, latex


def pacf_plot(df: pd.DataFrame, maxlags: int):
    for column in df:
        pacf = plot_pacf(df[column], lags=maxlags, method='ywm')
        plt.show()


def get2dplot(df: pd.DataFrame, save: bool):
    color = ["blue", "lightblue", "deepskyblue", "dodgerblue", "steelblue", "mediumblue", "darkblue", "slategrey",
             "gray", "black"]
    i = 0
    for column in df:
        df[column].plot(figsize=(15, 5), lw=1, color=color[i], label=column)
        i = i + 1
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
        adf_c = ts.adfuller(df[column], regression="c", autolag="AIC")  # constant only
        adf_ct = ts.adfuller(df[column], regression="ct", autolag="AIC")  # constant and trend
        #print('Constant only: ', adf_c[0:2])
        #print('Constant and trend: ', adf_ct[0:2])
        ADF_c.append(adf_c[0:2])
        ADF_ct.append(adf_ct[0:2])
    concat = [ADF_c, ADF_ct]
    tab = np.array(concat)
    table = tab.reshape(10, 4)

    # Define column names
    if indicator_diff == 1:
        cols = [['ADF Stat C Diff', 'p-value C Diff', 'ADF Stat CT Diff', 'p-value CT Diff']]
    elif indicator_diff == 0:
        cols = [['ADF Stat C', 'p-value C', 'ADF Stat CT', 'p-value CT']]

    # Return as pd.DataFrame
    results = pd.DataFrame(table, columns=cols)
    results.index = df.columns.values
    results = results.round(3)
    return results


def optimal_lag(x_train, y_train, maxlags, indicator):
    # Strip indices of dataframes
    x_train1 = x_train.reset_index(drop=True)
    y_train1 = y_train.reset_index(drop=True)
    dep = y_train1.T.reset_index(drop=True).T
    #dep = y_train1.iloc[:, 6]
    dep.name = 0

    AIC = []

    for i in range(1,maxlags,1):
        # Construct the model for different lags
        if indicator == 1:
            # ARX model
            #model = AutoReg(endog=dep, exog=x_train1, lags=i)
            model = AutoReg(endog=dep, exog=x_train1, lags=i)
        else:
            # AR model
            #model = AutoReg(endog=dep, lags=i)
            model = ARIMA(dep, order=(i,0,0))

        result = model.fit()
        # print('Lag Order =', i)
        # print('AIC : ', result.aic)
        # print('BIC : ', result.bic)
        # print('FPE : ', result.fpe)
        # print('HQIC: ', result.hqic, '\n')
        AIC.append(result.aic)
    idx = AIC.index(min(AIC))
    print('Optimal index is', idx)
    if indicator == 1:
        return idx, AutoReg(endog=dep, exog=x_train1, lags=idx).fit()
    else:
        #return idx, AutoReg(endog=dep, lags=idx).fit()
        return idx, ARIMA(dep, order=(idx, 0, 0)).fit()


def predict_ar(train_df, test_df, exog_train, exog_test, no_lags, indicator):
    if indicator == 1:
        mod = AutoReg(train_df, exog=exog_train, lags=no_lags)
    else:
        mod = AutoReg(train_df, lags=no_lags)
    AR_model = mod.fit()
    coef = AR_model.params
    print(AR_model.ar_lags)
    print(AR_model.summary())


    # Predict against test data
    t_train = train_df.index
    t_test = test_df.index

    if indicator == 1:
        pred = AR_model.predict(exog_oos=exog_test,start=len(train_df), end=len(train_df) + len(test_df) - 1, dynamic=True)
    else:
        pred = AR_model.predict(start=len(train_df), end=len(train_df) + len(test_df) - 1, dynamic=True)

    #pred = AR_model.predict(start=len(train_df), end=len(train_df) + len(test_df) - 1, dynamic=True)
    pred.index = t_test
    print(test_df - pred)

    # Plot the prediction vs test data
    pyplot.plot(pred)
    pyplot.plot(test_df, color='red')
    pyplot.show()


def plot_forecast(pred, test):
    plt.plot(pred, label="forecast")
    plt.plot(test, color='red', label="actual")
    plt.legend(loc="upper left")
    print("Test")
    plt.show()


# Forecast accuracy measures
def forecast_accuracy(forecast, actual, df_indicator):
    if df_indicator:
        mape = []
        me = []
        mae = []
        mpe = []
        rmse = []
        corr = []

        for column in actual:
            mape.append(np.mean(np.abs(forecast[column] - actual[column]) / np.abs(actual[column])))
            me.append(np.mean(forecast[column] - actual[column]))  # ME
            mae.append(np.mean(np.abs(forecast[column] - actual[column])))  # MAE
            mpe.append(np.mean((forecast[column] - actual[column]) / actual[column]))  # MPE
            rmse.append(np.mean((forecast[column] - actual[column]) ** 2) ** .5)  # RMSE
            corr.append(np.corrcoef(forecast[column], actual[column])[0, 1])  # corr
        print('mape: ', mape)
        print('me:', me)
        print('mae:', mae)
        print('mpe:', mpe)
        print('rmse:', rmse)
        print('corr:', corr)
        return

    else:
        mape = np.mean(np.abs(forecast - actual)/np.abs(actual))  # MAPE
        me = np.mean(forecast - actual)             # ME
        mae = np.mean(np.abs(forecast - actual))    # MAE
        mpe = np.mean((forecast - actual)/actual)   # MPE
        rmse = np.mean((forecast - actual)**2)**.5  # RMSE
        corr = np.corrcoef(forecast, actual)[0,1]   # corr
        print({'mape': mape, 'me': me, 'mae': mae,
               'mpe': mpe, 'rmse': rmse, 'corr': corr})
        return pd.DataFrame([[mape, me, mae, mpe, rmse, corr]], columns=['mape', 'me', 'mae', 'mpe', 'rmse', 'corr'])

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

def buildAE(y_train, y_test):
    # https://www.kaggle.com/code/saivarunk/dimensionality-reduction-using-keras-auto-encoder

    ncol = y_train.shape[1]

    encoding_dim = 3
    input_dim = Input(shape=(ncol,))

    # Encoder Layers
    encoded1 = Dense(3000, activation='relu')(input_dim)
    encoded2 = Dense(2750, activation='relu')(encoded1)
    encoded3 = Dense(2500, activation='relu')(encoded2)
    encoded4 = Dense(2250, activation='relu')(encoded3)
    encoded5 = Dense(2000, activation='relu')(encoded4)
    encoded6 = Dense(1750, activation='relu')(encoded5)
    encoded7 = Dense(1500, activation='relu')(encoded6)
    encoded8 = Dense(1250, activation='relu')(encoded7)
    encoded9 = Dense(1000, activation='relu')(encoded8)
    encoded10 = Dense(750, activation='relu')(encoded9)
    encoded11 = Dense(500, activation='relu')(encoded10)
    encoded12 = Dense(250, activation='relu')(encoded11)
    encoded13 = Dense(encoding_dim, activation='relu')(encoded12)

    # Decoder Layers
    decoded1 = Dense(250, activation='relu')(encoded13)
    decoded2 = Dense(500, activation='relu')(decoded1)
    decoded3 = Dense(750, activation='relu')(decoded2)
    decoded4 = Dense(1000, activation='relu')(decoded3)
    decoded5 = Dense(1250, activation='relu')(decoded4)
    decoded6 = Dense(1500, activation='relu')(decoded5)
    decoded7 = Dense(1750, activation='relu')(decoded6)
    decoded8 = Dense(2000, activation='relu')(decoded7)
    decoded9 = Dense(2250, activation='relu')(decoded8)
    decoded10 = Dense(2500, activation='relu')(decoded9)
    decoded11 = Dense(2750, activation='relu')(decoded10)
    decoded12 = Dense(3000, activation='relu')(decoded11)
    decoded13 = Dense(ncol, activation='sigmoid')(decoded12)

    # Combine Encoder and Deocder layers
    autoencoder = Model(inputs=input_dim, outputs=decoded13)

    # Compile the Model
    autoencoder.compile(optimizer='adam', loss='mae')
    # auto_encoder.compile(
    #     loss='mae',
    #     metrics=['mae'],
    #     optimizer='adam'
    # )
    autoencoder.summary()

    autoencoder.fit(y_train, y_train, epochs=10, batch_size=32, shuffle=False, validation_data=(y_test, y_test))

    encoder = Model(inputs=input_dim, outputs=encoded13)
    encoded_input = Input(shape=(encoding_dim,))
    encoded_train = pd.DataFrame(encoder.predict(y_train))
    encoded_train = encoded_train.add_prefix('feature_')

    return encoded_train





