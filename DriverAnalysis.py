# Load all data
import statsmodels.api as sm
from statsmodels.tools import add_constant

from Drivers import read_file, difference_series, optimal_lag
import pandas as pd
import numpy as np
from matplotlib import pyplot, pyplot as plt

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


diff5_swap = difference_series(df_swap, lag=5)
df_drivers_diff5 = difference_series(df_drivers, lag=5)
df_spread_diff5 = difference_series(spread, lag=5)

diff1_swap = difference_series(df_swap, lag=1)
df_drivers_diff1 = difference_series(df_drivers, lag=1)
df_spread_diff1 = difference_series(spread, lag=1)

diff20_swap = difference_series(df_swap, lag=20)
df_drivers_diff20 = difference_series(df_drivers, lag=20)
df_spread_diff20 = difference_series(spread, lag=20)
#df_drivers_diff = df_drivers.iloc[lag:]  # Cut off first n observations

# Get descriptive statistics, corelation tables, adf test results and figures for Data Section
#getDataTablesFigures()

# Create train, validation and test set
df5_swap = df_swap.iloc[5:]
df5_drivers = df_drivers.iloc[5:]
df5_drivers = df5_drivers.reset_index(drop=True)
#df_swap_dates = df_swap
df5_swap = df5_swap.reset_index(drop=True)
spread = spread.reset_index(drop=True)

df1_swap = df_swap.iloc[1:]
df1_drivers = df_drivers.iloc[1:]
df1_drivers = df1_drivers.reset_index(drop=True)
#df_swap_dates = df_swap
df1_swap = df1_swap.reset_index(drop=True)

df20_swap = df_swap.iloc[20:]
df20_drivers = df_drivers.iloc[20:]
df20_drivers = df20_drivers.reset_index(drop=True)
#df_swap_dates = df_swap
df20_swap = df20_swap.reset_index(drop=True)

    # Construct final df

df5_drivers = pd.DataFrame([df5_drivers['EcSu'],df_drivers_diff5['Sent'],df5_drivers['Stress'],df5_drivers['PoUn'],df_drivers_diff5['News'],df5_drivers['Infl']]).T
df1_drivers = pd.DataFrame([df1_drivers['EcSu'],df_drivers_diff1['Sent'],df1_drivers['Stress'],df1_drivers['PoUn'],df_drivers_diff1['News'],df1_drivers['Infl']]).T
df20_drivers = pd.DataFrame([df20_drivers['EcSu'],df_drivers_diff20['Sent'],df20_drivers['Stress'],df20_drivers['PoUn'],df_drivers_diff20['News'],df20_drivers['Infl']]).T

print(sm.OLS(df5_swap['10Y'], add_constant(df5_drivers)).fit().summary())
print(sm.OLS(diff5_swap['10Y'], add_constant(df5_drivers)).fit().summary())

print(sm.OLS(df1_swap['10Y'], add_constant(df1_drivers)).fit().summary())
print(sm.OLS(diff1_swap['10Y'], add_constant(df1_drivers)).fit().summary())

print(sm.OLS(df20_swap['10Y'], add_constant(df20_drivers)).fit().summary())
print(sm.OLS(diff20_swap['10Y'], add_constant(df20_drivers)).fit().summary())
t=1

color = ["blue", "lightblue", "lightskyblue", "deepskyblue", "dodgerblue", "steelblue", "mediumblue", "darkblue", "slategrey",
         "grey"]
i = 0
for column in diff5_swap:
    diff5_swap[column].plot(figsize=(15, 5), lw=1, color=color[i], label=column)
    i = i + 1
plt.legend(ncol=2)
plt.xlabel("Date")
plt.ylabel("Eurozone swap rate")
# plt.title("The Eurzone term structure of interest swap rate")
if bool:
    plt.savefig('swap2D.png')
plt.show()
plt.close()
