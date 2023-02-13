import pandas as pd
#%matplotlib inline
import numpy as np
import matplotlib.pyplot as plt
from  matplotlib import cm
import matplotlib.dates as mdates
import matplotlib.ticker as ticker

#df = pd.read_csv('Swap_Data.csv', sep=';', decimal=',')
df = pd.read_excel(open('SwapDriverData.xlsx', 'rb'),
              sheet_name='Swap')
dates = df.iloc[:, 0]

print(dates[500], dates[1000], dates[1500], dates[2000], dates[2500])
df = df.iloc[:, 1:11]
print(dates)
print(df)
print(df.dtypes)
#df = pd.to_numeric(df, errors='coerce')

#df = df.astype(float, errors='ignore')
#print(df.dtypes)
#print(df)
def format_date(x, pos=None):
    return dates.num2date(x).strftime('%Y-%m-%d') #use FuncFormatter to format dates

x = np.arange(len(df.columns))
y = np.arange(len(dates))
#y = np.array(dates)
X, Y = np.meshgrid(x, y)
Z = df
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
surf = ax.plot_surface(X, Y, Z, rstride=1, cstride=1, cmap=cm.coolwarm, linewidth=0, antialiased=True)
fig.colorbar(surf, shrink=0.5, aspect=5)
ax.set_xlabel('Tenor')
ax.set_zlabel('Swap rate')


def format_date(x, pos=None):
    return dates.num2date(x).strftime('%Y')

#xticks=['0', '1', '3', '5', '10', '20', '30']
#ax.set_xticklabels(xticks)
#yticks=['01-01-2012', '12-19-2013', '12-16-2015', '12-04-2017', '11-26-2019', '11-19-2021', '12-30-2021']
#ax.set_yticklabels(yticks)
#myFmt = mdates.DateFormatter('01-01-2012', '01-01-2021')
#ax.yaxis.set_major_formatter(myFmt)
#ax.set_xticks(dates)
print("here")
#ax.w_xaxis.set_major_locator(ticker.FixedLocator(dates)) # I want all the dates on my xaxis
#ax.w_xaxis.set_major_formatter(ticker.FuncFormatter(format_date))
plt.show()
