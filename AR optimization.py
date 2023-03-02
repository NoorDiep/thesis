import numpy as np
import pandas as pd

from Drivers import get_data

def ar_model(y, p):
    """
    Function to fit an autoregressive (AR) model of order p to a dependent variable y.
    """
    # number of observations
    nobs = y.shape[0]

    # construct the lagged dependent variable matrix X
    X = np.zeros((nobs - p, p + 1))
    X[:, 0] = 1
    for i in range(p):
        X[:, i + 1] = y[p - i - 1:nobs - i - 1]

    # slice y to match X
    y = y[p:nobs]

    # fit the model using ordinary least squares
    beta = np.linalg.lstsq(X, y, rcond=None)[0]

    # calculate the residuals
    e = y - np.dot(X, beta)

    # calculate the variance of the residuals
    sigma2 = np.sum(e ** 2) / (nobs - p)

    # calculate the BIC score
    bic = nobs * np.log(sigma2) + p * np.log(nobs)

    return bic, beta


def ar_select(y_train, y_val, max_p):
    """
    Function to select the optimal lag for an autoregressive (AR) model using validation data.
    """
    # number of columns in y
    nvars = y_train.shape[1]
    y_train.columns = range(y_train.columns.size)
    y_val.columns = range(y_val.columns.size)
    y_conc = y_train.append(y_val)


    # initialize the BIC matrix
    bic_mat = np.zeros((max_p, nvars))
    p_mat = np.zeros(nvars)
    beta_mat = []
    # loop over the columns of y
    for j in range(nvars):
        # loop over the candidate lags
        for p in range(1, max_p + 1):
            # fit the AR model on the training data
            bic, beta = ar_model(y_train[j], p)

            # store the BIC score
            bic_mat[p - 1, j] = bic

        # select the optimal lag for this column
        p_opt = np.argmin(bic_mat[:, j]) + 1
        p_mat[j] = p_opt

        # fit the AR model using the optimal lag on the combined training and validation data
        bic, beta = ar_model(y_conc[j], p_opt)
        beta_mat.append(beta)

        # print the results for this column
        print('Column %d: optimal lag = %d, BIC = %.3f' % (j + 1, p_opt, bic))

    return bic_mat, beta_mat, p_mat


def ar_forecast(y_train, y_val, y_test, p_opt, h):
    """
    Function to make forecasts h periods ahead using an autoregressive (AR) model with optimal lag p_opt.
    """
    # number of columns in y
    nvars = y_train.shape[1]

    # number of observations in each set
    n_train = y_train.shape[0]
    n_val = y_val.shape[0]
    n_test = y_test.shape[0]

    # combine the training and validation sets
    y_tv = np.vstack((y_train, y_val))
    n_tv = n_train + n_val

    f = []

    for k in p_opt:
        k = int(k)
        # initialize the forecasts matrix
        y_fcst = np.zeros((n_tv + n_test - k - h + 1, nvars))
        # loop over the columns of y
        for j in range(nvars):
            # fit the AR model using the training and validation sets
            bic, beta = ar_model(y_tv[:, j], k)

            # use the AR model to make forecasts h periods ahead
            for i in range(n_tv - k + 1, n_tv + n_test - k - h):
                # construct the lagged dependent variable matrix X for the test set
                X = np.zeros((h, k + 1))
                X[:, 0] = 1
                for l in range(k):
                    if i - k + l + 1 < 0:
                        X[:, l + 1] = y_train[i - k + l + 1, j]
                    elif i - k + l + 1 >= n_tv:
                        X[:, l + 1] = y_fcst[i - k + l + 1 - n_tv, j]
                    else:
                        X[:, l + 1] = y_tv[i - k + l + 1, j]

                    # make the forecast
                y_fcst[i - n_tv + k + h, j] = np.dot(X, beta)

        f.append(y_fcst[n_tv - k:n_tv + n_test - k - h + 1, :])

    # return the forecasts for the test set
    return f

    # return the forecasts for the test set
    return f



def ar1_forecast(y_train, y_test, beta_opt, p_opt, h):
    """
    Function to make forecasts h periods ahead using an autoregressive (AR) model with optimal lag p_opt.
    """
    # number of columns in y
    nvars = y_train.shape[1]

    # number of observations in each set
    n_train = y_train.shape[0]
    n_test = y_test.shape[0]

    f = []




    return


# Example usage
data, df, df_diff, date_train, date_test, date_train_diff, date_test_diff = get_data(lag=5)

X_train = df[0]
X_val = df[1]
X_test = df[2]
Y_train = df[3]
Y_val = df[4]
Y_test = df[5]

 # combine the training and validation sets
Y_tv = np.vstack((Y_train, Y_val))


max_p=20
# obtain the optimal lag using ar_select
bic_opt, beta_opt, p_opt = ar_select(Y_train, Y_val, max_p)
print(bic_opt)

# make forecasts using ar_forecast

# combine the training and validation sets

h=1
y_fcst = ar1_forecast(Y_tv, Y_test, beta_opt, p_opt, h)
print(y_fcst)