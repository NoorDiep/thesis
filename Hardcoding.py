import numpy as np
import pandas as pd
from VAR import get_data
from sklearn.linear_model import LinearRegression

def fit_ar_model(train, test, max_lag):
    train.columns = range(train.columns.size)
    test.columns = range(test.columns.size)

    try:
        num_cols = train.shape[1]
    except:
        num_cols = 1


    best_bic = np.inf
    best_preds = []
    best_p = []

    preds = []
    train_means = []
    train_stds = []

    for p in range(1, max_lag):
        preds = []
        bic = 0

        for i in range(num_cols):

            train_data = train[i]
            test_data = test[i]

            # Calculate mean and standard deviation of training data
            train_mean = np.mean(train_data)
            train_std = np.std(train_data)
            train_means.append(train_mean)
            train_stds.append(train_std)

            # Calculate autocorrelation coefficients up to lag p
            rho = [1.0]
            for k in range(1, p+1):
                rho_k = np.corrcoef(train_data[k:], train_data[:-k])[0, 1]
                rho.append(rho_k)

            # Construct design matrix X and response vector y for OLS regression
            X = np.zeros((len(train_data) - p, p))
            y = np.zeros(len(train_data) - p)
            for j in range(p):
                X[:, j] = train_data[p - j - 1:-j - 1]
            y = train_data[p:]

            # Estimate model parameters using OLS regression
            beta = np.linalg.inv(X.T.dot(X)).dot(X.T).dot(y)

            # Calculate BIC for this column
            sigma2 = np.mean((y - np.dot(X, beta)) ** 2)
            k = p + 1
            bic += np.log(len(train_data) - p) * k + np.log(sigma2) * len(train_data)

            # Make predictions on test set
            pred = []
            for j in len(test_data):
                if j < p:
                    pred.append(train_data[:-(p-j)])
                else:
                    y = np.dot(beta, np.flip(train_data[j-p:j]))
                    pred.append(y)

        # Calculate performance metrics
        if bic < best_bic:
            best_bic = bic
            best_preds = np.array(pred).T
            best_p = p

    # Calculate performance metrics
    mae = np.mean(np.abs(np.array(best_preds) - test), axis=0)
    mse = np.mean((np.array(best_preds) - test) ** 2, axis=0)
    rmse = np.sqrt(mse)

    return np.array(best_preds).T, train_means, train_stds, mae, mse, rmse, best_p


import numpy as np


def fit_arx_model(train_y, test_y, train_exog, test_exog, max_p, max_q):
    train_y.columns = range(train_y.columns.size)
    test_y.columns = range(test_y.columns.size)
    train_exog.columns = range(train_exog.columns.size)
    test_exog.columns = range(test_exog.columns.size)

    num_cols = train_y.shape[1]
    preds = []
    train_means = []
    train_stds = []
    best_bic = np.inf
    best_params = (0, 0)
    for i in range(num_cols):
        col_train_data = train_y[i]
        col_test_data = test_y[i]
        col_train_X = train_exog[i]
        col_test_X = test_exog[i]

        # Calculate mean and standard deviation of training data
        train_mean = np.mean(col_train_data)
        train_std = np.std(col_train_data)
        train_means.append(train_mean)
        train_stds.append(train_std)

        # Initialize variables to keep track of best model
        best_model_pred = None
        best_model_mae = np.inf
        best_model_mse = np.inf
        best_model_rmse = np.inf

        # Fit models for different values of p and q
        for p in range(1, max_p + 1):
            for q in range(0, max_q + 1):
                # Construct design matrix X and response vector y for OLS regression
                n = len(col_train_data) - max(p, q)
                X_train = np.zeros((n, p + q * col_train_X.size))
                y_train = np.zeros(n)
                for j in range(p):
                    X_train[:, j] = col_train_data[p - j - 1:-j - 1]
                for j in range(q):
                    X_train[:, p + j * col_train_X.shape[1]:p + (j + 1) * col_train_X.shape[1]] = col_train_X[
                                                                                                  q - j - 1:-j - 1, :]
                y_train = col_train_data[max(p, q):]

                # Estimate model parameters using OLS regression
                beta = np.linalg.inv(X_train.T.dot(X_train)).dot(X_train.T).dot(y_train)

                # Make predictions on test set
                pred = []
                for j in range(len(col_test_data)):
                    if j < max(p, q):
                        if q > j:
                            pred.append(col_train_data[-(q - j)])
                        else:
                            pred.append(np.dot(beta[:p], np.flip(col_train_data[-p - j:-j])))
                    else:
                        x = np.concatenate(
                            (np.flip(col_test_data[j - p:j]), np.flip(col_test_X[-q + j:, :].flatten()), np.array([1])))
                        y = np.dot(beta, x)
                        pred.append(y)

                # Calculate performance metrics and BIC
                mae = np.mean(np.abs(np.array(pred) - col_test_data))
                mse = np.mean((np.array(pred) - col_test_data) ** 2)
                rmse = np.sqrt(mse)
                k = p + q * col_train_X.shape[1] + 1
                n = len(col_train_data) - max(p, q)
                log_likelihood = -0.5 * (n * np.log(2 * np.pi * mse) + n)
                bic = np.log(n) * k - 2 * log_likelihood

                # Update best model if BIC is lower
                if bic < best_bic:
                    best_bic = bic
                    best_params = (p, q)
                    best_model_pred = pred
                    best_model_mae = mae
                    best_model_mse = mse
                    best_model_rmse = rmse

    return best_model_pred, best_model_mae, best_model_mse, best_model_rmse, best_params

def ar_model(y_train, y_test, max_p, forecast_periods):
    # Initialize variables
    best_p = [0]*y_train.shape[1]
    best_beta = [0]*y_train.shape[1]
    best_bic = np.array([np.inf]*y_train.shape[1])
    n = y_train.shape[0]
    best_mae = []
    best_mse = []
    best_rmse = []
    best_preds = []
    # Loop through all possible lag values
    for p in range(1, max_p+1):
        print(p)
        # Create lagged y matrix
        y_lagged = np.zeros((n-p, p*y_train.shape[1]))
        for i in range(p):
            y_lagged[:, i*y_train.shape[1]:(i+1)*y_train.shape[1]] = y_train[p-i-1:-i-1]

        # Estimate parameters with ordinary least squares
        beta = np.linalg.inv(y_lagged.T @ y_lagged) @ y_lagged.T @ y_train[p:]
        # Calculate residuals
        e = y_train[p:] - y_lagged @ beta
        # Calculate BIC score
        sigma2 = np.sum(e**2) / (n-p-y_train.shape[1])
        bic = np.log(sigma2) + p*y_train.shape[1]*np.log(n)/(n-p-y_train.shape[1])
        bic = np.array(bic)
        beta.columns = range(beta.columns.size)
        beta = pd.DataFrame(beta)
        for i in range(len(best_bic)):
            if bic[i] < best_bic[i]:
                best_p[i] = p
                best_bic[i] = bic[i]
                best_beta[i] = beta[i]

        # Predict on test set
        y_test = y_test.reset_index(drop=True)
        y_test = y_test.T.reset_index(drop=True).T
        y_test_np = y_test.to_numpy()
        if forecast_periods not in [1, 5, 30]:
            forecast_periods = 1
        y_pred = np.zeros((y_test.shape[0], y_train.shape[1]))
        for j in range(len(best_bic)):
            for i in range(best_p[j]):
                y_pred[i, j] = y_test_np[i, j]
            for i in range(best_p[j], y_test.shape[0]):
                y_lagged = np.zeros(best_p[j] * y_train.shape[1])
                for k in range(best_p[j]):
                    y_lagged[k * y_train.shape[1]:(k + 1) * y_train.shape[1]] = y_pred[i - best_p[j] + k, j]
                y_pred[i, j] = y_lagged @ best_beta[j]


                if forecast_periods > 1:
                    for f in range(1, forecast_periods):
                        y_lagged = np.zeros(best_p[j] * y_train.shape[1])
                        for k in range(best_p[j]):
                            y_lagged[k * y_train.shape[1]:(k + 1) * y_train.shape[1]] = y_pred[i - best_p[j] + k, j]
                        y_pred[i + f, j] = y_lagged @ best_beta[j]

        # Calculate out-of-sample performance
        best_mae.append(np.mean(np.abs(y_test[best_p[j]:] - y_pred[best_p[j]:, j])))
        best_mse.append(np.mean((y_test[best_p[j]:] - y_pred[best_p[j]:, j]) ** 2))
        best_rmse.append(np.sqrt(mse))
        best_preds.append(y_pred)
    # Return results
    return [best_p, best_beta, y_pred, best_mae, best_mse, best_rmse]


def ar_model1(y_train, y_test, max_p):
    # Initialize variables
    best_p = np.zeros(y_train.shape[1], dtype=int)
    best_beta = [None]*y_train.shape[1]
    best_bic = np.array([np.inf]*y_train.shape[1])
    n = y_train.shape[0]
    best_mae = []
    best_mse = []
    best_rmse = []
    best_preds = []

    y_train = y_train.to_numpy()
    y_test = y_test.to_numpy()
    # Loop through all possible lag values
    for j in range(y_train.shape[1]):
        for p in range(1, max_p+1):
            # Create lagged y matrix
            y_lagged = np.zeros((n-p, p))
            for i in range(p):
                y_lagged[:, i] = y_train[p-i-1:-i-1, j]

            # Estimate parameters with ordinary least squares
            beta = np.linalg.inv(y_lagged.T @ y_lagged) @ y_lagged.T @ y_train[p:, j]
            # Calculate residuals
            e = y_train[p:, j] - y_lagged @ beta
            # Calculate BIC score
            sigma2 = np.sum(e**2) / (n-p)
            bic = np.log(sigma2) + p*np.log(n)/(n-p)
            bic = np.array(bic)
            if bic < best_bic[j]:
                best_p[j] = p
                best_bic[j] = bic
                best_beta[j] = beta

        # Predict on test set
        y_pred = np.zeros(y_test.shape[0])
        for i in range(best_p[j]):
            y_pred[i,j] = y_test[i, j]
        for i in range(best_p[j], y_test.shape[0]):
            y_lagged = y_pred[i-best_p[j]:i, j][::-1]
            y_pred[i] = y_lagged @ best_beta[j]
        best_preds.append(y_pred)
        # Calculate out-of-sample performance
        best_mae.append(np.mean(np.abs(y_test[best_p[j]:, j] - y_pred[best_p[j]:])))
        best_mse.append(np.mean((y_test[best_p[j]:, j] - y_pred[best_p[j]:])**2))
        best_rmse.append(np.sqrt(best_mse[-1]))

    # Return results
    return [best_p, best_beta, np.array(best_preds).T, best_mae, best_mse, best_rmse]

def arx_model(y_train, y_test, x_train, x_test, max_p, max_q):
    y_train.columns = range(y_train.columns.size)
    y_test.columns = range(y_test.columns.size)
    x_train.columns = range(x_train.columns.size)
    x_test.columns = range(x_test.columns.size)

    # Initialize variables
    n = y_train.shape[0]
    m = x_train.shape[1]
    best_p = [0]*y_train.shape[1]
    best_q = [0]*y_train.shape[1]
    best_bic = [np.inf]*y_train.shape[1]
    # Loop through all possible lag combinations
    for p in range(1, max_p+1):
        for q in range(0, max_q+1):
            # Create lagged y matrix
            y_lagged = np.zeros((n-p, p*y_train.shape[1]))
            for i in range(p):
                y_lagged[:, i*y_train.shape[1]:(i+1)*y_train.shape[1]] = y_train[p-i-1:-i-1]
            # Create lagged x matrix
            if q > 0:
                x_lagged = np.zeros((n-p-q, q*m+p*y_train.shape[1]))

                for i in range(q):
                    x_lagged[:, i*m:(i+1)*m] = x_train[p+i:-q+i]
                for i in range(p):
                    x_lagged[:, q*m+i*y_train.shape[1]:(q+1)*m+i*y_train.shape[1]] = y_train[p-i-1:-q-i-1]
                X = np.hstack((np.ones((x_lagged.shape[0], 1)), x_lagged))
            else:
                X = np.hstack((np.ones((y_lagged.shape[0], 1)), y_lagged))

            # Estimate parameters with ordinary least squares
            beta = np.linalg.inv(X.T @ X) @ X.T @ y_train[p + q:, :]
            # Calculate residuals
            e = y_train[p + q:, :] - X @ beta
            # Calculate BIC score
            sigma2 = np.sum(e ** 2) / (n - p - q - m)
            bic = np.log(sigma2) + (p + q) * (y_train.shape[1] + m) * np.log(n) / (n - p - q - m)

            # Update best lag values if BIC score is lower
            for i in range(bic.size):
                if bic[i] < best_bic[i]:
                    print(bic)
                    best_p[i] = p
                    best_q[i] = q
                    best_bic[i] = bic

    # Predict on test set
    y_pred = np.zeros_like(y_test)
    for i in range(best_p):
        y_pred[i] = y_test[i]
    for i in range(best_p, y_test.shape[0]):
        x_lagged = np.zeros((1, best_q*m+best_p*y_train.shape[1]))
        if best_q > 0:
            for j in range(best_q):
                x_lagged[:, j*m:(j+1)*m] = x_test[i-best_p+j, :]
            for j in range(best_p):
                x_lagged[:, best_q*m+j*y_train.shape[1]:(best_q+1)*m+j*y_train.shape[1]] = y_pred[i-best_p+j, :]
        else:
            x_lagged[:, :best_p*y_train.shape[1]] = y_pred[i-best_p:i, :].flatten().reshape(1, -1)
        y_pred[i, :] = best_model.predict(np.hstack((1, x_lagged)))
    # Calculate performance metrics
    mae = np.mean(np.abs(y_test - y_pred))
    mse = np.mean((y_test - y_pred)**2)
    rmse = np.sqrt(mse)
    return

# Example usage
data, df, df_diff, date_train, date_test, date_train_diff, date_test_diff = get_data(lag=5)

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

#preds, train_means, train_stds, mae, mse, rmse = fit_ar_model(Y_train, Y_test, 3)
results = ar_model(Y_train, Y_test, 3, forecast_periods=1)
#preds, train_means, train_stds, mae, mse, rmse = ar_model1(Y_train, Y_test, 3)
#preds, train_means, train_stds, mae, mse, rmse = fit_arx_model(Y_train, Y_test, X_train, X_test,  3, 3)
#preds, train_means, train_stds, mae, mse, rmse = arx_model(Y_train, Y_test, X_train, X_test, 3,3)
print("Predictions:\n", preds)
print("MAE:", mae)
print("MSE:", mse)
print("RMSE:", rmse)

t=1

preds, train_means, train_stds, mae, mse, rmse = fit_arx_model(Y_train_diff, Y_test_diff, 3)
print("Differenced AR process results")
print("Predictions:\n", preds)
print("MAE:", mae)
print("MSE:", mse)
print("RMSE:", rmse)