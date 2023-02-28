import numpy as np
import pandas as pd
from VAR import get_data

def fit_ar_model(train, test, max_p):
    num_cols = test.shape[1]
    preds = None
    train_means = []
    train_stds = []
    best_p = None
    best_bic = np.inf

    # Initialize variables to keep track of best model
    best_model_mae = np.inf
    best_model_mse = np.inf
    best_model_rmse = np.inf

    for column in range(num_cols):
        train_data = train[column]
        test_data = test[column]

        # Calculate mean and standard deviation of training data
        train_mean = np.mean(train_data)
        train_std = np.std(train_data)
        train_means.append(train_mean)
        train_stds.append(train_std)

        for p in range(1,max_p+1):
            # Construct design matrix X and response vector y for OLS regression
            X = np.zeros((len(train_data) - p, p))
            y = np.zeros(len(train_data) - p)
            for j in range(p):
                X[:, j] = train_data[p - j - 1:-j - 1]
            y = train_data[p:]

            # Estimate model parameters using OLS regression
            beta = np.linalg.inv(X.T.dot(X)).dot(X.T).dot(y)

            # Make predictions on test set
            pred = []
            for j in range(len(test_data)):
                if j < p:
                    pred.append(train_data.iloc[-(p - j)])
                else:
                    y = np.dot(beta, np.flip(train_data[j - p:j]))
                    pred.append(y)

            # Calculate BIC for current p
            residuals = np.array(train_data[p:]) - np.dot(X, beta)
            n = len(train_data)
            k = p + 1
            loglik = -0.5 * n * (np.log(2 * np.pi) + np.log(np.mean(residuals ** 2)) + 1)
            bic = -2 * loglik + k * np.log(n)

            # Calculate performance metrics
            mae = np.mean(np.abs(np.array(pred) - test_data))
            mse = np.mean((np.array(pred) - test_data) ** 2)
            rmse = np.sqrt(mse)

            # Update best model if BIC is lower
            if bic < best_bic:
                best_bic = bic
                best_p = p
                preds = pred
                best_model_mae = mae
                best_model_mse = mse
                best_model_rmse = rmse

    return preds, best_model_mae, best_model_mse, best_model_rmse, best_p


def fit_arx_model(y_train, y_test, max_p, max_q, train_X, test_X):

    preds = []
    train_means = []
    train_stds = []
    best_bic = np.inf
    best_params = (0, 0)
    for column in y_train:
        col_train_data = y_train[column]
        col_test_data = y_test[column]
        col_train_X = train_X
        col_test_X = test_X

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
                X_train = np.zeros((n, p + q * col_train_X.shape[1]))
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
                            pred.append(col_train_data.iloc[-(q - j)])
                        else:
                            if p == 1 and j == 0:
                                pred.append(beta[0])
                            else:
                                pred.append(np.dot(beta[:p], np.flip(col_train_data[-p - j:-j])))
                    else:
                        x = np.concatenate(
                            (np.flip(col_test_data[j - p:j]), np.flip(col_test_X[-q + j:].values.flatten()), np.array([1])))
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
                    preds.append(best_model_pred)

    return preds, best_model_pred, best_params, best_bic, best_model_mae, best_model_mse, best_model_rmse

# Example usage
data, df, df_diff, date_train, date_test, date_train_diff, date_test_diff = get_data(lag=5)

X_train = df[0]
X_test = df[1]
Y_train = df[2]
Y_test = df[3]

X_train_diff = df_diff[0]
X_test_diff = df_diff[1]
Y_train_diff = df_diff[2]
Y_test_diff = df_diff[3]

############ ARX ############
#fit_arx_model(Y_train, Y_test, 10, 10, X_train, X_test)

############ AR ############
preds, mae, mse, rmse, best_p = fit_ar_model(Y_train['10Y'], Y_test['10Y'], 10)
print("Predictions:\n", preds)
print("MAE:", mae)
print("MSE:", mse)
print("RMSE:", rmse)

t=1

preds, train_means, train_stds, mae, mse, rmse = fit_ar_model(Y_train_diff, Y_test_diff, 3)
print("Differenced AR process results")
print("Predictions:\n", preds)
print("MAE:", mae)
print("MSE:", mse)
print("RMSE:", rmse)
