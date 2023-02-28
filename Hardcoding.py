import numpy as np
from VAR import get_data

def fit_ar_model(train, test, p):
    num_cols = data.shape[1]
    preds = []
    train_means = []
    train_stds = []
    for i in range(num_cols):
        train_data = train[:, i]
        test_data = test[:,i]

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

        # Make predictions on test set
        pred = []
        for j in range(len(test_data)):
            if j < p:
                pred.append(train_data[-(p-j)])
            else:
                y = np.dot(beta, np.flip(train_data[j-p:j]))
                pred.append(y)

        preds.append(pred)

    # Calculate performance metrics
    mae = np.mean(np.abs(np.array(preds) - test), axis=0)
    mse = np.mean((np.array(preds) - test) ** 2, axis=0)
    rmse = np.sqrt(mse)

    return np.array(preds).T, train_means, train_stds, mae, mse, rmse

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

preds, train_means, train_stds, mae, mse, rmse = fit_ar_model(Y_train['10Y'], Y_test['10Y'], 3)
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