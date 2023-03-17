import numpy as np
import pandas as pd
from statsmodels.tsa.stattools import adfuller as adf
from statsmodels.tsa.stattools import acf, pacf
from scipy.signal import find_peaks
from statsmodels.tsa.seasonal import seasonal_decompose
from sdt.changepoint import Pelt
import seaborn as sns
from numpy.polynomial import polynomial as poly
from statsmodels.tsa.tsatools import lagmat
from sklearn import preprocessing
import matplotlib.pyplot as plt
from sklearn.metrics import r2_score, mean_squared_error
from forecast_airquality import mean

if __name__ == "__main__":
    np.random.seed(123)
    data = pd.read_csv('AirQualityUCI/AirQualityUCI.csv', sep=';', decimal=',')
    data.drop(['Unnamed: 15', 'Unnamed: 16'], axis=1, inplace=True)
    data.replace(to_replace=',', value='.', regex=True, inplace=True)

    for i in 'C6H6(GT) T RH AH'.split():
        data[i] = pd.to_numeric(data[i], errors='coerce')
    data.replace(to_replace=-200, value=np.nan, inplace=True)
    data['Date'] = pd.to_datetime(data['Date'], dayfirst=True).dt.date
    data['Time'] = pd.to_datetime(data['Time'], format='%H.%M.%S').dt.time
    data.drop('NMHC(GT)', axis=1, inplace=True)
    data.drop_duplicates(inplace=True)
    data.dropna(how='any', axis=0, inplace=True)

    data.reset_index(drop=True, inplace=True)
    datetimecol = pd.to_datetime(data['Date'].astype(str) + ' ' + data['Time'].astype(str))
    data['DateTime'] = datetimecol
    data.drop(['Date', 'Time'], axis=1, inplace=True)
    cols = data.columns.tolist()
    cols = cols[-1:] + cols[:-1]
    data = data[cols]
    data['year'] = data['DateTime'].dt.year
    data['month'] = data['DateTime'].dt.month
    data['day'] = data['DateTime'].dt.day
    data['hour'] = data['DateTime'].dt.hour
    data.drop(['DateTime'], axis=1, inplace=True)

    # assume for now we're trying to predict the CO(GT) column
    train_valid_ratio = 0.7

    x, y = data.iloc[:, 1:], data.iloc[:, 0]

    train_samples = int(train_valid_ratio * len(data))
    train_idx = np.arange(train_samples)
    valid_idx = np.arange(train_samples, len(data))
    x_train, x_valid = x.iloc[:train_samples].to_numpy(), x.iloc[train_samples:].to_numpy()
    y_train, y_valid = y.iloc[:train_samples].to_numpy(), y.iloc[train_samples:].to_numpy()
    x, y = x.to_numpy(), y.to_numpy()
    y_train_df = pd.DataFrame(y_train)
    d = 0
    while True:
        diff = adf(y_train_df)
        if diff[1] <= 0.05:  # its stationary
            break
        else:
            y_train_df = y_train_df.diff()
            y_train_df.dropna(inplace=True)
            d += 1

    af = acf(y_train_df, alpha=0.05)
    peaks = find_peaks(af[0])
    values = af[0][peaks[0]]
    primary_period = peaks[0][np.argmax(values)]

    decompose = seasonal_decompose(y_train, period=primary_period)

    seasonal_component = decompose.seasonal[~np.isnan(decompose.seasonal)]
    D = d
    pcf = pacf(y_train_df, alpha=0.05)
    P = 0
    for i in range(1, len(pcf[0])):
        if pcf[1][i][0] <= pcf[0][i] <= pcf[1][i][1]:
            P = i
            break

    Q = 0
    for i in range(1, len(pcf[0])):
        if af[1][i][0] <= af[0][i] <= af[1][i][1]:
            Q = i
            break

    # now to get p and q
    pcf = pacf(y_train_df[:primary_period], alpha=0.05)
    p = 0
    for i in range(1, len(pcf[0])):
        if pcf[1][i][0] <= pcf[0][i] <= pcf[1][i][1]:
            p = i
            break

    q = 0
    af = acf(y_train_df[:primary_period], alpha=0.05)
    for i in range(1, len(pcf[0])):
        if af[1][i][0] <= af[0][i] <= af[1][i][1]:
            q = i
            break
    S = primary_period

    # decompose = seasonal_decompose(y_train, period=primary_period)
    det = Pelt(cost="l2", min_size=200, jump=1)
    chgpts = det.find_changepoints(y_train, 1)

    window_start = 0
    current_window_end_index = 0
    # window_end = chgpts[current_window_end_index]
    mse_valid_dec = []
    mse_valid_sarimax = []
    mse_valid_skforecast = []
    mse_valid_lstm = []
    while current_window_end_index <= min(5, len(chgpts)):
        if current_window_end_index == len(chgpts):  # last window
            y_window = y[window_start:]
            x_window = x[window_start:]
        else:
            y_window = y[window_start:chgpts[current_window_end_index]]
            x_window = x[window_start:chgpts[current_window_end_index]]

        min_max_scaler = preprocessing.MinMaxScaler()
        x_window = min_max_scaler.fit_transform(x_window)
        x_window_train = x_window[:int(train_valid_ratio * len(x_window))]
        x_window_valid = x_window[int(train_valid_ratio * len(x_window)):]
        y_window_train = y_window[:int(train_valid_ratio * len(y_window))]
        y_window_valid = y_window[int(train_valid_ratio * len(y_window)):]

        # get the polynomials for SARIMA training
        poly_p = np.ones(p + 1)
        poly_P = np.zeros(P * S + 1)
        poly_P_indices = np.arange(P * S + 1)
        poly_P[poly_P_indices % S == 0] = 1
        poly_d = poly.polypow([1, 1], d)
        poly_D = np.zeros(S + 1)
        poly_D[0] = 1
        poly_D[-1] = 1
        poly_D = poly.polypow(poly_D, D)
        ar_poly = np.polymul(np.polymul(np.polymul(poly_p, poly_P), poly_d), poly_D)
        ar_poly[ar_poly != 0] = 1

        poly_q = np.ones(q + 1)
        poly_Q = np.zeros(Q * S + 1)
        poly_Q_indices = np.arange(Q * S + 1)
        poly_Q[poly_Q_indices % S == 0] = 1
        ma_poly = np.polymul(poly_q, poly_Q)
        ma_poly[ma_poly != 0] = 1

        starting_t = max(len(ar_poly) - 1, len(ma_poly) - 1)
        lagged_y_train = lagmat(y_window_train, starting_t, "both", "in")
        ar_indices = np.arange(len(ar_poly))
        ar_sel = ar_poly != 0
        ar_selected_columns = ar_indices[ar_sel]
        lagged_y_train_selected = lagged_y_train[:, ar_selected_columns]
        errors_train = np.zeros((len(y_train), len(ma_poly) - 1))
        ma_indices = np.arange(len(ma_poly))
        ma_sel = ma_poly != 0
        ma_selected_columns = ma_indices[ma_sel]
        coeffs_ar = np.random.rand(1, len(ar_selected_columns) - 1).reshape((-1, 1))
        coeffs_ma = np.random.rand(1, len(ma_selected_columns) - 1).reshape((-1, 1))
        coeffs_exog = np.random.rand(1, len(x_window_train[0])).reshape((-1, 1))
        coeffs_bias = np.random.rand()

        lagged_x_window_train = x_window_train[starting_t:]
        """Step one optimization"""
        learning_rate = 0.001

        """Using SGD"""
        for i in range(90000):
            y_train_pred = np.matmul(lagged_y_train_selected[:, 1:], coeffs_ar) + np.matmul(
                lagged_x_window_train, coeffs_exog) + coeffs_bias
            residuals_ar = lagged_y_train_selected[:, [0]] - y_train_pred
            gradient_coeffs_ar = np.matmul(-lagged_y_train_selected[:, 1:].T, residuals_ar) / len(residuals_ar)
            gradient_coeffs_bias = -np.mean(residuals_ar)
            gradient_coeffs_exog = np.matmul(-lagged_x_window_train.T, residuals_ar) / len(residuals_ar)
            coeffs_ar = coeffs_ar - learning_rate * gradient_coeffs_ar
            coeffs_bias = coeffs_bias - learning_rate * gradient_coeffs_bias
            coeffs_exog = coeffs_exog - learning_rate * gradient_coeffs_exog

        """Using normal equation"""
        # X = np.concatenate(
        #     (lagged_y_train_selected[:, 1:], lagged_x_window_train, np.ones((len(lagged_x_window_train), 1))), axis=1)
        # coeff_res = np.matmul(np.linalg.pinv(np.matmul(X.T, X)), np.matmul(X.T, lagged_y_train_selected[:, [0]]))
        #
        # coeffs_ar = coeff_res[:len(lagged_y_train_selected[0])-1]
        # coeffs_exog = coeff_res[len(lagged_y_train_selected[0])-1: -1]
        # coeffs_bias = coeff_res[-1]
        y_train_pred = np.matmul(lagged_y_train_selected[:, 1:], coeffs_ar) + coeffs_bias + np.matmul(
            lagged_x_window_train, coeffs_exog)
        # plt.plot(y_train_pred, 'b')
        # plt.plot(lagged_y_train_selected[:, [0]], 'r')
        # plt.show()
        residuals_ar = lagged_y_train_selected[:, [0]] - y_train_pred
        mean_res = np.mean(residuals_ar)
        var_res = np.var(residuals_ar)
        stddev_res = np.sqrt(var_res)

        print("mean:", mean_res)
        print("stddevres", stddev_res)
        # coeffs_ar_step_1 = coeffs_ar
        # coeffs_exog_step_1 = coeffs_exog
        # coeffs_bias_step_1 = coeffs_bias

        """Step two optimization (joint)"""
        starting_t_ma_lags = len(ma_poly) - 1
        lagged_residuals = lagmat(residuals_ar[:-1, [0]], starting_t_ma_lags, "forward", "in")
        lagged_residuals = np.concatenate(
            (np.random.normal(mean_res, stddev_res, (1, lagged_residuals.shape[1])), lagged_residuals))
        # lagged_residuals = np.concatenate((np.zeros((1, lagged_residuals.shape[1])), lagged_residuals))
        lagged_residuals = lagged_residuals[:, :-1]
        y_prime = lagged_y_train_selected[:,
                  [0]] - residuals_ar  # this is what we need to optimize for in step 2  [y_s r_s] * [[ar],[ma]]

        x_new = lagged_y_train[:, 1:]
        x_new = np.concatenate((x_new, lagged_residuals), axis=1)
        columns_to_select = ar_selected_columns[1:] - 1
        offset_ma_coeffs = len(lagged_y_train[0, 1:])
        columns_to_select_2 = ma_selected_columns[1:] - 1 + offset_ma_coeffs
        columns_to_select_merged = np.concatenate((columns_to_select, columns_to_select_2))
        x_train_step_2 = np.concatenate((x_new[:, columns_to_select_merged], lagged_x_window_train), axis=1)

        # Discard earlier learned coeffs and put new ones
        coeffs_ar = np.random.rand(1, len(ar_selected_columns) - 1).reshape((-1, 1))
        coeffs_exog = np.random.rand(1, len(x_window_train[0])).reshape((-1, 1))
        # coeffs_ma = np.random.rand(1, len(ma_selected_columns) - 1).reshape((-1, 1))
        coeffs_bias = np.random.rand()
        coeffs_armax = np.concatenate((coeffs_ar, coeffs_ma, coeffs_exog))

        """Using SGD"""
        for i in range(90000):
            y_train_pred_step_2 = np.matmul(x_train_step_2, coeffs_armax) + coeffs_bias
            residuals_step_2 = y_prime - y_train_pred_step_2
            gradient_coeffs_armax = np.matmul(-x_train_step_2.T, residuals_step_2) / len(y_prime)
            gradient_coeffs_bias = -np.mean(residuals_step_2)
            coeffs_armax = coeffs_armax - learning_rate * gradient_coeffs_armax
            coeffs_bias = coeffs_bias - learning_rate * gradient_coeffs_bias

        """Using normal equation"""
        # X = np.concatenate((x_train_step_2, np.ones((len(x_train_step_2), 1))), axis=1)
        # coeffs_armax = np.matmul(np.linalg.pinv(np.matmul(X.T, X)), np.matmul(X.T, lagged_y_train_selected[:, [0]]))

        # y_train_pred_step_2 = np.matmul(x_train_step_2, coeffs_armax[:-1]) + coeffs_armax[-1]
        # coeffs_bias = coeffs_armax[-1]
        # coeffs_armax = coeffs_armax[:-1]
        # plt.plot(y_train_pred_step_2)
        # plt.plot(lagged_y_train_selected[:, [0]])
        # plt.plot(y_train_pred)
        # plt.show()


        last_input_window = x_train_step_2[-1]
        y_predicted = np.matmul(last_input_window, coeffs_armax) + coeffs_bias
        # step_1_model_prediction = np.matmul(x_train_step_2[-1, :len(columns_to_select)], coeffs_ar_step_1) + \
        #     np.matmul(lagged_x_window_train[-1], coeffs_exog_step_1) + coeffs_bias_step_1

        last_resid = lagged_y_train_selected[-1, [0]] - y_predicted
        # last_resid = lagged_y_train_selected[-1, [0]] - step_1_model_prediction

        y_valid_true = y_window_valid

        current_ar_window = lagged_y_train[-1, 1:]
        current_resid_window = lagged_residuals[-1]

        y_valid_predicted = []
        for i in range(len(y_valid_true)):
            current_ar_window = np.roll(current_ar_window, 1)
            current_ar_window[0] = y_predicted
            current_ar_window_selected = current_ar_window[ar_selected_columns[1:] - 1]
            current_resid_window = np.roll(current_resid_window, 1)
            current_resid_window[0] = last_resid
            current_resid_window_selected = current_resid_window[ma_selected_columns[1:] - 1]
            y_predicted = np.matmul(current_ar_window_selected, coeffs_armax[:len(ar_selected_columns) - 1]) + \
                          np.matmul(current_resid_window_selected, coeffs_armax[len(ar_selected_columns) - 1:len(
                              ar_selected_columns) - 1 + len(ma_selected_columns) - 1]) + \
                          coeffs_bias + np.matmul(x_window_valid[i], coeffs_armax[len(ar_selected_columns) - 1 + len(
                ma_selected_columns) - 1:]) + mean_res

            y_valid_predicted.append(y_predicted)
            last_resid = y_valid_true[i] - y_predicted
            # last_resid = y_valid_true[i] - np.matmul(current_ar_window_selected, coeffs_ar_step_1) + \
            # np.matmul(x_window_valid[i], coeffs_exog_step_1) + coeffs_bias_step_1

        plt.plot(y_valid_predicted)
        plt.plot(y_valid_true)
        if current_window_end_index < len(chgpts):
            window_start = chgpts[current_window_end_index]
        current_window_end_index += 1
        plt.show()
        mse_valid_sarimax.append(mean_squared_error(y_window_valid, y_valid_predicted))
    print(mean(mse_valid_sarimax))
