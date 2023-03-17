import numpy as np
import pandas as pd
from statsmodels.tsa.stattools import adfuller as adf
from statsmodels.tsa.stattools import acf, pacf
from scipy.signal import find_peaks
from statsmodels.tsa.seasonal import seasonal_decompose
import seaborn as sns
from numpy.polynomial import polynomial as poly
from statsmodels.tsa.tsatools import lagmat
import matplotlib.pyplot as plt


class SARIMA:
    def __init__(self, y_train, order=(1, 0, 0), seasonal_order=(1, 0, 0, 0)):
        self.y_train = y_train
        self.order = order
        self.seasonal_order = seasonal_order


if __name__ == "__main__":
    np.random.seed(123)
    data = sns.load_dataset('flights')
    data.drop(columns=['year', 'month'], inplace=True)
    train_valid_ratio = 0.7
    train_samples = int(train_valid_ratio * len(data))
    x_train, x_valid = np.arange(0, train_samples), np.arange(train_samples, len(data))
    y_train, y_valid = data.iloc[:train_samples, ].to_numpy().reshape((-1)), data.iloc[
                                                                             train_samples:, ].to_numpy().reshape((-1))
    train_idx = np.arange(train_samples)
    valid_idx = np.arange(train_samples, len(data))

    # differencing
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

    pcf = pacf(y_train_df, alpha=0.05)
    p = 0
    for i in range(1, len(pcf[0])):
        if pcf[0][i] >= pcf[1][i][0] and pcf[0][i] <= pcf[1][i][1]:
            p = i
            break

    q = 0
    af = acf(y_train_df, alpha=0.05)
    for i in range(1, len(pcf[0])):
        if af[0][i] >= af[1][i][0] and af[0][i] <= af[1][i][1]:
            q = i
            break

    peaks = find_peaks(af[0])
    values = af[0][peaks[0]]
    primary_period = peaks[0][np.argmax(values)]
    decompose = seasonal_decompose(y_train, period=primary_period)

    P = p
    D = d
    Q = q
    S = primary_period

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
    y = data.iloc[:, ].to_numpy().reshape((-1))
    lagged_y = lagmat(y, starting_t, "both", "in")
    train_samples = int(train_valid_ratio * len(lagged_y))
    y_train = lagged_y[:train_samples]
    y_valid = lagged_y[train_samples:]
    ar_indices = np.arange(len(ar_poly))
    ar_sel = ar_poly != 0
    ar_selected_columns = ar_indices[ar_sel]
    y_train_selected = y_train[:, ar_selected_columns]
    y_valid_selected = y_valid[:, ar_selected_columns]
    errors_train = np.zeros((len(y_train), len(ma_poly) - 1))
    ma_indices = np.arange(len(ma_poly))
    ma_sel = ma_poly != 0
    ma_selected_columns = ma_indices[ma_sel]
    coeffs_ar = np.random.rand(1, len(ar_selected_columns) - 1).reshape((-1, 1))
    coeffs_ma = np.random.rand(1, len(ma_selected_columns) - 1).reshape((-1, 1))
    coeffs_bias = np.random.rand()

    """Step one optimization"""
    learning_rate = 0.000001

    for i in range(40000):
        y_train_pred = np.matmul(y_train_selected[:, 1:], coeffs_ar) + coeffs_bias
        residuals_ar = y_train_selected[:, [0]] - y_train_pred
        gradient_coeffs_ar = np.matmul(-y_train_selected[:, 1:].T, residuals_ar) / len(residuals_ar)
        gradient_coeffs_bias = -np.mean(residuals_ar)
        coeffs_ar = coeffs_ar - learning_rate * gradient_coeffs_ar
        coeffs_bias = coeffs_bias - learning_rate * gradient_coeffs_bias

    y_train_pred = np.matmul(y_train_selected[:, 1:], coeffs_ar) + coeffs_bias
    residuals_ar = y_train_selected[:, [0]] - y_train_pred

    """Step two optimization (joint)"""
    starting_t_ma_lags = len(ma_poly) - 1
    lagged_residuals = lagmat(residuals_ar[:-1, [0]], starting_t_ma_lags, "forward", "in")
    lagged_residuals = np.concatenate((np.zeros((1, lagged_residuals.shape[1])), lagged_residuals))
    lagged_residuals = lagged_residuals[:, :-1]
    y_prime = y_train_selected[:,
              [0]] - residuals_ar  # this is what we need to optimize for in step 2  [y_s r_s] * [[ar],[ma]]

    x_new = y_train[:, 1:]
    x_new = np.concatenate((x_new, lagged_residuals), axis=1)
    columns_to_select = ar_selected_columns[1:] - 1
    offset_ma_coeffs = len(y_train[0, 1:])
    columns_to_select_2 = ma_selected_columns[1:] - 1 + offset_ma_coeffs
    columns_to_select_merged = np.concatenate((columns_to_select, columns_to_select_2))
    x_train_step_2 = x_new[:, columns_to_select_merged]

    # Discard earlier learned coeffs and put new ones
    coeffs_ar = np.random.rand(1, len(ar_selected_columns) - 1).reshape((-1, 1))
    # coeffs_ma = np.random.rand(1, len(ma_selected_columns) - 1).reshape((-1, 1))
    coeffs_bias = np.random.rand()
    coeffs_arma = np.concatenate((coeffs_ar, coeffs_ma))

    for i in range(40000):
        y_train_pred_step_2 = np.matmul(x_train_step_2, coeffs_arma) + coeffs_bias
        residuals_step_2 = y_prime - y_train_pred_step_2
        gradient_coeffs_arma = np.matmul(-x_train_step_2.T, residuals_step_2) / len(y_prime)
        gradient_coeffs_bias = -np.mean(residuals_step_2)
        coeffs_arma = coeffs_arma - learning_rate * gradient_coeffs_arma
        coeffs_bias = coeffs_bias - learning_rate * gradient_coeffs_bias

    # plt.plot(y_train_pred_step_2)
    # plt.plot(y_train_selected[:, [0]])
    # plt.plot(y_train_pred)

    # Validation set predictions (Recursive forecasting)
    last_input_window = x_train_step_2[-1]
    y_predicted = np.matmul(last_input_window, coeffs_arma) + coeffs_bias
    last_resid = y_train_selected[-1, [0]] - y_predicted

    y_valid_true = y_valid_selected[:, [0]]

    current_ar_window = y_train[-1, 1:]
    current_resid_window = lagged_residuals[-1]

    y_valid_predicted = []
    for i in range(len(y_valid_true)):
        current_ar_window = np.roll(current_ar_window, 1)
        current_ar_window[0] = y_predicted
        current_ar_window_selected = current_ar_window[ar_selected_columns[1:] - 1]
        current_resid_window = np.roll(current_resid_window, 1)
        current_resid_window[0] = last_resid
        current_resid_window_selected = current_resid_window[ma_selected_columns[1:] - 1]
        y_predicted = np.matmul(current_ar_window_selected, coeffs_arma[:len(ar_selected_columns) - 1]) + np.matmul(
            current_resid_window_selected, coeffs_arma[len(ar_selected_columns) - 1:]) + coeffs_bias
        y_valid_predicted.append(y_predicted)
        last_resid = y_valid_true[i, [0]] - y_predicted
    # current_window = np.roll(last_input_window[], 1)
    # current_window[0] = y_predicted
    #
    plt.plot(y_valid[:, [0]])
    plt.plot(y_valid_predicted)
    plt.show()
    print()
