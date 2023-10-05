import numpy as np
from numpy.polynomial import polynomial as poly
from statsmodels.tsa.tsatools import lagmat
import pandas as pd
from xgboost import XGBRegressor
from ssmatrix import *
from SSCalculate_Alternative import *
from sklearn.metrics import mean_squared_error


class SARIMAX_POLYPROCESSOR_VFL:
    def __init__(self, p, d, q, P, D, Q, S, y_window_train, x_window_train, n_clients, optim, lr=None, max_iter = None):
        self.splitclass = SSCalculate(n_clients)
        self.lagged_residuals = None
        self.last_input_window = None
        self.n_cli = n_clients
        self.y_predicted = None
        self.coeffs_armax = None
        self.last_resid = None
        self.residuals_ar = None
        self.optim = optim
        self.lr = lr
        self.max_iter = max_iter
        self.mean_res = None
        self.stddev_res = None
        self.poly_p = np.ones(p + 1)
        self.poly_P = np.zeros(P * S + 1)
        self.poly_P_indices = np.arange(P * S + 1)
        if S > 0:
            self.poly_P[self.poly_P_indices % S == 0] = 1
        elif S == 0:
            self.poly_P[0] = 1
        self.poly_d = poly.polypow([1, 1], d)
        self.poly_D = np.zeros(S + 1)
        self.poly_D[0] = 1
        self.poly_D[-1] = 1
        self.poly_D = poly.polypow(self.poly_D, D)
        self.ar_poly = np.polymul(np.polymul(np.polymul(self.poly_p, self.poly_P), self.poly_d), self.poly_D)
        self.ar_poly[self.ar_poly != 0] = 1
        self.poly_q = np.ones(q + 1)
        self.poly_Q = np.zeros(Q * S + 1)
        self.poly_Q_indices = np.arange(Q * S + 1)
        if S > 0:
            self.poly_Q[self.poly_Q_indices % S == 0] = 1
        elif S == 0:
            self.poly_Q[0] = 1
        self.ma_poly = np.polymul(self.poly_q, self.poly_Q)
        self.ma_poly[self.ma_poly != 0] = 1

        starting_t = max(len(self.ar_poly) - 1, len(self.ma_poly) - 1)
        self.lagged_y_train = lagmat(y_window_train, starting_t, "both", "in")
        self.ar_indices = np.arange(len(self.ar_poly))
        self.ar_sel = self.ar_poly != 0
        self.ar_selected_columns = self.ar_indices[self.ar_sel]
        self.lagged_y_train_selected = self.lagged_y_train[:, self.ar_selected_columns]
        errors_train = np.zeros((len(y_window_train), len(self.ma_poly) - 1))
        self.ma_indices = np.arange(len(self.ma_poly))
        self.ma_sel = self.ma_poly != 0
        self.ma_selected_columns = self.ma_indices[self.ma_sel]
        self.coeffs_ar = np.random.rand(1, len(self.ar_selected_columns) - 1).reshape((-1, 1))
        self.coeffs_ma = np.random.rand(1, len(self.ma_selected_columns) - 1).reshape((-1, 1))
        if isinstance(x_window_train, pd.DataFrame):
            self.coeffs_exog = np.random.rand(1, len(x_window_train.to_numpy()[0])).reshape((-1, 1))
        else:
            self.coeffs_exog = np.random.rand(1, len(x_window_train[0])).reshape((-1, 1))
        self.coeffs_bias = np.random.rand()
        self.lagged_x_window_train = x_window_train[starting_t:]

    def step1(self):
        if rank != 0:
            X = np.concatenate(
                (self.lagged_y_train_selected[:, 1:], self.lagged_x_window_train,
                 np.ones((len(self.lagged_x_window_train), 1))/(self.n_cli)), axis=1)



        elif rank == 0:
            X = np.concatenate(
                (self.lagged_y_train_selected[:, 1:], self.lagged_x_window_train,
                 np.zeros((len(self.lagged_x_window_train), 1))), axis=1)



        if self.optim == "NE":
            coeff_res = NormalEqSolver(X, self.lagged_y_train_selected[:, [0]], self.splitclass, self.n_cli)
        elif self.optim == "GD":
            est_coeffs = np.random.random((X.shape[1], 1))
            Y_temp = self.lagged_y_train_selected[:, [0]]
            for iter in range(self.max_iter):
                prediction2 = SMATMUL(X, est_coeffs, self.splitclass)
                transpose = np.transpose(X)
                diff = Y_temp - prediction2
                gradient_distributed = 2 * self.lr * SMATMUL(transpose, diff, self.splitclass)
                est_coeffs = est_coeffs + gradient_distributed

            coeff_res = est_coeffs

        self.coeffs_ar = coeff_res[:len(self.lagged_y_train_selected[0]) - 1]
        self.coeffs_exog = coeff_res[len(self.lagged_y_train_selected[0]) - 1: -1]
        self.coeffs_bias = coeff_res[-1]
        # y_train_pred = np.matmul(self.lagged_y_train_selected[:, 1:], self.coeffs_ar) + self.coeffs_bias + np.matmul(
        #     self.lagged_x_window_train, self.coeffs_exog)
        y_train_pred = SMATMUL(self.lagged_y_train_selected[:, 1:], self.coeffs_ar, self.splitclass) + self.coeffs_bias + SMATMUL(self.lagged_x_window_train, self.coeffs_exog, self.splitclass)


        # plt.plot(y_train_pred, 'b')
        # plt.plot(self.lagged_y_train_selected[:, [0]], 'r')
        # plt.show()
        self.residuals_ar = self.lagged_y_train_selected[:, [0]] - y_train_pred

        rar = self.residuals_ar
        AGGRES = AGGSHR(rar, 1)
        if rank == 1:
            self.mean_res = np.mean(AGGRES)
            self.var_res = np.var(AGGRES)
            self.stddev_res = np.sqrt(self.var_res)
            # self.var_res = np.mean(SMATMUL(AGGRES - self.mean_res, AGGRES - self.mean_res, self.splitclass))

        # var_res = np.var(self.residuals_ar)
        # self.stddev_res = np.sqrt(var_res)
        # print("mean:", self.mean_res)
        # print("stddevres", self.stddev_res)



    def step2(self, x_window_train):
        starting_t_ma_lags = len(self.ma_poly) - 1
        self.lagged_residuals = lagmat(self.residuals_ar[:-1, [0]], starting_t_ma_lags, "forward", "in")

        if rank == 1:
            self.lagged_residuals = np.concatenate(
                (np.random.normal(self.mean_res, self.stddev_res, (1, self.lagged_residuals.shape[1])),
                 self.lagged_residuals))
        self.lagged_residuals = SHR(self.lagged_residuals, 1, self.splitclass, self.n_cli)
        self.lagged_residuals = self.lagged_residuals[:, :-1]
        y_prime = self.lagged_y_train_selected[:,
                  [0]] - self.residuals_ar  # this is what we need to optimize for in step 2  [y_s r_s] * [[ar],[ma]]

        x_new = self.lagged_y_train[:, 1:]
        x_new = np.concatenate((x_new, self.lagged_residuals), axis=1)
        columns_to_select = self.ar_selected_columns[1:] - 1
        offset_ma_coeffs = len(self.lagged_y_train[0, 1:])
        columns_to_select_2 = self.ma_selected_columns[1:] - 1 + offset_ma_coeffs
        columns_to_select_merged = np.concatenate((columns_to_select, columns_to_select_2))
        x_train_step_2 = np.concatenate((x_new[:, columns_to_select_merged], self.lagged_x_window_train), axis=1)

        # Discard earlier learned coeffs and put new ones
        self.coeffs_ar = np.random.rand(1, len(self.ar_selected_columns) - 1).reshape((-1, 1))
        if isinstance(x_window_train, pd.DataFrame):
            self.coeffs_exog = np.random.rand(1, len(x_window_train.to_numpy()[0])).reshape((-1, 1))
        else:
            self.coeffs_exog = np.random.rand(1, len(x_window_train[0])).reshape((-1, 1))
        self.coeffs_bias = np.random.rand()
        self.coeffs_armax = np.concatenate((self.coeffs_ar, self.coeffs_ma, self.coeffs_exog))
        X = np.concatenate((x_train_step_2, np.ones((len(x_train_step_2), 1))), axis=1)

        if self.optim == "NE":
            self.coeffs_armax = NormalEqSolver(X, self.lagged_y_train_selected[:, [0]], self.splitclass, self.n_cli)
        elif self.optim == "GD":
            est_coeffs = np.random.random((X.shape[1], 1))
            Y_temp = self.lagged_y_train_selected[:, [0]]
            for iter in range(self.max_iter):
                prediction2 = SMATMUL(X, est_coeffs, self.splitclass)
                transpose = np.transpose(X)
                diff = Y_temp - prediction2
                gradient_distributed = 2 * self.lr * SMATMUL(transpose, diff, self.splitclass)
                est_coeffs = est_coeffs + gradient_distributed

            self.coeffs_armax = est_coeffs

        # self.coeffs_armax = NormalEqSolver(X, self.lagged_y_train_selected[:, [0]], self.splitclass, self.n_cli)
        # self.coeffs_armax = np.matmul(np.linalg.pinv(np.matmul(X.T, X)),
        #                               np.matmul(X.T, self.lagged_y_train_selected[:, [0]]))

        global bytes_sent
        bytes_sent += self.splitclass.bytes_sent
        res = comm.gather(bytes_sent, 1)
        # if rank == 1:
        #     print("Total bytes transmitted:", np.sum(res))

        y_train_pred_step_2 = SMATMUL(x_train_step_2, self.coeffs_armax[:-1], self.splitclass) + self.coeffs_armax[-1]

        y_train_pred = AGGSHR(y_train_pred_step_2, 1)
        y_actual = AGGSHR(self.lagged_y_train_selected[:, [0]], 1)
        if rank == 1:
            y_train_pred += 1
            mse = MSE(y_actual, y_train_pred)
            print("MSE:", mse, "bytes sent:", np.sum(res), "optim:", self.optim, "samples:", len(y_actual), "iter:", self.max_iter, "lr:", self.lr)
        bytes_sent = 0  # reset the communicated byte count



        self.coeffs_bias = self.coeffs_armax[-1]
        self.coeffs_armax = self.coeffs_armax[:-1]
        self.last_input_window = x_train_step_2[-1]
        # self.y_predicted = np.matmul(self.last_input_window, self.coeffs_armax) + self.coeffs_bias
        reshaped_last_window = np.reshape(self.last_input_window, (1, self.last_input_window.shape[0]))
        self.y_predicted = SMATMUL(reshaped_last_window, self.coeffs_armax, self.splitclass) + self.coeffs_bias
        self.last_resid = self.lagged_y_train_selected[-1, [0]] - self.y_predicted

    def forecast(self, exog, Y_test):
        current_ar_window = self.lagged_y_train[-1, 1:]
        current_resid_window = self.lagged_residuals[-1]
        y_test_predicted = []
        for i in range(0, len(Y_test)):
            current_ar_window = np.roll(current_ar_window, 1)
            if len(current_ar_window) > 0:
                current_ar_window[0] = self.y_predicted
            current_ar_window_selected = current_ar_window[self.ar_selected_columns[1:] - 1]
            current_resid_window = np.roll(current_resid_window, 1)
            if len(current_resid_window) > 0:
                current_resid_window[0] = self.last_resid
            current_resid_window_selected = current_resid_window[self.ma_selected_columns[1:] - 1]
            # self.y_predicted = np.matmul(current_ar_window_selected,
            #                              self.coeffs_armax[:len(self.ar_selected_columns) - 1]) + \
            #                    np.matmul(current_resid_window_selected,
            #                              self.coeffs_armax[len(self.ar_selected_columns) - 1:len(
            #                                  self.ar_selected_columns) - 1 + len(self.ma_selected_columns) - 1]) + \
            #                    self.coeffs_bias + np.matmul(exog[i], self.coeffs_armax[
            #                                                          len(self.ar_selected_columns) - 1 + len(
            #                                                              self.ma_selected_columns) - 1:])

            reshaped_ar_window_selected = np.reshape(current_ar_window_selected, (1, current_ar_window_selected.shape[0]))
            reshaped_resid_window_selected = np.reshape(current_resid_window_selected, (1, current_resid_window_selected.shape[0]))
            reshaped_exog = np.reshape(exog[i], (1, exog[i].shape[0]))
            # print(current_ar_window_selected.shape, self.coeffs_armax[:len(self.ar_selected_columns) - 1].shape)
            # exit()
            self.y_predicted = SMATMUL(reshaped_ar_window_selected, self.coeffs_armax[:len(self.ar_selected_columns) - 1], self.splitclass)
            self.y_predicted += SMATMUL(reshaped_resid_window_selected, self.coeffs_armax[len(self.ar_selected_columns) - 1:len(
                                             self.ar_selected_columns) - 1 + len(self.ma_selected_columns) - 1], self.splitclass)
            self.y_predicted += SMATMUL(reshaped_exog, self.coeffs_armax[
                                                                     len(self.ar_selected_columns) - 1 + len(
                                                                         self.ma_selected_columns) - 1:], self.splitclass)
            self.y_predicted += self.coeffs_bias

            y_test_predicted.append(self.y_predicted)
            # self.last_resid = Y_test[i] - self.y_predicted
            # self.last_resid = np.random.normal(self.mean_res, self.stddev_res)
            self.last_resid = 0.0

        return y_test_predicted


class SARIMAX_POLYPROCESSOR:
    def __init__(self, p, d, q, P, D, Q, S, y_window_train, x_window_train):
        self.lagged_residuals = None
        self.last_input_window = None
        self.y_predicted = None
        self.coeffs_armax = None
        self.last_resid = None
        self.residuals_ar = None
        self.mean_res = None
        self.stddev_res = None
        self.poly_p = np.ones(p + 1)
        self.poly_P = np.zeros(P * S + 1)
        self.poly_P_indices = np.arange(P * S + 1)
        if S > 0:
            self.poly_P[self.poly_P_indices % S == 0] = 1
        elif S == 0:
            self.poly_P[0] = 1
        self.poly_d = poly.polypow([1, 1], d)
        self.poly_D = np.zeros(S + 1)
        self.poly_D[0] = 1
        self.poly_D[-1] = 1
        self.poly_D = poly.polypow(self.poly_D, D)
        self.ar_poly = np.polymul(np.polymul(np.polymul(self.poly_p, self.poly_P), self.poly_d), self.poly_D)
        self.ar_poly[self.ar_poly != 0] = 1

        self.poly_q = np.ones(q + 1)
        self.poly_Q = np.zeros(Q * S + 1)
        self.poly_Q_indices = np.arange(Q * S + 1)
        if S > 0:
            self.poly_Q[self.poly_Q_indices % S == 0] = 1
        elif S == 0:
            self.poly_Q[0] = 1
        self.ma_poly = np.polymul(self.poly_q, self.poly_Q)
        self.ma_poly[self.ma_poly != 0] = 1

        starting_t = max(len(self.ar_poly) - 1, len(self.ma_poly) - 1)
        self.lagged_y_train = lagmat(y_window_train, starting_t, "both", "in")
        self.ar_indices = np.arange(len(self.ar_poly))
        self.ar_sel = self.ar_poly != 0
        self.ar_selected_columns = self.ar_indices[self.ar_sel]
        self.lagged_y_train_selected = self.lagged_y_train[:, self.ar_selected_columns]
        errors_train = np.zeros((len(y_window_train), len(self.ma_poly) - 1))
        self.ma_indices = np.arange(len(self.ma_poly))
        self.ma_sel = self.ma_poly != 0
        self.ma_selected_columns = self.ma_indices[self.ma_sel]
        self.coeffs_ar = np.random.rand(1, len(self.ar_selected_columns) - 1).reshape((-1, 1))
        self.coeffs_ma = np.random.rand(1, len(self.ma_selected_columns) - 1).reshape((-1, 1))
        if isinstance(x_window_train, pd.DataFrame):
            self.coeffs_exog = np.random.rand(1, len(x_window_train.to_numpy()[0])).reshape((-1, 1))
        else:
            self.coeffs_exog = np.random.rand(1, len(x_window_train[0])).reshape((-1, 1))
        self.coeffs_bias = np.random.rand()
        self.lagged_x_window_train = x_window_train[starting_t:]

    def step1(self):
        X = np.concatenate(
            (self.lagged_y_train_selected[:, 1:], self.lagged_x_window_train,
             np.ones((len(self.lagged_x_window_train), 1))), axis=1)

        # print(np.matmul(X.T,X))


        coeff_res = np.matmul(np.linalg.pinv(np.matmul(X.T, X)), np.matmul(X.T, self.lagged_y_train_selected[:, [0]]))

        self.coeffs_ar = coeff_res[:len(self.lagged_y_train_selected[0]) - 1]
        self.coeffs_exog = coeff_res[len(self.lagged_y_train_selected[0]) - 1: -1]
        self.coeffs_bias = coeff_res[-1]
        y_train_pred = np.matmul(self.lagged_y_train_selected[:, 1:], self.coeffs_ar) + self.coeffs_bias + np.matmul(
            self.lagged_x_window_train, self.coeffs_exog)


        # plt.plot(y_train_pred, 'b')
        # plt.plot(self.lagged_y_train_selected[:, [0]], 'r')
        # plt.show()
        self.residuals_ar = self.lagged_y_train_selected[:, [0]] - y_train_pred
        self.mean_res = np.mean(self.residuals_ar)
        var_res = np.var(self.residuals_ar)
        self.stddev_res = np.sqrt(var_res)

        # print("mean:", self.mean_res)
        # print("stddevres", self.stddev_res)

    def step2(self, x_window_train):
        starting_t_ma_lags = len(self.ma_poly) - 1
        self.lagged_residuals = lagmat(self.residuals_ar[:-1, [0]], starting_t_ma_lags, "forward", "in")
        self.lagged_residuals = np.concatenate(
            (np.random.normal(self.mean_res, self.stddev_res, (1, self.lagged_residuals.shape[1])),
             self.lagged_residuals))
        self.lagged_residuals = self.lagged_residuals[:, :-1]
        y_prime = self.lagged_y_train_selected[:,
                  [0]] - self.residuals_ar  # this is what we need to optimize for in step 2  [y_s r_s] * [[ar],[ma]]

        x_new = self.lagged_y_train[:, 1:]
        x_new = np.concatenate((x_new, self.lagged_residuals), axis=1)
        columns_to_select = self.ar_selected_columns[1:] - 1
        offset_ma_coeffs = len(self.lagged_y_train[0, 1:])
        columns_to_select_2 = self.ma_selected_columns[1:] - 1 + offset_ma_coeffs
        columns_to_select_merged = np.concatenate((columns_to_select, columns_to_select_2))
        x_train_step_2 = np.concatenate((x_new[:, columns_to_select_merged], self.lagged_x_window_train), axis=1)

        # Discard earlier learned coeffs and put new ones
        self.coeffs_ar = np.random.rand(1, len(self.ar_selected_columns) - 1).reshape((-1, 1))
        if isinstance(x_window_train, pd.DataFrame):
            self.coeffs_exog = np.random.rand(1, len(x_window_train.to_numpy()[0])).reshape((-1, 1))
        else:
            self.coeffs_exog = np.random.rand(1, len(x_window_train[0])).reshape((-1, 1))
        self.coeffs_bias = np.random.rand()
        self.coeffs_armax = np.concatenate((self.coeffs_ar, self.coeffs_ma, self.coeffs_exog))
        X = np.concatenate((x_train_step_2, np.ones((len(x_train_step_2), 1))), axis=1)
        self.coeffs_armax = np.matmul(np.linalg.pinv(np.matmul(X.T, X)),
                                      np.matmul(X.T, self.lagged_y_train_selected[:, [0]]))

        y_train_pred_step_2 = np.matmul(x_train_step_2, self.coeffs_armax[:-1]) + self.coeffs_armax[-1]
        self.coeffs_bias = self.coeffs_armax[-1]
        self.coeffs_armax = self.coeffs_armax[:-1]
        self.last_input_window = x_train_step_2[-1]
        self.y_predicted = np.matmul(self.last_input_window, self.coeffs_armax) + self.coeffs_bias
        self.last_resid = self.lagged_y_train_selected[-1, [0]] - self.y_predicted
        # plt.plot(y_train_pred_step_2)
        # plt.plot(self.lagged_y_train_selected[:, [0]])
        # plt.show()

    def forecast(self, exog, Y_test):
        current_ar_window = self.lagged_y_train[-1, 1:]
        current_resid_window = self.lagged_residuals[-1]
        y_test_predicted = []
        for i in range(0, len(Y_test)):
            current_ar_window = np.roll(current_ar_window, 1)
            if len(current_ar_window) > 0:
                current_ar_window[0] = self.y_predicted
            current_ar_window_selected = current_ar_window[self.ar_selected_columns[1:] - 1]
            current_resid_window = np.roll(current_resid_window, 1)
            if len(current_resid_window) > 0:
                current_resid_window[0] = self.last_resid
            current_resid_window_selected = current_resid_window[self.ma_selected_columns[1:] - 1]
            self.y_predicted = np.matmul(current_ar_window_selected,
                                         self.coeffs_armax[:len(self.ar_selected_columns) - 1]) + \
                               np.matmul(current_resid_window_selected,
                                         self.coeffs_armax[len(self.ar_selected_columns) - 1:len(
                                             self.ar_selected_columns) - 1 + len(self.ma_selected_columns) - 1]) + \
                               self.coeffs_bias + np.matmul(exog[i], self.coeffs_armax[
                                                                     len(self.ar_selected_columns) - 1 + len(
                                                                         self.ma_selected_columns) - 1:])

            y_test_predicted.append(self.y_predicted)
            # self.last_resid = Y_test[i] - self.y_predicted
            # self.last_resid = np.random.normal(self.mean_res, self.stddev_res)
            self.last_resid = 0.0

        return y_test_predicted


class SARITRAX:
    def __init__(self, p, d, q, P, D, Q, S, y_window_train, x_window_train):
        self.lagged_residuals = None
        self.last_input_window = None
        self.y_predicted = None
        self.coeffs_armax = None
        self.last_resid = None
        self.residuals_ar = None
        self.mean_res = None
        self.stddev_res = None
        self.poly_p = np.ones(p + 1)
        self.poly_P = np.zeros(P * S + 1)
        self.poly_P_indices = np.arange(P * S + 1)
        if S > 0:
            self.poly_P[self.poly_P_indices % S == 0] = 1
        elif S == 0:
            self.poly_P[0] = 1
        self.poly_d = poly.polypow([1, 1], d)
        self.poly_D = np.zeros(S + 1)
        self.poly_D[0] = 1
        self.poly_D[-1] = 1
        self.poly_D = poly.polypow(self.poly_D, D)
        self.ar_poly = np.polymul(np.polymul(np.polymul(self.poly_p, self.poly_P), self.poly_d), self.poly_D)
        self.ar_poly[self.ar_poly != 0] = 1

        self.poly_q = np.ones(q + 1)
        self.poly_Q = np.zeros(Q * S + 1)
        self.poly_Q_indices = np.arange(Q * S + 1)
        if S > 0:
            self.poly_Q[self.poly_Q_indices % S == 0] = 1
        elif S == 0:
            self.poly_Q[0] = 1
        self.ma_poly = np.polymul(self.poly_q, self.poly_Q)
        self.ma_poly[self.ma_poly != 0] = 1

        starting_t = max(len(self.ar_poly) - 1, len(self.ma_poly) - 1)
        self.lagged_y_train = lagmat(y_window_train, starting_t, "both", "in")
        self.ar_indices = np.arange(len(self.ar_poly))
        self.ar_sel = self.ar_poly != 0
        self.ar_selected_columns = self.ar_indices[self.ar_sel]
        self.lagged_y_train_selected = self.lagged_y_train[:, self.ar_selected_columns]
        errors_train = np.zeros((len(y_window_train), len(self.ma_poly) - 1))
        self.ma_indices = np.arange(len(self.ma_poly))
        self.ma_sel = self.ma_poly != 0
        self.ma_selected_columns = self.ma_indices[self.ma_sel]
        self.coeffs_ar = np.random.rand(1, len(self.ar_selected_columns) - 1).reshape((-1, 1))
        self.coeffs_ma = np.random.rand(1, len(self.ma_selected_columns) - 1).reshape((-1, 1))
        if isinstance(x_window_train, pd.DataFrame):
            self.coeffs_exog = np.random.rand(1, len(x_window_train.to_numpy()[0])).reshape((-1, 1))
        else:
            self.coeffs_exog = np.random.rand(1, len(x_window_train[0])).reshape((-1, 1))
        self.coeffs_bias = np.random.rand()
        self.lagged_x_window_train = x_window_train[starting_t:]
        self.xgreg = XGBRegressor(random_state=123, max_depth=5)

    def step1(self):
        X = np.concatenate(
            (self.lagged_y_train_selected[:, 1:], self.lagged_x_window_train), axis=1)

        self.xgreg.fit(X, self.lagged_y_train_selected[:, [0]])

        y_train_pred = self.xgreg.predict(X)

        # plt.plot(y_train_pred, 'b')
        # plt.plot(self.lagged_y_train_selected[:, [0]], 'r')
        # plt.show()
        self.residuals_ar = self.lagged_y_train_selected[:, [0]] - y_train_pred
        self.mean_res = np.mean(self.residuals_ar)
        var_res = np.var(self.residuals_ar)
        self.stddev_res = np.sqrt(var_res)
        # print("mean:", self.mean_res)
        # print("stddevres", self.stddev_res)

    def step2(self, x_window_train):
        starting_t_ma_lags = len(self.ma_poly) - 1
        self.lagged_residuals = lagmat(self.residuals_ar[:-1, [0]], starting_t_ma_lags, "forward", "in")
        self.lagged_residuals = np.concatenate(
            (np.random.normal(self.mean_res, self.stddev_res, (1, self.lagged_residuals.shape[1])),
             self.lagged_residuals))
        self.lagged_residuals = self.lagged_residuals[:, :-1]
        y_prime = self.lagged_y_train_selected[:,
                  [0]] - self.residuals_ar  # this is what we need to optimize for in step 2  [y_s r_s] * [[ar],[ma]]

        x_new = self.lagged_y_train[:, 1:]
        x_new = np.concatenate((x_new, self.lagged_residuals), axis=1)
        columns_to_select = self.ar_selected_columns[1:] - 1
        offset_ma_coeffs = len(self.lagged_y_train[0, 1:])
        columns_to_select_2 = self.ma_selected_columns[1:] - 1 + offset_ma_coeffs
        columns_to_select_merged = np.concatenate((columns_to_select, columns_to_select_2))
        x_train_step_2 = np.concatenate((x_new[:, columns_to_select_merged], self.lagged_x_window_train), axis=1)

        # Discard earlier learned coeffs and put new ones

        X = x_train_step_2
        self.xgreg.fit(X, self.lagged_y_train_selected[:, [0]])
        self.last_input_window = x_train_step_2[-1]
        self.y_predicted = self.xgreg.predict(np.reshape(self.last_input_window, (1, X.shape[1])))
        self.last_resid = self.lagged_y_train_selected[-1, [0]] - self.y_predicted

    def forecast(self, exog, Y_test):
        current_ar_window = self.lagged_y_train[-1, 1:]
        current_resid_window = self.lagged_residuals[-1]
        y_test_predicted = []
        for i in range(0, len(Y_test)):
            current_ar_window = np.roll(current_ar_window, 1)
            if len(current_ar_window) > 0:
                current_ar_window[0] = self.y_predicted
            current_ar_window_selected = current_ar_window[self.ar_selected_columns[1:] - 1]
            current_resid_window = np.roll(current_resid_window, 1)
            if len(current_resid_window) > 0:
                current_resid_window[0] = self.last_resid
            current_resid_window_selected = current_resid_window[self.ma_selected_columns[1:] - 1]

            x_inp = np.concatenate((current_ar_window_selected, current_resid_window_selected, exog[i]))
            x_inp = np.reshape(x_inp, (1, x_inp.shape[0]))

            self.y_predicted = self.xgreg.predict(x_inp)

            y_test_predicted.append(self.y_predicted)
            # self.last_resid = Y_test[i] - self.y_predicted
            # self.last_resid = np.random.normal(self.mean_res, self.stddev_res)
            self.last_resid = 0.0

        return y_test_predicted
