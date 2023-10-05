from sklearn.datasets import make_regression
import numpy as np

if __name__ == "__main__":
    num_features = 10
    num_samples = 10
    X, Y, coeffs = make_regression(num_samples, num_features, coef=True)
    Y = np.reshape(Y, (Y.shape[0], 1))
    coeffs = np.reshape(coeffs, (coeffs.shape[0], 1))

    est_coeffs = np.random.random((X.shape[1], 1))
    alpha = 0.001
    steps = 10000

    Y_predicted_true = np.matmul(X, coeffs)
    temp = Y_predicted_true - Y
    mse = np.matmul(temp.T, temp) / (len(Y))

    cal_mse = 1
    count_steps = 0
    while count_steps < steps:
        count_steps += 1
        prediction2 = np.matmul(X, est_coeffs)
        transpose = np.transpose(X)
        diff = Y - prediction2
        gradient_distributed = alpha * np.matmul(transpose, diff)
        est_coeffs = est_coeffs + gradient_distributed
        temp_est = np.matmul(X, est_coeffs) - Y
        cal_mse = np.matmul(temp_est.T, temp_est) / (len(Y))

    print(count_steps, cal_mse, mse)
    # print(est_coeffs)
    # print(coeffs)
