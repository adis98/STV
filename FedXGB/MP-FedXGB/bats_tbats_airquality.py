import numpy as np


def gradients(coefficients, outputs, x):
    predicted = np.matmul(coefficients, x)
    # errors = (outputs - predicted) ** 2
    # sq_loss = np.mean(errors) / 2
    grads = np.matmul(-coefficients.T, outputs - predicted) / len(outputs)
    return grads


def solve(coefficients, outputs, initial_x, learning_rate):
    grads = gradients(coefficients, outputs, initial_x)
    while max(np.abs(grads))*learning_rate > 0.00001:
        initial_x = initial_x - (learning_rate * grads)
        grads = gradients(coefficients, outputs, initial_x)
    return grads, initial_x


if __name__ == "__main__":
    a = 11.0
    b = 88.0
    c = 15.0
    d = 37.0
    e = 49.0
    f = 120.0

    np.random.seed(123)
    scale_a = np.random.randint(1, 2)
    scale_b = np.random.randint(1, 2)
    scale_c = np.random.randint(1, 2)
    scale_d = np.random.randint(1, 2)
    scale_e = np.random.randint(1, 2)
    scale_f = np.random.randint(1, 2)
    a_1 = a/(2 * scale_a)
    a_2 = a/(2 * scale_a)
    b_1 = b/(2 * scale_b)
    b_2 = b/(2 * scale_b)
    c_1 = c/(2 * scale_c)
    c_2 = c/(2 * scale_c)
    d_1 = d/(2 * scale_d)
    d_2 = d/(2 * scale_d)
    e_1 = e/(2 * scale_e)
    e_2 = e/(2 * scale_e)
    f_1 = f/(2 * scale_f)
    f_2 = f/(2 * scale_f)
    coefficients = np.array([[0, a, b], [0, d, e]])
    outputs = np.array([c, f])

    coefficients_1 = np.array([[0, a_1, b_1], [0, d_1, e_1]])
    outputs_1 = np.array([c_1, f_1])
    initial_x = np.array([0.0, 1.0, 1.0])
    coefficients_2 = np.array([[0, a_2, b_2], [0, d_2, e_2]])
    outputs_2 = np.array([c_2, f_2])
    learning_rate = 0.0001
    # grads, new_x = solve(coefficients, outputs, initial_x, learning_rate)
    grads = gradients(coefficients, outputs, initial_x)
    grads1 = gradients(coefficients_1, outputs_1, initial_x)
    grads2 = gradients(coefficients_2, outputs_2, initial_x)
    grad = (grads1 + grads2)/2
    while np.max(grad) * learning_rate > 0.00001:
        initial_x = initial_x - learning_rate * grad
        grads1 = gradients(coefficients_1, outputs_1, initial_x)
        grads2 = gradients(coefficients_2, outputs_2, initial_x)
        grad = (grads1 + grads2)/2
        # grads_1, new_x_1 = solve(coefficients_1, outputs_1, initial_x, learning_rate)
        # grads_2, new_x_2 = solve(coefficients_2, outputs_2, initial_x, learning_rate)
        # initial_x[1] = (a_1 * new_x_1[1] + a_2 * new_x_2[1])/(a_1 + a_2)
        # initial_x[2] = (b_1 * new_x_1[2] + b_2 * new_x_2[2])/(b_1 + b_2)



    # coefficients = np.array([[0, 2, 3], [0, 3, -3]])
    # outputs = np.array([5, 0])
    # initial_x = np.array([0, 0, 0])  # bias, x, and y
    # learning_rate = 0.01
    # grads, new_x = solve(coefficients, outputs, initial_x, learning_rate)

    print()
