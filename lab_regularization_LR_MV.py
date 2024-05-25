import numpy as np
import matplotlib.pyplot as plt
import copy
import math

# dataset
x = [[2104, 5, 1, 45], [1416, 3, 2, 40], [852, 2, 1, 35]]
y = [460, 232, 178]

x_train = np.array(x)
y_train = np.array(y)

# Create subplots for each feature
fig, axes = plt.subplots(2, 2, figsize=(12, 8))

# Feature 1 vs. Target
axes[0, 0].scatter(x_train[:, 0], y)
axes[0, 0].set_xlabel('Feature 1')
axes[0, 0].set_ylabel('Target')

# Feature 2 vs. Target
axes[0, 1].scatter(x_train[:, 1], y)
axes[0, 1].set_xlabel('Feature 2')
axes[0, 1].set_ylabel('Target')

# Feature 3 vs. Target
axes[1, 0].scatter(x_train[:, 2], y)
axes[1, 0].set_xlabel('Feature 3')
axes[1, 0].set_ylabel('Target')

# Feature 4 vs. Target
axes[1, 1].scatter(x_train[:, 3], y)
axes[1, 1].set_xlabel('Feature 4')
axes[1, 1].set_ylabel('Target')

plt.tight_layout()
plt.show()

# initialize w and b
b_init = 785.1811367994083
w_init = np.array([0.39133535, 18.75376741, -53.36032453, -26.42131618])
print(f"w_init shape: {w_init.shape}, b_init type: {type(b_init)}")

# compute prediction using vectorized version


def predict(x, w, b):
    p = np.dot(x, w) + b
    return p

# compute cost function


def compute_cost_linear_reg(X, y, w, b, lambda_=1):
    m = X.shape[0]
    n = len(w)
    cost = 0.
    for i in range(m):
        f_wb_i = np.dot(X[i], w) + b
        cost = cost + (f_wb_i - y[i])**2
    cost = cost / (2 * m)

    reg_cost = 0
    for j in range(n):
        reg_cost += (w[j]**2)
    reg_cost = (lambda_/(2*m)) * reg_cost

    total_cost = cost + reg_cost
    return total_cost


# testing cost function
np.random.seed(1)
X_tmp = np.random.rand(5, 6)
y_tmp = np.array([0, 1, 0, 1, 0])
w_tmp = np.random.rand(X_tmp.shape[1]).reshape(-1,)-0.5
b_tmp = 0.5
lambda_tmp = 0.7
cost_tmp = compute_cost_linear_reg(X_tmp, y_tmp, w_tmp, b_tmp, lambda_tmp)

print("Regularized cost:", cost_tmp)

# compute gradient


def compute_gradient_linear_reg(X, y, w, b, lambda_):

    m, n = X.shape  # (number of examples, number of features)
    dj_dw = np.zeros((n,))
    dj_db = 0.

    for i in range(m):
        err = (np.dot(X[i], w) + b) - y[i]
        for j in range(n):
            dj_dw[j] = dj_dw[j] + err * X[i, j]
        dj_db = dj_db + err
    dj_dw = dj_dw / m
    dj_db = dj_db / m

    for j in range(n):
        dj_dw[j] = dj_dw[j] + (lambda_/m) * w[j]

    return dj_db, dj_dw


# testing gradient
np.random.seed(1)
X_tmp = np.random.rand(5, 3)
y_tmp = np.array([0, 1, 0, 1, 0])
w_tmp = np.random.rand(X_tmp.shape[1])
b_tmp = 0.5
lambda_tmp = 0.7
dj_db_tmp, dj_dw_tmp = compute_gradient_linear_reg(
    X_tmp, y_tmp, w_tmp, b_tmp, lambda_tmp)

print(f"dj_db: {dj_db_tmp}", )
print(f"Regularized dj_dw:\n {dj_dw_tmp.tolist()}", )

# compute gradient descent


def gradient_descent(X, y, w_in, b_in, cost_function_rg, gradient_function_rg, alpha, num_iters):
    # An array to store cost J and w's at each iteration primarily for graphing later
    J_history = []
    w = copy.deepcopy(w_in)  # avoid modifying global w within function
    b = b_in
    lambda_ = 0.1
    for i in range(num_iters):

        # Calculate the gradient and update the parameters
        dj_db, dj_dw = compute_gradient_linear_reg(X, y, w, b, lambda_)  # None

        # Update Parameters using w, b, alpha and gradient
        w = w - alpha * dj_dw  # None
        b = b - alpha * dj_db  # None

        # Save cost J at each iteration
        if i < 100000:      # prevent resource exhaustion
            J_history.append(compute_cost_linear_reg(X, y, w, b, lambda_=0.1))

        # Print cost every at intervals 10 times or as many iterations if < 10
        if i % math.ceil(num_iters / 10) == 0:
            print(f"Iteration {i:4d}: Cost {J_history[-1]:8.2f}   ")

    return w, b, J_history  # return final w,b and J history for graphing


# initialize parameters
w_tmp = np.zeros_like(x_train[0])
b_tmp = 10
# some gradient descent settings
iterations = 100000
alpha = 0.0000007     # 5.0e-7
# run gradient descent
w_final, b_final, J_hist = gradient_descent(
    x_train, y_train, w_tmp, b_tmp, compute_cost_linear_reg, compute_gradient_linear_reg, alpha, iterations)
print(f"b,w found by gradient descent: {b_final:0.2f},{w_final} ")
m, _ = x_train.shape
for i in range(m):
    print(
        f"prediction: {np.dot(x_train[i], w_final) + b_final:0.2f}, target value: {y_train[i]}")

# testing - make prediction
predict1 = [3100, 4, 3, 47]
print(f"prediction: {np.dot(predict1, w_final) + b_final:0.2f}")
