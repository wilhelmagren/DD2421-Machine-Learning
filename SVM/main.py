# Imports
import numpy as np
import datasets as ds
import matplotlib.pyplot as plt
from scipy.optimize import minimize

# --- Global Variables ----------------------------------------------------------------------
N = 100  # Number of training samples
C = 50  # Upper bound for constraint, used for slack variables good when noisy data
bounds = [(0, C) for b in range(N)]  # Lower bound for alpha values in the B-array
start = np.zeros(N)  # Initial guess of the alpha-vector

# -- These global variables are fuking useless, but needed because of the lab definitions ---
target = []
inputs = []
P = []
# -------------------------------------------------------------------------------------------


def kernel(x, y, ker, arg):
    if ker == 'lin':
        return kernel_linear(x, y)
    elif ker == 'poly':
        return kernel_poly(x, y, arg)
    else:
        return kernel_rbf(x, y, arg)


def kernel_rbf(x, y, sig):
    return np.exp(-np.linalg.norm(x - y)/(2 * sig**2))


def kernel_poly(x, y, p):
    return np.power(np.dot(x.T, y) + 1, p)


def kernel_linear(x, y):
    return np.dot(x.T, y)


def prepare_p(data, target, kernel_tpe='lin', kernel_arg=None):
    """
        Precompute P and stuff
    """
    P = np.zeros([N, N], dtype=np.double)
    for i in range(N):
        for j in range(N):
            P[i][j] = np.dot(np.dot(target[i], target[j]), kernel(data[i], data[j], kernel_tpe, kernel_arg))
    return P


def objective(alpha):
    """
        @spec objective(np.array()) :: scalar
        func objective/1 takes the alpha-vector and calculates the 'dual-problem'.
            This is a version of the optimization problem that has computational advantages.
            We can utilize the 'kernel trick', eliminating the need for evaluating phi.
            This allows us to use transformations into very high-dimensional spaces without
            the penalty of excessive computation costs.
            Find the values alpha_i which minimizes the expression:
                1/2 SUM_i [SUM_j [alpha_i * alpha_j * target_i * target_j * K(x_i, x_j)]] - SUM_i [alpha_i]      (4)
    """
    return (np.dot(alpha, np.dot(alpha, P))/2) - np.sum(alpha)


def zero_fun(alpha):
    return np.dot(alpha, target)


def b_value(nonzero, kernel_tpe, kernel_arg):
    summer = 0
    for i, element in enumerate(nonzero):
        summer += element[0] * element[2] * kernel(nonzero[0][1], element[1], kernel_tpe, kernel_arg)
    return summer - nonzero[0][2]


def extract_nonzero_list(alpha, inpu, targets):
    return [(alpha[i], inpu[i], targets[i]) for i in range(N) if 10e-5 < alpha[i] < C]


def indicator(nonzero, x, y, b, kernel_tpe, kernel_arg):
    summer = 0
    for element in nonzero:
        summer += element[0] * element[2] * kernel(np.array([x, y]), element[1], kernel_tpe, kernel_arg)
    return summer - b


def plot(nonzero, b, ker_type, ker_args, xx, savefig=False):
    fig = plt.Figure()
    xgrid = np.linspace(-1.2, 1.2)
    ygrid = np.linspace(-1.2, 0.8)
    grid = np.array([[indicator(nonzero, x, y, b, ker_type, ker_args) for x in xgrid] for y in ygrid])
    plt.contour(xgrid, ygrid, grid, (-1.0, 0.0, 1.0), colors=('red', 'black', 'blue'),
                linewidths=(1, 2, 1))
    plt.title("{} kernel with sigma = {}, and C = {}".format(ker_type, ker_args, C))
    if savefig:
        #s = ""
        #if xx < 10:
        #    s = "00" + str(xx)
        #if 10 <= xx < 100:
        #    s = "0" + str(xx)
        filename = f"plot_{ker_type}_arg={xx}.png"
        plt.savefig(f'out/{filename}')  # Save a copy of the plot
        plt.close()
        return

    plt.show()


def perform_task(ker_tpe, ker_arg, dataset_arg, xx=0.0, verbose=False):
    np.random.seed(69)
    global inputs, target, P
    classA, classB, inputs, target = ds.generate_data(N, dataset_arg, verbose)
    ds.plot_classes(classA, classB, True)
    P = prepare_p(inputs, target, ker_tpe, ker_arg)
    XC = {'type': 'eq', 'fun': zero_fun}
    ret = minimize(objective, start, bounds=bounds, constraints=XC)
    if not ret['success']:
        print('<| bro you fucked up\n<| terminating...\n')
        raise ValueError('\t[ERROR] can\'t find it')
    alpha = ret['x']
    nonzero = extract_nonzero_list(alpha, inputs, target)
    # print(f"nonzero yo:\n {nonzero}")
    plot(nonzero, b_value(nonzero, ker_tpe, ker_arg), ker_tpe, ker_arg, xx, True)


def main():
    # rbf_tpe, rbf_arg, dataset_arg, verbose
    #dic = [float(x/10) for x in range(1, 20)]
    #for i in dic:
    #    xx = i*10
    #    print("<| Time to go, turn around {}".format(xx))
    #    perform_task('rbf', i, 4, i, False)
    perform_task('rbf', 1.2, 4, 1.2, False)


if __name__ == "__main__":
    main()
