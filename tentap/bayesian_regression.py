import numpy as np
from numpy.random import normal, uniform
from scipy.stats import multivariate_normal as mv_norm
import matplotlib.pyplot as plt

# ============================================================
# https://zjost.github.io/bayesian-linear-regression/#Strategy
# ============================================================


def real_fun(a_0, a_1, sigma, x):
    N = len(x)
    if sigma == 0:
        return a_0 + a_1*x
    else:
        return a_0 + a_1*x + normal(0, sigma, N)


class LinearBayes(object):
    def __init__(self, a_m0, m_S0, beta):
        self.prior = mv_norm(mean=a_m0, cov=m_S0)
        self.v_m0 = a_m0.reshape(a_m0.shape + (1,))  # Reshape to a column vector for matrix multiplications
        self.m_S0 = m_S0
        self.beta = beta
        self.v_mN = self.v_m0
        self.m_SN = self.m_S0
        self.posterior = self.prior

    def get_phi(self, a_x):
        m_phi = np.ones((len(a_x), 2))
        m_phi[:, 1] = a_x
        return m_phi

    def set_posterior(self, a_x, a_t):
        v_t = a_t.reshape(a_t.shape + (1,))  # Reshape to column vector for matrix multiplication
        m_phi = self.get_phi(a_x)
        self.m_SN = np.linalg.inv(np.linalg.inv(self.m_S0) + self.beta*m_phi.T.dot(m_phi))
        self.v_mN = self.m_SN.dot(np.linalg.inv(self.m_S0).dot(self.v_m0) + self.beta*m_phi.T.dot(v_t))
        self.posterior = mv_norm(mean=self.v_mN.flatten(), cov=self.m_SN)

    def prediction_limit(self, a_x, stdevs):
        N = len(a_x)
        m_x = self.get_phi(a_x).T.reshape((2, 1, N))
        predictions = []
        for idx in range(N):
            x = m_x[:,:, idx]
            sig_sq_x = 1/self.beta + x.T.dot(self.m_SN.dot(x))
            mean_x = self.v_mN.T.dot(x)
            predictions.append((mean_x + stdevs*np.sqrt(sig_sq_x)).flatten())
        return np.concatenate(predictions)

    def generate_data(self, a_x):
        N = len(a_x)
        m_x = self.get_phi(a_x).T.reshape((2, 1, N))
        predictions = []
        for idx in range(N):
            x = m_x[:, :, idx]
            sig_sq_x = 1 / self.beta + x.T.dot(self.m_SN.dot(x))
            mean_x = self.v_mN.T.dot(x)
            predictions.append(normal(mean_x.flatten(), np.sqrt(sig_sq_x)))
        return np.concatenate(predictions)

    def make_contour(self, a_x, a_y, real_parms=[], N=0):
        pos = np.empty(a_x.shape + (2,))
        pos[:, :, 0] = a_x
        pos[:, :, 1] = a_y

        plt.contourf(a_x, a_y, self.posterior.pdf(pos), 20)
        plt.xlabel('$w_0$', fontsize=16)
        plt.ylabel('$w_1$', fontsize=16)

        if real_parms:
            plt.scatter(real_parms[0], real_parms[1], marker='+', c='black', s=60)

        _ = plt.title('Distribution for Weight Parameters using %d datapoint(s)' % N, fontsize=10)
        plt.show()

    def make_scatter(self, a_x, a_t, real_parms, samples=None, stdevs=None):
        plt.scatter(a_x, a_t, alpha=0.5)
        plt.xlabel('x')
        plt.ylabel('t')

        plt.plot([-1, 1], real_fun(real_parms[0], real_parms[1], 0, np.array([-1., 1.])), 'r')

        _ = plt.title('Real Data from Noisey Linear Function')

        if samples:
            weights = self.posterior.rvs(samples)
            for weight in weights:
                plt.plot([-1, 1], real_fun(weight[0], weight[1], 0, np.array([-1., 1.])), 'black')
                _ = plt.title('Lines Sampled from Posterior Distribution vs Real Line and Data')

        if stdevs:
            a_xrange = np.linspace(-1, 1, 100)
            y_upper = self.prediction_limit(a_xrange, stdevs)
            y_lower = self.prediction_limit(a_xrange, -stdevs)
            plt.plot(a_xrange, y_upper, '+', c='green', linewidth=4.0)
            plt.plot(a_xrange, y_lower, '+', c='green', linewidth=4.0)
            _ = plt.title('Lines Sampled from Posterior Distribution vs Real Line and Data')
        plt.show()


a_0 = -0.3
a_1 = 0.5
noise_sigma = 0.2
beta = 1/noise_sigma**2
np.random.seed(20)
x_real = uniform(-1, 1, 1000)
t_real = real_fun(a_0, a_1, noise_sigma, x_real)

alpha = 2.0
v_m0 = np.array([0., 0.])
m_S0 = 1/alpha*np.identity(2)

linbayes = LinearBayes(v_m0, m_S0, beta)
N = 10
linbayes.set_posterior(x_real[0:N], t_real[0:N])

x, y = np.mgrid[-1:1:.01, -1:1:.01]
linbayes.make_contour(x, y, real_parms=[a_0, a_1], N=N)
