import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
from sklearn.svm import SVC

# =======================================================
N = 200
sigma = 0.2
# =======================================================


def ml_params(X, labels, W, verbose=False):
    assert(X.shape[1] == labels.shape[0])
    Ndims, Npts = np.shape(X)
    if verbose:
        print('<| The data has the shape {} x {}'.format(Ndims, Npts))
    classes = np.unique(labels)
    Nclasses = np.size(classes)
    mu = np.zeros([Nclasses, Ndims])
    sigma = np.zeros([Nclasses, Ndims, Ndims])

    if W is None:
        W = np.ones([Npts, 1])/float(Npts)
    else:
        assert(W.shape[0] == Npts)

    for jdx, k in enumerate(classes):
        idx = np.where(labels == k)[0]
        xlc = X[:, idx]
        wlc = W[idx, :]
        w_sum = sum(wlc)
        mu[jdx] += np.sum(wlc*xlc.T, axis=0)/w_sum
        tmp = np.square(xlc - np.transpose(mu[jdx].reshape([1, 3])))
        sigma[jdx] = np.diag(sum(wlc*tmp.T)/w_sum)
    if verbose:
        print('<| ==========\n<| This is the mu matrix:\n{}'.format(mu))
        print('<| ==========\n<| This is the sigma matrix:\n{}'.format(sigma))
    return mu, sigma


def generate_data(verbose=False):
    class_a = np.zeros([3, int(N/2)])
    class_b = np.zeros([3, int(N/2)])
    combined = np.zeros([3, N])
    for i in range(class_a.shape[0]):
        tmp_data_a = np.random.randn(int(N / 2), 1).flatten()
        tmp_data_b = np.random.randn(int(N / 2), 1).flatten()
        class_a[i, :] = sigma * tmp_data_a + [sigma for _ in range(int(N/2))]
        class_b[i, :] = sigma * tmp_data_b + [-sigma for _ in range(int(N/2))]
        combined[i, :int(N/2)] = class_a[i]
        combined[i, int(N/2):] = class_b[i]

    labels = np.concatenate([[1 for _ in range(int(N/2))],
                            [-1 for _ in range(int(N/2))]])
    if verbose:
        print('<| This is class_a:\n{}'.format(class_a))
        print('<| This is class_b:\n{}'.format(class_b))
        print('<| This is the combined data:\n{}'.format(combined))
        print('<| These are the labels:\n{}'.format(labels))
    return class_a, class_b, combined, labels


def plot_classes(a, b, savefig=False, plot_later=False):
    # Axes3D.scatter(xs, ys, zs=0, zdir='z', s=20, c=None, depthshade=True, *args, **kwargs)
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter([p for p in a[0]], [p for p in a[1]], [p for p in a[2]], c='blue')
    ax.scatter([p for p in b[0]], [p for p in b[1]], [p for p in b[2]], c='red')
    if savefig:
        plt.savefig('datasetplot.png')
    if not plot_later:
        plt.show()
    return


def data():
    rs = np.random.RandomState(69)
    X = np.zeros((N, 3))
    X[:int(N/2)] = rs.multivariate_normal(-np.ones(3) * 1.2, np.eye(3), size=int(N/2))
    X[int(N/2):] = rs.multivariate_normal(np.ones(3) * 2, np.eye(3), size=int(N/2))
    Y = np.zeros((N))
    Y[int(N/2):] = 1
    return X, Y


def main():
    X, Y = data()
    svc = SVC(kernel='linear')
    svc.fit(X, Y)
    # The equation of the separating hyperplane is given by all x in R^3 such that:
    # np.dot(svc.coef_[0], x) + b = 0. We should solve for the last coordinate
    # to plot the plane in terms of x and y.
    z = lambda x, y: (-svc.intercept_[0] - svc.coef_[0][0]*x - svc.coef_[0][1]*y) / svc.coef_[0][2]
    tmp = np.linspace(-5, 5, 100)
    xx, yy = np.meshgrid(tmp, tmp)
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.plot_surface(xx, yy, z(xx, yy))
    idx = np.where(Y == 0)[0]
    ax.plot3D(X[idx, 0], X[idx, 1], X[idx, 2], 'ob')
    idx = np.where(Y == 1)[0]
    ax.plot3D(X[idx, 0], X[idx, 1], X[idx, 2], 'sr')
    plt.show()


if __name__ == "__main__":
    main()
