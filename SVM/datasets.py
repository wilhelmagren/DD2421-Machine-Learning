import random
import numpy as np
import matplotlib.pyplot as plt


def plot_classes(classA, classB, savefig=False, plt_later=True):
    plt.plot([p[0] for p in classA], [p[1] for p in classA], 'b.')
    plt.plot([p[0] for p in classB], [p[1] for p in classB], 'r.')
    # plt.axis('equal')  # Force same scale on both axes
    if savefig:
        plt.savefig('svmplot.pdf')  # Save a copy of the plot
    if not plt_later:
        plt.show()
    return


def generate_data(N, arg, verbose=False):
    """
        0: lin separable
        1: lin not separable
        2: polyrbf separable
        3: polyrbf not separable
    """
    if arg == 0:
        return generate_data_separable(N, verbose)
    elif arg == 1:
        return generate_data_not_separable(N, verbose)
    elif arg == 2:
        return generate_data_polyrbf_separable(N, verbose)
    else:
        print('<| bro you fucked up\n<| terminating...')
        exit()


def generate_data_separable(N, verbose=True):
    """
        The class arrays will look like the following:
            classQ = [[x1, y1], [x2, y2], ..., [xN, yN]]
            with size N x 2

        The target array will look like the following:
            target = [t1, t2, ..., tN]
            with size N x 1

    """
    classA = np.concatenate((np.random.randn(50, 2) * 0.1 + np.array([0.8, 0.5]),
                             np.random.randn(50, 2) * 0.1 + np.array([-0.8, 0.5])))
    classB = np.random.rand(100, 2) * 0.3 + np.array([0.0, -0.5])
    # plot_classes(classA, classB, False, False)
    inputs = np.concatenate((classA, classB))
    target = np.concatenate((np.ones(classA.shape[0]),
                             -np.ones(classB.shape[0])))
    permute = list(range(N))
    random.shuffle(permute)
    inputs = inputs[permute, :]
    target = target[permute]
    if verbose:
        print(f"\t The class_A array looks like:\n {classA}\n\t and has the shape {classA.shape}\n")
        print(f"\t The class_B array looks like:\n {classB}\n\t and has the shape {classB.shape}\n")
        print(f"\t The inputs array looks like:\n {inputs}\n\t and has the shape {inputs.shape}\n")
        print(f"\t The target array looks like:\n {target}\n\t and has the shape {target.shape}\n")
    return classA, classB, inputs, target


def generate_data_not_separable(N, verbose=True):
    """
        The class arrays will look like the following:
            classQ = [[x1, y1], [x2, y2], ..., [xN, yN]]
            with size N x 2

        The target array will look like the following:
            target = [t1, t2, ..., tN]
            with size N x 1

    """
    classA = np.concatenate((np.random.randn(50, 2) * 0.1 + np.array([0.8, 0.0]),
                             np.random.randn(50, 2) * 0.1 + np.array([-0.8, 0.0])))
    classB = np.random.rand(100, 2) * 0.3 + np.array([0.0, -0.2])
    # plot_classes(classA, classB, False, False)
    inputs = np.concatenate((classA, classB))
    target = np.concatenate((np.ones(classA.shape[0]),
                             -np.ones(classB.shape[0])))
    permute = list(range(N))
    random.shuffle(permute)
    inputs = inputs[permute, :]
    target = target[permute]
    if verbose:
        print(f"\t The class_A array looks like:\n {classA}\n\t and has the shape {classA.shape}\n")
        print(f"\t The class_B array looks like:\n {classB}\n\t and has the shape {classB.shape}\n")
        print(f"\t The inputs array looks like:\n {inputs}\n\t and has the shape {inputs.shape}\n")
        print(f"\t The target array looks like:\n {target}\n\t and has the shape {target.shape}\n")
    return classA, classB, inputs, target


def generate_data_polyrbf_separable(N, verbose=True):
    """
        The class arrays will look like the following:
            classQ = [[x1, y1], [x2, y2], ..., [xN, yN]]
            with size N x 2

        The target array will look like the following:
            target = [t1, t2, ..., tN]
            with size N x 1

    """
    classA = np.concatenate((np.random.randn(25, 2) * 0.2 + np.array([-0.3, 0.2]),
                             np.random.randn(25, 2) * 0.2 + np.array([0.3, 0.2])))
    classB = np.random.rand(50, 2) * 0.5 + np.array([0.0, -1.0])
    # plot_classes(classA, classB, False, False)
    inputs = np.concatenate((classA, classB))
    target = np.concatenate((np.ones(classA.shape[0]),
                             -np.ones(classB.shape[0])))
    permute = list(range(N))
    random.shuffle(permute)
    inputs = inputs[permute, :]
    target = target[permute]
    if verbose:
        print(f"\t The class_A array looks like:\n {classA}\n\t and has the shape {classA.shape}\n")
        print(f"\t The class_B array looks like:\n {classB}\n\t and has the shape {classB.shape}\n")
        print(f"\t The inputs array looks like:\n {inputs}\n\t and has the shape {inputs.shape}\n")
        print(f"\t The target array looks like:\n {target}\n\t and has the shape {target.shape}\n")
    return classA, classB, inputs, target