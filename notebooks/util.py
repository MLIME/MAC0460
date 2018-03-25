import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D


def get_housing_prices_data(N, verbose=True):
    """
    Generates artificial linear data,
    where x = square meter, y = house price

    :param N: data set size
    :type N: int
    :param verbose: param to control print
    :type verbose: bool
    :return: design matrix, regression targets
    :rtype: np.array, np.array
    """
    cond = False
    while not cond:
        x = np.linspace(90, 1200, N)
        gamma = np.random.normal(30, 10, x.size)
        y = 50 * x + gamma * 400
        x = x.astype("float32")
        x = x.reshape((x.shape[0], 1))
        y = y.astype("float32")
        y = y.reshape((y.shape[0], 1))
        cond = min(y) > 0
    xmean, xsdt, xmax, xmin = np.mean(x), np.std(x), np.max(x), np.min(x)
    ymean, ysdt, ymax, ymin = np.mean(y), np.std(y), np.max(y), np.min(y)
    if verbose:
        print("\nX shape = {}".format(x.shape))
        print("\ny shape = {}\n".format(y.shape))
        print("X:\nmean {}, sdt {:.2f}, max {}, min {}".format(xmean,
                                                               xsdt,
                                                               xmax,
                                                               xmin))
        print("\ny:\nmean {}, sdt {:.2f}, max {}, min {}".format(ymean,
                                                                 ysdt,
                                                                 ymax,
                                                                 ymin))
    return x, y


def r_squared(y, y_hat):
    """
    Calculate the R^2 value

    :param y: regression targets
    :type y: np array
    :param y_hat: prediction
    :type y_hat: np array
    :return: r^2 value
    :rtype: float
    """
    y_mean = np.mean(y)
    ssres = np.sum(np.square(y - y_mean))
    ssexp = np.sum(np.square(y_hat - y_mean))
    sstot = ssres + ssexp
    return 1 - (ssexp / sstot)

    
def randomize_in_place(list1, list2, init=0):
    """
    Function to randomize two lists in the same way.

    :param list1: list
    :type list1: list or np.array
    :param list2: list
    :type list2: list or np.array
    :param init: seed
    :type init: int
    """
    np.random.seed(seed=init)
    np.random.shuffle(list1)
    np.random.seed(seed=init)
    np.random.shuffle(list2)