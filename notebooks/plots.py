import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D


def plot_points_regression(x,
                           y,
                           title,
                           xlabel,
                           ylabel,
                           prediction=None,
                           legend=False,
                           r_squared=None,
                           position=(90, 100)):
    """
    Plots the data points and the prediction,
    if there is one.

    :param x: design matrix
    :type x: np.array
    :param y: regression targets
    :type y: np.array
    :param title: plot's title
    :type title: str
    :param xlabel: x axis label
    :type xlabel: str
    :param ylabel: y axis label
    :type ylabel: str
    :param prediction: model's prediction
    :type prediction: np.array
    :param legend: param to control print legends
    :type legend: bool
    :param r_squared: r^2 value
    :type r_squared: float
    :param position: text position
    :type position: tuple
    """
    fig, ax = plt.subplots(1, 1, figsize=(8, 8))
    line1, = ax.plot(x, y, 'bo', label='Real data')
    if prediction is not None:
        line2, = ax.plot(x, prediction, 'r', label='Predicted data')
        if legend:
            plt.legend(handles=[line1, line2], loc=2)
    ax.set_title(title,
                 fontsize=20,
                 fontweight='bold')
    if r_squared is not None:
        bbox_props = dict(boxstyle="square,pad=0.3",
                          fc="white", ec="black", lw=0.2)
        t = ax.text(position[0], position[1], "$R^2 ={:.4f}$".format(r_squared),
                    size=15, bbox=bbox_props)

    ax.set_xlabel(xlabel, fontsize=20)
    ax.set_ylabel(ylabel, fontsize=20)
    plt.show()

def plot_cost_function_curve(X,
                             y,
                             cost_function,
                             title,
                             weights_list=None,
                             cost_list=None,
                             position=(20, 40),
                             range_points=(20, 40)):
    """
    Plots a cost surfice.
    It assumes that weight.shape == (2,). 

    :param X: design matrix
    :type X: np.ndarray
    :param y: regression targets
    :type y: np.ndarray
    :param cost_function: function to compute regression cost
    :type cost_function: lambda: (np.ndarray, np.ndarray, np.ndarray) -> float
    :param title: plot's title
    :type title: str
    :param weights_list: list of weights
    :type weights_list: list
    :param cost_list: list of costs
    :type cost_list: list
    :param position: surfice rotation position
    :type position: tuple
    :param range_points: range of values for w
    :type range_points: tuple
    """

    w_0, w_1 = 0, 0
    ms = np.linspace(w_0 - range_points[0] , w_0 + range_points[0], range_points[0])
    bs = np.linspace(w_1 - range_points[1] , w_1 + range_points[1], range_points[1])
    M, B = np.meshgrid(ms, bs)
    MB = np.stack((np.ravel(M), np.ravel(B)), axis=1)
    size = MB.shape[0] 
    MB = MB.reshape((size, 2, 1))
    zs = np.array([cost_function(X, y, MB[i]) 
                   for i in range(size)])
    Z = zs.reshape(M.shape)
    fig = plt.figure(figsize=(20, 10))
    ax = fig.add_subplot(111, projection='3d')
    ax.plot_surface(M, B, Z, rstride=1, cstride=1, color='b', alpha=0.2)
    ax.set_xlabel('w[0]', labelpad=30, fontsize=24, fontweight='bold')
    ax.set_ylabel('w[1]', labelpad=30, fontsize=24, fontweight='bold')
    ax.set_zlabel('J(w)', labelpad=30, fontsize=24, fontweight='bold')
    if weights_list is not None and cost_list is not None:
        ax.plot([weights_list[0][0]],
                [weights_list[0][1]],
                [cost_list[0]],
                markerfacecolor=(1.0, 0.0, 0.0, 1.0),
                markeredgecolor=(1.0, 0.0, 0.0, 1.0),
                marker='o',
                markersize=7)
        ax.plot([weights_list[-1][0]],
                [weights_list[-1][1]],
                [cost_list[-1]],
                markerfacecolor=(0.0, 0.0, 1.0, 1.0),
                markeredgecolor=(0.0, 0.0, 1.0, 1.0),
                marker='o',
                markersize=7)
        temp_red = 1.0
        temp_blue = 0.0
        size = len(weights_list)
        oldx = 0.0
        oldy = 0.0
        oldz = 0.0
        for w, cost in zip(weights_list, cost_list):
            rgba_color = (temp_red * 1.0, 0.0, temp_blue * 1.0, 1.0)
            ax.plot([w[0]],
                    [w[1]],
                    [cost],
                    markerfacecolor=rgba_color,
                    markeredgecolor=rgba_color,
                    marker='.',
                    markersize=4)
            if oldx + oldy + oldz != 0.0 :
                rgba_color_weak = list(rgba_color)
                rgba_color_weak[-1] = 0.3
                ax.plot([w[0], oldx],[w[1], oldy], [cost, oldz],color=rgba_color_weak)
            temp_red += - 1 / size
            temp_blue +=  1 / size
            oldx = w[0]
            oldy = w[1]
            oldz = cost    
    ax.view_init(elev=position[0], azim=position[1])
    ax.set_title(title,
             fontsize=20,
             fontweight='bold')
    plt.show()
    
def simple_step_plot(ylist,
                     yname,
                     title,
                     figsize=(4, 4),
                     labels=None):
    """
    Plots values over time.

    :param ylist: list of values lists
    :type ylist: list
    :param yname: value name
    :type yname: str
    :param title: plot's title
    :type title: str
    :param figsize: plot's size
    :type figsize: tuple
    :param labels: label for each values list in ylist
    :type range_points: list
    """
    y0 = ylist[0]
    x = np.arange(1, len(y0) + 1, 1)
    fig, ax = plt.subplots(1, 1, figsize=figsize)
    for y in ylist:
        ax.plot(x, y)
    plt.xlabel('step')
    plt.ylabel(yname)
    plt.title(title,
              fontsize=14,
              fontweight='bold')
    plt.grid(True)
    if labels is not None:
        plt.legend(labels,
           loc='upper right')
    plt.show()

    

