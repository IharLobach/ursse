import numpy as np

def normalize(a):
    return a/np.sum(a)


def hist_to_step(x, y):
    step = x[1]-x[0]
    xmin = x[0]-step/2
    xmax = x[-1]+step/2
    xrest = (x[1:]+x[:-1])/2
    xall = np.array([xmin, xmin]+list(np.repeat(xrest, 2))+[xmax, xmax])
    yfirst = 0
    ylast = 0
    yall = np.array([yfirst]+list(np.repeat(y, 2))+[ylast])
    return xall, yall


def myhist(ax, x, y, fill_alpha=0.4, **kwargs):
    xy = hist_to_step(x, y)
    res = ax.plot(*xy, **kwargs)
    col = res[0].get_color()
    ax.fill_between(*xy, color=col, alpha=fill_alpha)
