import numpy as np
import matplotlib.pyplot as plt


def show_me(array, title=None, units=None, show=False, verbose=False):
    try:
        plt.imshow(array)
        mean_str = ""
        if verbose:
            mean = float(np.nanmean(array))
            print("Mean %s is %.3f:" % (title, mean))
            mean_str = " (mean = %.3f)" % mean
        if title is not None:
            plt.title("%s%s" % (title, mean_str))  # mean value plotted only if verbose mode is chosen
        cb = plt.colorbar()
        if units is not None:
            cb.set_label(units)
        plt.savefig("/home/tepex/PycharmProjects/energy/png/%s.png" % title)
        if show:
            plt.show()
        plt.clf()
    except Exception as e:
        print(e)
