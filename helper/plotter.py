import matplotlib.pyplot as plt


def plot_loss_curve(par_values, train_scores, dev_scores, title="Loss Curve", xlab="Epoch", ylab="Loss", yexp=False):
    """
    Generate a simple plot of the test and training learning curve.
    :param par_values: list of checked values of the current parameter.
    :param train_scores : list of scores obtained in training set (same length as par_values).
    :param dev_scores : list of scores obtained in dev set (same length as par_values)
    :param title : title for the chart.
    :param xlab: name of horizontal axis
    :param ylab: name of vertical axis
    :param yexp : True for exponential vertical axis, False otherwise
    :return Figure object
    """
    fig = plt.figure()
    plt.title(title)
    plt.xlabel(xlab)
    plt.ylabel(ylab)

    plt.grid()
    plt.plot(par_values, train_scores, color="r", label="Training loss")
    plt.plot(par_values, dev_scores, color="g", label="Dev loss")

    if yexp:
        plt.yscale("log")

    plt.legend(loc="best")
    return fig


def plot_recall_curve(par_values, r1, r5, r10, title="R@k Curve", xlab="Epoch", ylab="R@k", yexp=False):
    """
    Generate a simple plot of the test and training learning curve.
    :param par_values: list of checked values of the current parameter.
    :param r1: recall@1 values for each epoch in dev set.
    :param r5: recall@5 values for each epoch in dev set.
    :param r10: recall@10 values for each epoch in dev set.
    :param title : title for the chart.
    :param xlab: name of horizontal axis
    :param ylab: name of vertical axis
    :param yexp : True for exponential vertical axis, False otherwise
    :return Figure object
    """
    fig = plt.figure()
    plt.title(title)
    plt.xlabel(xlab)
    plt.ylabel(ylab)

    plt.grid()
    plt.plot(par_values, r1, color="r", label="R@1")
    plt.plot(par_values, r5, color="g", label="R@5")
    plt.plot(par_values, r10, color="b", label="R@10")

    if yexp:
        plt.yscale("log")

    plt.legend(loc="best")
    return fig
