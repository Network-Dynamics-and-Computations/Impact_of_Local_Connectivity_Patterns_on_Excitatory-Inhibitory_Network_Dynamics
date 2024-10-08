import numpy as np
from numpy import linalg as la
import matplotlib.pylab as plt
from scipy.optimize import fsolve
import math
import statsmodels.api as sm
from scipy.stats import norm
import seaborn as sns
from scipy import stats
from matplotlib.patches import Ellipse
import matplotlib.transforms as transforms

### dynamics
def iidGaussian(stats,shapem):
	mu,sig = stats[0],stats[1]
	nx,ny = shapem[0],shapem[1]
	return np.random.normal(mu,sig,(nx,ny))

def confidence_ellipse(x, y, ax, n_std=3.0, facecolor='none', **kwargs):
    """
    Create a plot of the covariance confidence ellipse of `x` and `y`
    Parameters
    ----------
    x, y : array_like, shape (n, )
        Input data.
    ax : matplotlib.axes.Axes
        The axes object to draw the ellipse into.
    n_std : float
        The number of standard deviations to determine the ellipse's radiuses.
    Returns
    -------
    matplotlib.patches.Ellipse
    Other parameters
    ----------------
    kwargs : `~matplotlib.patches.Patch` properties
    """
    if x.size != y.size:
        raise ValueError("x and y must be the same size")

    cov = np.cov(x, y)
    pearson = cov[0, 1]/np.sqrt(cov[0, 0] * cov[1, 1])
    # Using a special case to obtain the eigenvalues of this
    # two-dimensionl dataset.
    ell_radius_x = np.sqrt(1 + pearson)
    ell_radius_y = np.sqrt(1 - pearson)
    ellipse = Ellipse((0, 0),
        width=ell_radius_x * 2,
        height=ell_radius_y * 2,
        facecolor=facecolor,
        **kwargs)

    # Calculating the stdandard deviation of x from
    # the squareroot of the variance and multiplying
    # with the given number of standard deviations.
    scale_x = np.sqrt(cov[0, 0]) * n_std
    mean_x = np.mean(x)

    # calculating the stdandard deviation of y ...
    scale_y = np.sqrt(cov[1, 1]) * n_std
    mean_y = np.mean(y)

    transf = transforms.Affine2D() \
        .rotate_deg(45) \
        .scale(scale_x, scale_y) \
        .translate(mean_x, mean_y)

    ellipse.set_transform(transf + ax.transData)
    return ax.add_patch(ellipse)

def list_to_dict(lst, string):
    """
    Transform a list of variables into a dictionary.
    Parameters
    ----------
    lst : list
        list with all variables.
    string : str
        string containing the names, separated by commas.
    Returns
    -------
    d : dict
        dictionary with items in which the keys and the values are specified
        in string and lst values respectively.
    """
    string = string[0]
    string = string.replace(']', '')
    string = string.replace('[', '')
    string = string.replace('\\', '')
    string = string.replace(' ', '')
    string = string.replace('\t', '')
    string = string.replace('\n', '')
    string = string.split(',')
    d = {s: v for s, v in zip(string, lst)}
    return d
def equations(x,data):
    '''
    Function: compute the largest real eigenvalue outlier of the matrix with chain motifs
    eqleft = (outlier**2+gaverage**2*tau)*(outlier**2-gaverage**2*tau*(N-1))
    eqright = outlier*eigvAm[0]*(outlier**2-gaverage**2*tau*(N-1))+gaverage**2*tau*eigvAm[0]*N
    f(outlier) = eqleft - eqright = 0
    params:
    data = [g, tau_chn, lambda_0, N]
    '''
    outlier = x
    gaverage,tau,eigvAm,N = data[0],data[1],data[2],data[3]
    eqleft = (outlier**2+gaverage**2*tau)*(outlier**2-gaverage**2*tau*(N-1))
    # eqright = outlier*eigvAm*(outlier**2-gaverage**2*tau*(N-1))+gaverage**2*tau*eigvAm*outlier*N
    eqright = outlier*eigvAm*(outlier**2+gaverage**2*tau)
    
    return (eqleft-eqright)