# Code developed by Konstanze Stuebner (kstueb@gmail.com) and Bodo Bookhagen (bodo.bookhagen@uni-potsdam.de)
#
# Python 3, using some additional python packages. If you have not setup an environment,
# you may want to add the following packages via conda
# conda install scipy numpy scikit-learn lmfit
# or generate a separate environment

import numpy as np
from scipy.stats import norm
from scipy.signal import find_peaks
from sklearn.neighbors import KernelDensity
from sklearn.model_selection import GridSearchCV
from lmfit.models import GaussianModel

import matplotlib.pyplot as plt

VALID_KDE_Weight = ['adf','1/error','1/error**2','none']
VALID_ADF_Sigma = ['error','sqrtError']

class BoulderAges():
    """Age density function as sum of Gaussians.

    parameters:
        Ages, Errors : 1D array

        KDEweight : str ['adf'|'1/error'|'1/error**2'|'none']
            KDE can be weighted by the age density function ('adf'), 1/error,
            or not weighted at all. If KDEweight='adf' then ADFsigma should be
            specified. Default is 'adf'.

        ADFsigma : str ['error'|'sqrtError']
            The age density function is calculated as sum of Gaussians for each
            boulder age, where the width of the Gaussians is specified by
            ADFsigma. Default is 'error'.


    example:
        test = BoulderAges(Ages, Errors)
        x = np.arange(0,100)
        plt.plot(x,test.ADF(x),'-b',label='Age DF')
        plt.plot(Ages,test.ADF(Ages),'.b')

        kde = test.KDE()
        plt.plot(x,scoreKDE(kde,x),'r',label='KDE, bw={:.0f}'.format(kde.bandwidth))
        plt.legend()
    """

    def __init__(self, Ages, Errors, *, KDEweight='adf', ADFsigma='sqrtError'):
        self.Ages = Ages
        self.Errors = Errors
        self.KDEweight = KDEweight
        self.ADFsigma = ADFsigma

        if Ages.shape!=Errors.shape:
            raise ValueError("Ages and Errors must be 1D arrays of the same length")
        if KDEweight not in VALID_KDE_Weight:
            raise ValueError("Invalid KDEweight: {:s}".format(KDEweight))
        if ADFsigma not in VALID_ADF_Sigma:
            raise ValueError("Invalid ADFsigma: {:s}".format(ADFsigma))
        return

    def ADF(self, xx):
        """Age density function as sum of Gaussians."""

        if self.ADFsigma=='error':
            Sigma=self.Errors
        elif self.ADFsigma=='sqrtError':
            Sigma=np.sqrt(self.Errors)
        else:
            raise ValueError("Invalid ADFsigma: {:s}".format(ADFsigma))
        yy = 0
        for Age_i,Sigma_i in zip(self.Ages,Sigma):
            yy += norm.pdf(xx,loc=Age_i,scale=Sigma_i)
        return yy/len(self.Ages)

    def KDE(self, bandwidth=None):
        """Kernel Density Estimation."""

        if self.KDEweight=='adf':
            Weights = self.ADF(self.Ages)
        elif self.KDEweight=='1/error':
            Weights = 1/self.Errors
        elif self.KDEweight=='1/error**2':
            Weights = 1/(self.Errors**2)
        elif self.KDEweight=='none':
            Weights = None
        else:
            raise ValueError("Invalid KDEweight: {:s}".format(KDEweight))

        if bandwidth is None:
            grid = GridSearchCV(KernelDensity(kernel='gaussian'),
                        {'bandwidth': np.linspace(5., 50., 46)},
                        cv=len(self.Ages)) #cross-validation
            grid.fit(self.Ages.reshape(-1,1), sample_weight=Weights)
            kde = grid.best_estimator_
            bandwidth = grid.best_params_['bandwidth']
        else:
            kde = KernelDensity(kernel='gaussian', bandwidth=bandwidth).fit(
                self.Ages.reshape(-1,1), sample_weight=Weights)
        return kde

def scoreKDE(kde, xx):
    """Values of the kde at positions xx."""
    if not isinstance(kde,KernelDensity):
        raise TypeError("'kde' must be a KernelDensity")
    return np.exp(kde.score_samples(xx.reshape(-1, 1)))

def fit_Gaussians_to_curve(x_data, y_data, peaks):
    """Perform Gaussian curve fitting.
    Return: lmfit.model.ModelResult
    Note that peaks older than max(x_data) will be ignored.

    parameters:
        x_data, y_data : 1D arrays of the smoothed boulder age distribution (kde)
        peaks: ndarray of approximate peak locations

    example:
        peaks = get_auto_peaks(x, scoreKDE(kde,x))
        result = fit_Gaussians_to_curve(x, scoreKDE(kde,x), peaks=peaks)
    """

    x_data = x_data.reshape(-1); y_data = y_data.reshape(-1)

    prefix = ['G'+str(i)+'_' for i in range(100)]
    prefix = prefix[:len(peaks)]
    model = [None]*len(prefix)
    for i,p in enumerate(prefix):
        model[i] = GaussianModel(prefix=p)
    #initialize parameters of first Gaussian component
    params = model[0].make_params(center=peaks[0], sigma=1, amplitude=1)
    params[prefix[0]+'amplitude'].min = 0.
    params[prefix[0]+'center'].min = 0.
    params[prefix[0]+'center'].max = max(x_data)
    if len(peaks>1): #add parameters of remaining Gaussian components
        for i in range(1,len(peaks)):
            params += model[i].make_params(center=peaks[i], sigma=1, amplitude=1)
            params[prefix[i]+'amplitude'].min = 0.
            params[prefix[i]+'center'].min = 0.
            params[prefix[i]+'center'].max = max(x_data)

    compositeModel = model[0]
    for i in range(1,len(peaks)):
        compositeModel += model[i]

    return compositeModel.fit(y_data, params, x=x_data)

def iterate_Gaussian_fitting(x_data, y_data, nA, verbose=False):
    """Repeat fit_Gaussians_to_curve() with n+1 peaks until the misfit <1%
    (see Gaussian_misfit() for definition).
    Note that peaks older than max(x_data) will be ignored.

    parameters:
        x_data, y_data : 1D arrays of the smoothed boulder age distribution (kde)
        nA : len(Ages); no more than ceil(sqrt(nA)) peaks are allowed.
        verbose : boolean; if 'True' report progress in the terminal

    example:
        result = iterate_Gaussian_fitting(x, scoreKDE(kde,x), len(Ages))
    """
    if verbose:
        print('\nIterations:')
    peaks = get_auto_peaks(x_data, y_data)
    A = 100
    i = 0
    while A>1: #misfit >1%
        i += 1
        result = fit_Gaussians_to_curve(x_data, y_data, peaks)
        A = Gaussian_misfit(x_data, y_data, result.best_fit)
        if verbose:
            print('#{:d}'.format(i))
            report = Gaussian_fitting_report(result)
            for r in report:
                print('{:.1f} +/- {:.1f} ky ({:.0f}%)'.format(r['Mu'], r['Sigma'], r['Area']))
            print('misfit {:.1f}%\n'.format(A))
        peaks = get_peaks_from_lmfitresult(result)
        sorted = np.argsort(result.residual)
        peaks = np.append(peaks,x_data[sorted[0]])
        #if len(peaks)>np.ceil(np.sqrt(nA)):
        #    if verbose:
        #        print('max. number of peaks ({:d})'.format(len(peaks)-1))
        #    break

    #remove Gaussians with less than 1% Area and recalculate
    peaks = []
    report = Gaussian_fitting_report(result)
    for r in report:
        if r['Area']>1:
            peaks = np.append(peaks,r['Mu'])
    result = fit_Gaussians_to_curve(x_data, y_data, peaks)
    return result

def get_auto_peaks(x_data, y_data):
    """Return age peaks as ndarray."""
    peaks,_ = find_peaks(y_data)
    return x_data[peaks]

def get_peaks_from_lmfitresult(result):
    """Return age peaks from lmfit.model.ModelResult as ndarray."""
    prefix = [r.prefix for r in result.components]
    return np.array([result.best_values[p+'center'] for p in prefix])

def Gaussian_misfit(x_data,y_data,y_fit):
    """Estimate misfit between KDE and sum-of-Gaussian model.
    Return area of misfit to the KDE in percent.

    parameters:
        x_data, y_data, y_fit : 1D arrays of KDE (data) and model fit (fit)

    example:
        A = Gaussian_misfit(x_data, y_data, result.best_fit)
    """
    return np.sum(np.abs(y_fit-y_data))/np.sum(y_data)*100

def Gaussian_fitting_report(result):
    """Return list of dictionaries with keys prefix, Mu, Sigma, Apc for each
    component in the Gaussian composite model sorted by decending age (Mu).

    example:
        report = Gaussian_fitting_report(result)
        for r in report:
            print('{:.1f} +/- {:.1f} ky ({:.0f}%)'.format(r['Mu'], r['Sigma'], r['Area']))
    """

    prefix = [r.prefix for r in result.components]
    Mu = np.array([result.best_values[p+'center'] for p in prefix])
    Sigma = np.array([result.best_values[p+'sigma'] for p in prefix])

    sorted = np.flip(np.argsort(Mu))
    Mu = Mu[sorted]
    Sigma = Sigma[sorted]
    prefix = [prefix[i] for i in sorted]

    Atotal = np.sum(result.best_fit) #model total area
    report = []
    for p,m,s in zip(prefix, Mu, Sigma):
        Apc = np.sum(result.eval_components()[p])/Atotal*100
        report.append({'prefix':p, 'Mu':m, 'Sigma':s, 'Area':Apc})
    return report

def plot_lmfit_model(xx, result, misfit, axis=None, thresholdArea=5):
    """Plot the lmfit.model.ModelResult results

    parameters:
        xx : 1D array
        result : output of fit_Gaussians_to_curve() or iterate_Gaussian_fitting()
        misfit :
        axis : axis object to add the plot to (optional)
        thresholdArea : the oldest peak with an area > thresholdArea is the
            preferred moraine age and plotted in red

    example:
        x = np.arange(0,100)
        result = iterate_Gaussian_fitting(x, scoreKDE(kde,x), len(Ages))
        A = Gaussian_misfit(x, scoreKDE(kde,x), result.best_fit)
        plot_lmfit_model(x, result, A)

    """

    if axis: ax = axis
    else: _,ax = plt.subplots()

    #total model:
    label = 'Model, misfit {:.1f}%'.format(misfit)
    ax.plot(xx, result.best_fit, 'k-', label=label)

    report = Gaussian_fitting_report(result)
    prefix = [r['prefix'] for r in report] #prefix sorted by decending Mu

    #find oldest peak with a an area of >5%
    for r in report:
        if r['Area']>thresholdArea:
            best_p = r['prefix']
            break

    #Gaussian components:
    Atotal = np.sum(result.best_fit) #model total area
    Mu = np.empty_like(prefix,dtype=float) #mean of each Gaussian
    Sigma = np.empty_like(prefix,dtype=float) #sigma of each Gaussian
    Apc = np.empty_like(prefix,dtype=float) #area of each Gaussian in percent
    for i, p in enumerate(reversed(prefix)):
        Mu[i] = result.best_values[p+'center']
        Sigma[i] = result.best_values[p+'sigma']
        Apc[i] = np.sum(result.eval_components()[p])/Atotal*100
        label = '{:s}: {:.1f}$\pm${:.1f} ky, {:.1f}%'.format(p[:-1], Mu[i], Sigma[i], Apc[i])
        if p==best_p:
            color='red'
        else:
            color='grey'
        ax.plot(xx, result.eval_components()[p], color=color, label=label)
    ax.legend()
    return ax



def main():
    plt.close('all')

    np.random.seed(12345)
    #Generating a random age distribution with mean age of 50 and deviation of 15. 8 samples
    Ages = np.random.normal(loc=50,scale=15,size=8)
    Errors = 0.1 * Ages.reshape(len(Ages)) + np.random.rand(1) * 5

    test = BoulderAges(Ages, Errors)
    #By default, the KDE is weighted with the age density function (ADF), which is
    #the sum of Gaussians computed with Mu=boulder age and Sigma=sqrt(boulder error).
    #This weighting can be changed, e.g.:
    #test.ADFsigma='error'
    #test.KDEweight='none'

    _,ax = plt.subplots()
    x = np.arange(0,100)
    ax.plot(x,test.ADF(x),'-b',label='Age DF')
    ax.plot(Ages,test.ADF(Ages),'.r')

    kde = test.KDE()

    ax.plot(x,scoreKDE(kde,x),'r',label='KDE, bw={:.0f}'.format(kde.bandwidth))
    #data points and errorbars at convenient y-scaling
    y = np.arange(1,len(Ages)+1)/2/len(Ages)*ax.get_ylim()[1]
    ax.plot(Ages, y, 'k+')
    ax.errorbar(x=Ages, y=y, xerr=Errors, fmt='none', ecolor='k')
    plt.legend()
    plt.title('Random test sample, n={:d}'.format(len(Ages)))
    plt.xlabel('Age (ky)')
    plt.ylabel('Probability')
    plt.grid()


    result = iterate_Gaussian_fitting(x, scoreKDE(kde,x), len(Ages), verbose=True)
    A = Gaussian_misfit(x, scoreKDE(kde,x), result.best_fit)

    ax = plot_lmfit_model(x, result, A)
    ax.plot(Ages, y, 'k+')
    ax.errorbar(x=Ages, y=y, xerr=Errors, fmt='none', ecolor='k')
    plt.title('Random test sample, Gaussian fitting')
    plt.xlabel('Age (ky)')
    plt.ylabel('Probability')
    plt.grid()

    plt.show()

if __name__ == "__main__":
    main()
