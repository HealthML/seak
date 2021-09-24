"""Contains classes and functions for Likelihood ratio tests.

The class :class:`LRTnoK` implements the set-based likelihood ratio test for continuous phenotypes. A background kernel is not supported.

Null model:

.. math:: y = {\\alpha} X + {\\epsilon}

Alternative model:

.. math:: y = {\\alpha} X + {\\gamma} G_1 + {\\epsilon}

With: :math:`X`: covariates, dimension :math:`nxc` (:math:`n:=` number of individuals, :math:`c:=` number of covariates),
:math:`G_1`: set of variants to test, dimensions :math:`nxk` (:math:`n:=` number of individuals, :math:`k:=` number of variants in set to test association for).

"""

import numpy as np
from scipy.stats import chi2
from scipy.linalg import eigh
from numpy.linalg import svd, LinAlgError

from fastlmm.util.stats import linreg, chi2mixture
from fastlmm.inference import lmm

from seak.optimizer import optimize_lambda

import logging

def make_lambdagrid(lambda0, gridlength, log_grid_lo, log_grid_hi):

    '''
    Port of the R-function to create the lambda grid (used inside RLRTsim).

    Creates a grid of lambda values to optimize over.
    '''

    if lambda0 == 0:
        grid = np.exp(np.linspace(log_grid_lo, log_grid_hi, num=gridlength - 1))
        return np.concatenate([np.zeros(1), grid])
    else:
        # Note: I'm not sure this was ever used recently, as there seems to be a bug in the R-code (?)
        leftratio = np.min([np.max([np.log(lambda0) / (log_grid_hi - log_grid_lo), 2.]), 0.8])
        leftlength = np.max(np.round(leftratio * gridlength) - 1, 2)
        leftdistance = lambda0 - np.exp(log_grid_lo)

        # make sure leftlength doesn't split the left side into too small parts:
        if leftdistance < (leftlength * 10 * np.finfo(float).eps):
            leftlength = np.max([np.round(leftdistance / (10 * np.finfo(float).eps)), 2])

        # leftdistance approx. 1 ==> make a regular grid, since
        # (1 +- epsilon)^((1:n)/n) makes a too concentrated grid
        if (np.abs(leftdistance - 1) < 0.3):
            leftgrid = np.linspace(np.exp(log_grid_lo), lambda0, num=leftlength + 1)[:-1]
        else:
            leftdiffs = np.where([leftdistance > 1] * leftlength - 1,
                                  leftdistance ** (np.arange(2, leftlength + 1) / leftlength) - leftdistance ** (1. / leftlength),
                                  leftdistance ** (np.arange(leftlength - 1, 1) - leftdistance ** leftlength))
            leftgrid = lambda0 - leftdiffs[::-1]

        rightlength = gridlength - leftlength
        rightdistance = np.exp(log_grid_hi) - lambda0
        rightdiffs = rightdistance ** (np.arange(2, rightlength + 1) / rightlength - rightdistance ** (1 / rightlength))
        rightgrid = lambda0 + rightdiffs

    return np.concatenate([np.zeros(1), leftgrid, lambda0, rightgrid])


def rlrsim_loop(p, k, n, nsim, g, q, mu, lambd, lambda0, xi=None, REML=True):

    '''
    Python port of the RLRsim.cpp function using scipy & numpy

    This is a "naive" port with the same loops used in the C++ code
    It's recommended to use rlrsim() instead, as it's much faster. This function is mainly here for documentation purposes.

    See :func:`rlrsim`

    '''

    dfChiK = n - p - k if (n - p - k) > 0 else 0

    if REML:
        n0 = n - p
        xi = mu
    else:
        assert xi is not None, "xi can't be None if REML == False !"
        n0 = n

    # pre-compute stuff that stays constant over simulations

    # broadcast to shape (g, k)
    lambdamu = lambd[:, np.newaxis] * mu
    lambdamuP1 = lambdamu + 1.

    # broadcast to shape (g, k)
    fN = ((lambd - lambda0)[:, np.newaxis] * mu) / lambdamuP1
    fD = (1 + lambda0 * mu) / lambdamuP1

    # shape (g,)
    sumlog1plambdaxi = np.sum(np.log1p(lambd[:,np.newaxis] * xi), axis=1)

    # simulations
    res = np.zeros(nsim)
    lambdaind = np.zeros(nsim, dtype=np.int)

    # replace this loop with map?
    for i_s in range(nsim):

        LR = 0.

        ChiSum = 0

        ChiK = chi2(dfChiK).rvs(1)  # Xnpk
        Chi1 = chi2(1).rvs(k)  # ws^2

        if not REML:
            ChiSum = Chi1.sum()

        # loop over lambda values:
        for i_g in range(g):

            N = (fN[i_g, :] * Chi1).sum()
            D = (fD[i_g, :] * Chi1).sum()
            D += ChiK

            LR = n0 * np.log1p(N / D) - sumlog1plambdaxi[i_g]

            if LR >= res[i_s]:
                res[i_s] = LR
                lambdaind[i_s] = i_g
            else:
                break

        if not REML:
            res[i_s] = res[i_s] + n * np.log1p(chi2(q).rvs(1) / (ChiSum + ChiK))

    result = {'res' : res,
              'lambdaind': lambdaind,
              'lambdamu': lambdamu,
              'fN': fN,
              'fD': fD,
              'sumlog1plambdaxi': sumlog1plambdaxi,
              'n': n,
              'p': p,
              'dfChik': dfChiK,
              'REML': REML
              }

    return result



def rlrsim_np(p, k, n, nsim, g, q, mu, lambd, lambda0, xi=None, REML=True):


    '''
    Python port of the RLRsim.cpp function using scipy & numpy

    This version uses broadcasting to speed up operations.
    It uses a lot of memory for large nsim, but is way faster than rlrsim_loop()

    :param int p: number of covariates in X
    :param int k: number of variables to test
    :param int n: number of individuals
    :param int nsim: number of sumulations to perform
    :param int g: lambda grid length (not used anymore...)
    :param int q: number of free parameters in the null-model (?)
    :param numpy.ndarray mu: array of eigenvalues
    :param numpy.ndarray lambd: array of lambda values
    :param float lambda0: lambda0 value
    :xi: ?
    :param bool REML: perform REML (default=True)

    '''

    dfChiK = n - p - k if (n - p - k) > 0 else 0

    if REML:
        n0 = n - p
        xi = mu
    else:
        assert xi is not None, "xi can't be None if REML == False !"
        n0 = n

    # pre-compute stuff that stays constant over simulations

    # broadcast to shape (g, k)
    lambdamu = lambd[:, np.newaxis] * mu
    # (g, k)
    lambdamuP1 = lambdamu + 1.

    # broadcast to shape (g, k)
    fN = ((lambd - lambda0)[:, np.newaxis] * mu) / lambdamuP1
    # (g, k)
    fD = (1 + lambda0 * mu) / lambdamuP1

    # shape (g,)
    sumlog1plambdaxi = np.sum(np.log1p(lambd[:, np.newaxis] * xi), axis=1)

    # simulations

    # need nsim values of ChiK
    # shape (nsim, )
    ChiK = chi2(dfChiK).rvs(nsim)

    # need (nsim, k) values of Chi1
    Chi1 = np.reshape(chi2(1).rvs(nsim * k), (nsim, 1, k))

    if not REML:
        # need (nsim, ) values of ChiSum
        ChiSum = Chi1.sum(axis=1)

    # fN (g, k) *  Chi1 (nsim, 1, k) -> N (nsim, g, k) -sum-> N (nsim, g)
    N = np.sum(fN * Chi1, axis=2)
    # fD (g, k) * Chi1 (nsim, 1, k) -> D (nsim, g, k) -sum-> D (nsim, g)
    D = np.sum(fD * Chi1, axis=2)
    # D (nsim, g) + Chik(nsim,) -> D (nsim, g)
    D += ChiK[:,np.newaxis]

    # n0 (1,) * log1p(N/D) (nsim, g) -> LR (nsim, g)
    LR = n0 * np.log1p(N / D)
    # LR (nsim, g) - ...(g,) -> LR (nsim, g)
    LR -= sumlog1plambdaxi

    # (nsim, g) -> lambdaind (nsim,)
    lambdaind = np.argmax(LR, axis=1)
    # (nsim, g) -> res (nsim,)
    res = LR[np.arange(LR.shape[0]), lambdaind]

    if not REML:
        res += n * np.log1p(chi2(q).rvs(nsim) / (ChiSum + ChiK))

    result = {
        'res': res,
        'lambdaind': lambdaind,
        'lambdamu': lambdamu,
        'fN': fN,
        'fD': fD,
        'sumlog1plambdaxi': sumlog1plambdaxi,
        'n': n,
        'p': p,
        'dfChik': dfChiK,
        'REML': REML
    }

    return result


def rlrsim(p, k, n, nsim, g, q, mu, lambd, lambda0, xi=None, REML=True):


    '''
    Python port of the RLRsim.cpp function using scipy & numpy

    This version uses broadcasting + cython to speed up operations. It is about 3 times faster than the numpy-version.

    :param int p: number of covariates in X
    :param int k: number of variables to test
    :param int n: number of individuals
    :param int nsim: number of sumulations to perform
    :param int g: lambda grid length
    :param int q: number of free parameters in the null-model (?)
    :param numpy.ndarray mu: array of eigenvalues
    :param numpy.ndarray lambd: array of lambda values
    :param float lambda0: lambda0 value
    :xi: ?
    :param bool REML: perform REML (default=True)

    '''

    dfChiK = n - p - k if (n - p - k) > 0 else 0

    if REML:
        n0 = n - p
        xi = mu
    else:
        assert xi is not None, "xi can't be None if REML == False !"
        n0 = n

    # pre-compute stuff that stays constant over simulations
    # g is the axis along lambda values that we optimize over

    # broadcast to shape (g, k)
    lambdamu = lambd[:, np.newaxis] * mu
    # (g, k)
    lambdamuP1 = lambdamu + 1.

    # broadcast to shape (g, k)
    fN = ((lambd - lambda0)[:, np.newaxis] * mu) / lambdamuP1
    fD = (1 + lambda0 * mu) / lambdamuP1

    # shape (g,)
    sumlog1plambdaxi = np.sum(np.log1p(lambd[:, np.newaxis] * xi), axis=1)

    # simulations

    # need nsim values of ChiK (nsim, )
    ChiK = chi2(dfChiK).rvs(nsim)

    # need (nsim, k) values of Chi1
    Chi1 = np.reshape(chi2(1).rvs(nsim * k), (nsim, k))

    if not REML:
        # need (nsim, ) values of ChiSum
        ChiSum = Chi1.sum(axis=1)
    else:
        ChiSum = np.zeros((nsim,), dtype=float)
        
    # this is the part that we want to optimize:
    res = np.zeros(nsim, dtype=float)
    lambdaind = np.zeros(nsim, dtype=int)
        
    optimize_lambda(res, lambdaind, nsim, g, n0, ChiSum, ChiK, Chi1, sumlog1plambdaxi, fN, fD)

    if not REML:
        res += n * np.log1p(chi2(q).rvs(nsim) / (ChiSum + ChiK))

    result = {
        'res': res,
        'lambdaind': lambdaind,
        'lambdamu': lambdamu,
        'fN': fN,
        'fD': fD,
        'sumlog1plambdaxi': sumlog1plambdaxi,
        'n': n,
        'p': p,
        'dfChik': dfChiK,
        'REML': REML
    }

    return result


def RLRTSim(X, Z, Xdagger, sqrtSigma=None, lambda0=np.nan, seed=2020, nsim=10000, use_approx=0, log_grid_hi=8,
            log_grid_lo=-10, gridlength=200):

    '''
    Python port of the LRTSim function using scipy & numpy
    see https://cran.r-project.org/web/packages/RLRsim/RLRsim.pdf for details, also
    Crainiceanu, C. and Ruppert, D. (2004) Likelihood ratio tests in linear mixed models with one variance component

    :param numpy.ndarray X: The fixed effects design matrix of the model under the alternative
    :param numpy.ndarray Z: The random effects design matrix of the model under the alternative (i.e. G_1)
    :param numpy.ndarray Xdagger: Moore-Penrose pseudo-inverse of X
    :param numpy.ndarray sqrtSigma: The upper triangular Cholesky factor of the correlation matrix of the random effect
    :param numpy.ndarray lambda0:
    :param seed: numpy random seed to use
    :param int nsim: Number of values to simulate
    :param int use_approx: Not implemented (yet)
    :param log_grid_hi: Higher value of the grid on the log scale.
    :param log_grid_lo: Lower value of the grid on the log scale.
    :param int gridlength: Length of the grid for the grid search over lambda

    :return: A dictionary containing the simulations and other parameters
    :rtype: dict

    Output dictionary:
      "res":  np.array of simulated test statistics
       ...:   parameters and intermediate variables used to generate simulated test statistics

    '''

    # this is a port of the R-function ...
    # sqrtSigma is the upper triangular factor of the cholesky decomposition

    if np.isnan(lambda0):
        lambda0 = 0

    if lambda0 > np.exp(log_grid_hi):
        log_grid_hi = np.log(10 * lambda0)
        # print warning

    if (lambda0 != 0) & (lambda0 < np.exp(log_grid_lo)):
        log_grid_lo = np.log(-10 * lambda0)
        # print warning

    if seed is not None:
        # is this safe?
        np.random.seed(seed)

    n, p = np.shape(X)

    K = min(n, Z.shape[1])

    sqrtSigma = np.eye(Z.shape[1]) if sqrtSigma is None else sqrtSigma

    # project out X
    rZ = Z - X.dot(Xdagger.dot(Z))
    np.matmul(rZ, sqrtSigma.T, out=rZ)

    try:
        mu = svd(rZ, full_matrices=False, compute_uv=False)
        if np.any(mu < -0.1):
            logging.warning("kernel contains a negative Eigenvalue")
        mu *= mu
    except LinAlgError:  # revert to Eigenvalue decomposition
        logging.warning(
            "Got SVD exception, trying eigenvalue decomposition of square of Z. Note that this is a little bit less accurate")
        mu_ = eigh(rZ.T.dot(rZ), eigvals_only=True)
        if np.any(mu_ < -0.1):
            logging.warning("kernel contains a negative Eigenvalue")
        mu = mu_ * mu_
        #is it ok if some are 0?

    # normalize
    mu /= np.max(mu)

    if use_approx != 0:
        raise NotImplementedError
    # else:
    # the original has different approximate procedures that could be re-implemented here

    lambda_grid = make_lambdagrid(lambda0, gridlength, log_grid_lo, log_grid_hi)

    res = rlrsim(p=p,
                 k=K,
                 n=n,
                 nsim=nsim,
                 g=gridlength,
                 q=0,
                 mu=mu,
                 lambd=lambda_grid,
                 lambda0=lambda0,
                 xi=mu,
                 REML=True)

    res['lambda'] = lambda_grid[res['lambdaind']] if 'lambdaind' in res else np.zeros((1,))

    return res


def pv_chi2mixture(stat, scale, dof, mixture, alteqnull=None):

    '''
    Returns p-values(s) using chi2 with custom parameters.

    see :func:`fit_chi2mixture`

    wraps chi2.sf(stat/scale, dof) * mixture

    :param stat: The LRT test statistic
    :type stat: Union[float, np.array]
    :param float scale: scaling factor applied to the test statistic
    :param float dof: degrees of freedom
    :param float mixture: mixture weight

    :return: p-value
    :rtype: float

    '''

    if alteqnull is None:
        alteqnull = stat == 0.

    pv = np.array(chi2.sf(stat/scale,dof) * mixture)
    pv[alteqnull] = 1.

    return pv


def fit_chi2mixture(sims, qmax=0.1):

    '''
    Takes simulated test statistics as input. Fits a chi2 mixture to them using "quantile regression" (Listgarten 2013).
    returns a dictionary.

    :param numpy.ndarray sims: array of tests statistics
    :param float qmax: fraction of largest test statistics to fit the distribution with

    :return: Dictionary with chi2 mixture parameters
    :rtype: dict

    Output dictionary:
      "mse"   : mean squared error
      "dof"   : degrees of freedom
      "scale" : scale
      "imax"  : number of values used to fit the ditribution
    '''

    mix = chi2mixture.chi2mixture(lrt=sims, qmax=qmax, fitdof=True)
    # method described in LRT paper
    res = mix.fit_params_Qreg()
    res['mixture'] = mix.mixture

    return res


class LRTnoK():

    '''Single kernel Likelihood Ratio Test (LRT) for continuous phenotypes.

    Sets up null model for given phenotypes and covariates.
    Use :func:`altmodel` to fit an alternative model, and calculate the LRT test statistic.
    Use :func:`pv_sim` to simulate test statistics and generate empirical p-values.

    :param numpy.ndarray X: :math:`nxc` matrix containing the covariates (:math:`n:=` number of individuals, :math:`c:=` number of covariates)
    :param numpy.ndarray Y: :math:`nx1` matrix containing the phenotype (:math:`n:=` number of individuals)
    :param bool REML: use restricted maximum likelihood? (only "True" has been tested)

    :ivar X: X
    :ivar Y: Y
    :ivar Xdagger: Moore-Penrose pseudo-inverse of X
    :ivar model0: dictionary with nLL of the null model
    :ivar model1: dictionary with nLL and other parameters of the alternative model

    '''

    def __init__(self, X, Y, REML=True):

        self.X = X
        if np.ndim(Y) == 1:
            self.Y = Y[:, np.newaxis]
        elif np.ndim(Y) == 2:
            self.Y = Y
        else:
            raise ValueError("Can't handle multi-dimensional Y!")

        self.Xdagger = np.linalg.pinv(X) # needed for rlrsim
        self._nullmodel(REML=REML)
        self.model1 = None
        self.likalt={}


    def _nullmodel(self, REML=True):
        # initialize the null model
        self.model0 = {}
        model = linreg(self.X, self.Y[:,0])
        self.model0['h2'] = np.nan
        self.model0['nLL'] = model['nLL']

    def altmodel(self, G1):

        """Fit alternative model. Returns a dictionary with model parameters and test statistic.

        :param numpy.ndarray G1:  :math:`G_1`: set of variants (i.e. variables) to test, dimensions :math:`nxk` (:math:`n:=` number of individuals, :math:`k:=` number of variants in set to test association for)

        :return: Dictionary of fitted model parameters
        :rtype: dict

        Output dictionary:
        'nLL'       : negative log-likelihood
        'sigma2'    : the model variance sigma^2
        'stat'      : LRT test statistic
        'alteqnull' : h2 is 0 or negative (i.e. p-value is 1.)
        'beta'      : [D*1] array of fixed effects weights beta
        'h2'        : mixture weight between Covariance and noise
        'REML'      : True: REML was computed, False: ML was computed
        'a2'        : mixture weight between K0 and K1
        'scale'     : Scale parameter that multiplies the Covariance matrix (default 1.0)
        --------------------------------------------------------------------------
        """
        # use FaST-LMM's implementation
        lmm1 = lmm.LMM()
        lmm1.setG(G1)
        lmm1.setX(self.X)
        lmm1.sety(self.Y[:,0])
        lik1 = lmm1.findH2() #The alternative model has one kernel and needs to find only h2

        lik1['alteqnull'] = lik1['h2'] <= 0.0
        lik1['stat'] = 2.0*(self.model0['nLL'] - lik1['nLL'])

        self.model1 = lmm1
        self.model1_lik = lik1

        return lik1

    def pv_5050(self, lik1=None):

        """
        Return p-value(s) calculated assuming a 50/50 mixture of chi2 df0 and chi2 df1. Considered a conservative estimate.
        If lik1 is None, returns the p-value for the current alternative model.

        :param lik1: likelihood ratio test statistic
        :type lik1: Union[float, numpy.ndarray]

        :rtype: Union[float, numpy.ndarray]

        """

        if lik1 is None:
            lik1 = self.model1_lik

        if lik1['alteqnull']:
            pv = 1.
        else:
            pv = chi2(1).sf(lik1['stat']) * 0.5

        return pv


    def pv_sim(self, nsim=100000, seed=420):

        '''
        Runs "nsim" simulations of the test statistic IF the likelihood of the alternative model is greater than that of the null model, i.e. self.model_lik['alteqnull'] is False.
        returns a dictionary with the simulations and empirical p-value. The empirical p-value is 1. if self.model_lik['alteqnull'] is True, and the array of simulated test statistics will be empty.

        :param int nsim: Number of simulated test statistics to return
        :param int seed: numpy random seed to use for simulations

        :return: Dictionary with containing the simulated test statistics, and empirical p-value
        :rtype: dict

        Output dictionary:
          "res": array of simulated test statistics
          "pv":  empirical p-value of current alternative model

        '''

        lik1 = self.model1_lik

        if lik1['alteqnull']:
            pv = 1.
            result = {
                'res': np.array([]),
                'pv': pv
            }
            return result
        else:
            sims = RLRTSim(self.X, self.model1.G, self.Xdagger, nsim=nsim, seed=seed)
            pv = np.mean(lik1['stat'] < sims['res'])
            sims['pv'] = pv
            return sims

