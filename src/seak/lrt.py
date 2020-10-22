

import numpy as np
from scipy.stats import chi2
from numpy.linalg import svd
from numpy.linalg import eigvals

from fastlmm.util.stats import linreg
from fastlmm.inference import lmm

def make_lambdagrid(lambda0, gridlength, log_grid_lo, log_grid_hi):

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
            leftgrid = np.linspace(np.exp(log_grid_lo), lambda0, length=leftlength + 1)[:-1]
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


def rlrsim(p, k, n, nsim, g, q, mu, lambd, lambda0, xi=None, REML=True):

    # this is a port of the c++ function ...

    '''
    Python port of the RLRsim.cpp function using scipy & numpy

    :param int p: number of covariates in X
    :param int k: number of variables to test
    :param int n: number of individuals
    :param int nsim: number of sumulations to perform
    :param int g: lambda grid length
    :param int q: number of free parameters in the null-model (?)
    :param np.ndarray mu: array of eigenvalues
    :param np.ndarray lambd: array of lambda values
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
    lambdamuP1 = lambdamu + 1.

    # broadcast to shape (g, k)
    fN = ((lambd - lambda0)[:, np.newaxis] * mu) / lambdamuP1
    fD = (1 + lambda0 * mu)[:, np.newaxixs] / lambdamuP1

    # shape (g,)
    sumlog1plambdaxi = np.sum(np.log1p(lambd[:np.newaxis] * xi), axis=1)

    # simulations
    res = np.zeros(nsim)
    lambdaind = np.zeros(nsim)

    # replace this loop with map?
    for i_s in range(nsim):

        LR = 0.

        ChiSum = 0

        ChiK = chi2(dfChiK).rvs(1)  # Xnpk
        Chi1 = chi2(1).rvs(k)  # ws^2

        # rchisq(n, df, ncp = 0)

        if not REML:
            ChiSum = Chi1.sum()

        # loop over lambda values:
        # replace this loop with broadcasting?
        for i_g in range(g):

            N = fN[i_g, :] * Chi1
            D = fD[i_g, :] * Chi1

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
              'Chi1': Chi1,
              'ChiK': ChiK,
              'n': n,
              'p': p,
              'dfChik': dfChiK,
              'REML': REML
              }

    return result


def rlrsim_precomputed(res, nsim):

    '''
    Python port of the RLRsim.cpp function using scipy & numpy
    Takes pre-computed matrices from a previous simulation as input
    '''

    dfChiK = res['dfChiK']
    REML = res['REML']

    if REML:
        n0 = res['n'] - res['p']
    else:
        n0 = res['n']

    # pre-compute stuff that stays constant over simulations

    # shape (g, k)
    lambdamu = res['lambdamu']

    # shape (g, k)
    fN = res['fN']
    fD = res['fD']

    # shape (g,)
    sumlog1plambdaxi = res['sumlog1plambdaxi']

    # simulations
    res = np.zeros(nsim)
    lambdaind = np.zeros(nsim)

    # replace this loop with map?
    for i_s in range(nsim):

        LR = 0.

        ChiSum = 0

        ChiK = chi2(dfChiK).rvs(1)  # Xnpk
        Chi1 = chi2(1).rvs(fN.shape[1])  # ws^2

        if not REML:
            ChiSum = Chi1.sum()

        # loop over lambda values:
        # replace this loop with broadcasting?

        for i_g in range(lambdamu.shape[0]):

            N = fN[i_g, :] * Chi1
            D = fD[i_g, :] * Chi1

            D += ChiK

            LR = n0 * np.log1p(N / D) - sumlog1plambdaxi[i_g]

            if LR >= res[i_s]:
                res[i_s] = LR
                lambdaind[i_s] = i_g
            else:
                break

        if not REML:
            res[i_s] = res[i_s] + n0 * np.log1p(chi2(res['q']).rvs(1) / (ChiSum + ChiK))

    result = {'res' : res,
              'lambdaind': lambdaind,
              'lambdamu': lambdamu,
              'fN': fN,
              'fD': fD,
              'sumlog1plambdaxi': sumlog1plambdaxi,
              'Chi1': Chi1,
              'ChiK': ChiK,
              'n': res['n'],
              'p': res['p'],
              'dfChik': dfChiK,
              'REML': REML
              }

    return result



def RLRTSim(X, Z, Xdagger, sqrtSigma=None, lambda0=np.nan, seed=2020, nsim=10000, use_approx=0, log_grid_hi=8,
            log_grid_lo=-10, gridlength=200):

    '''
    Python port of the RLRTsim function using scipy & numpy

    :param np.ndarray X: covariate matrix X (design matrix)
    :param np.ndarray Z: variables to test
    :param np.ndarray Xdagger: Xdagger
    :param np.ndarray sqrtSigma: upper triangular factor of the cholesky decomposition of the correlation matrix Sigma
    :param np.ndarray lambda0:
    :param seed: numpy random seed to use
    :param int nsim: number of simulations to run
    :param int use_approx: Not implemented (yet)
    :param log_grid_hi:
    :param log_grid_lo:
    :param int gridlength: length of the lambda grid

    '''

    # this is a port of the R-function ...
    # sqrtSigma is the upper triangular factor of the cholesky decomposition

    if np.isnan(lambda0):
        lambda0 = 0

    if len(lambda0) > 1:
        raise ValueError('len(lambda0) > 1')

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

    mu = eigvals(Z - sqrtSigma.T.dot(X.dot(Xdagger.dot(Z)))) ** 2

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

    return {'res': res, 'lambda': lambda_grid[res['lambdaind']]}


class LRTnoK():

    '''
    barebone version of "lrt" in FaST-LMM
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

        self.model0 = {}
        model = linreg(self.X, self.Y[:,0])
        self.model0['h2'] = np.nan
        self.model0['nLL'] = model['nLL']

    def altmodel(self, G1):

        '''
        Output dictionary:
        'nLL'       : negative log-likelihood
        'sigma2'    : the model variance sigma^2
        'stat'      : rlrt test statistic
        'alteqnull' : h2 is 0 or negative
        'beta'      : [D*1] array of fixed effects weights beta
        'h2'        : mixture weight between Covariance and noise
        'REML'      : True: REML was computed, False: ML was computed
        'a2'        : mixture weight between K0 and K1
        'dof'       : Degrees of freedom of the Multivariate student-t
                        (default None uses multivariate Normal likelihood)
        'scale'     : Scale parameter that multiplies the Covariance matrix (default 1.0)
        --------------------------------------------------------------------------
        '''
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

        '''
        p-value calculated assuming a 50/50 mixture of chi2 df0 and chi2 df1
        '''

        if lik1 is None:
            lik1 = self.model1_lik

        if lik1['alteqnull']:
            pv = 1.
        else:
            pv = chi2.sf(lik1['stat']) * 0.5

        return pv


    def pv_sim(self, nsim=100000):

        '''
        p-value calculated using 'nsim' simulations
        '''

        lik1 = self.model1_lik

        if lik1['alteqnull']:
            pv = 1.
            return {'p':pv,'sim':np.array([])}
        else:
            sims = RLRTSim(self.X, self.model1.G, self.Xdagger, nsim=nsim)
            pv = np.mean(lik1['stat'] < sims)
            return {'p':pv, 'sim':sims}
