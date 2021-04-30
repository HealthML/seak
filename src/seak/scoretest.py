"""Contains classes for variance component score tests.

A single kernel and two-kernel score-based test is available for a linear link function (continuous phenotypes) (:class:`ScoretestNoK` and :class:`Scoretest2K`).
With the second kernel :math:`K_0` correcting for population structure.
A single kernel score-based test is available for a logistic link function (binary phenotypes) (:class:`ScoretestLogit`)

Null model single kernel:

.. math:: y = {\\alpha} X + {\\epsilon}

Alternative model single kernel:

.. math:: y = {\\alpha} X + {\\gamma} G_1 + {\\epsilon}

Null model two-kernel:

.. math:: y = {\\alpha} X + {\\beta} G_0 + {\\epsilon}

Alternative model two-kernel:

.. math:: y = {\\alpha} X + {\\beta} G_0 + {\\gamma} G_1 + {\\epsilon}

With: :math:`X`: covariates, dimension :math:`nxc` (:math:`n:=` number of individuals, :math:`c:=` number of covariates),
:math:`G_0`: variants to construct the background kernel :math:`K_0` from, correcting for population structure, dimensions :math:`nxm` (:math:`n:=` number of individuals, :math:`m:=` number of variants/variants),
:math:`G_1`: set of variants to test, dimensions :math:`nxk` (:math:`n:=` number of individuals, :math:`k:=` number of variants in set to test association for).

.. note:: For all classes of the module :mod:`scoretest` the class attributes are instance attributes! This is a bug in the automatic documentation.

.. note::
   The source code of the mathematical implementation is adopted from the `FaST-LMM <https://pypi.org/project/fastlmm/>`_ Python package  from Microsoft Corporation.
   Source code can be found in :mod:`fastlmm.association.score`.

"""

# imports
import logging
import time

import numpy as np
import scipy as sp
import scipy.linalg as LA
from scipy.optimize import brentq as root
from scipy.stats import chi2, norm
import statsmodels.api as sm
from seak.cppextension import wrap_qfc

from seak.mingrid import minimize1D

# set logging configs
logging.basicConfig(format='%(asctime)s - %(lineno)d - %(message)s')


class DaviesError(Exception):
    """raise this error if Davie's method does not converge"""
    pass

class Scoretest:
    """Superclass for all score-based set-association tests.

    Defines all common attributes and implements methods that all subclasses share, such as methods for p-value
    computation for alternative :func:`pv_alt_model` model.

    Interface for Scoretest subclasses.

    :ivar numpy.ndarray Y: :math:`nx1` matrix containing the phenotype (:math:`n:=` number of individuals)
    :ivar numpy.ndarray X: :math:`nxc` matrix containing the covariates (:math:`n:=` number of individuals, :math:`c:=` number of covariates)
    :ivar int N: number of individuals
    :ivar int P: number of phenotypes
    :ivar int D: number of covariates
    :ivar int Neff: effective sample size (number of individuals) as used in calculations of unbiased estimators
    """

    __slots__ = ["Y", "X", "N", "P", "D", "Neff"]

    def __init__(self, phenotypes, covariates):
        """Constructor.

        If no covariates are given or if the covariates do not contain a bias column, a bias column is added.
        Sets up all attributes common to all score-based set-association tests.

        :param numpy.ndarray phenotypes: :math:`nx1` matrix containing the phenotype (:math:`n:=` number of individuals)
        :param numpy.ndarray covariates: :math:`nxc` matrix containing the covariates (:math:`n:=` number of individuals, :math:`c:=` number of covariates)
        """
        self.Y = self._set_phenotypes(phenotypes)  # phenotypes
        self.N = self.Y.shape[0]  # number of individuals
        self.P = self.Y.shape[1]  # number of phenotypes
        self.X = self._set_covariates(covariates)  # covariates
        self.D = self.X.shape[1]  # num of covariates
        self.Neff = self.N - self.D  # unbiased estimator for variance

        if self.P != 1:
            logging.error('More than one phenotype given.')

        if self.X.shape[0] != self.N:
            logging.error('Number of individuals in phenotype and covariates does not match.')

    def _set_phenotypes(self, phenotypes):
        """Casts phenotypes to two dimensions."""
        if phenotypes.ndim == 1:
            phenotypes = phenotypes.reshape(len(phenotypes), 1)
        return phenotypes

    def _set_covariates(self, covariates):
        """Appends bias (offset) column to covariates if not present."""
        if covariates is None:
            X = np.ones((self.N, 1))
        elif not self.has_bias(covariates):
            X = sp.hstack((np.ones((self.N, 1)), covariates))
        else:
            X = covariates
        return X

    @staticmethod
    def has_bias(covariates):
        """Checks whether at least one all constant column (bias) is present in the covariates."""
        # Might have multiple invariant columns, though! Not dropped.
        for i in range(covariates.shape[1]):
            if np.all(covariates[:, i] == covariates[0, i]) and covariates[0, i] != 0:
                return True
        return False

    @staticmethod
    def _linreg(Y, X=None, Xdagger=None):
        """Efficient multiplication with symmetric covariate orthogonal projection matrix.

        Corresponds to Lippert et al., 2014, Supplement p. 10 Proposition 9
        S*a = a - X(Xdagger*a)
        with S = (I_N - X*(X.T*X)^-1*X.T) rank: N-D, symmetric covariate orthogonal projection matrix
        Source: fastlmm.association.score.py

        :param numpy.ndarray Y: factor that is multiplied with S
        :param numpy.ndarray X: covariate matrix
        :param numpy.ndarray Xdagger: precomputed Moore-Penrose pseudo-inverse of the covariate matrix or None, gets computed if not provided
        :return: result of multiplication (regressed out covariate effects); and Moore-Penrose pseudo-inverse of the covariate matrix
        :rtype: numpy.ndarray, numpy.ndarray

        The resulting term is nothing but the ordinary least squares (OLS) regression residuals after regressing out X.
        Note, that the pseudoinverse of the covariates, X†, need only be computed once (in O(ND2)) and then can be re-used across different a.
        For multiplication of matrices by S the result is applied by treating each row or column as a vector in the multiplication.
        Lippert et al., 2014

        """
        if X is None:
            RxY = Y - Y.mean(0)
            return RxY, None
        else:
            if Xdagger is None:
                Xdagger = np.linalg.pinv(X)
            RxY = Y - X.dot(Xdagger.dot(Y))
            return RxY, Xdagger

    @staticmethod
    def _hat(Y, X=None, Xdagger=None):
        """Efficient multiplication with hat-matrix.
        """

        if X is None:
            Yhat = Y.mean(0)
            return Yhat, None
        else:
            if Xdagger is None:
                Xdagger = np.linalg.pinv(X)
            Yhat = X.dot(Xdagger.dot(Y))
            return Yhat, Xdagger


    def pv_alt_model(self, G1, G2=None, method='davies'):
        """Computes p-value of the alternative model.

        :param numpy.ndarray G1: set of variants to test, dimensions :math:`nxk` (:math:`n:=` number of individuals, :math:`k:=` number of variants to test association for)
        :return: p-value of the alternative model
        :rtype: float
        """
        if G1.shape[0] != self.N:
            logging.error('Number of individuals in phenotype and genotypes to be tested does not match.')

        if G2 is None:
            squaredform, GPG = self._score(G1)
        else:
            if G1.shape[0] != G2.shape[0]:
                logging.error('Number of individuals in G1 and G2 do not match.')
            squaredform, GPG = self._score_conditional(G1, G2)

        if method == 'davies':
            pv = self._pv_davies(squaredform, GPG)
        elif method == 'saddle':
            pv = self._pv_saddle(squaredform, GPG)
        else:
            raise NotImplementedError('method "{}" not implemented.'.format(method))

        return pv

    def _score(self, G1):
        """Method that returns the score-based test statistic (squaredform) and the matrix (1/2)xG.TxPthetaxG (GPG) from
        which the null distribution can be efficiently computed for a set of genotypes.

        :param numpy.ndarray G1: set of variants to test, dimensions :math:`nxk` (:math:`n:=` number of individuals, :math:`k:=` number of variants to test association for)
        :return: squaredform, GPG
        :raises: NotImplementedError - Interface.
        """
        raise NotImplementedError

    def _score_conditional(self, G1, G2):
        """Method that returns the conditional score-based test statistic (squaredform) for G1 conditioned on G2 and the
        matrix (1/2)xG.TxPthetaxG (GPG) from which the null distribution can be efficiently computed for a set of genotypes.

        :param numpy.ndarray G1: set of variants to test, dimensions :math:`nxk` (:math:`n:=` number of individuals, :math:`k:=` number of variants to test association for)
        :return: squaredform, GPG
        :raises: NotImplementedError - Interface.
        """
        raise NotImplementedError


    @staticmethod
    def _pv_davies(squaredform, GPG):
        """Given the test statistic and GPG computes the corresponding p-value."""
        eigvals = LA.eigh(GPG, eigvals_only=True)
        pv = Scoretest._pv_davies_eig(squaredform, eigvals)
        return pv

    @staticmethod
    def _pv_davies_eig(squaredform, eigvals):
        """Given the test statistic and the eigenvalues of GPG computes the corresponding p-value using Davie's method"""

        if squaredform == 0.:
            return 1.

        if len(eigvals) == 1:
            # skip
            return (chi2(df=1., scale=eigvals).sf(squaredform))[0]

        try:
            result = Scoretest._qf(squaredform, eigvals)
        except DaviesError:
            logging.warning('Using "saddle" instead of "davies" (likely because "davies" did not converge)')
            result = [Scoretest._pv_saddle_eig(squaredform, eigvals)]  # if Davies method does not converge use
        # removed keyword argument that corresponds to default.

        if result[0] == 0.:
            logging.warning('Using "saddle" instead of "davies" (Davies returned 0.)')
            try:
                result = [Scoretest._pv_saddle_eig(squaredform, eigvals)]
            except ValueError:
                logging.warning('failed to use "saddle" but "davies" returned 0., reporting -1.')
                result = [-1.0]

        return result[0]

    @staticmethod
    def _qf(chi2val, coeffs, dof=None, noncentrality=None, sigma=0.0, lim=1000000, acc=1e-7):
        """Given the test statistic (squaredform) and the eigenvalues of GPG computes the corresponding p-value, calls a C script."""

        from seak.cppextension import wrap_qfc

        size = coeffs.shape[0]
        if dof is None:
            dof = np.ones(size, dtype='int32')
        if noncentrality is None:
            noncentrality = np.zeros(size)
        ifault = np.zeros(1, dtype='int32')
        trace = np.zeros(7)

        pval = 1.0 - wrap_qfc.qf(coeffs, noncentrality, dof, sigma, chi2val, lim, acc, trace, ifault)

        if ifault[0] > 0:
            logging.warning('ifault {} encountered during p-value computation'.format(ifault[0]))
            raise DaviesError

        return pval, ifault[0], trace

    @staticmethod
    def _pv_saddle(squaredform, GPG):
        eigvals = LA.eigh(GPG, eigvals_only=True)
        pv = Scoretest._pv_saddle_eig(squaredform, eigvals)
        return pv

    @staticmethod
    def _pv_saddle_eig(squaredform, eigvals, delta=None):
        """Given the test statistic and the eigenvalues of GPG computes the corresponding p-value using saddle-point approximation."""
        # (D. KUONEN 1999) as implemented in the skatMeta package
        x = squaredform
        lambd = eigvals
        delta = np.zeros(len(lambd)) if delta is None else delta

        if x == 0.:
            # skip
            return 1.

        if len(lambd) == 1:
            # skip
            return (chi2(df=1., loc=delta, scale=lambd).sf(x))[0]

        d = np.max(lambd)
        lambd /= d
        x /= d

        def k0(zeta):
            return -1 * np.sum(np.log(1 - 2 * zeta * lambd)) / 2 + np.sum((delta * lambd * zeta) / (1 - 2 * zeta * lambd))

        def kprime0(zeta):
            return np.sum(lambd / (1 - 2 * zeta * lambd)) + np.sum((delta*lambd)/(1-2*zeta*lambd) + 2*(delta*zeta*lambd**2)/(1-2*zeta*lambd)**2)

        def kpprime0(zeta):
            return 2 * np.sum(lambd**2 / (1-2*zeta*lambd)**2) + np.sum((4*delta*lambd**2) / (1-2*zeta*lambd)**2 + 8*delta*zeta*lambd**3 / (1-2*zeta*lambd)**3)

        n = len(lambd)

        if np.any(lambd < 0.):
            lmin = np.max(1 / (2 * lambd[lambd < 0.])) * 0.99999
        elif x > np.sum(lambd):
            lmin = -0.01
        else:
            lmin = -n / (2 * x)

        lmax = np.min(1 / (2 * lambd[lambd > 0.])) * 0.99999

        try:
            def func(zeta):
                return kprime0(zeta) - x
            hatzeta = root(func, lmin, lmax, maxiter=1000)
        except RuntimeError as e:
            logging.warning('P-value computation did not converge:\n{}'.format(e))
            return np.nan

        w = np.sign(hatzeta) * np.sqrt(2 * (hatzeta * x - k0(hatzeta)))
        v = hatzeta * np.sqrt(kpprime0(hatzeta))

        if np.abs(hatzeta) < 1e-4:
            return np.nan
        else:
            return norm(loc=0., scale=1.).sf(w + np.log(v/w)/w)




class ScoretestNoK(Scoretest):
    """Single kernel score-based set-association test for continuous phenotypes.

    Sets up null model for given phenotypes and covariates.
    If no covariates are given or if the covariates do not contain a bias column, a bias column is added.
    Compute p-value for alternative model with :func:`pv_alt_model`.

    Null model single kernel:

    .. math:: Y = {\\alpha} X + {\\epsilon}

    Alternative model single kernel:

    .. math:: Y = {\\alpha} X + {\\gamma} G_1 + {\\epsilon}

    With: :math:`X`: covariates, dimension :math:`nxc` (:math:`n:=` number of individuals, :math:`c:=` number of covariates)
    :math:`G_1`: set of variants to test, dimensions :math:`nxk` (:math:`n:=` number of individuals, :math:`k:=` number of variants in set to test association for)

    :param numpy.ndarray phenotypes: :math:`nx1` matrix containing the phenotype (:math:`n:=` number of individuals)
    :param numpy.ndarray covariates: :math:`nxc` matrix containing the covariates (:math:`n:=` number of individuals, :math:`c:=` number of covariates)

    :ivar RxY: OLS residuals of the phenotype after regressing out fixed effects (covariates X)
    :ivar Xdagger: Moore-Penrose pseudo-inverse of the covariate matrix X
    :ivar sigma2: environmental variance
    """

    __slots__ = ["RxY", "Xdagger", "sigma2"]

    def __init__(self, phenotypes, covariates):
        """Constructor."""
        super().__init__(phenotypes, covariates)
        self.RxY, self.Xdagger, self.sigma2 = self._compute_null_model()

    def coef(self, G1):

        '''
        Returns single-variant regression coefficients and their estimated variances

        :param numpy.ndarray G1: N x 1 vector
        :return dict: dictionary with two slots: "beta" and "var_beta"
        '''

        if G1.ndim == 1:
            G1 = G1[:, np.newaxis]

        assert G1.shape[1] == 1, 'Error: this is only supported for single variables'

        RxG, self.Xdagger = super()._linreg(Y=G1, X=self.X, Xdagger=self.Xdagger)

        denom = G1.T.dot(RxG)

        beta_hat = G1.T.dot(self.RxY) / denom

        residuals = (self.RxY - RxG.dot(beta_hat))
        sigma2 = (residuals * residuals).sum() / (self.N - self.D - 1)

        var_beta_hat = sigma2 / denom

        return {'beta':beta_hat, 'var_beta':var_beta_hat}

    def _compute_null_model(self):
        """Computes parameters of null model."""
        # residual of y regressed on X, which here, is equivalent to sigma2*Py (P is the projection matrix, which is idempotent)
        # note: Xdagger is pseudo inverse of X

        RxY, Xdagger = super()._linreg(Y=self.Y, X=self.X, Xdagger=None)

        # estimate for residual (environmental) variance
        sigma2 = (RxY * RxY).sum() / (self.Neff * self.P)

        return RxY, Xdagger, sigma2

    def _score(self, G1):
        """Computes squaredform and GPG, input for p-value computation. """

        # for the 1K case, P reduces to 1/sigma2*S
        # multiplication with S is achieved by getting the residuals regressed on X.
        # SG, needed for "GPG":
        RxG, self.Xdagger = super()._linreg(Y=G1, X=self.X, Xdagger=self.Xdagger)

        # needed for the squared form:
        GtRxY = G1.T.dot(self.RxY)

        ## original note: P is never computed explicitly, only via residuals such as Py=1/sigma2(I-Xdagger*X)y and
        ## PG=1/sigma2(I-Xdagger*X)G
        ## also note that "RxY"=Py=1/sigma2*(I-Xdagger*X)y is nothing more (except for 1/sigma2) than the residual of y
        ## regressed on X (i.e. y-X*beta), and similarly for PG="RxG"

        # note: because GtRxY has shape (D, 1), the code below is the same as (GtRxY.transpose()).dot(GtRxY)/(2 * sigma2^2):
        ## original note: yPKPy=yPG^T*GPy=(yPG^T)*(yPG^T)^T

        squaredform = ((GtRxY * GtRxY).sum()) * (0.5 / (self.sigma2 * self.sigma2))

        # we are only interested in the eigenvalues of GPG
        # np.dot(RxG.T, RxG) and np.dot(RxG, RxG.T) have the same non-zero eigenvalues!
        if G1.shape[0] > G1.shape[1]:
            # full rank, i.e. D > N
            GPG = np.dot(RxG.T, RxG)  # GPG is always a square matrix in the smaller dimension
        else:
            # low rank, i.e. D < N
            GPG = np.dot(RxG, RxG.T)

        GPG /= self.sigma2 * 2.0  # what we will take eigenvalues of for Davies, scale because P is 0.5 * 1/sigmae2 * S

        return squaredform, GPG

    def _score_conditional(self, G1, G2):
        """Computes squaredform and GPG, input for p-value computation. """

        # for the 1K case, P reduces to 1/sigma2*S
        # multiplication with S is achieved by getting the residuals regressed on X.
        # SG, needed for "GPG":

        n1 = G1.shape[1]

        Gc = np.concatenate([G1, G2], axis=1)

        # SG
        RxGc, Xdagger = super()._linreg(Y=Gc, X=self.X, Xdagger=self.Xdagger)

        # score statistics:
        GtRxY = Gc.T.dot(self.RxY)
        G2tRxY = GtRxY[n1:]

        GPG = np.dot(RxGc.T, RxGc)

        G1tPG1 = GPG[0:n1, 0:n1]
        G2tPG2 = GPG[n1:, n1:]
        G1tPG2 = GPG[0:n1, n1:]
        G2tPG1 = GPG[n1:, 0:n1]

        # conditioning of the test statistics:
        G1tPG2_G2tPG2inv = G1tPG2.dot(np.linalg.inv(G2tPG2))

        # conditional G1tPG1 -> GPG
        GPG = G1tPG1 - G1tPG2_G2tPG2inv.dot(G2tPG1)

        # conditional squaredform
        expected_teststat = G1tPG2_G2tPG2inv.dot(G2tRxY)
        G1tRxY_cond = GtRxY[:n1] - expected_teststat

        squaredform = ((G1tRxY_cond * G1tRxY_cond).sum()) / (2.0 * self.sigma2 * self.sigma2)
        GPG /= (self.sigma2 * 2.0)

        return squaredform, GPG

class ScoretestLogit(Scoretest):
    """Single kernel score-based set-association test for binary phenotypes.

    Sets up null model for given phenotypes and covariates.
    If no covariates are given or if the covariates do not contain a bias column, a bias column is added.
    Compute p-value for alternative model with :func:`pv_alt_model`.

    :param numpy.ndarray phenotypes: :math:`nx1` matrix containing the phenotype (:math:`n:=` number of individuals)
    :param numpy.ndarray covariates: :math:`nxc` matrix containing the covariates (:math:`n:=` number of individuals, :math:`c:=` number of covariates)
    """

    __slots__ = ["pY", "stdY", "VX", "pinvVX"]

    def __init__(self, phenotypes, covariates):
        super().__init__(phenotypes, covariates)
        # check if is binary
        uniquey = np.unique(self.Y)
        if not np.sort(uniquey).tolist() == [0, 1]:
            raise Exception("must use binary data in {0,1} for logit tests, found:" + str(self.Y))
        self.pY, self.stdY, self.VX, self.pinvVX = self._compute_null_model()

    def _compute_null_model(self):
        """Computes parameters of null model."""
        logreg_mod = sm.Logit(self.Y[:, 0], self.X)
        logreg_result = logreg_mod.fit(disp=0)
        pY = logreg_result.predict(self.X)
        stdY = np.sqrt(pY * (1.0 - pY))
        VX = self.X * np.lib.stride_tricks.as_strided(stdY, (stdY.size, self.X.shape[1]), (stdY.itemsize, 0))
        pinvVX = np.linalg.pinv(VX)
        return pY, stdY, VX, pinvVX

    def _score(self, G1):
        """Computes squaredform and GPG, input for p-value computation."""
        RxY = (self.Y.flatten() - self.pY)  # residual of y regressed on X, which here, is equivalent to sigma2*Py
        # (P is the projection matrix, which is idempotent)
        VG = G1 * np.lib.stride_tricks.as_strided(self.stdY, (self.stdY.size, G1.shape[1]), (self.stdY.itemsize, 0))
        GY = G1.T.dot(RxY)
        squaredform = (GY * GY).sum() / (2.0 * self.P)
        RxVG, Xd = super()._linreg(VG, X=self.VX, Xdagger=self.pinvVX)

        if G1.shape[0] < G1.shape[1]:
            GPG = RxVG.dot(RxVG.T) / (2.0 * self.P)
        else:
            GPG = RxVG.T.dot(RxVG) / (2.0 * self.P)
        return squaredform, GPG


class Scoretest2K(Scoretest):
    """Two-kernel score-based set-association test for continuous phenotypes.

    Sets up null model for given phenotypes, covariates and background kernel :math:`K_0` or background genotypes :math:`G_0`.
    If no covariates are given or if the covariates do not contain a bias column, a bias column is added.
    Compute p-value for alternative model with :func:`pv_alt_model`.

    Null model two-kernel:

    .. math:: Y = {\\alpha} X + {\\beta} G_0 + {\\epsilon}

    Alternative model two-kernel:

    .. math:: Y = {\\alpha} X + {\\beta} G_0 + {\\gamma} G_1 + {\\epsilon}

    With: :math:`X`: covariates, dimension :math:`nxc` (:math:`n:=` number of individuals, :math:`c:=` number of covariates)
    :math:`G_0`: variants to construct the background kernel :math:`K_0` from, correcting for population structure, dimensions :math:`nxm` (:math:`n:=` number of individuals, :math:`m:=` number of variants/variants)
    :math:`G_1`: set of variants to test, dimensions :math:`nxk` (:math:`n:=` number of individuals, :math:`k:=` number of variants in set to test association for)

    :param numpy.ndarray phenotypes: :math:`nx1` matrix containing the phenotype (:math:`n:=` number of individuals)
    :param numpy.ndarray covariates: :math:`nxc` matrix containing the covariates (:math:`n:=` number of individuals, :math:`c:=` number of covariates)
    :param numpy.ndarray K0: genetic similarity matrix/GRM :math:`K_0` accounting for confounding
    :param numpy.ndarray G0: genotype matrix :math:`G_0` used to contruct math:`K_0` to account for confounding
    :param boolean forcefullrank: for testing purposes only

    :ivar S: eigenvalues of PKP
    :ivar Xdagger: Moore-Penrose pseudo-inverse of the covariate matrix X
    :ivar sigma2e: environmental variance
    :ivar sigma2g: genetic variance
    """

    __slots__ = ["K0", "G0", "Xdagger", "S", "U", "lowrank", "UY", "UUY", "YUUY", "optparams", "sigma2e", "sigma2g"]

    def __init__(self, phenotypes, covariates=None, K0=None, G0=None, forcefullrank=False):
        """Constructor."""
        # note: super().__init__ simply fills the slots for covariates etc. no computation done yet:
        super().__init__(phenotypes, covariates)
        self.G0 = G0
        self.K0 = K0
        # spectral decomposition, needed to efficiently compute the matrix square root of P, see Lippert 2014, suppl. 7.3
        self.Xdagger, self.S, self.U, self.lowrank, self.UY, self.YUUY = self._compute_spectral_decomposition(forcefullrank=forcefullrank)
        self.optparams = self._compute_null_model()
        self.sigma2e = (1.0 - self.optparams["h2"]) * self.optparams["sigma2"]
        self.sigma2g = self.optparams["h2"] * self.optparams["sigma2"]

    def _compute_spectral_decomposition(self, forcefullrank=False):
        """Computes spectral decomposition of K0 or G0."""
        lowrank = False
        Xdagger = None

        # we want to compute the spectral decomposition of P (needed for matrix square root)
        # sigma2g * P = (S(Kg+delta*I)S)^dagger
        # we can compute the spectral decomposition of the right hand


        if self.K0 is not None:
            # K0 already computed, i.e. the full rank case, work with K0
            # compute SVD of SKS
            # see Lemma 10 in Lippert et al. 2014
            ar = np.arange(self.K0.shape[0])

            self.K0[ar, ar] += 1.0

            PxKPx, Xdagger = super()._linreg(Y=self.K0, X=self.X, Xdagger=Xdagger)
            PxKPx, self.Xdagger = super()._linreg(Y=PxKPx.T, X=self.X, Xdagger=Xdagger)

            # S are the eigenvalues
            # U are the eigenvectors
            [S, U] = LA.eigh(PxKPx)
            self.K0[ar, ar] -= 1.0
            U = U[:, self.D:self.N]
            S = S[self.D:self.N] - 1.0

        elif 0.7 * (self.Neff) <= self.G0.shape[1] or forcefullrank:
            # K0 gets computed, work with K0

            self.K0 = self.G0.dot(self.G0.T)
            # compute SVD of SKS
            # see Lemma 10 in Lippert et al. 2014
            # the rest is identical to the case above...
            ar = np.arange(self.K0.shape[0])
            self.K0[ar, ar] += 1.0

            PxKPx, Xdagger = super()._linreg(Y=self.K0, X=self.X, Xdagger=Xdagger)
            PxKPx, self.Xdagger = super()._linreg(Y=PxKPx.T, X=self.X, Xdagger=Xdagger)

            # S are the eigenvalues
            # U are the eigenvectors
            self.K0[ar, ar] -= 1.0
            [S, U] = LA.eigh(PxKPx)
            U = U[:, self.D:self.N]
            S = S[self.D:self.N] - 1.0

        else:
            # work with G0 instead of K0, this is the low-rank case

            PxG, Xdagger = super()._linreg(Y=self.G0, X=self.X, Xdagger=Xdagger)
            [U, S, V] = LA.svd(PxG, False, True)
            inonzero = S > 1E-10
            # S are the eigenvalues
            # U are the eigenvectors
            S = S[inonzero] * S[inonzero]
            U = U[:, inonzero]
            lowrank = True

        UY = U.T.dot(self.Y)

        if lowrank:
            Yres, Xdagger = super()._linreg(Y=self.Y, X=self.X, Xdagger=Xdagger)
            UUY = Yres - U.dot(UY)
            YUUY = (UUY * UUY).sum()
        else:
            YUUY = None

        return Xdagger, S, U, lowrank, UY, YUUY

    def _compute_null_model(self):
        """Computes parameters of null model."""
        resmin = [None]

        def f(x, resmin=resmin, **kwargs):
            res = self._nLLeval(h2=x)
            if (resmin[0] is None) or (res['nLL'] < resmin[0]['nLL']):
                resmin[0] = res
            return res['nLL']

        minimize1D(f, evalgrid=None, nGrid=20, minval=0.0, maxval=0.99999)

        # dictionary containing the model parameters at the optimal h2
        optparams = resmin[0]

        return optparams

    def _nLLeval(self, h2=0.0):
        """
        evaluate -ln( N( U^T*y | U^T*X*beta , h2*S + (1-h2)*I ) ),
        where K = USU^T
        --------------------------------------------------------------------------
        Input:
        h2      : mixture weight between K and Identity (environmental noise)
        --------------------------------------------------------------------------
        Output dictionary:
        'nLL'       : negative log-likelihood
        'sigma2'    : the model variance sigma^2
        'h2'        : mixture weight between Covariance and noise
        --------------------------------------------------------------------------
        """
        if (h2 < 0.0) or (h2 >= 1.0):
            return {'nLL': 3E20,
                    'h2': h2
                    }
        k = self.S.shape[0]

        Sd = h2 * self.S + (1.0 - h2)
        UYS = self.UY / np.lib.stride_tricks.as_strided(Sd, (Sd.size, self.UY.shape[1]), (Sd.itemsize, 0))

        YKY = (UYS * self.UY).sum()

        logdetK = np.emath.log(Sd).sum()

        if (self.lowrank):  # low rank part
            YKY += self.YUUY / (1.0 - h2)
            logdetK += np.emath.log(1.0 - h2) * (self.Neff * self.P - k)

        sigma2 = YKY / (self.Neff * self.P)
        nLL = 0.5 * (logdetK + self.Neff * self.P * (np.emath.log(2.0 * sp.pi * sigma2) + 1))
        result = {
            'nLL': nLL,
            'sigma2': sigma2,
            'h2': h2
        }
        return result


    def _score(self, G1):
        """Computes squaredform and GPG with a background kernel."""

        # SG
        RxG, Xdagger = super()._linreg(Y=G1, X=self.X, Xdagger=self.Xdagger)

        # UtSG
        UG = self.U.T.dot(RxG)

        if self.lowrank:
            UUG = RxG - self.U.dot(UG)

        # Compare to Lippert 2014, Lemma 11. Rescale eigenvalues according to mixing parameters
        # The inverse of the diagonal matrix of eigenvalues (Lambda + delta*I) is calculated trivially:
        Sd = 1.0 / (self.S * self.sigma2g + self.sigma2e)

        # matrix multiplication of UtSG with (Lambda + delta*I)^-1, which is called Sd here
        SUG = UG * np.lib.stride_tricks.as_strided(Sd, (Sd.size, UG.shape[1]), (Sd.itemsize, 0))

        GPY = SUG.T.dot(self.UY)
        if self.lowrank:
            GPY += UUG.T.dot(self.UUY) / self.sigma2e

        # see GPY in (14)
        squaredform = 0.5 * (GPY * GPY).sum()

        if G1.shape[0] > G1.shape[1]:
            GPG = SUG.T.dot(UG)
        else:
            GPG = SUG.dot(UG.T)


        if self.lowrank:
            if G1.shape[0] > G1.shape[1]:
                GPG_lowr = UUG.T.dot(UUG) / self.sigma2e
            else:
                GPG_lowr = UUG.dot(UUG.T) / self.sigma2e
            GPG += GPG_lowr

        GPG *= 0.5

        # in the original they compute expectationsqform (expected value) and varsqform (variance) fo the squared form.
        # these were not used.

        return squaredform, GPG

    def _score_conditional(self, G1, G2):
        """Computes squaredform and GPG with a background kernel."""

        n1 = G1.shape[1]

        Gc = np.concatenate([G1, G2], axis=1)

        # SG
        RxG, Xdagger = super()._linreg(Y=Gc, X=self.X, Xdagger=self.Xdagger)

        # UtSG
        UG = self.U.T.dot(RxG)

        if self.lowrank:
            UUG = RxG - self.U.dot(UG)

        # Compare to Lippert 2014, Lemma 11. Rescale eigenvalues according to mixing parameters
        # The inverse of the diagonal matrix of eigenvalues (Lambda + delta*I) is calculated trivially:
        Sd = 1.0 / (self.S * self.sigma2g + self.sigma2e)

        # matrix multiplication of UtSG with (Lambda + delta*I)^-1, which is called Sd here
        SUG = UG * np.lib.stride_tricks.as_strided(Sd, (Sd.size, UG.shape[1]), (Sd.itemsize, 0))

        GPY = SUG.T.dot(self.UY)
        if self.lowrank:
            GPY += UUG.T.dot(self.UUY) / self.sigma2e

        G2tPY = GPY[n1:]

        GPG = SUG.T.dot(UG)

        if self.lowrank:
            GPG_lowr = UUG.T.dot(UUG) / self.sigma2e
            GPG += GPG_lowr

        G1tPG1 = GPG[0:n1, 0:n1]
        G2tPG2 = GPG[n1:, n1:]
        G1tPG2 = GPG[0:n1, n1:]
        G2tPG1 = GPG[n1:, 0:n1]

        # conditioning of the test statistics:
        G1tPG2_G2tPG2inv = G1tPG2.dot(np.linalg.inv(G2tPG2))

        # conditional squaredform
        expected_teststat = G1tPG2_G2tPG2inv.dot(G2tPY)
        G1tPY_cond = GPY[:n1] - expected_teststat
        squaredform = 0.5 * (G1tPY_cond * G1tPY_cond).sum()

        # conditional G1tPG1 -> GPG
        GPG = G1tPG1 - G1tPG2_G2tPG2inv.dot(G2tPG1)
        GPG *= 0.5

        return squaredform, GPG

