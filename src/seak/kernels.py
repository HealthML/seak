"""

Functions that can incorporate prior knowledge into set-association tests.

Any function that takes a genotype matrix G, variant effects V, and optional additional arguments and returns a set of
transformed genotypes/variables to be tested.

"""

# imports
import numpy as np
from sklearn.cluster import AgglomerativeClustering


# Kernel functions
def diffscore_max(G, V, scale_sqrt=True):
    """Uses the largest absolute variant effect predictions (veps) per SNV.

    Linear weighted kernel.

    :param numpy.ndarray G: SNVs to be tested (genotypes), :math:`nxm` (:math:`n:=` number of individuals, :math:`m:=` number of SNVs)
    :param numpy.ndarray V: veps :math:`mxk` (:math:`m:=` number of SNVs, :math:`k:=` number of veps); dimension :math:`m` needs to be equal in G and V
    :return: linear weighted genotype matrix (:math:`GxV`)
    :rtype: numpy.ndarray
    """
    V = np.abs(V)
    if scale_sqrt:
        V = np.sqrt(V)
    # Maximum out of any diffscore prediction
    V = np.amax(V, axis=1)
    V = np.diag(V, k=0)
    return np.matmul(G, V)


def single_column_kernel(i, scale_sqrt=True):
    """Returns a kernel function with weights that correspond to a single column of the veps numpy.ndarray.

    :param int i: column of which weights should be selected
    :return: kernel function that linearly weights the genotypes with a single column of veps (e.g. corresponding to a specific transcription factor), needs to be called with genotypes and veps array
    :rtype: function
    """
    def skk(G, V):
        """Scales the SNVs by the variant effect prediction (vep) corresponding to a single column of the vep file scaled by a factor of 100 as linear weights.

        Linear weighted kernel.

        :param numpy.ndarray G: SNVs to be tested (genotypes), :math:`nxm` (:math:`n:=` number of individuals, :math:`m:=` number of SNVs)
        :param numpy.ndarray V: veps :math:`mxk` (:math:`m:=` number of SNVs, :math:`k:=` number of veps); dimension :math:`m` needs to be equal in G and V
        :return: linear weighted genotype matrix (:math:`GxV`)
        :rtype: numpy.ndarray
        """
        V = V[:, i]
        V = np.abs(V)
        if scale_sqrt:
            V = np.sqrt(V)
        V = np.diag(V, k=0)
        return np.matmul(G, V)
    return skk


def linear(G, V):
    """ Linear kernel, does not scale the genotypes."""
    return G


def linear_weighted(G, V):
    """Linear weighted kernel.

    Scales the SNVs by the variant effect predictions (veps) corresponding of a 1-dimensional array which gets
    transformed into a diagonal weight matrix as linear weights.

    :param numpy.ndarray G: SNVs to be tested (genotypes), :math:`nxm` (:math:`n:=` number of individuals, :math:`m:=` number of SNVs)
    :param numpy.ndarray V: veps :math:`mx1` (:math:`m:=` number of SNVs) with dimension 1 and shape: :math:`m`
    :return: linear weighted genotype matrix (:math:`GxV`)
    :rtype: numpy.ndarray
    """
    V = np.diag(V, k=0)
    return np.matmul(G, V)


class LocalCollapsing:

    """Class for local collapsing of GV, based on a set of positions

    :param float distance_threshold: maximum distance allowed between SNPs within the same cluster

    """

    def __init__(self, distance_threshold=100.):
        self.clust = AgglomerativeClustering(n_clusters=None, affinity='manhattan', linkage='complete', distance_threshold=distance_threshold)

    def collapse(self, G, pos, weights=1.):

        """
        Collapses G locally, based on pos

        :param np.ndarray G: Matrix containing the values to be collapsed, typically shape = (n_individuals, n_SNP)
        :param np.ndarray pos: array-like, passed to sklearn.cluster.AgglomerativeClustering
                               typically shape (n_samples, 1) for base pair positional clustering,
                               but supports (n_samples, n_features) or (n_samples, n_samples).
        :param np.array weights: single variant weights, shape = (n_SNP, ), default: equal weights for all variants

        :return: Collapsed matrix with shape (n_individuals, n_clusters), and the cluster assignments (n_individuals, )
        :rtype: tuple (np.ndarray, np.ndarray)

        """

        n_snp = G.shape[1]

        if pos.ndim == 1:
            pos = pos.reshape((-1, 1))

        clusters = self.clust.fit_predict(pos)

        C = np.zeros((n_snp, self.clust.n_clusters_))
        C[np.arange(n_snp),clusters] = weights

        return np.matmul(G,C), clusters
