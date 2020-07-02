"""Computes a background kernel matrix :math:`K_0` or respective genotype matrix :math:`G_0` from
`binary PLINK 1 <https://www.cog-genomics.org/plink/2.0/input#bed>`_ files (:class:`GRMLoader`) or
loads a precomputed matrix :math:`K_0` from a NumPy file (:class:`GRM_from_file`).

Needed, if a linear mixed model is used to correct for confounding.
In low rank case (:math:`n>k`): loads a genotype matrix :math:`G_0`.
In full rank case (:math:`n<k`): computes background kernel :math:`K_0`.
With :math:`n:=` number of individuals and :math:`k:=` number of SNVs.

GRM: genetic relatedness matrix.
"""
# imports
import logging

import numpy as np
from pandas_plink import read_plink

from seak.data_loaders import VariantLoader

# set logging configs
logging.basicConfig(format='%(asctime)s - %(lineno)d - %(message)s')


class GRMLoader:
    """Constructs a background kernel :math:`K_0` from given binary PLINK 1 genotype files using the leave-one-out-chromosome (LOCO) strategy.

    Initially no background kernel is constructed, only the instance attributes are initialized. The kernel gets
    constructed when calling the method :func:`compute_background_kernel` which should only be called after calling the
    :func:`update_ind` method manually or the :func:`seak.data_loaders.intersect_and_update_datasets`.
    This way, individuals which are neither contained in the test nor in the background kernel data set are excluded.

    In full rank case, loads the SNPs in blocks to construct the kernel.
    In low rank case, loads all SNPs into memory at once.

    :param str path_to_plink_files_with_prefix: path prefix to genotype PLINK files for background kernel construction, if several files should be loaded please use the wildcard character *
    :param int blocksize: how many genotypes to load at once; should be chosen dependent on RAM available
    :param str/int LOCO_chrom_id: identifier of the chromosome/region that is used in the respective test set and should be excluded from the background kernel or None if all variants should be included
    :param bool forcelowrank: enforce low rank data loading behavior for testing purposes

    .. note:: The leave-one-chromosome-out (LOCO) strategy can be disabled with :attr:`LOCO_chrom_id`.
    .. note:: The input file directory should only contain a single FAM file (requirement of `pandas_plink <https://pandas-plink.readthedocs.io/en/latest/api/pandas_plink.read_plink.html>`_).
    """
    def __init__(self, path_to_plink_files_with_prefix, blocksize, LOCO_chrom_id=None, forcelowrank=False):
        """Constructor."""
        self.forcelowrank = forcelowrank  # only for testing purposes!
        self.GRM_bim, self.GRM_fam, self.GRM_bed = read_plink(path_to_plink_files_with_prefix)
        self.GRM_bim['bim_nindex'] = self.GRM_bim.index
        self.GRM_fam['fam_nindex'] = self.GRM_fam.index
        self.GRM_fam['iid'] = self.GRM_fam['iid'].astype(str)
        self.GRM_fam = self.GRM_fam.rename(columns={'iid': 'individual_id'})
        self.GRM_fam.set_index(keys='individual_id', inplace=True)
        self.variants_to_include = self._get_LOCO_SNV_indices(LOCO_chrom_id)
        self.blocksize = blocksize
        self.nb_ind = None
        self.nb_SNVs_unf = None
        self.G0 = None
        self.K0 = None
        self.nb_SNVs_f = None
        self.samples_overlapped = False

    def _get_LOCO_SNV_indices(self, LOCO_chrom_id):
        """Returns list of indices that should be included in the GRM.

        :param str/int LOCO_chrom_id: identifier of the chromosome/region that is used in the respective test set and should be excluded from the background kernel or None if all variants should be included
        :return: numerical indices of the SNVs to exclude from the background kernel computation
        :rtype: numpy.ndarray or ndarray-like
        """
        if LOCO_chrom_id is None:
            return self.GRM_bim.loc[:, 'bim_nindex'].values
        else:
            LOCO_SNV_nindices = self.GRM_bim.loc[:, 'bim_nindex'][~(self.GRM_bim['chrom'].astype(str) == str(LOCO_chrom_id))].values
            return LOCO_SNV_nindices

    def update_individuals(self, iids):
        """Sets individuals to include into the background kernel data set based on individual ids (:attr:`iids`).

        :param iids: numpy.Series of individual ids that should be retained for background kernel computation
        """
        self.GRM_fam = self.GRM_fam.loc[iids]
        self.samples_overlapped = True

    def get_iids(self):
        """Returns all individual ids.

        :return:
        :rtype: pandas.Index
        """
        return self.GRM_fam.index

    def _build_G0(self):
        """Low rank case: constructs :math:`G_0` from provided bed file (PLINK 1).

        :return: normalized genotypes :math:`G_0` and number of SNVs that where loaded
        :rtype: numpy.ndarray, int
        """
        temp_genotypes = self.GRM_bed[self.variants_to_include, :].compute()
        temp_genotypes = temp_genotypes[:, self.GRM_fam['fam_nindex'].values].T
        temp_genotypes = VariantLoader.mean_imputation(temp_genotypes)
        filter_invariant = temp_genotypes == temp_genotypes[0, :]
        filter_invariant = ~filter_invariant.all(0)
        filter_all_nan = ~np.all(np.isnan(temp_genotypes), axis=0)
        total_filter = filter_invariant & filter_all_nan
        temp_genotypes = temp_genotypes[:, total_filter]
        temp_genotypes = VariantLoader.standardize(temp_genotypes)
        nb_SNVs_filtered = temp_genotypes.shape[1]
        # Normalize
        return temp_genotypes / np.sqrt(nb_SNVs_filtered), nb_SNVs_filtered

    def _build_K0_blocked(self):
        """Full rank case: Builds background kernel :math:`K_0` by loading blocks of SNPs from provided bed file (PLINK 1).

        :return: normalized background kernel :math:`K_0` and number of SNVs that where used to built the kernel
        :rtype: numpy.ndarray, int
        """
        K0 = np.zeros([self.nb_ind, self.nb_ind], dtype=np.float32)
        nb_SNVs_filtered = 0
        stop = self.nb_SNVs_unf
        for start in range(0, stop, self.blocksize):
            if start+self.blocksize >= stop:
                temp_genotypes = self.GRM_bed[self.variants_to_include[start:], ].compute()
            else:
                temp_genotypes = self.GRM_bed[self.variants_to_include[start:start + self.blocksize], ].compute()
            print("self.GRM_fam['fam_nindex'].values)", self.GRM_fam['fam_nindex'].values)
            print("type(self.GRM_fam['fam_nindex'].values)", type(self.GRM_fam['fam_nindex'].values))
            temp_genotypes = temp_genotypes[:, self.GRM_fam['fam_nindex'].values].T
            temp_genotypes = VariantLoader.mean_imputation(temp_genotypes)
            filter_invariant = temp_genotypes == temp_genotypes[0, :]
            filter_invariant = ~filter_invariant.all(0)
            filter_all_nan = ~np.all(np.isnan(temp_genotypes), axis=0)
            total_filter = filter_invariant & filter_all_nan
            temp_genotypes = temp_genotypes[:, total_filter]
            temp_genotypes = VariantLoader.standardize(temp_genotypes)
            temp_n_SNVS = temp_genotypes.shape[1]
            nb_SNVs_filtered += temp_n_SNVS
            temp_K = np.matmul(temp_genotypes, temp_genotypes.T)
            K0 += temp_K
        # Normalize
        return K0 / nb_SNVs_filtered, nb_SNVs_filtered

    def compute_background_kernel(self):
        """Computes background kernel :math:`K_0` for given set of genotypes (binary PLINK 1 files).

        Overlap with data of set to be tested should have been carried out before, such that individuals in both data
        sets match.
        Does not return anything but sets instance attributes for either the background kernel :math:`K_0` or the
        background kernel genotype matrix :math:`G_0`.
        """
        if not self.samples_overlapped:
            logging.warning('Data to construct background kernel was not overlapped with data of set to be tested.')
        self.nb_ind = self.GRM_fam.shape[0]
        self.nb_SNVs_unf = len(self.variants_to_include)
        print('# of individuals: {}'.format(self.nb_ind))
        print('# of SNVs: {}'.format(self.nb_SNVs_unf))
        # low rank
        if self.nb_ind > self.nb_SNVs_unf or self.forcelowrank:
            self.G0, self.nb_SNVs_f = self._build_G0()
            self.K0 = None
        # full rank
        else:
            self.G0 = None
            self.K0, self.nb_SNVs_f = self._build_K0_blocked()


class GRMLoader_from_file:
    """Load a precomputed GRM from a NumPy file (.npy or .npz).

     Must either be of low rank :math:`n>k` or a square :math:`nxn` matrix with :math:`n:=` number of individuals and :math:`k:=` number of SNVs.
     """

    def __init__(self, path_to_GRM_npy, forcelowrank=False):
        """Loads a background kernel from a .npy or .npz files

        :param path_to_GRM_npy: path to precomputed GRM
        :param forcelowrank: for testing purposes only, loads genotypes even though there are more SNVs than individuals
        """

        self.nb_ind, self.nb_SNVs, self.G0, self.K0 = self._load_background_kernel(path_to_GRM_npy, forcelowrank)

    def _load_background_kernel(self, path_to_GRM_npy, forcelowrank):
        """Loads background kernel from numpy file

        :param path_to_GRM_npy: path to precomputed GRM from numpy file
        :param forcelowrank: for testing purposes only, loads genotypes even though there are more SNVs than individuals
        :return: number of individuals, number of SNVs, :math:`G0` or :math:`K0`
        :rtype: int, int, None or numpy.ndarray
        """
        GRM = np.load(path_to_GRM_npy)
        nb_ind = GRM.shape[0]
        nb_SNVs = GRM.shape[1]
        print('# of individuals: {}'.format(nb_ind))
        print('# of SNVs: {}'.format(nb_SNVs))
        # low rank
        if nb_ind > nb_SNVs:
            G0 = GRM
            K0 = None
        # full rank
        elif nb_ind == nb_SNVs:
            G0 = None
            K0 = GRM
        elif nb_ind <= nb_SNVs and forcelowrank:
            G0 = GRM
            K0 = None
        else:
            raise ValueError('The provided data for the GRM/K0/G0 has inappropriate dimensions (either nxk with '
                             'n:= individuals and k:= SNVs and n > k for G0, or nxn for K0.')
        return nb_ind, nb_SNVs, G0, K0
