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
import pandas as pd
# from pandas_plink import read_plink

from pysnptools.snpreader import Bed, SnpReader
from pysnptools.kernelreader import KernelHdf5, KernelNpz, KernelData
from pysnptools.standardizer import Unit, DiagKtoN

# set logging configs
logging.basicConfig(format='%(asctime)s - %(lineno)d - %(message)s')

# TODO: define general kernel interface as with data_loaders

class GRMLoaderSnpReader:
    """Constructs a background kernel :math:`K_0` from given binary PLINK 1 genotype files using the leave-one-out-chromosome (LOCO) strategy.

    Initially no background kernel is constructed, only the instance attributes are initialized. The kernel gets
    constructed when calling the method :func:`compute_background_kernel` which should only be called after calling the
    :func:`update_ind` method manually or the :func:`seak.data_loaders.intersect_and_update_datasets`.
    This way, individuals which are neither contained in the test nor in the background kernel data set are excluded.

    In full rank case, loads the SNPs in blocks to construct the kernel.
    In low rank case, loads all SNPs into memory at once.

    :param str path_to_plink_files_with_prefix: path prefix to genotype PLINK files for background kernel construction
    :param int blocksize: how many genotypes to load at once; should be chosen dependent on RAM available
    :param str/int LOCO_chrom_id: identifier of the chromosome/region that is used in the respective test set and should be excluded from the background kernel or None if all variants should be included
    :param bool forcelowrank: enforce low rank data loading behavior for testing purposes

    .. note:: The leave-one-chromosome-out (LOCO) strategy can be disabled with :attr:`LOCO_chrom_id`.
    """

    def __init__(self, path_or_bed, blocksize, LOCO_chrom_id=None, forcelowrank=False):
        """Constructor."""
        self.forcelowrank = forcelowrank  # only for testing purposes!

        if isinstance(path_or_bed, str):
            self.bed = Bed(path_or_bed, count_A1=True)
        else:
            assert isinstance(path_or_bed, SnpReader), 'path_or_bed must either be a path to a bed-file, or an instance of SnpReader.'

        self.bed.pos[:, 0] = self.bed.pos[:, 0].astype('str')  # chromosome should be str, stored positions are 1-based
        self.iid_fid = pd.DataFrame(self.bed.iid, index=self.bed.iid[:, 1].astype(str), columns=['fid', 'iid'])

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
            return np.arange(self.bed.sid_count, dtype=int)
        else:
            return np.where(~(self.bed.pos[:, 0].astype(str) == LOCO_chrom_id))[0]

    def update_individuals(self, iids):
        """Sets individuals to include into the background kernel data set based on individual ids (:attr:`iids`).

        :param iids: numpy.Series of individual ids that should be retained for background kernel computation
        """
        iid_fid = self.iid_fid.loc[iids]
        self.bed = self.bed[self.bed.iid_to_index(iid_fid.values), :]
        self.samples_overlapped = True

    def get_iids(self):
        """Returns all individual ids.
        :return:
        :rtype: numpy.ndarray
        """
        return self.iid_fid.index.values

    def _build_G0(self):
        """Low rank case: constructs :math:`G_0` from provided bed file (PLINK 1).

        :return: normalized genotypes :math:`G_0` and number of SNVs that where loaded
        :rtype: numpy.ndarray, int
        """

        temp_genotypes = self.bed[:, self.variants_to_include].read().standardize(Unit()).val

        # Replaced the code below with PySnpTools internal standardizer
        #filter_invariant = ~(temp_genotypes == temp_genotypes[0, :]).all(0)
        #filter_invariant = ~filter_invariant.all(0)
        #filter_all_nan = ~np.all(np.isnan(temp_genotypes), axis=0)
        #total_filter = filter_invariant & filter_all_nan
        #temp_genotypes = temp_genotypes[:, total_filter]
        #temp_genotypes = VariantLoader.standardize(temp_genotypes)
        #nb_SNVs_filtered = temp_genotypes.shape[1]
        # Normalize
        #return temp_genotypes / np.sqrt(nb_SNVs_filtered), nb_SNVs_filtered

        # TODO: is invariant-filtering really necessary here?
        invariant = (temp_genotypes == temp_genotypes[0, :]).all(0)

        n_filtered = (~invariant).sum()
        temp_genotypes /= np.sqrt(n_filtered)

        return temp_genotypes[:, ~invariant], n_filtered

    def _build_K0_blocked(self):
        """Full rank case: Builds background kernel :math:`K_0` by loading blocks of SNPs from provided bed file (PLINK 1).

        :return: normalized background kernel :math:`K_0` and number of SNVs that where used to built the kernel
        :rtype: numpy.ndarray, int
        """

        # TODO: make use of PySnpTools KernelReader functionality

        K0 = np.zeros([self.nb_ind, self.nb_ind], dtype=np.float32)
        nb_SNVs_filtered = 0
        stop = self.nb_SNVs_unf

        for start in range(0, stop, self.blocksize):

            if start+self.blocksize >= stop:
                temp_genotypes = self.bed[:, self.variants_to_include[start:]].read().standardize(Unit()).val
            else:
                temp_genotypes = self.bed[:, self.variants_to_include[start:start + self.blocksize]].read().standardize(Unit()).val

            # Replaced the code below with the PySnpTools internal standardizer
            # temp_genotypes = VariantLoader.mean_imputation(temp_genotypes)
            # filter_invariant = temp_genotypes == temp_genotypes[0, :]
            # filter_invariant = ~filter_invariant.all(0)
            # filter_all_nan = ~np.all(np.isnan(temp_genotypes), axis=0)
            # total_filter = filter_invariant & filter_all_nan
            # temp_genotypes = temp_genotypes[:, total_filter]
            # temp_genotypes = VariantLoader.standardize(temp_genotypes)
            # temp_n_SNVS = temp_genotypes.shape[1]
            # nb_SNVs_filtered += temp_n_SNVS

            # TODO: is invariant-filtering really necessary here?
            invariant = (temp_genotypes == temp_genotypes[0, :]).all(0)

            K0 += np.matmul(temp_genotypes[:, ~invariant], temp_genotypes[:, ~invariant].T)
            nb_SNVs_filtered += (~invariant).sum()

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
        self.nb_ind = self.bed.iid_count
        self.nb_SNVs_unf = self.bed.sid_count
        print('# of individuals for background kernel: {}'.format(self.nb_ind))
        print('# of (unfiltered) SNVs for background kernel: {}'.format(self.nb_SNVs_unf))
        # low rank
        if self.nb_ind > self.nb_SNVs_unf or self.forcelowrank:
            self.G0, self.nb_SNVs_f = self._build_G0()
            self.K0 = None
        # full rank
        else:
            self.G0 = None
            self.K0, self.nb_SNVs_f = self._build_K0_blocked()
        print('# of filtered SNVs for background kernel: {}'.format(self.nb_SNVs_f))

    def write_kernel(self, path, filetype='hdf5'):
        """Write constructed background kernel :math:`K_0` to file, using eihter pysnptools.kernelreader.KernelHdf5 or pysnptools.kernelreader.KernelNpz.

        :param str path: Path to the output file to be created.
        :param str filetype: Either 'hdf5' or 'npz'
        """
        if self.K0 is None:
            if self.G0 is not None:
                raise ValueError('G0 is initialized: Number of individuals < number of variants. In this case no kernel is constructed.')
            raise ValueError('K0 is not initialized, need to call compute_background_kernel() first')
        elif filetype == 'hdf5':
            KernelHdf5.write(path, KernelData(self.iid_fid.values, val=self.K0))
        elif filetype == 'npz':
            KernelNpz.write(path, KernelData(self.iid_fid.values, val=self.K0))
        else:
            raise ValueError('filetype has to be either "npz" or "hdf5", got {}'.format(filetype))


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
