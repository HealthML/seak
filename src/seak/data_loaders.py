"""Contains all data IO and processing functionalities to read and manipulate genotype, covariate and variant effect prediction data.

Implements interfaces for all data types such that you can write your custom data loading classes (:class:`VariantLoader`, :class:`AnnotationLoader`, :class:`CovariatesLoader`).

Accepted input file format for genotype data is the `binary PLINK 1 <https://www.cog-genomics.org/plink/2.0/input#bed>`_ format.
Accepted input file format for covariate and phenotype data is the CSV format.
Accepted input file format for variant effect predictions is the HDF5 format, following the syntax of the output of
:func:`Janggu.predict_variant_effect` (`Janggu documentation <https://janggu.readthedocs.io/en/latest/tutorial.html#id1>`_) which needs to be accompanied by a index file in TSV format.

.. note:: The program assumes that genotype files are 1-based and fully closed, and variant effect prediction files 0-based and half-open.
"""

# imports
import logging

import h5py  # load VEPs
import numpy as np
import pandas as pd
from functools import reduce
from pysnptools.snpreader import Bed, SnpReader
# from pandas_plink import read_plink  # utilities to read 1-based plink files

# set logging configs
logging.basicConfig(format='%(asctime)s - %(lineno)d - %(message)s')

# TODO: allow for multi-intersection based on intervalarray or similar
# TODO: make intersections based on coordintes "fool proof" -> convert XYM chromosomes to numbers when intersecting with snpreader

def intersect_ids(ar1, ar2):
    """
    returns the intersection (common elements) of ar1 and ar2, in the order that they first appear in ar1.

    :param np.ndarray ar1: Array of ids
    :param np.ndarray ar2: Array of ids
    """

    ids, query_idx, _ = np.intersect1d(ar1, ar2, return_indices=True)
    return ids[np.argsort(query_idx)]

class VariantLoader:
    """Interface for loading genotype data for variants that implements data preprocessing methods."""

    def genotypes_by_region(self, coordinates):
        """Given a region of interest returns the variants that lie in the respective region.

        The order of the individuals (rows, axis 0) matches that returned by :func:`get_iids`.
        The order of the variants (columns, axis 1) matches that returned by :func:`get_vids`.
        To change the default order of individuals or genotypes use :func:`update_individuals` or
        :func:`update_variants`, respectively.

        :param dict coordinates: a dictionary of genomic coordinates {"chr": str, "start": int, "end": int}
        :return: tuple of genotypes for region  of interest (numpy.ndarray (:math:`n*m` with :math:`n:=` number of individuals and :math:`m:=` number of SNVs)); and pandas.DataFrame with names/vids of respective genotypes as index
        :rtype: (numpy.ndarray, dict, pandas.Index)
        :raises: NotImplementedError: Interface.
        """
        raise NotImplementedError

    def genotypes_by_id(self, vids):
        """Retrieves genotypes for a set of variant ids (:attr:`vids`).

        The ordering of the individuals (rows, axis 0) matches that returned by :func:`get_iids`.
        The ordering of the variants (columns, axis 1) matches that returned by :func:`get_vids`.
        To change the default ordering of variants use :func:`update_individuals`.

        :param pandas.Index vids: one or more variant ids to retrieve genotypes for
        :return: tuple of genotypes for region of interest (numpy.ndarray (:math:`n*m` with :math:`n:=` number of individuals and :math:`m:=` number of SNVs)); and pandas.DataFrame with names/vids of respective genotypes as index
        :rtype: (numpy.ndarray, pandas.Index)
        :raises: NotImplementedError: Interface.
        """
        raise NotImplementedError

    def update_individuals(self, iids, exclude=False):
        """Set individuals to include/exclude.

        When :attr:`exclude` == False (default), this also changes the order to match the order in :attr:`iids`.

        :param pandas.Index iids: Individual ids to include/exclude
        :raises NotImplementedError: Interface.
        """
        raise NotImplementedError

    def update_variants(self, vids=None, coordinates=None, exclude=False):
        """Set variants to include/exclude based on variant ids (:attr:`vids`) or genetic coordinates (:attr:`coordinates`).

        When :attr:`exclude` == False (default), this also changes the order to match the order in :attr:`vids`.

        :param pandas.Index vids: Variant ids to include/exclude
        :param dict coordinates: genomic coordinates {"chr": str, "start": int, "end": int} to include/exclude
        :raises NotImplementedError: Interface.
        """
        raise NotImplementedError

    def get_iids(self):
        """Returns all individual ids.

        :return: Returns all individual ids.
        :rtype: np.ndarray
        :raises NotImplementedError: Interface.
        """
        raise NotImplementedError

    def get_vids(self):
        """Returns all variant ids.

        :return: Returns all variant ids.
        :rtype: np.ndarray
        :raises NotImplementedError: Interface.
        """
        raise NotImplementedError

    @staticmethod
    def mean_imputation(X):
        """Replaces missing genotype values (np.nan) by the respective mean value for that variant.

        Excludes missing values from computation.

        :param numpy.ndarray X: 2D array with dimensions :math:`n*m` with :math:`n:=` number of individuals and :math:`m:=` number of SNVs.
        :return: array with missing values mean imputed along specified axis
        :rtype: numpy.ndarray
        """
        mx = np.ma.masked_invalid(X)
        return mx.filled(np.nanmean(mx, axis=0))

    @staticmethod
    def standardize(X, inplace=False):
        """Z-score normalizes the genotype data, excluding missing values from computation. If inplace == True, performs
        operations inplace and returns a reference to X itself.

src/seak/data_loaders.p        :param numpy.ndarray X: 2D array with dimensions :math:`n*m` with :math:`n:=` number of individuals and :math:`m:=` number of SNVs.
        ::
        :return: z-score normalized array
        :rtype: numpy.ndarray
        """
        mean = np.nanmean(X, axis=0)
        std = np.nanstd(X, axis=0)

        if inplace:
            X -= mean
            X /= std
            return X
        else:
            return (X - mean) / std

    @staticmethod
    def center(X, inplace=False):
        """Mean centers genotype values, excluding missing values from computation.

        :param numpy.ndarray X: 2D array with dimensions :math:`n*m` with :math:`n:=` number of individuals and :math:`m:=` number of SNVs.
        :return: mean-centered array
        :rtype: numpy.ndarray
        """
        mean = np.nanmean(X, axis=0)

        if inplace:
            X -= mean
            return X
        else:
            return X - mean

    @staticmethod
    def scale(X, inplace=False):
        """Scales the genotype data by standard deviation, excluding missing values from computation.

        :param numpy.ndarray X: 2D array wit dimensions :math:`n*m` with :math:`n:=` number of individuals and :math:`m:=` number of SNVs.
        :return: normalized array
        :rtype: numpy.ndarray
        """
        std = np.nanstd(X, axis=0)

        if inplace:
            X /= std
            return X
        else:
            return X / std

    @staticmethod
    def preprocess_genotypes(genotypes, vids, impute_mean=True,
                             center=False, scale=False, normalize=False,
                             invert_encoding=True, recode_maf=False,
                             remove_singletons=False, remove_doubletons=False,
                             min_maf=0, max_maf=1):
        """Filters input variants and respective :attr:`vids` according to selected parameters.

        MAF: minor allele frequency.

        :param numpy.ndarray genotypes: :math:`n*m` genotype matrix (with :math:`n:=` number of individuals and :math:`m:=` number of SNVs)
        :param pandas.Series or pandas.DataFrame vids: information on the variants to be passed along (typically the variant ids)
        :param bool impute_mean: whether missing values should be replaced by the mean
        :param bool center: whether to mean center the data
        :param bool scale: whether to scale the data by the standard deviation
        :param bool normalize: whether to mean center and standardize the data by standard deviation
        :param bool invert_encoding: whether to invert the encoding of the genotypes (pandas_plink which is used for data loading counts major alleles by default, thus by default we invert the encodings, such that the minor allele is counted)
        :param bool recode_maf: encode genotypes according to sample, minor allele is counted (additive 0, 1, 2)
        :param bool remove_singletons: whether to remove singleton genotypes (minor allele count 1); by default also removes singletons in case of reverted genotypes for the sample
        :param bool remove_doubletons: whether to remove doubleton genotypes (minor allele count 2); by default also removes doubletons in case of reverted genotypes for the sample
        :param float min_maf: variants with MAF smaller than min_maf in the sample are excluded
        :param float max_maf:  variants with MAF larger than max_maf in the sample are excluded
        :return: filtered genotypes, index of variants that remain after filtering
        :rtype: numpy.ndarray, pandas.Index
        """

        ## TODO: make more memory efficient by performing some operations inplace
        ## potentially separate filtering from processing (?)

        assert not (
                invert_encoding & recode_maf), 'Genotypes can either be completely inverted or recoded according to ' \
                                               'minor allele frequency but not both.'

        if recode_maf:
            genotypes = np.where((np.nanmean(genotypes, axis=0) / 2 > 0.5), abs(genotypes - 2),
                                 genotypes)
        if invert_encoding:
            # As pandas-plink reads counts of the major allele --> this is inverted here!
            genotypes = np.abs(genotypes - 2)

        # MAF filteringsrc/seak/data_loaders.p, filtering of singletons and doubletons
        # Assumes additive model as 0, 1, 2 (count of minor allele)! (i.e. if pandas_plink was used to read in data,
        # invert_encoding must be used)

        # Remove singletons and/or doubletons regardless of encoding (i.e. if MAC of current encoding or MAC of inverted
        # encoding is one for singletons or two for doubletons.
        filter_mac = None
        if (remove_singletons and remove_doubletons) or remove_doubletons:
            filter_mac = (np.nansum(genotypes, axis=0) > 2) & \
                         (np.nansum(genotypes, axis=0) < genotypes.shape[0] * 2 - 2)
        elif remove_singletons:
            filter_mac = (np.nansum(genotypes, axis=0) > 1) & \
                         (np.nansum(genotypes, axis=0) < genotypes.shape[0] * 2 - 1)

        # MAF filtering
        filter_maf = None
        maf = np.nansum(genotypes, axis=0) / (genotypes.shape[0] * 2)
        if min_maf != 0 and max_maf != 1:
            filter_maf = (maf > min_maf) | (maf < max_maf)
        elif min_maf != 0:
            filter_maf = maf > min_maf
        elif max_maf != 1:
            filter_maf = maf < max_maf

        # Mean imputation
        if impute_mean:
            # Impute missing genotypes
            genotypes = VariantLoader.mean_imputation(genotypes)

        # Drop invariant SNPs
        ## TODO: this is broken if we dont  impute nans.
        filter_invariant = genotypes == genotypes[0, :]
        filter_invariant = ~filter_invariant.all(0)
        filter_all_nan = ~np.all(np.isnan(genotypes), axis=0)

        # Combine filter based on chosen options
        if filter_mac is not None and filter_maf is not None:
            total_filter = filter_invariant & filter_all_nan & filter_mac & filter_maf
        elif filter_mac is not None:
            total_filter = filter_invariant & filter_all_nan & filter_mac
        elif filter_maf is not None:
            total_filter = filter_invariant & filter_all_nan & filter_maf
        else:
            total_filter = filter_invariant & filter_all_nan

        genotypes = genotypes[:, total_filter]

        if genotypes.shape[1] == 0:
            return None, None

        if normalize or (center & scale):
            genotypes = VariantLoader.standardize(genotypes)
        elif center:
            genotypes = VariantLoader.center(genotypes)
        elif scale:
            genotypes = VariantLoader.scale(genotypes)

        vids = vids[total_filter]

        return genotypes, vids


class AnnotationLoader:
    """Interface for loading variant annotations that implements data preprocessing methods."""

    # TODO: it would be nice to define a set of pre-processing steps done before the annotations are loaded:

    def anno_by_id(self, vids):
        """Retrieves annotations for a set of variant ids (:attr:`vids`).

        The order will match that of :attr:`vids`.

        :param pandas.Index vids: names of variants to retrieve annotations for
        :return: variant effect predictions for given variants
        :rtype: numpy.ndarray
        :raises NotImplementedError: Interface.
        """
        raise NotImplementedError

    def update_variants(self, vids=None, coordinates=None, exclude=False):
        """Set variants to include/exclude based on variant ids (:attr:`vids`) or genetic coordinates (:attr:`coordinates`).

        When :attr:`exclude` == False (default), this also changes the order to match the order of :attr:`vids`.

        :param pandas.Index vids: Variant ids to include/exclude
        :param dict coordinates: a dictionary of genomic coordinates {"chr": str, "start": int, "end": int, "name": str}
        :raises NotImplementedError: Interface.
        """
        raise NotImplementedError

    def get_vids(self):
        """Returns all variant ids.

        :return: Returns all variant ids.
        :rtype: pandas.Index
        :raises NotImplementedError: Interface.
        """


class CovariatesLoader:
    """Stores covariates (fixed effects) and phenotypes."""

    def update_individuals(self, iids, exclude=False):
        """Set individuals to include/exclude.

        When :attr:`exclude` == False (default), this also changes the order to match the order in :attr:`iids`.

        :param pandas.Index iids: Individual ids to include/exclude
        """

    def get_iids(self):
        """Returns all individual ids.

        :return: Returns all individual ids.
        :rtype: pandas.Index
        """
        raise NotImplementedError


class VariantLoaderSnpReader(VariantLoader):
    """Loads variants (SNVs) by wrapping SnpReader objects.

    Can be initialized from either an SnpReader, or a path to a Plink bed-file.

    Has methods to subset the dataset based on variant ids and individual ids.
    Can retrieve genotypes of a specific region or by variant id.

    :param str path_or_bed: SnpReader or path to binary PLINK 1 bed-file
    """

    __slots__ = ['bed', 'iid_fid']

    def __init__(self, path_or_bed):
        """Constructor."""

        if isinstance(path_or_bed, str):
            self.bed = Bed(path_or_bed, count_A1=False)
        else:
            assert isinstance(path_or_bed,
                              SnpReader), 'path_or_bed must either be a path to a bed-file, or an instance of SnpReader.'
            self.bed = path_or_bed
        self.bed.pos[:, 0] = self.bed.pos[:, 0].astype('str')  # chromosome should be str, stored positions are 1-based. This doesnt' work! :(
        self.iid_fid = pd.DataFrame(self.bed.iid, index=self.bed.iid[:, 1], columns=['fid', 'iid'])

    def update_variants(self, vids=None, coordinates=None, exclude=False):
        """Set variants to include/exclude based on variant ids (:attr:`vids`) or genetic coordinates (:attr:`coordinates`).

        When :attr:`exclude` == False (default), this also changes the order to match the order in :attr:`vids`.
        Make sure that the vids are in the order of the veps in the respective index file if working with HDF5 files.
        --> Reason: HDF5 files can only be accessed indexing elements in increasing order!

        repeated IDs are ignored.

        :param vids: Variant ids to include/exclude
        :param dict coordinates: genomic coordsrc/seak/data_loaders.pinates {"chr": str, "start": int, "end": int} to include/exclude
        """
        if vids is None and coordinates is None:
            logging.error('Either variant ids or genomic coordinates must be specified to set variants. Both are None.')
        elif vids is not None and coordinates is not None:
            logging.error('Either variant ids or genomic coordinates must be specified to set variants, not both.')
        elif vids is not None:
            vids, idx_query, idx_bed = np.intersect1d(vids, self.bed.sid, return_indices=True)
            if exclude:
                mask = np.ones(self.bed.sid_count, dtype=bool)
                mask[idx_bed] = False
                self.bed = self.bed[:, mask]
            else:
                self.bed = self.bed[:, idx_bed[np.argsort(idx_query)]]
        else:
            try:
                mask = (self.bed.pos[:, 0] == float(coordinates['chrom'])) & (self.bed.pos[:, 2] > coordinates['start']) & (
                    self.bed.pos[:, 2] <= coordinates['end'])  # pos is one based, coordinates are 0 based half open
            except ValueError as e:
                raise ValueError('Chromosome identifiers need to be convertible to numbers!: \n{}'.format(e))
            if exclude:
                self.bed = self.bed[:, ~mask]
            else:
                self.bed = self.bed[:, mask]

    def update_individuals(self, iids, exclude=False):
        """Set individuals to include/exclude.

        When :attr:`exclude` == False (default), this also changes the order to match the order in :attr:`iids`:.

        :param iids: Individual ids to include/exclude
        """

        iid_fid = self.iid_fid.loc[iids]

        if exclude:
            mask = np.ones(self.bed.iid_count, dtype=bool)
            mask[self.bed.iid_to_index(iid_fid.values)] = False
            self.bed = self.bed[mask, :]
            self.iid_fid.drop(iids, 'index', inplace=True)
        else:
            self.bed = self.bed[self.bed.iid_to_index(iid_fid.values),:]
            self.iid_fid = iid_fid

    def genotypes_by_region(self, coordinates, return_pos=False):
        """Given a region of interest returns the variants that lie in the respective region.

        The order of the individuals (rows, axis 0) matches that returned by :func:`get_iids`.
        The order of the variants (columns, axis 1) matches that returned by :func:`get_vids`.
        To change the default order of individuals or genotypes use :func:`update_individuals` or :func:`update_variants`,
        respectively
        If no variants lie in the requested region the method returns (None, None).

        :param dict coordinates: a dictionary of genomic coordinates {"chr": str, "start": int, "end": int}
        :param bool return_pos: a boolean indicating whether variant positions should be returned
        :return: tuple of genotypes for region of interest (numpy.ndarray (number of individuals x number of SNVs)); and pandas.Index with names/vids of respective variants; If `return_pos` is True, a pandas.Series with variant positions.
        :rtype: (numpy.ndarray, pandas.Index, pandas.Series)
        """
        # only keep genotypes/veps_index_bed for selected region (genotype info)
        try:
            mask = (self.bed.pos[:, 0] == float(coordinates['chrom'])) & (self.bed.pos[:, 2] > coordinates['start']) & (
                    self.bed.pos[:, 2] <= coordinates['end']) # pos is one based, coordinates are 0 based half open
        except ValueError as e:
            raise ValueError('Chromosome identifiers need to be convertible to numbers!: \n{}'.format(e))


        if mask.sum() == 0:
            logging.warning('No variants were found for the region {}.'.format(coordinates))
            if return_pos:
                return None, None, None
            else:
                return None, None

        temp_geno = self.bed[:, mask].read()  # where data is actually read

        if return_pos:
            return temp_geno.val, temp_geno.sid, temp_geno.pos[:, 2] - 1 # return 0 based positions
        else:
            return temp_geno.val, temp_geno.sid

    def genotypes_by_id(self, vids, return_pos=False, missing_ok=False):
        """Retrieves genotypes for a set of variant ids (:attr:`vids`).

        The ordering of the individuals (rows, axis 0) matches that of :attr:`vids`.
        The ordering of the variants (columns, axis 1) matches that returned by :func:`get_vids`.
        To change the default ordering of variants use :func:`update_variants`.

        :param pandas.Index vids: one or more variant ids to retrieve genotypes for
        :return: tuple of genotypes for region of interest (numpy.ndarray (number of individuals x number of SNVs)); and pandas.DataFrame with names/vids of respective genotypes as index
        :rtype: (numpy.ndarray, pandas.Index)
        """

        if isinstance(vids, str):
            vids = np.array([vids])

        if missing_ok:
            vids, idx_query, _ = np.intersect1d(vids, self.bed.sid, return_indices=True)
            vids = vids[np.argsort(idx_query)]

        if len(vids) == 0:
            if return_pos:
                return None, None, None
            else:
                return None, None

        temp_geno = self.bed[:, self.bed.sid_to_index(vids)].read()

        if return_pos:
            return temp_geno.val, temp_geno.sid, temp_geno.pos[:, 2] - 1 # return 0-based positions
        else:
            return temp_geno.val, temp_geno.sid


    def get_iids(self):
        """Returns all individual ids.

        :return:
        :rtype: pandas.Index
        """
        return self.bed.iid[:, 1]

    def get_vids(self):
        """Returns all variant ids.

        :return:
        :rtype: pandas.Index
        """
        return self.bed.sid


class Hdf5Loader(AnnotationLoader):
    """Loads variant annotations from HDF5 files accompanied by a UCSC BED file or TSV with variant information.


    The HDF5 files need to follow the syntax of the output of :func:`Janggu.predict_variant_effect` (`Janggu documentation <https://janggu.readthedocs.io/en/latest/tutorial.html#id1>`_).
    You need to specify a key to the data.
    Instances of the class still needs to be overlapped with respective genotypes and vice versa.
    `UCSC BED <https://genome.ucsc.edu/FAQ/FAQformat#format1>`__ files/or TSV require first three fields: chrom, chromStart, chromEnd; we also require field name, that needs
    to match the names of the genotype data.
    Above that: if TSV and not USCS BED file is given make sure the genomic coordinates are 0-based an half-open.

    :param str path_to_vep_bed: path to UCSC BED file containing meta info of variant effect predictions
    :param str path_to_vep_hdf5: path to HDF5 file containing the variant effect predictions
    :param str hdf5_key: name of variant effect predict to load
    :param bool from_janggu: whether janggu was used to generate the veps, if yes, naming convention is presupposed
    """
    __slots__ = ['veps_index_df', 'veps', 'from_janggu', '_mask']

    def __init__(self, path_to_vep_bed, path_to_vep_hdf5, hdf5_key, from_janggu=False):
        """Constructor."""
        self.veps_index_df = self._read_veps_from_bed(path_to_vep_bed, from_janggu=from_janggu)
        self.veps = h5py.File(path_to_vep_hdf5, 'r')[hdf5_key]
        self._mask = np.ones(self.veps.shape[1]).astype(bool)

    def set_mask(self, mask):
        """
        Sets a boolean mask along the annotation-axis to be applied when loading data. Can be used to sub-set annotations.

        :param np.ndarray mask: boolean mask to be applied to the columns (second dimension) of the annotations loaded.
        """
        assert len(mask) == len(self._mask), 'expected mask to have length {}, got {}'.format(self.veps.shape[1], len(mask))
        self._mask = mask

    def _read_veps_from_bed(self, path_to_vep_bed, from_janggu):
        """Loads veps from UCSC BED file."""
        col_names_USCS_bed_file = ['chrom', 'start', 'end', 'name']
        veps_index_df = pd.read_csv(path_to_vep_bed, sep='\t', header=None, usecols=[0, 1, 2, 3],
                                    names=col_names_USCS_bed_file,
                                    low_memory=False, dtype={'chrom': str, 'name': str})
        veps_index_df['veps_nindex'] = veps_index_df.index
        if from_janggu:
            veps_index_df['inferred_name'] = veps_index_df.name.str.split(pat='_[ACGT]+>[ACGT]+$', n=1, expand=True).values[:,0]
            veps_index_df.set_index(keys='inferred_name', inplace=True)
        else:
            veps_index_df.set_index(keys='name', inplace=True)

        return veps_index_df

    def update_variants(self, vids=None, coordinates=None, exclude=False):
        """Set variants to include/exclude based on variant ids (:attr:`vids`) or genetic coordinates (:attr:`coordinates`).

        When :attr:`exclude` == False (default), this also changes the order to match the order in :attr:`vids`.
        Make sure that the vids are in the order of the veps in the respective index file if working with HDF5 files.
        --> Reason: HDF5 files can only be accessed indexing elements in increasing order!

        :param pandas.Index vids: Variant ids to include/exclude
        :param dict coordinates: genomic coordinates {"chr": str, "start": int, "end": int} to include/exclude
        """
        if vids is None and coordinates is None:
            logging.error('Either variant ids or genomic coordinates must be specified to set variants. Both are None.')
        elif vids is not None and coordinates is not None:
            logging.error('Either variant ids or genomic coordinates must be specified to set variants, not both.')
        elif vids is not None:
            if exclude:
                try:
                    self.veps_index_df = self.veps_index_df.drop(vids)
                except KeyError:
                    logging.warning(
                        'At least one variant (vid) that you tried to exclude does not exist in data set.')
                    self.veps_index_df = self.veps_index_df.drop(vids, errors='ignore')
            else:
                self.veps_index_df = self.veps_index_df.loc[vids]
        else:
            if exclude:
                try:
                    # TODO: at the moment this only get varaints that are fully contained in the region, change?
                    temp_veps_index_df = self.veps_index_df[(self.veps_index_df['chrom'] == coordinates['chrom']) &
                                                            (self.veps_index_df['start'] >= coordinates['start']) &
                                                            (self.veps_index_df['end'] <= coordinates['end'])]
                    print('here', temp_veps_index_df.index)
                    self.veps_index_df = self.veps_index_df.drop(temp_veps_index_df.index)
                except KeyError:
                    logging.warning('No veps lie within region that you want to exclude.')
            else:
                # TODO: at the moment this only get varaints that are fully contained in the region, change?
                self.veps_index_df = self.veps_index_df[
                    (self.veps_index_df['chrom'] == coordinates['chrom']) &
                    (self.veps_index_df['start'] >= coordinates['start']) &
                    (self.veps_index_df['end'] <= coordinates['end'])]
                if self.veps_index_df.empty:
                    logging.error('No veps lie within region that you want to include.')

    def anno_by_id(self, vids, shuffle=None):
        """Retrieves annotations for a set of variant ids (:attr:`vids`).

        The order will match that of vids.

        :param pandas.Index vids: names of variants to retrieve annotations for
        :param int shuffle: For testing purpose only; default None; shuffles loaded veps along axis 0 or 1
        :return: variant effect predictions for given variants
        :rtype: numpy.ndarray
        """
        try:
            veps_indices = self.veps_index_df.loc[vids]['veps_nindex'].values
            veps = self.veps[veps_indices, :][:, self._mask]
        except AttributeError:  # handle case where only a single prediction is requested
            veps_indices = self.veps_index_df.loc[vids]['veps_nindex']
            veps = self.veps[[veps_indices], :][:, self._mask]

        if shuffle is not None:
            # Shuffles effects for variants among individuals
            if shuffle == 0:
                # inplace modification
                np.random.shuffle(veps)
            # Shuffles effects for certain individual along different variant effect predictions
            if shuffle == 1:
                # inplace modification
                np.apply_along_axis(np.random.shuffle, 1, veps)
            else:
                print('Please choose parameter None, 0 or 1 for shuffling of variant effect predictions.')

        return veps

    def get_vids(self):
        """Returns all variant ids.

        :return:
        :rtype: pandas.Index
        """
        return self.veps_index_df.index


class EnsemblVEPLoader(AnnotationLoader):
    # TODO: would be cool if this worked directly with the output of ./filter_vep

    """Creates an AnnotationLoader from outputs columns of the Ensembl Variant Effect Predictor (VEP).

    Specifically the columns 'Uploaded_variation', 'Location', 'Allele' and 'Gene'. Additionally will take a numpy array or pandas dataframe with effects for those variants.

    The format of the columns are described on:
    https://www.ensembl.org/info/docs/tools/vep/vep_formats.html#defaultout

        Uploaded variation - as chromosome_start_alleles
        Location - in standard coordinate format (chr:start or chr:start-end)
        Allele - the variant allele used to calculate the consequence
        Gene - Ensembl stable ID of affected gene

    Positions in "Location" are one-based. "Uploaded_variation" is used as the variant ID.

    :param uploaded_variation: Series containing "Uploaded variation"
    :param location: Series containing "Location"
    :param allele: Series containing "Allele"
    :param gene: Series containing "Gene"
    :param data: DataFrame or ndarray containing the variant effect predictions

    """

    __slots__ = ['vep_df', 'pos_df']

    def __init__(self, uploaded_variation, location, gene, allele=None, data=None):

        # TODO: currenlty doesn't use allele information...

        if data is None:
            data = np.ones(len(uploaded_variation))

        self.veps_df = pd.DataFrame(data=data,
                                    index=pd.MultiIndex.from_arrays([gene, uploaded_variation], names=['gene', 'vid']))
        # returns chrom (str) and one-based position (int)
        chrom, start, end = self._parse_location(location)
        try:
            self.pos_df = pd.DataFrame({'chrom': chrom, 'start': start - 1, 'end': end, 'gene': gene.values},
                                       index=self.veps_df.index.get_level_values('vid'))
        except AttributeError:
            self.pos_df = pd.DataFrame({'chrom': chrom,  'start': start - 1, 'end': end, 'gene': gene},
                                       index=self.veps_df.index.get_level_values('vid'))

    def _parse_location(self, loc):

        loc = pd.Series(loc)
        loc = loc.str.split(':', expand=True)

        chrom = np.asarray(loc.iloc[:,0].values)

        loc = loc.iloc[:,1].str.split('-', expand=True)

        start = np.asarray(loc.iloc[:,0].astype(np.int32).values)
        try:
            end = np.asarray(loc.iloc[:,1].astype(np.float32).values) # contains nans
        except IndexError:
            end = start

        # return start and end positions (1-based, fully closed)
        return chrom, start, np.where(np.isnan(end), start, end).astype(np.int32)

    def _overlaps(self, coordinates):
        # TODO: at the moment this only get varaints that are fully contained in the region, change?
        mask = (self.pos_df['chrom'] == coordinates['chrom']) & (self.pos_df['start'] >= coordinates['start']) & (
                self.pos_df['end'] <= coordinates['end'])
        return mask

    def anno_by_id(self, vids=None, gene=None):
        '''
        Retrieves gene-specific variant effect predictions based on variant IDs (vids) and gene

        :param vids: variant ids
        :param gene: gene ids

        :param default: if not None, default value to return when no variant effect predictions are found for a requested variant.
        '''

        if gene is None:
            if vids is None:
                logging.error('Either vids, gene, or both must be specified to access variants by id.')
            else:
                return self.veps_df.loc[(slice(None), vids), :]
        else:
            if vids is None:
                return self.veps_df.loc[(gene, slice(None)), :]
            else:
                if isinstance(vids, str):
                    return self.veps_df.loc[(gene, [vids]),]  # this makes sure it has the correct dimensions
                elif isinstance(gene, str) or len(gene) == 1:
                    return self.veps_df.loc[(gene, vids),]
                else:
                    assert len(vids) == len(
                        gene), "If both vids and gene are longer than 1, they need to be the same length to avoid ambiguity!"
                return self.veps_df.loc[list(zip(gene, vids))]

    def anno_by_interval(self, coordinates, gene=None):
        '''
        Retrieves gene-specific variant effect predictions based on position and gene

        :param dict coordinates: genomic coordinates {"chr": str, "start": int, "end": int}
        :param gene: gene id

        '''

        if gene is not None:
            assert isinstance(gene, str), 'gene must be a single identifier (str) or None.'

        mask = self._overlaps(coordinates)
        vids = self.pos_df[mask].index.values

        return self.anno_by_id(vids, gene)

    def update_variants(self, vids=None, coordinates=None, exclude=False):
        '''Set variants to include/exclude based on variant ids (:attr:`vids`) or genetic coordinates (:attr:`coordinates`).

        :param pandas.Index vids: Variant ids to include/exclude
        :param dict coordinates: genomic coordinates {"chr": str, "start": int, "end": int} to include/exclude
        '''
        if vids is None and coordinates is None:
            logging.error('Either variant ids or genomic coordinates must be specified to set variants. Both are None.')
        elif vids is not None and coordinates is not None:
            logging.error('Either variant ids or genomic coordinates must be specified to set variants, not both.')
        elif vids is not None:
            if exclude:
                self.veps_df = self.veps_df.loc[self.veps_df.index.drop(vids, level=1)]
                self.pos_df = self.pos_df.loc[self.veps_df.index.get_level_values('vid'),]
            else:
                self.pos_df = self.pos_df.loc[vids,]
                self.veps_df = self.veps_df.loc[zip(self.pos_df['gene'], self.pos_df.index.values),]
        else:
            mask = self._overlaps(coordinates)
            if exclude:
                if np.all(~mask):
                    logging.warning('No veps lie within region that you want to exclude.')
                else:
                    vids = self.pos_df[mask].index.values
                    self.pos_df = self.pos_df.loc[~mask]
                    self.veps_df = self.veps_df.loc[self.veps_df.index.drop(vids, level=1)]
            else:
                if np.all(~mask):
                    logging.error('No veps lie within region that you want to include.')
                else:
                    self.pos_df = self.pos_df.loc[mask]
                    self.veps_df = self.veps_df.loc[zip(self.pos_df['gene'], self.pos_df.index.values),]

    def get_vids(self):
        return self.pos_df.index.values

class CovariatesLoaderCSV(CovariatesLoader):
    """Stores covariates (fixed effects) and phenotypes.

    Initializes instance of a CovariatesLoader class from a CSV file.

    :param str phenotype_of_interest: column name of phenotype that should be loaded
    :param str path_to_covariates: path to CSV file with phenotype (if path_to_phenotypes is None) and covariates
    :param list(str) covariate_column_names: list of the covariate column names
    :param str path_to_phenotypes: path to CSV file with phenotype (if stored separately from covariates)
    """

    __slots__ = ['cov', 'phenotype_of_interest', 'covariate_column_names']
    # cov: complete data for all individuals of interest with no missing values, no duplicates, no constant columns
    # column named 'iid' with ids to merge with genotypes dataset; 'cov_index'

    def __init__(self, phenotype_of_interest, path_to_covariates, covariate_column_names, sep=',', path_to_phenotypes=None):
        """Constructor."""
        if isinstance(phenotype_of_interest, str):
            phenotype_of_interest = [phenotype_of_interest]
        self.phenotype_of_interest = phenotype_of_interest
        self.cov = pd.read_csv(path_to_covariates, sep=sep)

        self.cov = self.cov.rename(columns={self.cov.columns[0]: 'iid'})
        self.cov['iid'] = self.cov['iid'].astype(str)

        self.cov.set_index(keys='iid', inplace=True, drop=False)

        if path_to_phenotypes is not None:
            pheno = pd.read_csv(path_to_phenotypes, sep=sep)
            pheno.rename(columns={pheno.columns[0]: 'iid'}, inplace=True)
            pheno['iid'] = pheno['iid'].astype(str)
            pheno.set_index('iid', inplace=True)
            pheno = pheno[phenotype_of_interest]
            self.cov = self.cov.join(pheno)

        # Only consider individuals for which information on all covariates and phenotype is available!

        self.covariate_column_names = covariate_column_names

        # Drop all individuals with incomplete covariate or phenotype information
        cov_pheno_and_ids = self.covariate_column_names[:] + ['iid'] + self.phenotype_of_interest
        nb_of_ind = self.cov.shape[0]
        self.cov = self.cov[cov_pheno_and_ids].dropna()
        nb_of_ind_complete = self.cov.shape[0]
        logging.warning('{} individuals/samples were dropped because of incomplete covariate or phenotype information.'.format(nb_of_ind-nb_of_ind_complete))


    def update_individuals(self, iids, exclude=False):
        """Set individuals to include/exclude.

        When :attr:`exclude` == False (default), this also changes the order to match the order in :attr:`iids`.

        :param pandas.Index iids: Individual ids to include/exclude
        """
        if exclude:
            try:
                self.cov = self.cov.drop(iids)
            except KeyError:
                logging.warning('At least one individual (iid) that you tried to exclude does not exist in data set.')
                self.cov = self.cov.drop(iids, errors='ignore')
        else:
            self.cov = self.cov.loc[iids]

    def get_one_hot_covariates_and_phenotype(self, test_type, pheno=None):
        """Returns numpy.ndarray of the phenotype and one hot encoded covariates with invariant covariates
        removed and a bias column.

        Make sure :func:`update_cov` was called beforehand, such that :attr:`self.cov` has the same order as the genotypes.
        Internally calls function :func:`pandas.get_dummies` which converts categorical variable into indicator variables.

        :param string test_type: Either 'logit', '2K' or 'noK'
        :return: data for phenotype and one hot encoded covariates with invariant covariates removed and bias column
        :rtype: numpy.ndarray
        """

        if pheno is None and len(self.phenotype_of_interest) > 1:
            raise ValueError('pheno can not be None if there is more than one phenotype.')

        pheno = self.phenotype_of_interest[0] if pheno is None else pheno

        #print('Get one hot covariates and phenotype')
        one_hot_covariates = pd.get_dummies(self.cov[self.covariate_column_names], prefix_sep='_', drop_first=True)

        # Drop invariant covariates
        one_hot_covariates = one_hot_covariates.loc[:, one_hot_covariates.apply(pd.Series.nunique) != 1]
        one_hot_covariates = np.asarray(one_hot_covariates)

        # Add offset/bias column
        X = np.hstack((np.ones((one_hot_covariates.shape[0], 1)), one_hot_covariates))

        if test_type == 'logit':
            phenotype = np.asarray(pd.get_dummies(self.cov[pheno], dtype=float, drop_first=True))
            if phenotype.shape[1] != 1:
                logging.error('It seems like your phenotype was not encoded binary.')
            phenotype = phenotype.reshape(len(phenotype), 1)
        else:
            phenotype = np.asarray(self.cov[pheno])
            phenotype = phenotype.reshape(len(phenotype), 1)
        return phenotype, X

    def get_iids(self):
        """Returns all individual ids.

        :return:
        :rtype: pandas.Index
        """
        return self.cov.index

class RegionLoader:
    """Interface for loading genomic regions whose subclasses can be used to group variants into sets to test jointly.

    Subclasses need only implement __iter__, which yields 0-based, right-open (->left-closed) dictionaries of genomic intervals {"chr": str, "start": int, "end": int, "name": str}

    """

    def __iter__(self):
        """
        Functions that iterates over the regions contained in the RegionLoader.
        Yields dictionaries of single regions, {"chr": str, "start": int, "end": int, "name": str}
        """
        raise NotImplementedError

    pass


class BEDRegionLoader(RegionLoader):

    """Reads all regions in a `UCSC BED <https://genome.ucsc.edu/FAQ/FAQformat.html#format1>`__-formatted file into memory.

    :ivar pandas.DataFrame regions: DataFrame storing regions in memory {"chr": str, "start": int, "end": int, "name": str}

    """
    __slots__ = ['regions']

    def __init__(self, path_to_regions_UCSC_BED, chrom_to_load=None, drop_non_numeric_chromosomes=False):
        self.regions = self._read_regions(path_to_regions_UCSC_BED, chrom_to_load, drop_non_numeric_chromosomes)

    def _read_regions(self, path_to_regions_UCSC_BED, chrom_to_load, drop_non_numeric_chromosomes):
        col_names_USCS_bed_file_reference_genome = ['chrom', 'start', 'end', 'name']
        regions = pd.read_csv(path_to_regions_UCSC_BED, sep='\t', header=None, usecols=[0, 1, 2, 3],
                              names=col_names_USCS_bed_file_reference_genome, low_memory=False,
                              dtype={'chrom': str, 'name': str, 'chromStart': int, 'chromEnd': int})
        if drop_non_numeric_chromosomes:
            # Remove all not numeric chromosome values!
            filter_non_autosomal_chromosomes = regions['chrom'].apply(lambda x: x.isdigit())
            regions = regions[filter_non_autosomal_chromosomes]
        if chrom_to_load is not None:
            regions = regions[regions['chrom'] == str(chrom_to_load)]
        return regions

    def __iter__(self):
        for idx, row in self.regions.iterrows():
            yield row.to_dict()


def intersect_and_update_datasets(test, variantloader, covariateloader, annotationloader=None, grmloader=None):
    """Creates smallest common subset of class instances provided with respect to variants and individuals.

    Modifies instances inplace.

    For 2K case also computed background kernel :math:`K_0`.

    :param string test: ['2K', 'noK', 'logit']
    :param VariantLoader variantloader: instance of respective class to be modified
    :param CovariatesLoader covariateloader: instance of respective class to be modified
    :param AnnotationLoader annotationloader: instance of respective class to be modified
    :param GRMLoader grmloader: instance of respective class to be modified
    """
    assert test in ['2K', 'noK', 'logit'], 'Invalid test type chosen.'
    if test == '2K':
        assert grmloader is not None, 'You need to provide a GRMloader instance (background kernel) for the 2K test.'
        if annotationloader is not None:
            # Overlap individuals: genotypes and covariates
            genotypes_covariates_GRM_intersection = reduce(intersect_ids, (variantloader.get_iids(),covariateloader.get_iids(), grmloader.get_iids()))

            # Overlap genotypes with VEPs
            veps_genotypes_intersection = intersect_ids(annotationloader.get_vids(), variantloader.get_vids())

            # Update respective instances
            variantloader.update_variants(veps_genotypes_intersection)
            annotationloader.update_variants(veps_genotypes_intersection)

            variantloader.update_individuals(genotypes_covariates_GRM_intersection)
            covariateloader.update_individuals(genotypes_covariates_GRM_intersection)
            grmloader.update_individuals(genotypes_covariates_GRM_intersection)

            grmloader.compute_background_kernel()
        else:
            logging.warning('You have not specified any variant effect predictions or annotations.')
            # Overlap individuals: genotypes and covariates
            genotypes_covariates_GRM_intersection = reduce(intersect_ids, (variantloader.get_iids(), covariateloader.get_iids(), grmloader.get_iids()))

            # Update respective instances
            variantloader.update_individuals(genotypes_covariates_GRM_intersection)
            covariateloader.update_individuals(genotypes_covariates_GRM_intersection)
            grmloader.update_individuals(genotypes_covariates_GRM_intersection)

            grmloader.compute_background_kernel()

    else:
        if annotationloader is not None:
            # Overlap individuals: genotypes and covariates
            genotypes_covariates_intersection = intersect_ids(variantloader.get_iids(), covariateloader.get_iids())

            # Overlap genotypes with VEPs
            veps_genotypes_intersection = intersect_ids(annotationloader.get_vids(), variantloader.get_vids())

            # Update respective instances
            variantloader.update_variants(veps_genotypes_intersection)
            annotationloader.update_variants(veps_genotypes_intersection)

            variantloader.update_individuals(genotypes_covariates_intersection)
            covariateloader.update_individuals(genotypes_covariates_intersection)
        else:
            logging.warning('You have not specified any variant effect predictions or annotations.')
            # Overlap individuals: genotypes and covariates
            genotypes_covariates_intersection = intersect_ids(variantloader.get_iids(), covariateloader.get_iids())

            # Update respective instances
            variantloader.update_individuals(genotypes_covariates_intersection)
            covariateloader.update_individuals(genotypes_covariates_intersection)

