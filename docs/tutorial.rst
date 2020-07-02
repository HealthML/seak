.. _Tutorial:

=========
Tutorial
=========

Performing a two variance component score tests with seak
---------------------------------------------------------
This example illustrates how to perform a two variance component score test with :mod:`seak`.


.. code-block:: python

  # Small example for usage of two variance component model with background kernel
  # Imports
  import pkg_resources

  import pandas as pd

  from seak import construct_background_kernel
  from seak import data_loaders
  from seak import kernels
  from seak import scoretest

  # Paths to input data
  # Path to package dummy data
  data_path = pkg_resources.resource_filename('seak', 'data/')

  # Path to veps
  path_to_VEP_bed = data_path + "dummy_veps.bed"
  path_to_VEP_hdf5 = data_path + "dummy_veps.hdf5"

  # Path to genotypes
  path_to_covariates = data_path + "dummy_covariates_fixed.csv"
  path_to_plink_files_with_prefix = data_path + "full_rank_continuous"

  # Path to regions
  path_to_reference_genes_bed = data_path + "dummy_regions.bed"

  # Path to background kernel
  path_to_plink_bg_kernel = data_path + "full_rank_background_kernel"

  # Load data
  # Variant effect predictions
  hdf5_loader = data_loaders.Hdf5Loader(path_to_vep_bed=path_to_VEP_bed,
                                        path_to_vep_hdf5=path_to_VEP_hdf5,
                                        hdf5_key='diffscore')

  # Genotypes
  plink_loader = data_loaders.PlinkLoader(path_to_plink_files_with_prefix=path_to_plink_files_with_prefix)

  # Genomic regions
  ucsc_region_loader = data_loaders.BEDRegionLoader(path_to_regions_UCSC_BED=path_to_reference_genes_bed,
                                                    chrom_to_load=1,
                                                    drop_non_numeric_chromosomes=True)

  # Covariate data
  covariate_loader_csv = data_loaders.CovariatesLoaderCSV(phenotype_of_interest='pheno_full_rank_continuous',
                                                          path_to_covariates=path_to_covariates,
                                                          covariate_column_names=['cov1', 'cov2'])

  # Background_kernel
  background_kernel = construct_background_kernel.GRMLoader(path_to_plink_files_with_prefix=path_to_plink_bg_kernel,
                                                            blocksize=200,
                                                            LOCO_chrom_id=1)

  # Intersects and updates data sets, computes background kernel for matched data
  data_loaders.intersect_and_update_datasets(
      test='2K',
      variantloader=plink_loader,
      covariateloader=covariate_loader_csv,
      annotationloader=hdf5_loader,
      grmloader=background_kernel)

  # Get one-hot encoded phenotypes and covariates for association tests
  Y, X = covariate_loader_csv.get_one_hot_covariates_and_phenotype(test_type='2K')
  null_model = scoretest.Scoretest2K(Y, X, background_kernel.K0, background_kernel.G0)

  # Save information of interest in pandas dataframe
  results = pd.DataFrame(columns=['name', 'chrom', 'start', 'end', 'p_value', 'n_SNVs'])

  # Iterate over regions and compute p-values for sets of variants
  for index, region in ucsc_region_loader.regions.iterrows():
      temp_genotypes_info_dict = region.to_dict()

      # Get set of variants based on region annotations
      temp_genotypes, temp_vids = plink_loader.genotypes_by_region(region)

      # Check whether any variants lie within respective region
      if temp_genotypes is None:
          continue

      # Preprocess genotypes
      G, temp_vids = data_loaders.VariantLoader.preprocess_genotypes(genotypes=temp_genotypes,
                                                                     vids=temp_vids,
                                                                     impute_mean=True,
                                                                     center=False,
                                                                     scale=False)

      # Check whether any variants remain after preprocessing
      if G is None:
        continue

    # Get respective variant effect predictions based on vids
    V = hdf5_loader.anno_by_id(temp_vids)

    # Apply kernel function of choice to genotypes G including prior knowledge
    # in form of variant effect predictions V
    GV = kernels.phi(kernels.diffscore_max, G, V)

    # Compute p-value
    temp_p_value = null_model.pv_alt_model(GV)


    # Save temporary results
    temp_genotypes_info_dict['p_value'] = temp_p_value
    temp_genotypes_info_dict['n_SNVs'] = G.shape[1]

    # Append temporary results to final dataframe
    results = results.append(temp_genotypes_info_dict, ignore_index=True)

  results

  #        name chrom start  end   p_value n_SNVs
  # 0   region1     1     0   10  0.401793     10
  # 1   region2     1    10   20  0.376850     10
  # 2   region3     1    20   30  0.545177     10
  # 3   region4     1    30   40  0.286926     10
  # 4   region5     1    40   50  0.276154     10
  # 5   region6     1    50   60  0.649827     10
  # 6   region7     1    60   70  0.872293     10
  # 7   region8     1    70   80  0.533158     10
  # 8   region9     1    80   90  0.324089     10
  # 9  region10     1    90  100  0.066610      9
