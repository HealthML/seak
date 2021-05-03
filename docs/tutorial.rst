.. _Tutorial:

=========
Tutorial
=========

Score test and likelihood ratio test with a single variance component
---------------------------------------------------------------------


.. code-block:: python

  import pkg_resources

  import pandas as pd
  import h5py
  
  import numpy as np
  from seak import data_loaders
  from seak import kernels
  from seak import scoretest
  from seak import lrt
  
  
  # Paths to input data
  # Path to package dummy data
  data_path = pkg_resources.resource_filename('seak', 'data/')
  
  # Path to veps
  path_to_VEP_bed = data_path + "dummy_veps.bed"
  path_to_VEP_hdf5 = data_path + "dummy_veps.hdf5"
  
  # Path to genotypes
  path_to_covariates = data_path + "dummy_covariates_fixed.csv"
  bedfilepath = data_path + "full_rank_continuous.bed"
  
  # Path to regions
  path_to_reference_genes_bed = data_path + "dummy_regions.bed"
  
  # Path to background kernel
  path_to_bg_kernel_bed = data_path + "full_rank_background_kernel.bed"
  
  # Load data
  # Variant effect predictions
  # path_to_VEP_bed is a UCSC BED-file containing the variant positions and names  
  # path_to_VEP_hdf5 is a HDF5 file containing variant effect predictions.
  
  hdf5_loader = data_loaders.Hdf5Loader(path_to_vep_bed=path_to_VEP_bed,
                                        path_to_vep_hdf5=path_to_VEP_hdf5,
                                        hdf5_key='diffscore')
  
  # the diffscore dataset contains 100 variants with 160 predictions each, which can be directly accessed with the "veps" attribute
  hdf5_loader.veps.shape
  
  # the loader can be indexed with variant ids
  hdf5_loader.anno_by_id(hdf5_loader.get_vids()[0:3]).shape
  
  # we can set a mask to only load sepecific predictions
  mask = np.zeros(hdf5_loader.veps.shape[1])
  mask[9] = 1. # get only the 10th prediction
  mask = mask.astype(bool)
  hdf5_loader.set_mask(mask)
  
  # the loader now only returns the 10th prediction
  hdf5_loader.anno_by_id(hdf5_loader.get_vids()[0:3]).shape
  
  # Genotypes
  plink_loader = data_loaders.VariantLoaderSnpReader(bedfilepath)
  
  # this loader can also be accessed by variant ids
  G, vid = plink_loader.genotypes_by_id(plink_loader.get_vids()[0])
  # additionally, this loader can also be accessed by region, as is shown later...
  
  # individual ids can be accessed as follows
  plink_loader.get_iids()[0:3] # first 3 individuals
  
  # Covariate data
  # this loader handles standard operations like adding a bias column and converting categorical variables using one-hot encoding
  # path_to_covariates is a csv containing the phenotype (pheno_full_rank_continuous) and covariates (cov1, cov2)
  covariate_loader_csv = data_loaders.CovariatesLoaderCSV(phenotype_of_interest='pheno_full_rank_continuous',
                                                          path_to_covariates=path_to_covariates,
                                                          covariate_column_names=['cov1', 'cov2'])
  # phenotypes can also be contained in a seperate file defined with path_to_phenotypes (default: read phenotypes and covariates from the same file)
  
  
  # Intersects the different datasets
  # intersects variants (between the variantloader and annotationloader)
  # intersects individuals (between the variantloader and covariatesloader)
  # we could also do this manually. This function is added for convenience
  # by specifying "noK" we indicate that we will not use a "background kernel"
  data_loaders.intersect_and_update_datasets(test='noK',
                                             variantloader=plink_loader,
                                             covariateloader=covariate_loader_csv,
                                             annotationloader=hdf5_loader)
  
  # Get one-hot encoded phenotypes and covariates for association tests
  Y, X = covariate_loader_csv.get_one_hot_covariates_and_phenotype(test_type='noK')
  
  # initialize the null model for the score test
  model_score = scoretest.ScoretestNoK(Y, X)
  
  # initialize the null model for the likelihood ratio test
  model_lrt = lrt.LRTnoK(X, Y)
  
  regions = data_loaders.BEDRegionLoader(path_to_regions_UCSC_BED=path_to_reference_genes_bed, chrom_to_load=1, drop_non_numeric_chromosomes=True)
  
  def get_G1(r):
      temp_genotypes, temp_vids, temp_pos = plink_loader.genotypes_by_region(r, return_pos=True)
      if temp_genotypes is None:
          return temp_genotypes, temp_vids, temp_pos # all will be None
      G, vids = plink_loader.preprocess_genotypes(genotypes=temp_genotypes,
                                                       vids=temp_vids,
                                                       impute_mean=True,
                                                       center=True,
                                                       scale=False)
      if G is None:
          return G, vids, temp_pos # all will be None
      else:
          return G, temp_vids, temp_pos[[x in temp_vids for x in vids]]
      
  
  results = []
  simulations = []
      
  # Iterate over regions
  for i, region in enumerate(regions):
  
      result = {}
      
      # Get set of variants based on region annotations
      G1, vids, pos = get_G1(region)
  
      if G1 is None:
          continue
          
      # Get respective variant effect predictions based on vids
      V = hdf5_loader.anno_by_id(vids)
      
      # set weights
      weights = np.sqrt(np.abs(V))[:,0]
  
      # weighted linear kernel
      GV = G1.dot(np.diag(weights))
      
      result['region'] = region['name']
      result['n_variants'] = GV.shape[1]
      
      # score test
      # p-value for the score-test
      result['pv_score'] = model_score.pv_alt_model(GV)
      
      # LRT 
      # fit alternative model
      altmodel = model_lrt.altmodel(G1)
      
      # LRT test statistic
      result['lrt_stat'] = altmodel['stat']
      
      if altmodel['alteqnull']:
          # alternative model is less likely than the null model -> p-value = 1.
          result['pv_lrt_empirical'] = 1.
          result['lrt_alteqnull'] = 1
      else:
          # alternative is more likeliy than the null model, simulate test statistics
          sim = model_lrt.pv_sim(nsim=10000, seed=i) # simulate 10,000 test statistics for the current alternative
          simulations.append(sim['res'])
          result['pv_lrt_empirical'] = sim['pv']
          result['lrt_alteqnull'] = 0
          
      results.append(result)
  
  results = pd.DataFrame.from_dict(results)
  
  # fit a chi2 mixture distribution to the simulated test statistics:
  chi2param = lrt.fit_chi2mixture(np.concatenate(simulations), qmax=0.1)
  
  chi2param
  
  results['pv_lrt'] = 1.
  
  # p-values for the LRT calculated with the mixture distribution 
  results.loc[~results.lrt_alteqnull.astype(bool),'pv_lrt'] = lrt.pv_chi2mixture(results.loc[~results.lrt_alteqnull.astype(bool),'lrt_stat'].values, chi2param['scale'], chi2param['dof'], chi2param['mixture'])
  
  results
