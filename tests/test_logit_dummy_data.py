# Note: all dummy data bim files start at 0 for position, even though file format should be 1-based.
def test_full_rank_logit():
    server = False
    if server:
        import data_loaders
        import scoretest
        import kernels
    else:
        from seak import data_loaders
        from seak import scoretest
        from seak import kernels
    import time
    import pandas as pd
    import numpy as np
    import pkg_resources

    data_path = pkg_resources.resource_filename('seak', 'data/')

    # Path to veps
    path_to_VEP_bed = data_path + "dummy_veps.bed"
    path_to_VEP_hdf5 = data_path + "dummy_veps.hdf5"

    # Path to genotypes
    path_to_covariates = data_path + "dummy_covariates_fixed.csv"
    path_to_plink_files_with_prefix = data_path + "full_rank_logit"

    # Path to regions
    path_to_reference_genes_bed = data_path + "dummy_regions.bed"

    # Load data
    # VEPs
    hdf5_loader = data_loaders.Hdf5Loader(path_to_vep_bed=path_to_VEP_bed, path_to_vep_hdf5=path_to_VEP_hdf5,
                                          hdf5_key='diffscore')

    # Genotypes
    plink_loader = data_loaders.PlinkLoader(path_to_plink_files_with_prefix=path_to_plink_files_with_prefix)

    # Genes
    ucsc_region_loader = data_loaders.BEDRegionLoader(path_to_regions_UCSC_BED=path_to_reference_genes_bed,
                                                      chrom_to_load=1, drop_non_numeric_chromosomes=True)

    # Covariates
    covariate_loader_csv = data_loaders.CovariatesLoaderCSV(phenotype_of_interest='pheno_full_rank_logit',
                                                            path_to_covariates=path_to_covariates,
                                                            covariate_column_names=['cov1', 'cov2'])

    # Overlap individuals: genotypes and covariates
    print('Overlaps')
    print('Individuals')
    genotypes_covariates_intersection = plink_loader.get_iids().intersection(covariate_loader_csv.get_iids())

    print(genotypes_covariates_intersection.shape)
    print(genotypes_covariates_intersection)

    # Overlap genotypes with VEPs
    print('Genotypes')
    print(len(plink_loader.bim.index))
    print(len(hdf5_loader.veps_index_df.index))
    veps_genotypes_intersection = hdf5_loader.get_vids().intersection(plink_loader.get_vids())
    print(len(veps_genotypes_intersection))

    # Update respective instances
    print('Updates')
    print('plink_loader.bim.shape', plink_loader.bim.shape)
    plink_loader.update_variants(veps_genotypes_intersection)
    print('plink_loader.bim.shape', plink_loader.bim.shape)
    print('hdf5_loader.veps_index_df.shape', hdf5_loader.veps_index_df.shape)

    hdf5_loader.update_variants(veps_genotypes_intersection)
    print('hdf5_loader.veps_index_df.shape', hdf5_loader.veps_index_df.shape)

    print('plink_loader.fam.shape', plink_loader.fam.shape)
    plink_loader.update_individuals(genotypes_covariates_intersection)
    print('plink_loader.fam.shape', plink_loader.fam.shape)
    print('covariate_loader_csv.cov.shape', covariate_loader_csv.cov.shape)
    covariate_loader_csv.update_individuals(genotypes_covariates_intersection)
    print('covariate_loader_csv.cov.shape', covariate_loader_csv.cov.shape)

    Y, X = covariate_loader_csv.get_one_hot_covariates_and_phenotype(test_type='logit')
    null_model = scoretest.ScoretestLogit(Y, X)
    results = pd.DataFrame(columns=['name', 'chrom', 'start', 'end', 'p_value', 'n_SNVs', 'time'])

    for index, region in ucsc_region_loader.regions.iterrows():
        t_test_gene_start = time.time()
        temp_genotypes_info_dict = region.to_dict()
        temp_genotypes, temp_vids = plink_loader.genotypes_by_region(region)

        if temp_genotypes is None:
            continue

        G, temp_vids = data_loaders.VariantLoader.preprocess_genotypes(temp_genotypes, temp_vids, impute_mean=True,
                                                                       normalize=False, invert_encoding=True,
                                                                       recode_maf=False)
        if G is None:
            continue

        V = hdf5_loader.anno_by_id(temp_vids)

        GV = kernels.phi(kernels.diffscore_max, G, V)
        temp_p_value = null_model.pv_alt_model(GV)
        temp_genotypes_info_dict['p_value'] = temp_p_value
        temp_genotypes_info_dict['n_SNVs'] = G.shape[1]
        t_test_gene_end = time.time()
        temp_time = float(t_test_gene_end - t_test_gene_start)
        temp_genotypes_info_dict['time'] = temp_time
        results = results.append(temp_genotypes_info_dict, ignore_index=True)

    # results.to_csv('./test_full_rank_logit.csv')
    print(results)
    reference_result = pd.read_csv(data_path + 'reference_results/test_full_rank_logit.csv', index_col=0)
    print(np.corrcoef(reference_result['p_value'], results['p_value']))
    print(np.all((np.isclose(reference_result['p_value'], results['p_value']))))
    assert np.all((np.isclose(reference_result['p_value'], results['p_value']))), 'The last change in code changes the result!!'


def test_full_rank_logit_automatic_intersection():
    server = False
    if server:
        import data_loaders
        import scoretest
        import kernels
    else:
        from seak import data_loaders
        from seak import scoretest
        from seak import kernels
    import time
    import pandas as pd
    import numpy as np
    import pkg_resources

    data_path = pkg_resources.resource_filename('seak', 'data/')

    # Path to veps
    path_to_VEP_bed = data_path + "dummy_veps.bed"
    path_to_VEP_hdf5 = data_path + "dummy_veps.hdf5"

    # Path to genotypes
    path_to_covariates = data_path + "dummy_covariates_fixed.csv"
    path_to_plink_files_with_prefix = data_path + "full_rank_logit"

    # Path to regions
    path_to_reference_genes_bed = data_path + "dummy_regions.bed"

    # Load data
    # VEPs
    hdf5_loader = data_loaders.Hdf5Loader(path_to_vep_bed=path_to_VEP_bed, path_to_vep_hdf5=path_to_VEP_hdf5,
                                          hdf5_key='diffscore')

    # Genotypes
    plink_loader = data_loaders.PlinkLoader(path_to_plink_files_with_prefix=path_to_plink_files_with_prefix)

    # Genes
    ucsc_region_loader = data_loaders.BEDRegionLoader(path_to_regions_UCSC_BED=path_to_reference_genes_bed,
                                                      chrom_to_load=1, drop_non_numeric_chromosomes=True)

    # Covariates
    covariate_loader_csv = data_loaders.CovariatesLoaderCSV(phenotype_of_interest='pheno_full_rank_logit',
                                                            path_to_covariates=path_to_covariates,
                                                            covariate_column_names=['cov1', 'cov2'])

    # Intersect and update data sets
    data_loaders.intersect_and_update_datasets('logit', plink_loader, covariate_loader_csv, hdf5_loader)

    Y, X = covariate_loader_csv.get_one_hot_covariates_and_phenotype(test_type='logit')
    null_model = scoretest.ScoretestLogit(Y, X)
    results = pd.DataFrame(columns=['name', 'chrom', 'start', 'end', 'p_value', 'n_SNVs', 'time'])

    for index, region in ucsc_region_loader.regions.iterrows():
        t_test_gene_start = time.time()
        temp_genotypes_info_dict = region.to_dict()
        temp_genotypes, temp_vids = plink_loader.genotypes_by_region(region)

        if temp_genotypes is None:
            continue

        G, temp_vids = data_loaders.VariantLoader.preprocess_genotypes(temp_genotypes, temp_vids, impute_mean=True,
                                                                       normalize=False, invert_encoding=True,
                                                                       recode_maf=False)
        if G is None:
            continue

        V = hdf5_loader.anno_by_id(temp_vids)

        GV = kernels.phi(kernels.diffscore_max, G, V)
        temp_p_value = null_model.pv_alt_model(GV)
        temp_genotypes_info_dict['p_value'] = temp_p_value
        temp_genotypes_info_dict['n_SNVs'] = G.shape[1]
        t_test_gene_end = time.time()
        temp_time = float(t_test_gene_end - t_test_gene_start)
        temp_genotypes_info_dict['time'] = temp_time
        results = results.append(temp_genotypes_info_dict, ignore_index=True)

    # results.to_csv('./test_full_rank_logit.csv')
    print(results)
    reference_result = pd.read_csv(data_path + 'reference_results/test_full_rank_logit.csv', index_col=0)
    print(np.corrcoef(reference_result['p_value'], results['p_value']))
    print(np.all((np.isclose(reference_result['p_value'], results['p_value']))))
    assert np.all((np.isclose(reference_result['p_value'], results['p_value']))), 'The last change in code changes the result!!'
