# Note: all dummy data bim files start at 0 for position, even though file format should be 1-based.

# TODO: these tests are overly complicated and hard to debug. should be replaced by actual UNIT-tests

def test_full_rank_continuous():
    # full rank bg kernel
    # imports
    import pkg_resources
    import time

    import numpy as np
    import pandas as pd

    from seak import construct_background_kernel
    from seak import data_loaders
    from seak import kernels
    from seak import scoretest

    from seak.data_loaders import intersect_ids

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
    # VEPs
    hdf5_loader = data_loaders.Hdf5Loader(path_to_vep_bed=path_to_VEP_bed, path_to_vep_hdf5=path_to_VEP_hdf5,
                                          hdf5_key='diffscore')

    # Genotypes
    plink_loader = data_loaders.VariantLoaderSnpReader(path_to_plink_files_with_prefix+'.bed')

    # Genes
    ucsc_region_loader = data_loaders.BEDRegionLoader(path_to_regions_UCSC_BED=path_to_reference_genes_bed,
                                                      chrom_to_load=1, drop_non_numeric_chromosomes=True)

    # Covariates
    covariate_loader_csv = data_loaders.CovariatesLoaderCSV(phenotype_of_interest='pheno_full_rank_continuous',
                                                            path_to_covariates=path_to_covariates,
                                                            covariate_column_names=['cov1', 'cov2'])

    # Background_kernel
    # background_kernel = construct_background_kernel.GRMLoader(path_to_plink_bg_kernel, 10, 1)
    background_kernel = construct_background_kernel.GRMLoaderSnpReader(path_to_plink_bg_kernel + '.bed', 10, '1')

    # Overlap individuals: genotypes and covariates
    genotypes_covariates_GRM_intersection = data_loaders.intersect_ids(plink_loader.get_iids(), covariate_loader_csv.get_iids())
    # Overlap genotypes with VEPs
    veps_genotypes_intersection = data_loaders.intersect_ids(hdf5_loader.get_vids(), plink_loader.get_vids())

    # Update respective instances
    plink_loader.update_variants(veps_genotypes_intersection)
    hdf5_loader.update_variants(veps_genotypes_intersection)
    plink_loader.update_individuals(genotypes_covariates_GRM_intersection)
    covariate_loader_csv.update_individuals(genotypes_covariates_GRM_intersection)
    background_kernel.update_individuals(genotypes_covariates_GRM_intersection)

    background_kernel.compute_background_kernel()
    print('nb_SNVs_unf: {}, nb_SNVs_f: {}'.format(background_kernel.nb_SNVs_unf, background_kernel.nb_SNVs_f))
    print('sum(diag(K)): {}'.format(np.diag(background_kernel.K0).sum()))

    Y, X = covariate_loader_csv.get_one_hot_covariates_and_phenotype(test_type='2K')
    null_model = scoretest.Scoretest2K(Y, X, background_kernel.K0, background_kernel.G0)
    results = pd.DataFrame(columns=['name', 'chrom', 'start', 'end', 'p_value', 'n_SNVs', 'time'])

    for region in ucsc_region_loader:
        t_test_gene_start = time.time()
        temp_genotypes_info_dict = region
        temp_genotypes, temp_vids = plink_loader.genotypes_by_region(region, return_pos=False)
        if temp_genotypes is None:
            continue

        G, temp_vids = data_loaders.VariantLoader.preprocess_genotypes(temp_genotypes, temp_vids, impute_mean=True,
                                                                       normalize=False, invert_encoding=True,
                                                                       recode_maf=False)
        if G is None:
            continue

        V = hdf5_loader.anno_by_id(temp_vids)

        GV = kernels.diffscore_max(G, V, False)
        temp_p_value = null_model.pv_alt_model(GV)
        temp_genotypes_info_dict['p_value'] = temp_p_value
        temp_genotypes_info_dict['n_SNVs'] = G.shape[1]
        t_test_gene_end = time.time()
        temp_time = float(t_test_gene_end - t_test_gene_start)
        temp_genotypes_info_dict['time'] = temp_time
        results = results.append(temp_genotypes_info_dict, ignore_index=True)

    # results.to_csv('./test_full_rank_continuous_2K.csv')
    reference_result = pd.read_csv(data_path + 'reference_results/test_full_rank_continuous_2K_computed.csv', index_col=0)
    print(data_path + 'reference_results/test_full_rank_continuous_2K.csv')

    print('expected result:')
    print(reference_result)
    print('actual result:')
    print(results)
    print('p-value corrcoef:')
    print(np.corrcoef(reference_result['p_value'], results['p_value']))

    # results.to_csv('test_full_rank_continuous_2K_computed.csv')

    assert np.all((np.isclose(reference_result['p_value'], results['p_value']))), 'The last change in code changes the result!!'


def test_low_rank_continuous_low_rank_bg():
    # low rank bg kernel
    # imports
    import pkg_resources
    import time

    import numpy as np
    import pandas as pd

    from seak import construct_background_kernel
    from seak import data_loaders
    from seak import kernels
    from seak import scoretest
    from seak.data_loaders import intersect_ids

    data_path = pkg_resources.resource_filename('seak', 'data/')

    # Path to veps
    path_to_VEP_bed = data_path + "dummy_veps.bed"
    path_to_VEP_hdf5 = data_path + "dummy_veps.hdf5"

    # Path to genotypes
    path_to_covariates = data_path + "dummy_covariates_fixed.csv"
    path_to_plink_files_with_prefix = data_path + "low_rank_continuous.bed"

    # Path to regions
    path_to_reference_genes_bed = data_path + "dummy_regions.bed"

    # Path to background kernel
    path_to_plink_bg_kernel = data_path + "full_rank_background_kernel"

    # Load data
    # VEPs
    hdf5_loader = data_loaders.Hdf5Loader(path_to_vep_bed=path_to_VEP_bed, path_to_vep_hdf5=path_to_VEP_hdf5,
                                          hdf5_key='diffscore')

    # Genotypes
    plink_loader = data_loaders.VariantLoaderSnpReader(path_to_plink_files_with_prefix)

    # Genes
    ucsc_region_loader = data_loaders.BEDRegionLoader(path_to_regions_UCSC_BED=path_to_reference_genes_bed,
                                                      chrom_to_load=1, drop_non_numeric_chromosomes=True)

    # Covariates
    covariate_loader_csv = data_loaders.CovariatesLoaderCSV(phenotype_of_interest='pheno_low_rank_continuous',
                                                            path_to_covariates=path_to_covariates,
                                                            covariate_column_names=['cov1', 'cov2'])

    # Background_kernel
    # background_kernel = construct_background_kernel.GRMLoader(path_to_plink_bg_kernel, 200, 1, forcelowrank=True)
    background_kernel = construct_background_kernel.GRMLoaderSnpReader(path_to_plink_bg_kernel + '.bed', 200, '1')

    # Overlap individuals: genotypes and covariates
    genotypes_covariates_GRM_intersection = intersect_ids(plink_loader.get_iids(), intersect_ids(covariate_loader_csv.get_iids(), background_kernel.get_iids()))
    print(genotypes_covariates_GRM_intersection)
    # Overlap genotypes with VEPs
    veps_genotypes_intersection = intersect_ids(hdf5_loader.get_vids(), plink_loader.get_vids())
    # Update respective instances
    plink_loader.update_variants(veps_genotypes_intersection)
    hdf5_loader.update_variants(veps_genotypes_intersection)
    plink_loader.update_individuals(genotypes_covariates_GRM_intersection)
    covariate_loader_csv.update_individuals(genotypes_covariates_GRM_intersection)
    background_kernel.update_individuals(genotypes_covariates_GRM_intersection)

    background_kernel.compute_background_kernel()
    # np.save('/Users/piarautentrauch/github_repos/seak/src/seak/data/precomputed_background_G0.npy', background_kernel.G0)

    Y, X = covariate_loader_csv.get_one_hot_covariates_and_phenotype(test_type='2K')
    null_model = scoretest.Scoretest2K(Y, X, background_kernel.K0, background_kernel.G0)
    results = pd.DataFrame(columns=['name', 'chrom', 'start', 'end', 'p_value', 'n_SNVs', 'time'])

    print('nb_SNVs_unf: {}, nb_SNVs_f: {}'.format(background_kernel.nb_SNVs_unf, background_kernel.nb_SNVs_f))
    print('sum(diag(K)): {}'.format(np.diag(background_kernel.K0).sum()))

    for region in ucsc_region_loader:
        t_test_gene_start = time.time()
        temp_genotypes_info_dict = region
        temp_genotypes, temp_vids = plink_loader.genotypes_by_region(region)

        if temp_genotypes is None:
            continue

        G, temp_vids = data_loaders.VariantLoader.preprocess_genotypes(temp_genotypes, temp_vids, impute_mean=True,
                                                                       normalize=False, invert_encoding=True,
                                                                       recode_maf=False)
        if G is None:
            continue

        V = hdf5_loader.anno_by_id(temp_vids)

        GV = kernels.diffscore_max(G, V, False)
        temp_p_value = null_model.pv_alt_model(GV)
        temp_genotypes_info_dict['p_value'] = temp_p_value
        temp_genotypes_info_dict['n_SNVs'] = G.shape[1]
        t_test_gene_end = time.time()
        temp_time = float(t_test_gene_end - t_test_gene_start)
        temp_genotypes_info_dict['time'] = temp_time
        results = results.append(temp_genotypes_info_dict, ignore_index=True)

    # results.to_csv('./test_low_rank_continuous_2K_low_rank_bg.csv')
    reference_result = pd.read_csv(data_path + 'reference_results/test_low_rank_continuous_2K_low_rank_bg_computed.csv', index_col=0)

    print('expected result:')
    print(reference_result)
    print('actual result:')
    print(results)
    print('p-value corrcoef:')
    print(np.corrcoef(reference_result['p_value'], results['p_value']))

    # results.to_csv('test_low_rank_continuous_2K_low_rank_bg_computed.csv')

    assert np.all((np.isclose(reference_result['p_value'],
                              results['p_value']))), 'The last change in code changes the result!!'



def test_full_rank_continuous_automatic_intersection():
    # full rank bg kernel
    # imports
    import pkg_resources
    import time

    import numpy as np
    import pandas as pd

    from seak import construct_background_kernel
    from seak import data_loaders
    from seak import kernels
    from seak import scoretest

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
    # VEPs
    hdf5_loader = data_loaders.Hdf5Loader(path_to_vep_bed=path_to_VEP_bed, path_to_vep_hdf5=path_to_VEP_hdf5,
                                          hdf5_key='diffscore')

    # Genotypes
    plink_loader = data_loaders.VariantLoaderSnpReader(path_to_plink_files_with_prefix+'.bed')

    # Genes
    ucsc_region_loader = data_loaders.BEDRegionLoader(path_to_regions_UCSC_BED=path_to_reference_genes_bed,
                                                      chrom_to_load=1, drop_non_numeric_chromosomes=True)

    # Covariates
    covariate_loader_csv = data_loaders.CovariatesLoaderCSV(phenotype_of_interest='pheno_full_rank_continuous',
                                                            path_to_covariates=path_to_covariates,
                                                            covariate_column_names=['cov1', 'cov2'])

    # Background_kernel
    background_kernel = construct_background_kernel.GRMLoaderSnpReader(path_to_plink_bg_kernel+'.bed', 200, '1')

    # Intersect and update data sets
    data_loaders.intersect_and_update_datasets('2K', plink_loader, covariate_loader_csv, hdf5_loader, background_kernel)

    Y, X = covariate_loader_csv.get_one_hot_covariates_and_phenotype(test_type='2K')
    null_model = scoretest.Scoretest2K(Y, X, background_kernel.K0, background_kernel.G0)
    results = pd.DataFrame(columns=['name', 'chrom', 'start', 'end', 'p_value', 'n_SNVs', 'time'])

    for region in ucsc_region_loader:
        t_test_gene_start = time.time()
        temp_genotypes_info_dict = region
        temp_genotypes, temp_vids = plink_loader.genotypes_by_region(region)
        if temp_genotypes is None:
            continue
        G, temp_vids = data_loaders.VariantLoader.preprocess_genotypes(temp_genotypes, temp_vids, impute_mean=True,
                                                                       normalize=False, invert_encoding=True,
                                                                       recode_maf=False)
        if G is None:
            continue

        V = hdf5_loader.anno_by_id(temp_vids)

        GV = kernels.diffscore_max(G, V, False)
        temp_p_value = null_model.pv_alt_model(GV)
        temp_genotypes_info_dict['p_value'] = temp_p_value
        temp_genotypes_info_dict['n_SNVs'] = G.shape[1]
        t_test_gene_end = time.time()
        temp_time = float(t_test_gene_end - t_test_gene_start)
        temp_genotypes_info_dict['time'] = temp_time
        results = results.append(temp_genotypes_info_dict, ignore_index=True)

    # results.to_csv('./test_full_rank_continuous_2K.csv')
    print(results)
    reference_result = pd.read_csv(data_path + 'reference_results/test_full_rank_continuous_2K.csv', index_col=0)
    print(np.corrcoef(reference_result['p_value'], results['p_value']))
    print(np.all((np.isclose(reference_result['p_value'], results['p_value']))))

    # results.to_csv('test_full_rank_continuous_2K.csv')

    assert np.all((np.isclose(reference_result['p_value'], results['p_value']))), 'The last change in code changes the result!!'
