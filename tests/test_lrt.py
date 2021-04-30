

def test_lrt():

    import numpy as np

    from seak.lrt import LRTnoK
    from numpy import isclose

    np.random.seed(1)

    # random covariates
    X = np.random.randn(1000,10)

    # random genotypes
    G_1 = np.random.binomial(1,0.01,size=(1000,10))

    # part of Y explained by co variates
    Y = X.dot(np.random.randn(10,1)*0.05)

    # part of Y explained by G_1 (3/10 causal SNPs)
    Y += G_1.dot(np.array([1. if i > 7 else 0. for i in range(10)])[:,np.newaxis]*0.5)

    # part of Y explained by random noise
    Y += np.random.randn(1000,1)

    lrt = LRTnoK(X, Y)

    # print(lrt.model0)

    assert isclose(lrt.model0['nLL'], 1385.7447556588409), 'Null model nLL changed. should be ~1385.7447556588409, is {}. Check LRTnoK.__init__'.format(lrt.model0['nLL'])

    altmodel = lrt.altmodel(G_1)

    # print(altmodel)

    assert isclose(altmodel['nLL'], 1385.7118679498765), 'Alt model nLL changed. should be ~1385.7118679498765, is {}. Check LRTnoK.altmodel()'.format(altmodel['nLL'])
    assert isclose(altmodel['stat'], 0.06577541792876218), 'Alt model LRT test statistic changed. should be ~0.06577541792876218, is {}. Check LRTnoK.altmodel()'.format(altmodel['stat'])

    sims = lrt.pv_sim(nsim=1000, seed=21)

    # print(sims['pv'])

    assert sims['pv'] == 0.353, 'pv_sim() output changed. should be 0.353, is {}. Check LRTnoK.pv_sim()'.format(sims['pv'])







