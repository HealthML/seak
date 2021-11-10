========
Overview
========

:mod:`seak`, which stands for **se**\ quence **a**\ nnotations in **k**\ ernel-based tests, is an open-source Python
software package for performing set-based genotype-phenotype association tests. It allows for the flexible incorporation
of prior knowledge, such as variant effect predictions or other annotations, into these association tests.

The mathematical implementation of these tests is based on :mod:`FaST-LMM-Set` :cite:`Listgarten2013` :cite:`Lippert2014`.

Two types of association tests are available, namely to score test (:mod:`seak.scoretest`) and the likelihood ratio test (LRT, :mod:`seak.lrt`).
While the score test is computationally more efficient, the LRT has potentially higher power :cite:`Listgarten2013`.

The score test is available for continuous (:class:`seak.scoretest.ScoretestLogit`) and binary phenotypes (:class:`seak.scoretest.ScoretestNoK`),
and can correct for (cryptic) relatedness and population stratification using a two random effects model (:class:`seak.scoretest.Scoretest2K`, continuous phenotypes only).
P-values are calculated using either Davie's exact method :cite:`Davies1980`, or saddle point approximation :cite:`Kuonen1999` (as implemented in the `skatMeta R-package <https://github.com/cran/skatMeta>`_).

The LRT is implemented for continuous phenotypes (:class:`seak.lrt.LRTnoK`).  LRT test statistics can be sampled using the fast implementations described in :cite:`Scheipl2008`. This class also provides support for gene-specific hypothesis testing, using a combination of the approaches outlined in :cite:`zhou2016boosting`, :cite:`Lippert2014` and :cite:`Listgarten2013`.

The module :mod:`seak.cct` implements the Cauchy Combination Test (CCT, :cite:`liu2020cauchy`), as implemented in the `STAAR R-package <https://github.com/xihaoli/STAAR>`_.

If you use the functions in the modules listed above, please also cite the original authors.

Seak provides interfaces for data loading functionalities (:mod:`seak.data_loaders`) in order to maximize flexibility. This way users can easily adapt the package to the input data types of their choice.

* Free software: Apache Software License 2.0

Installation
============
The installation of :mod:`seak` requires Python 3.7+ and the packages `numpy <https://pypi.org/project/numpy/>`_ and `cython <https://pypi.org/project/Cython/>`_. All other dependencies are installed automatically when installing the package.

Clone the repository. Then, on the command line::

    pip install -e ./seak


Documentation
=============
For a reference documenting all public modules included in :mod:`seak` meant for general usage see:
:ref:`API reference`.

Tutorial
========
A small example illustrating how to perform score- and likelihood ratio tests is shown in: :ref:`Tutorial`.

A pipeline using :mod:`seak` to perform functionally informed association tests on UK Biobank data is available `here <https://github.com/HealthML/faatpipe>`_

References
=============

For more information on FaST-LMM visit `FaST-LMM <https://github.com/fastlmm/FaST-LMM>`_.

.. bibliography:: references.bib
    :style: unsrt
