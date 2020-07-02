========
Overview
========

**Seak**, which stands for **se**\ quence **a**\ nnotations in **k**\ ernel-based tests, is an open-source Python
software package for performing set-based genotype-phenotype association tests. It allows for the flexible incorporation
of prior knowledge, such as variant effect predictions, or other annotations, into variant association tests via kernel
functions.  The mathematical implementation of these tests is based on
**FaST-LMM-** (Listgarten et al., 2013; Lippert et al., 2014). It can correct for population and family structure as well as
cryptic relatedness in a two random effects model (**seak.scoretest.Scoretest2K**) for continuous phenotypes.
Furthermore, it also implements a one random effect model with a linear (**seak.scoretest.ScoretestNoK**)
and a logistic (**seak.scoretest.ScoretestLogit**) link function. Associations are tested in variance component score tests.

Seak provides interfaces for all data loading functionalities (**seak.data_loaders**) in order to maximize flexibility. This way users can easily adapt the package to the input data types of their choice.

* Free software: Apache Software License 2.0

Installation
============
The installation of **seak** requires Python 3.7+ and the packages `numpy <https://pypi.org/project/numpy/>`_ and `cython <https://pypi.org/project/Cython/>`_. All other dependencies are installed automatically when installing the package.

Clone the repository. Then, on the command line::

    pip install -e ./seak


Documentation
=============
You can find a full documentation of **seak** including an API reference on https://seak.readthedocs.io/.

References
=============

The score-based association tests implemented in **seak.scoretest** are adapted from `FaST-LMM <https://github.com/fastlmm/FaST-LMM>`_.

Lippert, C., Xiang, J., Horta, D., Widmer, C., Kadie, C., Heckerman, D., & Listgarten, J. (2014). Greater power and computational efficiency for kernel-based association testing of sets of genetic variants. *Bioinformatics*, *30*(22), 3206–3214. https://doi.org/10.1093/bioinformatics/btu504

Listgarten, J., Lippert, C., Kang, E. Y., Xiang, J., Kadie, C. M., & Heckerman, D. (2013). A powerful and efficient set test for genetic markers that handles confounders. *Bioinformatics*, *29*(12), 1526–1533. https://doi.org/10.1093/bioinformatics/btt177

