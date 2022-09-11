========
Overview
========

**Seak**, which stands for **se**\ quence **a**\ nnotations in **k**\ ernel-based tests, is an open-source Python
software package for performing set-based genotype-phenotype association tests. It allows for the flexible incorporation
of prior knowledge, such as variant effect predictions, or other annotations, into variant association tests via kernel
functions.  The mathematical implementation of these tests is based on
**FaST-LMM-** (Listgarten et al., 2013; Lippert et al., 2014). Fast simulation of LRT test statistics is based on RLRsim (Scheipl et al., 2008)

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

`seak <https://www.nature.com/articles/s41467-022-32864-2>`_

Check out the link above our Nature Communications paper using `seak`

`FaST-LMM <https://github.com/fastlmm/FaST-LMM>`_.

Lippert, Christoph, et al. "Greater power and computational efficiency for kernel-based association testing of sets of genetic variants." *Bioinformatics* 30.22 (2014): 3206-3214.

Listgarten, Jennifer, et al. "A powerful and efficient set test for genetic markers that handles confounders." *Bioinformatics* 29.12 (2013): 1526-1533.

`RLRsim <https://cran.r-project.org/web/packages/RLRsim/RLRsim.pdf>`_

Scheipl, Fabian, Sonja Greven, and Helmut Kuechenhoff. "Size and power of tests for a zero random effect variance or polynomial regression in additive and linear mixed models." *Computational statistics & data analysis* 52.7 (2008): 3283-3299.

A more complete list of references can be found on `readthedocs <https://seak.readthedocs.io/en/latest/readme.html>`_ .
