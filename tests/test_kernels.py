import numpy as np

from seak import kernels


def test_single_column_kernel():
    V = np.asarray([[0, 1, 2, ], [0, 1, 2], [0, 1, 2]])
    G = np.asarray([[0, 1, 2], [2, 0, 1], [1, 1, 1], [0, 1, 2], [0, 0, 2]]) * 100.
    i = 2
    result = kernels.single_column_kernel(i, False)(G, V)
    expected_result = np.asarray([[0, 200, 400],
                                  [400, 0, 200],
                                  [200, 200, 200],
                                  [0, 200, 400],
                                  [0, 0, 400]])

    assert np.all(np.isclose(result, expected_result))


def test_diffscore_max_kernel():
    V = np.asarray([[0, 1, 2, ], [0, 1, 2], [0, 1, 2]])
    G = np.asarray([[0, 1, 2], [2, 0, 1], [1, 1, 1], [0, 1, 2], [0, 0, 2]]) * 100.
    result = kernels.diffscore_max(G, V, False)
    expected_result = np.asarray([[0, 200, 400],
                                  [400, 0, 200],
                                  [200, 200, 200],
                                  [0, 200, 400],
                                  [0, 0, 400]])

    assert np.all(np.isclose(result, expected_result))


def test_linear_kernel():
    V = np.asarray([[0, 1, 2, ], [0, 1, 2], [0, 1, 2]])
    G = np.asarray([[0, 1, 2], [2, 0, 1], [1, 1, 1], [0, 1, 2], [0, 0, 2]])
    result = kernels.linear(G, V)
    assert np.all(np.isclose(result, G))
